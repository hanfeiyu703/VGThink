from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from peft import LoraConfig, PeftModel, get_peft_model
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from e2egmner.modeling import (
    GroundedQwen25VLForConditionalGeneration,
    checkpoint_has_eceb,
    load_eceb_checkpoint,
    load_eceb_config,
    save_eceb_checkpoint,
    unwrap_grounded_qwen,
)
from e2egmner.training import EntityPatchInteractorConfig


def _configs() -> tuple[Qwen2_5_VLConfig, EntityPatchInteractorConfig]:
    text_config = {
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": True,
        "rope_parameters": {
            "type": "mrope",
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
            "rope_theta": 1_000_000.0,
        },
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 16,
        "out_hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 1,
        "spatial_merge_size": 1,
        "window_size": 4,
        "fullatt_block_indexes": [0],
    }
    return (
        Qwen2_5_VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=120,
            video_token_id=121,
            vision_start_token_id=118,
            vision_end_token_id=119,
        ),
        EntityPatchInteractorConfig(
            hidden_size=16,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            spatial_merge_size=1,
            max_entities_per_sample=4,
        ),
    )


def _model() -> GroundedQwen25VLForConditionalGeneration:
    torch.manual_seed(19)
    model_config, bridge_config = _configs()
    return GroundedQwen25VLForConditionalGeneration(model_config, eceb_config=bridge_config).eval()


def _batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[118, 120, 120, 120, 120, 119, 10, 11, 12]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.linspace(-1.0, 1.0, steps=4 * 12).reshape(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "tweet_token_mask": torch.tensor(
            [[False, False, False, False, False, False, True, True, True]]
        ),
    }


def _target() -> list[list[dict]]:
    return [
        [
            {
                "visible": True,
                "loss_enabled": True,
                "mention_positions": [6, 7],
                "patch_weights": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            }
        ]
    ]


class GroundedQwen25VLTest(unittest.TestCase):
    def test_stage1_adapter_enters_zero_gated_eceb_without_changing_logits(self) -> None:
        model_config, bridge_config = _configs()
        target_modules = (
            r"^(model\.language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))$"
        )
        with tempfile.TemporaryDirectory() as directory:
            base_dir = str(Path(directory) / "base")
            adapter_dir = str(Path(directory) / "adapter")
            torch.manual_seed(37)
            upstream = Qwen2_5_VLForConditionalGeneration(model_config).eval()
            upstream.save_pretrained(base_dir)
            stage1 = get_peft_model(
                upstream,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=2,
                    lora_alpha=4,
                    target_modules=target_modules,
                ),
            )
            with torch.no_grad():
                for name, parameter in stage1.named_parameters():
                    if "lora_B" in name:
                        parameter.fill_(0.01)
            stage1.save_pretrained(adapter_dir)

            native = PeftModel.from_pretrained(
                Qwen2_5_VLForConditionalGeneration.from_pretrained(base_dir),
                adapter_dir,
            ).eval()
            grounded_base = GroundedQwen25VLForConditionalGeneration.from_pretrained(
                base_dir, eceb_config=bridge_config
            )
            grounded = PeftModel.from_pretrained(grounded_base, adapter_dir).eval()
            self.assertEqual(
                unwrap_grounded_qwen(grounded).entity_patch_interactor.alpha.item(), 0.0
            )
            batch = _batch()
            native_inputs = {
                key: value for key, value in batch.items() if key != "tweet_token_mask"
            }
            with torch.no_grad():
                native_logits = native(**native_inputs).logits
                grounded_logits = grounded(**batch).logits
            self.assertTrue(torch.equal(native_logits, grounded_logits))

    def test_zero_gate_is_identity_and_generation_uses_nonzero_bridge(self) -> None:
        model = _model()
        batch = _batch()
        native_inputs = {key: value for key, value in batch.items() if key != "tweet_token_mask"}
        with torch.no_grad():
            baseline = model(**native_inputs)
            zero_gate = model(**batch)
            model.entity_patch_interactor.alpha.fill_(0.25)
            bridged = model(**batch)
            generated = model.generate(**batch, max_new_tokens=2, do_sample=False, use_cache=True)
        self.assertTrue(torch.equal(zero_gate.logits, baseline.logits))
        self.assertFalse(torch.equal(bridged.logits, baseline.logits))
        self.assertTrue(
            torch.equal(generated[:, : batch["input_ids"].shape[1]], batch["input_ids"])
        )

    def test_gold_regions_affect_only_auxiliary_loss(self) -> None:
        model = _model()
        batch = _batch()
        with torch.no_grad():
            model.entity_patch_interactor.alpha.fill_(0.7)
            without = model(**batch)
            with_target = model(**batch, eceb_targets=_target())
        self.assertTrue(torch.equal(with_target.logits, without.logits))
        self.assertEqual(without.eceb_loss.item(), 0.0)
        self.assertGreater(with_target.eceb_loss.item(), 0.0)

        with self.assertRaisesRegex(ValueError, "return_dict=True"):
            model(**batch, return_dict=False)

    def test_peft_and_eceb_sidecars_round_trip(self) -> None:
        wrapped = get_peft_model(
            _model(),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        with torch.no_grad():
            unwrap_grounded_qwen(wrapped).entity_patch_interactor.alpha.fill_(0.375)

        with tempfile.TemporaryDirectory() as directory:
            wrapped.save_pretrained(directory)
            save_eceb_checkpoint(wrapped, directory)
            self.assertTrue(checkpoint_has_eceb(directory))

            model_config, _ = _configs()
            base = GroundedQwen25VLForConditionalGeneration(
                model_config, eceb_config=load_eceb_config(directory)
            )
            restored = PeftModel.from_pretrained(base, directory, is_trainable=False).eval()
            load_eceb_checkpoint(restored, directory)
            alpha = unwrap_grounded_qwen(restored).entity_patch_interactor.alpha
            self.assertAlmostEqual(alpha.item(), 0.375, places=6)
            with torch.no_grad():
                generated = restored.generate(
                    **_batch(), max_new_tokens=1, do_sample=False, use_cache=True
                )
            self.assertGreater(generated.shape[1], _batch()["input_ids"].shape[1])

    def test_missing_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bridge_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "entity_patch_interactor"):
                checkpoint_has_eceb(directory)


if __name__ == "__main__":
    unittest.main()
