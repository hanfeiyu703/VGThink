"""Qwen2.5-VL wrapper with an inference-active entity/patch evidence bridge.

The bridge runs only when image pixels and a tweet-token mask are present.  In
cached generation that is the prefill call; the resulting prompt states then
remain available through the normal language-model KV cache.  Gold entity and
region annotations are accepted only by the auxiliary loss and never alter the
fusion path.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any, Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)

from ..training.entity_patch_interactor import (
    EntityPatchInteractor,
    EntityPatchInteractorConfig,
)


ECEB_CONFIG_NAME = "bridge_config.json"
ECEB_WEIGHTS_NAME = "entity_patch_interactor.safetensors"


class GroundedQwen25VLOutput(Qwen2_5_VLCausalLMOutputWithPast):
    """Standard Qwen output plus differentiable ECEB supervision."""

    def __init__(self, *args, eceb_loss=None, eceb_stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eceb_loss = eceb_loss
        self.eceb_stats = eceb_stats


def _config_to_dict(config: EntityPatchInteractorConfig) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return asdict(config)


def load_eceb_config(checkpoint_dir: str) -> EntityPatchInteractorConfig:
    config_path = os.path.join(checkpoint_dir, ECEB_CONFIG_NAME)
    weights_path = os.path.join(checkpoint_dir, ECEB_WEIGHTS_NAME)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"ECEB config not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"ECEB checkpoint is incomplete: {config_path} exists but {weights_path} is missing"
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EntityPatchInteractorConfig(**payload)


def checkpoint_has_eceb(checkpoint_dir: Optional[str]) -> bool:
    if not checkpoint_dir:
        return False
    config_path = os.path.join(checkpoint_dir, ECEB_CONFIG_NAME)
    weights_path = os.path.join(checkpoint_dir, ECEB_WEIGHTS_NAME)
    if os.path.exists(config_path) != os.path.exists(weights_path):
        missing = weights_path if os.path.exists(config_path) else config_path
        raise FileNotFoundError(f"Incomplete ECEB checkpoint; missing {missing}")
    return os.path.isfile(config_path)


def unwrap_grounded_qwen(model) -> "GroundedQwen25VLForConditionalGeneration":
    """Return the custom base model through an optional PEFT wrapper."""

    candidate = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not isinstance(candidate, GroundedQwen25VLForConditionalGeneration):
        raise TypeError(
            f"Expected GroundedQwen25VLForConditionalGeneration, got {type(candidate).__name__}"
        )
    return candidate


def save_eceb_checkpoint(model, checkpoint_dir: str) -> None:
    grounded = unwrap_grounded_qwen(model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(checkpoint_dir, ECEB_CONFIG_NAME)
    weights_path = os.path.join(checkpoint_dir, ECEB_WEIGHTS_NAME)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(_config_to_dict(grounded.eceb_config), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in grounded.entity_patch_interactor.state_dict().items()
    }
    save_file(state, weights_path)


def load_eceb_checkpoint(model, checkpoint_dir: str, *, strict: bool = True) -> None:
    grounded = unwrap_grounded_qwen(model)
    saved_config = load_eceb_config(checkpoint_dir)
    if _config_to_dict(saved_config) != _config_to_dict(grounded.eceb_config):
        raise ValueError(
            "ECEB config mismatch between instantiated model and checkpoint: "
            f"model={_config_to_dict(grounded.eceb_config)} "
            f"checkpoint={_config_to_dict(saved_config)}"
        )
    state = load_file(os.path.join(checkpoint_dir, ECEB_WEIGHTS_NAME), device="cpu")
    grounded.entity_patch_interactor.load_state_dict(state, strict=strict)
    embedding = grounded.get_input_embeddings().weight
    grounded.entity_patch_interactor.to(device=embedding.device, dtype=embedding.dtype)


class GroundedQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with a tweet-token-to-image-patch bridge before the LM."""

    # Transformers 5.x registers these two legacy checkpoint renamings by the
    # *exact* upstream class name.  Subclassing would otherwise bypass them and
    # silently leave the entire language model randomly initialized when
    # loading a Qwen 2.5-VL checkpoint saved by Transformers 4.x.
    _UPSTREAM_KEY_MAPPING = {
        r"^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("key_mapping", cls._UPSTREAM_KEY_MAPPING)
        return super().from_pretrained(*args, **kwargs)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize Qwen modules and ECEB's direct parameters.

        Transformers initializes checkpoint-missing modules by recursively
        calling ``_init_weights``.  Its generic implementation covers Linear,
        MultiheadAttention and normalization modules, but intentionally cannot
        initialize arbitrary parameters registered directly on a module.  The
        ECEB ``alpha`` gate and learned None embedding are such parameters.  If
        a native Qwen checkpoint is loaded into this subclass, leaving them to
        the generic missing-key path materializes uninitialized memory (and can
        make the nominally zero-gated bridge non-zero or NaN).
        """

        super()._init_weights(module)
        if isinstance(module, EntityPatchInteractor):
            torch.nn.init.normal_(
                module.none_embedding,
                mean=0.0,
                std=float(module.config.hidden_dim) ** -0.5,
            )
            torch.nn.init.zeros_(module.alpha)

    def __init__(self, config, eceb_config: Optional[EntityPatchInteractorConfig | dict] = None):
        super().__init__(config)
        if eceb_config is None:
            hidden_size = int(config.text_config.hidden_size)
            eceb_config = EntityPatchInteractorConfig(hidden_size=hidden_size)
        elif isinstance(eceb_config, dict):
            eceb_config = EntityPatchInteractorConfig(**eceb_config)
        self.eceb_config = eceb_config
        self.entity_patch_interactor = EntityPatchInteractor(eceb_config)
        self.entity_patch_interactor.apply(self._init_weights)

    @staticmethod
    def _normalize_tweet_mask(
        tweet_token_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if tweet_token_mask.ndim != 2 or tweet_token_mask.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "tweet_token_mask must have shape [batch, sequence], got "
                f"{tuple(tweet_token_mask.shape)} for input_ids {tuple(input_ids.shape)}"
            )
        mask = tweet_token_mask.to(device=input_ids.device, dtype=torch.bool)
        sequence_delta = int(input_ids.shape[1]) - int(mask.shape[1])
        if sequence_delta < 0:
            raise ValueError(
                f"tweet_token_mask is longer than input_ids: {mask.shape[1]} > {input_ids.shape[1]}"
            )
        if sequence_delta:
            mask = torch.nn.functional.pad(mask, (0, sequence_delta), value=False)
        return mask

    def _bridge_prefill(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        tweet_token_mask: torch.Tensor,
        eceb_targets: Optional[list[list[dict[str, Any]]]],
        vision_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if input_ids is None:
            raise ValueError("ECEB prefill requires input_ids")
        if image_grid_thw is None:
            raise ValueError("ECEB prefill requires image_grid_thw")
        if image_grid_thw.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "ECEB v1 supports exactly one image per sample: "
                f"batch={input_ids.shape[0]} image_grids={image_grid_thw.shape[0]}"
            )

        inputs_embeds = self.get_input_embeddings()(input_ids)
        vision_outputs = self.model.get_image_features(
            pixel_values,
            image_grid_thw,
            **vision_kwargs,
        )
        image_embeds = torch.cat(vision_outputs.pooler_output, dim=0).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        token_image_mask = image_mask.squeeze(-1)
        normalized_tweet_mask = self._normalize_tweet_mask(tweet_token_mask, input_ids)
        if bool((normalized_tweet_mask & token_image_mask).any()):
            raise ValueError("tweet_token_mask must not include image placeholder tokens")

        return self.entity_patch_interactor(
            inputs_embeds=inputs_embeds,
            tweet_token_mask=normalized_tweet_mask,
            image_token_mask=token_image_mask,
            image_grid_thw=image_grid_thw,
            targets=eceb_targets,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        tweet_token_mask: Optional[torch.Tensor] = None,
        eceb_targets: Optional[list[list[dict[str, Any]]]] = None,
        **kwargs,
    ):
        should_bridge = (
            pixel_values is not None and tweet_token_mask is not None and inputs_embeds is None
        )
        if not should_bridge:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                second_per_grid_ts=second_per_grid_ts,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        if kwargs.get("return_dict") is False:
            raise ValueError("ECEB requires return_dict=True")
        if pixel_values_videos is not None:
            raise ValueError("ECEB v1 supports image inputs only, not video inputs")

        fused_embeds, eceb_loss, eceb_stats = self._bridge_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            tweet_token_mask=tweet_token_mask,
            eceb_targets=eceb_targets,
            vision_kwargs=kwargs,
        )
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeds,
            labels=labels,
            use_cache=use_cache,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            second_per_grid_ts=second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return GroundedQwen25VLOutput(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            rope_deltas=output.rope_deltas,
            eceb_loss=eceb_loss,
            eceb_stats=eceb_stats,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        *args,
        tweet_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
        if tweet_token_mask is not None and model_inputs.get("pixel_values") is not None:
            model_inputs["tweet_token_mask"] = tweet_token_mask
        return model_inputs
