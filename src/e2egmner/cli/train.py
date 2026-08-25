"""Stage-I/II LoRA training for VGThink."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from ..data import Qwen25VLSFTCollator, TwitterGroundedMNERJsonl
from ..engine import run_dev_eval
from ..modeling import (
    GroundedQwen25VLForConditionalGeneration,
    checkpoint_has_eceb,
    load_eceb_checkpoint,
    load_eceb_config,
    save_eceb_checkpoint,
)
from ..training import EntityPatchInteractorConfig
from ..utils import get_first_param_device, pick_vision_inputs, seed_everything, supports_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage I SFT or Stage II ECEB alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--attn_impl", choices=["sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default=(
            r"^(model\.language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))$"
        ),
    )
    parser.add_argument("--use_eceb", action="store_true")
    parser.add_argument("--eceb_hidden_dim", type=int, default=256)
    parser.add_argument("--eceb_num_heads", type=int, default=8)
    parser.add_argument("--eceb_num_layers", type=int, default=2)
    parser.add_argument("--eceb_dropout", type=float, default=0.10)
    parser.add_argument("--eceb_weight", type=float, default=0.05)
    parser.add_argument("--eceb_warmup_ratio", type=float, default=0.10)
    parser.add_argument("--eceb_lr", type=float, default=1e-4)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--dev_batch_size", type=int, default=2)
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_iou", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    return parser.parse_args()


def _precision(args: argparse.Namespace, is_cuda: bool):
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    if not args.bf16 and not args.fp16 and is_cuda:
        args.bf16 = supports_bf16()
        args.fp16 = not args.bf16
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    return dtype, is_cuda, torch.bfloat16 if args.bf16 else torch.float16


def _hidden_size(config) -> int:
    text_config = getattr(config, "text_config", config)
    value = getattr(text_config, "hidden_size", None)
    if value is None:
        raise ValueError("Cannot resolve language hidden size from model config")
    return int(value)


def _load_system_prompt(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8").strip() if path else None


def _subset(dataset, maximum: int | None):
    if maximum is None:
        return dataset
    if maximum <= 0:
        raise ValueError("Sample limits must be positive")
    return Subset(dataset, range(min(maximum, len(dataset))))


def _loader(
    path: str,
    *,
    image_root: str | None,
    collator,
    batch_size: int,
    shuffle: bool,
    workers: int,
    maximum: int | None,
    pin_memory: bool,
):
    dataset = TwitterGroundedMNERJsonl(path, image_root=image_root)
    dataset = _subset(dataset, maximum)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        collate_fn=collator,
    )


def _causal_loss(
    logits: torch.Tensor, labels: torch.Tensor, token_weights: torch.Tensor
) -> torch.Tensor:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = token_weights[:, 1:].to(shifted_logits.dtype)
    valid = shifted_labels.ne(-100)
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    denominator = shifted_weights[valid].sum().clamp_min(1.0)
    return (losses * shifted_weights * valid.to(losses.dtype)).sum() / denominator


def _save(model, processor, path: str, use_eceb: bool) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    if use_eceb:
        save_eceb_checkpoint(model, path)


def _build_model(args: argparse.Namespace, dtype: torch.dtype):
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    adapter_has_eceb = bool(args.adapter_path and checkpoint_has_eceb(args.adapter_path))
    if args.use_eceb:
        if adapter_has_eceb:
            eceb_config = load_eceb_config(args.adapter_path)
        else:
            base_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
            eceb_config = EntityPatchInteractorConfig(
                hidden_size=_hidden_size(base_config),
                hidden_dim=args.eceb_hidden_dim,
                num_heads=args.eceb_num_heads,
                num_layers=args.eceb_num_layers,
                dropout=args.eceb_dropout,
            )
        base = GroundedQwen25VLForConditionalGeneration.from_pretrained(
            args.model_name, eceb_config=eceb_config, **model_kwargs
        )
    else:
        if adapter_has_eceb:
            raise ValueError("An ECEB checkpoint requires --use_eceb")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    for parameter in base.parameters():
        parameter.requires_grad = False

    if args.adapter_path:
        model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if args.use_eceb:
        if adapter_has_eceb:
            load_eceb_checkpoint(model, args.adapter_path)
        grounded = model.get_base_model()
        for parameter in grounded.entity_patch_interactor.parameters():
            parameter.requires_grad = True
    return model


def main() -> None:
    args = parse_args()
    if args.use_eceb and not args.adapter_path:
        raise ValueError("Stage II requires --adapter_path from Stage I")
    if min(args.epochs, args.batch_size, args.grad_accum_steps) <= 0:
        raise ValueError("Epochs, batch size, and accumulation must be positive")
    seed_everything(args.seed)
    is_cuda = args.device.startswith("cuda")
    dtype, use_amp, autocast_dtype = _precision(args, is_cuda)
    system_prompt = _load_system_prompt(args.system_prompt_file)

    processor_source = args.model_name
    if args.adapter_path and any(
        (Path(args.adapter_path) / name).is_file()
        for name in ("processor_config.json", "preprocessor_config.json")
    ):
        processor_source = args.adapter_path
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = _build_model(args, dtype)
    model.to(args.device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    collator = Qwen25VLSFTCollator(
        processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        system_prompt=system_prompt,
    )
    train_loader = _loader(
        args.train_jsonl,
        image_root=args.image_root,
        collator=collator,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        maximum=args.max_train_samples,
        pin_memory=is_cuda,
    )
    dev_loader = (
        _loader(
            args.dev_jsonl,
            image_root=args.image_root,
            collator=collator,
            batch_size=args.dev_batch_size,
            shuffle=False,
            workers=args.num_workers,
            maximum=args.max_dev_samples,
            pin_memory=is_cuda,
        )
        if args.dev_jsonl
        else None
    )

    eceb_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "entity_patch_interactor" in name
    ]
    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "entity_patch_interactor" not in name
    ]
    if not lora_parameters:
        raise RuntimeError("No trainable LoRA parameters were found")
    groups = [
        {
            "params": lora_parameters,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]
    if args.use_eceb:
        if not eceb_parameters:
            raise RuntimeError("No trainable ECEB parameters were found")
        groups.append(
            {
                "params": eceb_parameters,
                "lr": args.eceb_lr,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, args.epochs * steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )
    eceb_warmup_steps = max(1, int(total_steps * args.eceb_warmup_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.fp16)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    global_step = 0
    best_score = float("-inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        accumulated = 0
        for batch in progress:
            device = get_first_param_device(model)
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)
            forward_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                **pick_vision_inputs(batch),
            }
            if args.use_eceb:
                forward_inputs.update(
                    tweet_token_mask=batch["tweet_token_mask"],
                    eceb_targets=batch["eceb_targets"],
                )
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                output = model(**forward_inputs)
                weights = (
                    batch["field_token_weights"]
                    if args.use_eceb
                    else torch.ones_like(batch["labels"], dtype=torch.float32)
                )
                lm_loss = _causal_loss(output.logits, batch["labels"], weights)
                eceb_loss = output.eceb_loss if args.use_eceb else lm_loss.new_zeros(())
                eceb_scale = (
                    args.eceb_weight * min(1.0, (global_step + 1) / eceb_warmup_steps)
                    if args.use_eceb
                    else 0.0
                )
                total_loss = lm_loss + eceb_scale * eceb_loss
                loss = total_loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated += 1

            if accumulated == args.grad_accum_steps:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_parameters + eceb_parameters, args.max_grad_norm
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                accumulated = 0
                progress.set_postfix(
                    loss=float(total_loss.detach()),
                    lm=float(lm_loss.detach()),
                    eceb=float(eceb_loss.detach()),
                    step=global_step,
                )

        if accumulated:
            correction = args.grad_accum_steps / accumulated
            for parameter in lora_parameters + eceb_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_parameters + eceb_parameters, args.max_grad_norm
                )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        _save(model, processor, str(output_dir), args.use_eceb)
        if dev_loader is not None:
            metrics = run_dev_eval(
                model,
                processor,
                dev_loader,
                str(output_dir),
                epoch,
                use_amp,
                autocast_dtype,
                args.eval_max_new_tokens,
                args.eval_iou,
            )
            score = float(metrics["dev/gmner/f1"])
            if score > best_score:
                best_score = score
                _save(model, processor, str(output_dir / "best"), args.use_eceb)
        else:
            _save(model, processor, str(output_dir / "best"), args.use_eceb)


if __name__ == "__main__":
    main()
