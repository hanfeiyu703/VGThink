"""Stage-III group-relative optimization with ECEB active in every forward."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from peft import PeftModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from ..data import Qwen25VLSFTCollator, TwitterGroundedMNERJsonl
from ..engine import run_dev_eval
from ..modeling import (
    GroundedQwen25VLForConditionalGeneration,
    checkpoint_has_eceb,
    load_eceb_checkpoint,
    load_eceb_config,
    save_eceb_checkpoint,
)
from ..rl import GMNERReward
from ..utils import get_first_param_device, pick_vision_inputs, seed_everything, supports_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGThink Stage-III verifiable sequence refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--attn_impl", choices=["sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--sft_anchor_coef", type=float, default=0.10)
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


def _repeat_vision(vision: dict[str, torch.Tensor], count: int):
    repeated: dict[str, torch.Tensor] = {}
    for key, value in vision.items():
        if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
            repeated[key] = torch.cat([value] * count, dim=0)
        else:
            repeated[key] = value.repeat_interleave(count, dim=0)
    return repeated


def _response_attention(
    generated: torch.Tensor,
    prompt_attention: torch.Tensor,
    prompt_width: int,
    eos_ids: list[int],
) -> torch.Tensor:
    response = generated[:, prompt_width:]
    response_attention = torch.ones_like(response, dtype=prompt_attention.dtype)
    for row in range(response.shape[0]):
        eos_positions = torch.nonzero(
            torch.stack([response[row].eq(value) for value in eos_ids]).any(dim=0),
            as_tuple=False,
        )
        if eos_positions.numel():
            response_attention[row, int(eos_positions[0]) + 1 :] = 0
    return torch.cat((prompt_attention, response_attention), dim=1)


@torch.inference_mode()
def _rollouts(model, processor, batch: dict, args, use_amp, autocast_dtype):
    count = args.num_rollouts
    prompt_ids = batch["prompt_input_ids"].repeat(count, 1)
    prompt_attention = batch["prompt_attention_mask"].repeat(count, 1)
    tweet_mask = batch["prompt_tweet_token_mask"].repeat(count, 1)
    vision = _repeat_vision(pick_vision_inputs(batch), count)
    was_training = model.training
    model.eval()
    try:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention,
                tweet_token_mask=tweet_mask,
                **vision,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                use_cache=True,
                logits_to_keep=1,
            )
    finally:
        model.train(was_training)
    prompt_width = prompt_ids.shape[1]
    response_ids = generated[:, prompt_width:]
    response_texts = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_eos = getattr(model.generation_config, "eos_token_id", None)
    raw_eos = processor.tokenizer.eos_token_id if raw_eos is None else raw_eos
    eos_ids = [int(raw_eos)] if isinstance(raw_eos, int) else [int(value) for value in raw_eos]
    attention = _response_attention(generated, prompt_attention, prompt_width, eos_ids)
    return (
        generated,
        attention,
        tweet_mask,
        vision,
        prompt_width,
        [text.strip() for text in response_texts],
    )


def _response_logprob_means(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tweet_mask: torch.Tensor,
    vision: dict[str, torch.Tensor],
    prompt_width: int,
) -> torch.Tensor:
    logits_start = max(0, prompt_width - 1)
    logits_to_keep = torch.arange(logits_start, input_ids.shape[1], device=input_ids.device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tweet_token_mask=tweet_mask,
        logits_to_keep=logits_to_keep,
        **vision,
    )
    labels = input_ids.clone()
    labels[:, :prompt_width] = -100
    labels[attention_mask.eq(0)] = -100
    shifted_logits = output.logits[:, :-1].float()
    logits_end = logits_start + output.logits.shape[1]
    shifted_labels = labels[:, logits_start + 1 : logits_end]
    valid = shifted_labels.ne(-100)
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    return -(losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _anchor_loss(model, batch: dict) -> torch.Tensor:
    labels = batch["labels"]
    positions = torch.nonzero(labels.ne(-100), as_tuple=False)
    if not positions.numel():
        raise ValueError("SFT anchor has no supervised tokens")
    logits_start = max(0, int(positions[:, 1].min()) - 1)
    logits_to_keep = torch.arange(logits_start, labels.shape[1], device=labels.device)
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        tweet_token_mask=batch["tweet_token_mask"],
        logits_to_keep=logits_to_keep,
        **pick_vision_inputs(batch),
    )
    shifted_logits = output.logits[:, :-1].float()
    logits_end = logits_start + output.logits.shape[1]
    shifted_labels = labels[:, logits_start + 1 : logits_end]
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )


def _save(model, processor, path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    save_eceb_checkpoint(model, path)


def _load_data(
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
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("Sample limits must be positive")
        dataset = Subset(dataset, range(min(maximum, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        collate_fn=collator,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Stage III currently requires --batch_size 1")
    if not checkpoint_has_eceb(args.adapter_path):
        raise ValueError("Stage III requires a complete Stage-II ECEB checkpoint")
    if args.num_rollouts < 2:
        raise ValueError("--num_rollouts must be at least 2")
    if not 0.0 < args.top_p <= 1.0 or args.temperature <= 0:
        raise ValueError("Sampling requires temperature > 0 and top_p in (0, 1]")
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    seed_everything(args.seed)
    is_cuda = args.device.startswith("cuda")
    if not args.bf16 and not args.fp16 and is_cuda:
        args.bf16 = supports_bf16()
        args.fp16 = not args.bf16
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    use_amp = is_cuda
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    processor_source = (
        args.adapter_path
        if any(
            (Path(args.adapter_path) / name).is_file()
            for name in ("processor_config.json", "preprocessor_config.json")
        )
        else args.model_name
    )
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    base = GroundedQwen25VLForConditionalGeneration.from_pretrained(
        args.model_name,
        eceb_config=load_eceb_config(args.adapter_path),
        **model_kwargs,
    )
    for parameter in base.parameters():
        parameter.requires_grad = False
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    load_eceb_checkpoint(model, args.adapter_path)
    for parameter in model.get_base_model().entity_patch_interactor.parameters():
        parameter.requires_grad = True
    model.to(args.device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        if args.system_prompt_file
        else None
    )
    collator = Qwen25VLSFTCollator(
        processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        system_prompt=system_prompt,
    )
    train_loader = _load_data(
        args.train_jsonl,
        image_root=args.image_root,
        collator=collator,
        batch_size=1,
        shuffle=True,
        workers=args.num_workers,
        maximum=args.max_train_samples,
        pin_memory=is_cuda,
    )
    dev_loader = (
        _load_data(
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.fp16)
    reward_function = GMNERReward(iou_threshold=args.eval_iou)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    global_step = 0
    best_score = float("-inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        accumulated = 0
        progress = tqdm(train_loader, desc=f"stage3 {epoch + 1}/{args.epochs}")
        for batch in progress:
            device = get_first_param_device(model)
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)
            (
                rollout_ids,
                rollout_attention,
                rollout_tweet_mask,
                rollout_vision,
                prompt_width,
                responses,
            ) = _rollouts(model, processor, batch, args, use_amp, autocast_dtype)
            breakdowns = [
                reward_function(response, batch["gold_texts"][0]) for response in responses
            ]
            rewards = torch.tensor(
                [item.total for item in breakdowns],
                device=device,
                dtype=torch.float32,
            )
            standard_deviation = rewards.std(unbiased=False)
            advantages = (
                (rewards - rewards.mean()) / (standard_deviation + 1e-6)
                if float(standard_deviation) > 1e-6
                else torch.zeros_like(rewards)
            )
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                log_probabilities = _response_logprob_means(
                    model,
                    rollout_ids,
                    rollout_attention,
                    rollout_tweet_mask,
                    rollout_vision,
                    prompt_width,
                )
                policy_loss = -(advantages.detach() * log_probabilities).mean()
                anchor_loss = _anchor_loss(model, batch)
                total_loss = policy_loss + args.sft_anchor_coef * anchor_loss
                loss = total_loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Stage-III loss: {loss.item()}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated += 1
            if accumulated == args.grad_accum_steps:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
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
                reward=float(rewards.mean()),
                consistency=sum(item.consistency_reward for item in breakdowns) / len(breakdowns),
                step=global_step,
            )

        if accumulated:
            correction = args.grad_accum_steps / accumulated
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        _save(model, processor, str(output_dir))
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
                _save(model, processor, str(output_dir / "best"))
        else:
            _save(model, processor, str(output_dir / "best"))


if __name__ == "__main__":
    main()
