"""Evaluate a vanilla or ECEB LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..data import Qwen25VLSFTCollator, TwitterGroundedMNERJsonl
from ..engine import evaluate, run_eceb_diagnostics
from ..modeling import (
    GroundedQwen25VLForConditionalGeneration,
    checkpoint_has_eceb,
    load_eceb_checkpoint,
    load_eceb_config,
)
from ..utils import seed_everything, supports_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_impl", choices=["sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--eceb_diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    seed_everything(args.seed)
    is_cuda = args.device.startswith("cuda")
    if not args.bf16 and not args.fp16 and is_cuda:
        args.bf16 = supports_bf16()
        args.fp16 = not args.bf16
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    processor_source = (
        checkpoint
        if any(
            (checkpoint / name).is_file()
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
    adapter_path = checkpoint / "adapter_config.json"
    has_eceb = checkpoint_has_eceb(str(checkpoint))
    if adapter_path.is_file():
        base = (
            GroundedQwen25VLForConditionalGeneration.from_pretrained(
                args.model_name,
                eceb_config=load_eceb_config(str(checkpoint)),
                **model_kwargs,
            )
            if has_eceb
            else Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
        )
        model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=False)
        if has_eceb:
            load_eceb_checkpoint(model, str(checkpoint))
    else:
        if has_eceb:
            raise ValueError("ECEB checkpoints must include a LoRA adapter")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(checkpoint), **model_kwargs)
    model.to(args.device)
    model.eval()

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
    dataset = TwitterGroundedMNERJsonl(args.test_jsonl, image_root=args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=is_cuda,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    metrics = evaluate(
        model=model,
        processor=processor,
        data_loader=loader,
        output_dir=args.output_dir,
        split="test",
        tag=args.tag,
        use_amp=is_cuda,
        autocast_dtype=autocast_dtype,
        max_new_tokens=args.max_new_tokens,
        iou_threshold=args.iou,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"metrics_{args.tag}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if args.eceb_diagnostics:
        if not has_eceb:
            raise ValueError("--eceb_diagnostics requires an ECEB checkpoint")
        run_eceb_diagnostics(
            model=model,
            data_loader=loader,
            output_path=str(output / f"eceb_{args.tag}.json"),
            use_amp=is_cuda,
            autocast_dtype=autocast_dtype,
        )


if __name__ == "__main__":
    main()
