"""Shared generation and metric loop for development and test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from ..evaluation.geometry import restore_assistant_text_bboxes_to_original
from ..evaluation.metrics import (
    count_correct_eeg,
    count_correct_gmner,
    count_correct_mner,
    prf,
)
from ..evaluation.parsing import extract_answer_text, parse_triples
from ..utils.runtime import append_text_log, get_first_param_device, pick_vision_inputs


def model_uses_eceb(model) -> bool:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return hasattr(base, "entity_patch_interactor")


def _move_tensors(batch: dict, device: torch.device) -> None:
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)


@torch.inference_mode()
def evaluate(
    *,
    model,
    processor,
    data_loader,
    output_dir: str,
    split: str,
    tag: str,
    use_amp: bool,
    autocast_dtype,
    max_new_tokens: int,
    iou_threshold: float = 0.5,
    strict_gold: bool = True,
    log_file: Optional[str] = None,
) -> dict[str, float | int | str]:
    """Greedily decode records, restore coordinates, and compute all three F1s."""

    was_training = model.training
    model.eval()
    device = get_first_param_device(model)
    collator = data_loader.collate_fn
    use_eceb = model_uses_eceb(model)
    output_path = Path(output_dir) / f"{split}_pred_{tag}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "gmner_correct": 0,
        "mner_correct": 0,
        "eeg_correct": 0,
        "predicted": 0,
        "gold": 0,
        "invalid_regions": 0,
        "restore_failures": 0,
    }
    with output_path.open("w", encoding="utf-8") as writer:
        for batch in tqdm(data_loader, desc=f"[{split}] {tag}", leave=False):
            _move_tensors(batch, device)
            generation_inputs = {
                "input_ids": batch["prompt_input_ids"],
                "attention_mask": batch["prompt_attention_mask"],
                **pick_vision_inputs(batch),
            }
            if use_eceb:
                generation_inputs["tweet_token_mask"] = batch["prompt_tweet_token_mask"]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                generated = model.generate(
                    **generation_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    logits_to_keep=1,
                )
            prompt_width = generation_inputs["input_ids"].shape[1]
            response_ids = generated[:, prompt_width:]
            responses = processor.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for index, response in enumerate(responses):
                response = (response or "").strip()
                gold_original = batch["gold_texts_orig"][index]
                original_width, original_height = batch["orig_sizes"][index]
                restored = restore_assistant_text_bboxes_to_original(
                    response,
                    orig_w=original_width,
                    orig_h=original_height,
                    min_pixels=collator.min_pixels,
                    max_pixels=collator.max_pixels,
                    factor=collator.factor,
                    transform_obj_points=True,
                )
                if restored is None:
                    totals["restore_failures"] += 1
                    prediction_answer = ""
                else:
                    prediction_answer = extract_answer_text(restored)
                gold_answer = extract_answer_text(gold_original)
                predictions = parse_triples(
                    prediction_answer, strict=False, where=f"{split}_prediction"
                )
                gold = parse_triples(gold_answer, strict=strict_gold, where=f"{split}_gold")
                totals["invalid_regions"] += sum(
                    not prediction.region_valid for prediction in predictions
                )
                totals["predicted"] += len(predictions)
                totals["gold"] += len(gold)
                totals["gmner_correct"] += count_correct_gmner(predictions, gold, iou_threshold)
                totals["mner_correct"] += count_correct_mner(predictions, gold)
                totals["eeg_correct"] += count_correct_eeg(predictions, gold, iou_threshold)
                writer.write(
                    json.dumps(
                        {
                            "id": batch["record_ids"][index],
                            "image": batch["image_paths"][index],
                            "text": batch["user_texts"][index],
                            "response": response,
                            "response_original_coordinates": restored,
                            "gold_processed_coordinates": batch["gold_texts"][index],
                            "gold": gold_original,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    result: dict[str, float | int | str] = {}
    for task in ("gmner", "mner", "eeg"):
        precision, recall, f1 = prf(totals[f"{task}_correct"], totals["predicted"], totals["gold"])
        result[f"{split}/{task}/precision"] = precision
        result[f"{split}/{task}/recall"] = recall
        result[f"{split}/{task}/f1"] = f1
    result[f"{split}/predicted"] = totals["predicted"]
    result[f"{split}/gold"] = totals["gold"]
    result[f"{split}/invalid_regions"] = totals["invalid_regions"]
    result[f"{split}/restore_failures"] = totals["restore_failures"]
    result[f"{split}/prediction_path"] = str(output_path)
    print(json.dumps(result, indent=2))
    if log_file:
        append_text_log(log_file, json.dumps(result, sort_keys=True))
    model.train(was_training)
    return result


def run_dev_eval(
    model,
    processor,
    dev_loader,
    output_dir: str,
    epoch_idx: int,
    use_amp: bool,
    autocast_dtype,
    eval_max_new_tokens: int,
    iou_threshold: float = 0.5,
    strict_gold: bool = True,
    log_file: Optional[str] = None,
):
    return evaluate(
        model=model,
        processor=processor,
        data_loader=dev_loader,
        output_dir=output_dir,
        split="dev",
        tag=f"epoch{epoch_idx + 1}",
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
        max_new_tokens=eval_max_new_tokens,
        iou_threshold=iou_threshold,
        strict_gold=strict_gold,
        log_file=log_file,
    )


def run_test_eval(
    model,
    processor,
    test_loader,
    output_dir: str,
    tag: str,
    use_amp: bool,
    autocast_dtype,
    eval_max_new_tokens: int,
    iou_threshold: float = 0.5,
    strict_gold: bool = True,
    log_file: Optional[str] = None,
):
    return evaluate(
        model=model,
        processor=processor,
        data_loader=test_loader,
        output_dir=output_dir,
        split="test",
        tag=tag,
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
        max_new_tokens=eval_max_new_tokens,
        iou_threshold=iou_threshold,
        strict_gold=strict_gold,
        log_file=log_file,
    )


@torch.inference_mode()
def run_eceb_diagnostics(
    *,
    model,
    data_loader,
    output_path: str,
    use_amp: bool,
    autocast_dtype,
) -> dict:
    """Aggregate patch/None assignment diagnostics without using labels as inputs."""

    if not model_uses_eceb(model):
        raise TypeError("ECEB diagnostics require a bridge checkpoint")
    was_training = model.training
    model.eval()
    device = get_first_param_device(model)
    weighted: dict[str, float] = {}
    counts = {"entities": 0.0, "visible": 0.0, "none": 0.0}
    for batch in tqdm(data_loader, desc="[eceb] diagnostics", leave=False):
        _move_tensors(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tweet_token_mask=batch["tweet_token_mask"],
                eceb_targets=batch["eceb_targets"],
                logits_to_keep=1,
                **pick_vision_inputs(batch),
            )
        stats = output.eceb_stats or {}
        entities = float(stats.get("eceb/entities", 0.0))
        visible = float(stats.get("eceb/visible_entities", 0.0))
        none = float(stats.get("eceb/none_entities", 0.0))
        counts["entities"] += entities
        counts["visible"] += visible
        counts["none"] += none
        for name, denominator in (
            ("raw_loss", entities),
            ("point_acc", entities),
            ("visible_point_acc", visible),
            ("none_acc", none),
            ("inside_mass", visible),
            ("p_none_visible", visible),
            ("p_none_none", none),
        ):
            weighted[name] = (
                weighted.get(name, 0.0) + float(stats.get(f"eceb/{name}", 0.0)) * denominator
            )
    metrics = {
        name: total
        / max(
            1.0,
            counts[
                "entities"
                if name in {"raw_loss", "point_acc"}
                else "none"
                if name in {"none_acc", "p_none_none"}
                else "visible"
            ],
        )
        for name, total in weighted.items()
    }
    payload = {"metrics": metrics, "counts": counts}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    model.train(was_training)
    return payload


__all__ = [
    "evaluate",
    "model_uses_eceb",
    "run_dev_eval",
    "run_eceb_diagnostics",
    "run_test_eval",
]
