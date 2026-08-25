"""Qwen2.5-VL collation for response SFT and ECEB supervision."""

from __future__ import annotations

from copy import deepcopy
from html import unescape
import re
from typing import Any, Optional, Sequence

import torch
from PIL import Image

from ..evaluation.geometry import rescale_assistant_text_bboxes
from ..training.entity_patch_interactor import merged_patch_grid_shape


POST_MARKER_RE = re.compile(r"^Text:\s*", flags=re.MULTILINE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
OBJ_BLOCK_RE = re.compile(r"<obj>.*?</obj>", flags=re.IGNORECASE | re.DOTALL)
UNGROUNDABLE_BLOCK_RE = re.compile(
    r"<ungroundable>.*?</ungroundable>", flags=re.IGNORECASE | re.DOTALL
)
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
ANSWER_FIELD_RE = re.compile(
    r"\|\s*(?P<type>PER|ORG|LOC|MISC|OTHER)\s*\|\s*"
    r"(?P<region>None|\[(?:[^\[\]]|\[[^\[\]]*\])*\])",
    flags=re.IGNORECASE,
)
HTML_ENTITY_PREFIX_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+)\s*$")

DEFAULT_SYSTEM_PROMPT = (
    "You are an information extraction assistant for grounded multimodal named "
    "entity recognition. Keep only entities explicitly mentioned in the text. "
    "In <think>, bind each visible entity to <obj>mention|[x,y]</obj> using "
    "processed-image pixel coordinates and each "
    "unsupported entity to <ungroundable>mention</ungroundable>. Then emit exactly "
    "one <answer> block containing mention|type|[x1,y1,x2,y2] or "
    "mention|type|None records separated by semicolons. Output one <think> block "
    "followed by one <answer> block and no text outside them."
)


def rasterize_region_patch_weights(
    target: dict[str, Any],
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Compute exact fractional patch coverage for normalized official boxes."""

    temporal, grid_h, grid_w = merged_patch_grid_shape(grid_thw, spatial_merge_size)
    if target.get("polygons"):
        raise ValueError("The minimal VGThink release supports official boxes only")

    boxes: list[tuple[float, float, float, float]] = []
    for raw_box in target.get("boxes") or []:
        if len(raw_box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
        y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
        if x1 < x2 and y1 < y2:
            boxes.append((x1, y1, x2, y2))

    def union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
        x_edges = sorted({value for rectangle in rectangles for value in rectangle[::2]})
        area = 0.0
        for left, right in zip(x_edges, x_edges[1:]):
            if right <= left:
                continue
            intervals = sorted(
                (y1, y2) for x1, y1, x2, y2 in rectangles if x1 < right and x2 > left
            )
            covered_y = 0.0
            if intervals:
                lower, upper = intervals[0]
                for next_lower, next_upper in intervals[1:]:
                    if next_lower > upper:
                        covered_y += upper - lower
                        lower, upper = next_lower, next_upper
                    else:
                        upper = max(upper, next_upper)
                covered_y += upper - lower
            area += (right - left) * covered_y
        return area

    patch_area = 1.0 / (grid_h * grid_w)
    coverage: list[float] = []
    for row in range(grid_h):
        patch_y1, patch_y2 = row / grid_h, (row + 1) / grid_h
        for column in range(grid_w):
            patch_x1, patch_x2 = column / grid_w, (column + 1) / grid_w
            clipped = [
                (
                    max(x1, patch_x1),
                    max(y1, patch_y1),
                    min(x2, patch_x2),
                    min(y2, patch_y2),
                )
                for x1, y1, x2, y2 in boxes
                if x1 < patch_x2 and x2 > patch_x1 and y1 < patch_y2 and y2 > patch_y1
            ]
            coverage.append(min(1.0, max(0.0, union_area(clipped) / patch_area)))
    weights = torch.tensor(coverage, dtype=torch.float32)
    if not bool(weights.gt(0).any()):
        for raw_point in target.get("points") or []:
            if len(raw_point) < 2:
                continue
            col = min(grid_w - 1, max(0, int(float(raw_point[0]) * grid_w)))
            row = min(grid_h - 1, max(0, int(float(raw_point[1]) * grid_h)))
            weights[row * grid_w + col] = 1.0
            break
    if temporal > 1:
        weights = weights.repeat(temporal)
    return weights


class Qwen25VLSFTCollator:
    """Create masked response labels, post-token masks, and optional ECEB targets."""

    def __init__(
        self,
        processor,
        max_length: int = 2048,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        factor: int = 28,
        system_prompt: Optional[str] = None,
        spatial_merge_size: int = 2,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)
        self.factor = int(factor)
        self.system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.spatial_merge_size = max(1, int(spatial_merge_size))
        image_processor = getattr(processor, "image_processor", None)
        self.min_pixels = int(
            min_pixels
            if min_pixels is not None
            else getattr(image_processor, "min_pixels", 256 * 28 * 28)
        )
        self.max_pixels = int(
            max_pixels
            if max_pixels is not None
            else getattr(image_processor, "max_pixels", 1024 * 28 * 28)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_messages(self, user_text: str, assistant_text: Optional[str] = None):
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def _expand_texts(self, texts: list[str], full_inputs: dict[str, Any]) -> list[str]:
        if not (
            "image_grid_thw" in full_inputs
            and hasattr(self.processor, "replace_image_token")
            and hasattr(self.processor, "get_text_with_replacements")
        ):
            raise RuntimeError("The active processor cannot expose image-token text offsets")
        replacements = [
            self.processor.replace_image_token(full_inputs, image_idx=index)
            for index in range(len(texts))
        ]
        expanded, _ = self.processor.get_text_with_replacements(
            list(texts), images_replacements=replacements
        )
        return expanded

    def _tokenize_expanded(
        self,
        texts: list[str],
        full_inputs: dict[str, Any],
        *,
        padding_side: str,
        offsets: bool,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        expanded = self._expand_texts(texts, full_inputs)
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            encoded = self.tokenizer(
                expanded,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=offsets,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = old_side
        return encoded, expanded

    @staticmethod
    def _post_char_span(rendered: str, user_text: str) -> Optional[tuple[int, int]]:
        matches = list(POST_MARKER_RE.finditer(user_text))
        if len(matches) != 1:
            return None
        occurrences: list[int] = []
        start = 0
        while True:
            found = rendered.find(user_text, start)
            if found < 0:
                break
            occurrences.append(found)
            start = found + 1
        if len(occurrences) != 1:
            return None
        user_start = occurrences[0]
        post_start = user_start + matches[0].end()
        post_end = user_start + len(user_text)
        return (post_start, post_end) if post_start < post_end else None

    @staticmethod
    def _positions_in_span(offsets: Sequence[Sequence[int]], start: int, end: int) -> list[int]:
        """Return tokens touching a character span.

        Byte-level tokenizers commonly attach the leading space to the first
        word token. Requiring full containment would therefore erase the first
        post token and every entity mention at a whitespace boundary.
        """

        overlapping: list[int] = []
        for index, pair in enumerate(offsets):
            token_start, token_end = int(pair[0]), int(pair[1])
            if token_end <= token_start:
                continue
            if token_start < end and token_end > start:
                overlapping.append(index)
        return overlapping

    def _post_masks(
        self,
        *,
        user_texts: list[str],
        full_texts: list[str],
        full_offsets: torch.Tensor,
        full_ids: torch.Tensor,
        full_attention: torch.Tensor,
        prompt_texts: list[str],
        prompt_offsets: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        prompt_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
        special_ids = set(int(value) for value in self.tokenizer.all_special_ids)
        for index, user_text in enumerate(user_texts):
            full_span = self._post_char_span(full_texts[index], user_text)
            prompt_span = self._post_char_span(prompt_texts[index], user_text)
            if full_span is None or prompt_span is None:
                continue
            full_positions = self._positions_in_span(full_offsets[index].tolist(), *full_span)
            prompt_positions = self._positions_in_span(prompt_offsets[index].tolist(), *prompt_span)
            if not full_positions or not prompt_positions:
                continue
            if any(
                not bool(full_attention[index, position])
                or int(full_ids[index, position]) in special_ids
                for position in full_positions
            ):
                continue
            if any(
                not bool(prompt_attention[index, position])
                or int(prompt_ids[index, position]) in special_ids
                for position in prompt_positions
            ):
                continue
            if not torch.equal(
                full_ids[index, full_positions], prompt_ids[index, prompt_positions]
            ):
                continue
            full_mask[index, full_positions] = True
            prompt_mask[index, prompt_positions] = True
        return full_mask, prompt_mask

    @staticmethod
    def _align_mention_spans(post: str, mentions: Sequence[str]) -> list[Optional[tuple[int, int]]]:
        """Bind ordered annotation records to distinct occurrences in the post."""

        used: list[tuple[int, int]] = []

        def boundary_ok(start: int, end: int) -> bool:
            first, last = post[start], post[end - 1]
            if start and first.isalnum() and post[start - 1].isalnum():
                return False
            if end < len(post) and last.isalnum() and post[end].isalnum():
                return False
            return True

        def available(start: int, end: int) -> bool:
            return boundary_ok(start, end) and all(
                end <= occupied_start or start >= occupied_end
                for occupied_start, occupied_end in used
            )

        aligned: list[Optional[tuple[int, int]]] = []
        for raw_mention in mentions:
            mention = str(raw_mention or "").strip()
            variants = list(dict.fromkeys(value for value in (mention, unescape(mention)) if value))
            candidates: list[tuple[int, int]] = []
            for ignore_case in (False, True):
                flags = re.IGNORECASE if ignore_case else 0
                for variant in variants:
                    patterns = [re.escape(variant)]
                    pieces = variant.split()
                    if len(pieces) > 1:
                        patterns.append(r"\s+".join(re.escape(piece) for piece in pieces))
                    for pattern in dict.fromkeys(patterns):
                        candidates.extend(
                            match.span() for match in re.finditer(pattern, post, flags=flags)
                        )
                choice = next(
                    (span for span in sorted(set(candidates)) if available(span[0], span[1])),
                    None,
                )
                if choice is not None:
                    used.append(choice)
                    aligned.append(choice)
                    break
            else:
                aligned.append(None)
        return aligned

    def _eceb_targets(
        self,
        *,
        raw_targets: list[list[dict[str, Any]]],
        user_texts: list[str],
        expanded_texts: list[str],
        offsets: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[list[dict[str, Any]]]:
        output: list[list[dict[str, Any]]] = []
        for sample_index, sample_targets in enumerate(raw_targets):
            rendered_span = self._post_char_span(
                expanded_texts[sample_index], user_texts[sample_index]
            )
            marker = list(POST_MARKER_RE.finditer(user_texts[sample_index]))
            post = user_texts[sample_index][marker[0].end() :] if len(marker) == 1 else ""
            mention_spans = self._align_mention_spans(
                post, [str(target.get("text") or "") for target in sample_targets]
            )
            prepared: list[dict[str, Any]] = []
            for raw_target, mention_span in zip(sample_targets, mention_spans):
                target = deepcopy(raw_target)
                positions: list[int] = []
                if rendered_span is not None and mention_span is not None:
                    positions = self._positions_in_span(
                        offsets[sample_index].tolist(),
                        rendered_span[0] + mention_span[0],
                        rendered_span[0] + mention_span[1],
                    )
                target["mention_positions"] = positions
                if bool(target.get("loss_enabled", True)) and not positions:
                    raise ValueError(
                        f"Cannot align ECEB target {target.get('text')!r} to source post {post!r}"
                    )
                if bool(target.get("visible")):
                    target["patch_weights"] = rasterize_region_patch_weights(
                        target,
                        image_grid_thw[sample_index],
                        spatial_merge_size=self.spatial_merge_size,
                    )
                prepared.append(target)
            output.append(prepared)
        return output

    def _field_token_weights(
        self,
        *,
        expanded_texts: list[str],
        offsets: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Implement the paper's numeric/type/format/None field weights."""

        weights = torch.ones_like(labels, dtype=torch.float32)

        def add_spans(
            sample_index: int,
            spans: list[tuple[int, int]],
            value: float,
        ) -> None:
            for start, end in spans:
                for position in range(int(offsets.shape[1])):
                    token_start, token_end = (
                        int(offsets[sample_index, position, 0]),
                        int(offsets[sample_index, position, 1]),
                    )
                    if (
                        token_end > token_start
                        and token_start < end
                        and token_end > start
                        and int(labels[sample_index, position]) != -100
                    ):
                        weights[sample_index, position] = max(
                            float(weights[sample_index, position]), value
                        )

        for sample_index, text in enumerate(expanded_texts):
            answer_matches = list(ANSWER_BLOCK_RE.finditer(text))
            obj_matches = list(OBJ_BLOCK_RE.finditer(text))
            ungroundable_matches = list(UNGROUNDABLE_BLOCK_RE.finditer(text))
            numeric_spans: list[tuple[int, int]] = []
            structural_spans: list[tuple[int, int]] = []
            type_spans: list[tuple[int, int]] = []
            null_spans: list[tuple[int, int]] = []

            for obj_match in obj_matches:
                start, end = obj_match.span()
                pipe = text.rfind("|", start, end)
                field_end = text.lower().rfind("</obj>", start, end)
                if pipe < 0 or field_end < pipe:
                    continue
                structural_spans.append((pipe, pipe + 1))
                coordinate_field = text[pipe + 1 : field_end]
                numeric_spans.extend(
                    (pipe + 1 + match.start(), pipe + 1 + match.end())
                    for match in NUMBER_RE.finditer(coordinate_field)
                )
                structural_spans.extend(
                    (index, index + 1)
                    for index in range(pipe + 1, field_end)
                    if text[index] in "[],"
                )

            for answer_match in answer_matches:
                body_start = answer_match.start(1)
                body = answer_match.group(1)
                if body.strip().casefold() == "none":
                    none_match = re.search(r"\bNone\b", body, flags=re.IGNORECASE)
                    if none_match is not None:
                        null_spans.append(
                            (body_start + none_match.start(), body_start + none_match.end())
                        )
                for field_match in ANSWER_FIELD_RE.finditer(body):
                    type_spans.append(
                        (
                            body_start + field_match.start("type"),
                            body_start + field_match.end("type"),
                        )
                    )
                    field_text = field_match.group(0)
                    structural_spans.extend(
                        (
                            body_start + field_match.start() + index,
                            body_start + field_match.start() + index + 1,
                        )
                        for index, character in enumerate(field_text)
                        if character == "|"
                    )
                    region_start = body_start + field_match.start("region")
                    region = field_match.group("region")
                    if region.casefold() == "none":
                        null_spans.append((region_start, region_start + len(region)))
                    else:
                        numeric_spans.extend(
                            (region_start + match.start(), region_start + match.end())
                            for match in NUMBER_RE.finditer(region)
                        )
                        structural_spans.extend(
                            (region_start + index, region_start + index + 1)
                            for index, character in enumerate(region)
                            if character in "[],"
                        )
                for delimiter in re.finditer(r"[;；]", body):
                    if delimiter.group(0) == ";" and HTML_ENTITY_PREFIX_RE.search(
                        body[: delimiter.start()]
                    ):
                        continue
                    start = body_start + delimiter.start()
                    structural_spans.append((start, start + 1))

            # Only tag delimiters are structural, not all prose inside a block.
            structural_tag_spans = [
                (match.start(), match.end())
                for match in re.finditer(r"</?(?:think|answer|obj)>", text, flags=re.IGNORECASE)
            ]
            for block in ungroundable_matches:
                null_spans.extend(
                    (
                        block.start() + tag.start(),
                        block.start() + tag.end(),
                    )
                    for tag in re.finditer(r"</?ungroundable>", block.group(0), flags=re.IGNORECASE)
                )
            add_spans(sample_index, structural_tag_spans + structural_spans, 1.2)
            add_spans(sample_index, type_spans, 1.5)
            add_spans(sample_index, numeric_spans + null_spans, 2.0)
        return weights

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        images: list[Image.Image] = []
        assistant_texts: list[str] = []
        original_texts: list[str] = []
        original_sizes: list[tuple[int, int]] = []
        for item in batch:
            width, height = item["image"].size
            scaled = rescale_assistant_text_bboxes(
                item["assistant_text"],
                orig_w=width,
                orig_h=height,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                factor=self.factor,
                transform_obj_points=True,
            )
            if scaled is None:
                raise ValueError(
                    f"Cannot rescale assistant target for record {item.get('record_id')!r}"
                )
            images.append(item["image"])
            assistant_texts.append(scaled)
            original_texts.append(item["assistant_text"])
            original_sizes.append((width, height))
        user_texts = [str(item["user_text"]) for item in batch]
        prompt_rendered = [
            self.processor.apply_chat_template(
                self.build_messages(user), tokenize=False, add_generation_prompt=True
            )
            for user in user_texts
        ]
        full_rendered = [
            self.processor.apply_chat_template(
                self.build_messages(user, answer),
                tokenize=False,
                add_generation_prompt=False,
            )
            for user, answer in zip(user_texts, assistant_texts)
        ]

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            full_inputs = self.processor(
                text=full_rendered,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = old_side
        full_offsets, expanded_full = self._tokenize_expanded(
            full_rendered, full_inputs, padding_side="right", offsets=True
        )
        prompt_inputs, expanded_prompt = self._tokenize_expanded(
            prompt_rendered, full_inputs, padding_side="left", offsets=True
        )
        if not torch.equal(full_offsets["input_ids"], full_inputs["input_ids"]):
            raise RuntimeError("Tokenizer offsets do not align with multimodal input IDs")

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        labels = input_ids.clone()
        labels.fill_(-100)
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[index, int(prompt_length) :] = input_ids[index, int(prompt_length) :]
            labels[index, attention_mask[index].eq(0)] = -100

        tweet_mask, prompt_tweet_mask = self._post_masks(
            user_texts=user_texts,
            full_texts=expanded_full,
            full_offsets=full_offsets["offset_mapping"],
            full_ids=input_ids,
            full_attention=attention_mask,
            prompt_texts=expanded_prompt,
            prompt_offsets=prompt_inputs["offset_mapping"],
            prompt_ids=prompt_inputs["input_ids"],
            prompt_attention=prompt_inputs["attention_mask"],
        )

        full_inputs["labels"] = labels
        full_inputs["field_token_weights"] = self._field_token_weights(
            expanded_texts=expanded_full,
            offsets=full_offsets["offset_mapping"],
            labels=labels,
        )
        full_inputs["prompt_input_ids"] = prompt_inputs["input_ids"]
        full_inputs["prompt_attention_mask"] = prompt_inputs["attention_mask"]
        full_inputs["tweet_token_mask"] = tweet_mask
        full_inputs["prompt_tweet_token_mask"] = prompt_tweet_mask
        full_inputs["eceb_targets"] = self._eceb_targets(
            raw_targets=[item.get("region_targets") or [] for item in batch],
            user_texts=user_texts,
            expanded_texts=expanded_full,
            offsets=full_offsets["offset_mapping"],
            image_grid_thw=full_inputs["image_grid_thw"],
        )
        full_inputs["gold_texts"] = assistant_texts
        full_inputs["gold_texts_orig"] = original_texts
        full_inputs["orig_sizes"] = original_sizes
        full_inputs["user_texts"] = user_texts
        full_inputs["image_paths"] = [item.get("image_path") for item in batch]
        full_inputs["record_ids"] = [item.get("record_id") for item in batch]
        return full_inputs


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Qwen25VLSFTCollator",
    "rasterize_region_patch_weights",
]
