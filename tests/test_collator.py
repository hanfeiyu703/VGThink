from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
import unittest

from PIL import Image
import torch

from e2egmner.data import Qwen25VLSFTCollator
from e2egmner.data.collator import rasterize_region_patch_weights


class _CharacterTokenizer:
    base = 1000
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.base + ord(character) for character in text]

    def __call__(
        self,
        texts,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
        return_offsets_mapping: bool = False,
    ):
        self.assert_call(padding, truncation, return_tensors)
        if isinstance(texts, str):
            texts = [texts]
        rows = [self.encode(text)[:max_length] for text in texts]
        offsets = [
            [[index, index + 1] for index in range(min(len(text), max_length))] for text in texts
        ]
        width = max(map(len, rows))
        ids, attention, padded_offsets = [], [], []
        for row, row_offsets in zip(rows, offsets):
            pad_width = width - len(row)
            pads = [self.pad_token_id] * pad_width
            pad_attention = [0] * pad_width
            pad_offsets = [[0, 0]] * pad_width
            if self.padding_side == "left":
                ids.append(pads + row)
                attention.append(pad_attention + [1] * len(row))
                padded_offsets.append(pad_offsets + row_offsets)
            else:
                ids.append(row + pads)
                attention.append([1] * len(row) + pad_attention)
                padded_offsets.append(row_offsets + pad_offsets)
        result = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attention),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(padded_offsets)
        return result

    @staticmethod
    def assert_call(padding: bool, truncation: bool, return_tensors: str) -> None:
        if not padding or not truncation or return_tensors != "pt":
            raise AssertionError("unexpected tokenizer options")

    @classmethod
    def decode(cls, ids: torch.Tensor) -> str:
        return "".join(chr(int(value) - cls.base) for value in ids.tolist())


class _CharacterProcessor:
    image_placeholder = "<IMAGE>"
    image_replacement = "V"

    def __init__(self) -> None:
        self.tokenizer = _CharacterTokenizer()
        self.image_processor = SimpleNamespace(min_pixels=56 * 56, max_pixels=56 * 112)

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        if tokenize:
            raise AssertionError("test processor expects rendered text")
        rendered = []
        for message in messages:
            rendered.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, str):
                rendered.append(content)
            else:
                for part in content:
                    rendered.append(
                        self.image_placeholder if part["type"] == "image" else part["text"]
                    )
            rendered.append(f"</{message['role']}>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)

    def replace_image_token(self, inputs: dict[str, Any], image_idx: int) -> str:
        if image_idx >= inputs["image_grid_thw"].shape[0]:
            raise IndexError(image_idx)
        return self.image_replacement

    def get_text_with_replacements(self, texts, *, images_replacements):
        expanded = [
            text.replace(self.image_placeholder, replacement)
            for text, replacement in zip(texts, images_replacements)
        ]
        return expanded, images_replacements

    def __call__(self, *, text, images, **tokenizer_options):
        replacements = [self.image_replacement] * len(images)
        expanded, _ = self.get_text_with_replacements(text, images_replacements=replacements)
        result = self.tokenizer(expanded, **tokenizer_options)
        result["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images))
        result["pixel_values"] = torch.zeros((4 * len(images), 12))
        return result


def _example(user: str, assistant: str, *, target=None):
    return {
        "record_id": "sample",
        "image": Image.new("RGB", (62, 24), "white"),
        "image_path": "images/sample.png",
        "user_text": user,
        "assistant_text": assistant,
        "region_targets": [] if target is None else [target],
    }


class CollatorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = _CharacterProcessor()
        self.collator = Qwen25VLSFTCollator(
            self.processor,
            max_length=2048,
            system_prompt="system",
        )

    def test_masks_coordinates_weights_and_official_region_target_align(self) -> None:
        output = self.collator(
            [
                _example(
                    "Text: Alpha visited Paris",
                    "<think><obj>Alpha|[27,12]</obj></think><answer>Alpha|PER|[4,3,50,21]</answer>",
                    target={
                        "text": "Alpha",
                        "etype": "PER",
                        "visible": True,
                        "loss_enabled": True,
                        "boxes": [[0.0, 0.0, 0.5, 0.5]],
                        "points": [[0.25, 0.25]],
                    },
                )
            ]
        )
        self.assertIsNotNone(output)
        tweet_ids = output["input_ids"][0, output["tweet_token_mask"][0]]
        prompt_tweet_ids = output["prompt_input_ids"][0, output["prompt_tweet_token_mask"][0]]
        self.assertEqual(_CharacterTokenizer.decode(tweet_ids), "Alpha visited Paris")
        self.assertTrue(torch.equal(tweet_ids, prompt_tweet_ids))
        self.assertTrue(output["labels"][0, output["tweet_token_mask"][0]].eq(-100).all())

        scaled = output["gold_texts"][0]
        self.assertIn("<obj>Alpha|[49,28]</obj>", scaled)
        self.assertIn("<answer>Alpha|PER|[7,7,90,49]</answer>", scaled)
        weights = output["field_token_weights"][output["labels"].ne(-100)]
        for expected in (1.2, 1.5, 2.0):
            self.assertTrue(torch.isclose(weights, torch.tensor(expected)).any())

        target = output["eceb_targets"][0][0]
        self.assertTrue(target["mention_positions"])
        self.assertTrue(all(output["tweet_token_mask"][0, target["mention_positions"]]))
        self.assertEqual(target["patch_weights"].numel(), 1)
        self.assertGreater(target["patch_weights"].item(), 0.0)

    def test_leading_space_token_can_bind_the_first_mention(self) -> None:
        self.assertEqual(self.collator._positions_in_span([[0, 6]], 1, 6), [0])

    def test_mentions_use_distinct_occurrences_and_html_fallback(self) -> None:
        post = "Walsh asked walsh about Fox & ; Fin Financial Group"
        spans = self.collator._align_mention_spans(
            post, ["Walsh", "walsh", "Fox &amp ; Fin Financial Group"]
        )
        self.assertEqual(
            [post[slice(*span)] for span in spans if span is not None],
            ["Walsh", "walsh", "Fox & ; Fin Financial Group"],
        )

    def test_field_weights_distinguish_mentions_from_serialized_fields(self) -> None:
        text = (
            "<think><obj>U2|[1,2]</obj> <ungroundable>X2</ungroundable></think>"
            "<answer>U2|ORG|[1,2,3,4]; X2|PER|None</answer>"
        )
        encoded = self.processor.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        weights = self.collator._field_token_weights(
            expanded_texts=[text],
            offsets=encoded["offset_mapping"],
            labels=encoded["input_ids"],
        )[0]

        mention_digit_positions = [
            match.start("digit") for match in re.finditer(r"(?:U|X)(?P<digit>2)(?=[|<])", text)
        ]
        self.assertTrue(all(weights[index].item() == 1.0 for index in mention_digit_positions))
        for index, character in enumerate(text):
            if character in "[],;|":
                self.assertAlmostEqual(weights[index].item(), 1.2, places=5)
        ungroundable_open = text.index("<ungroundable>")
        self.assertEqual(weights[ungroundable_open].item(), 2.0)

    def test_ambiguous_text_marker_disables_bridge_mask(self) -> None:
        output = self.collator(
            [
                _example(
                    "Text: first\nText: second",
                    "<think>none</think><answer>None</answer>",
                )
            ]
        )
        self.assertIsNotNone(output)
        self.assertFalse(output["tweet_token_mask"].any())
        self.assertFalse(output["prompt_tweet_token_mask"].any())

    def test_official_box_uses_exact_fractional_patch_coverage(self) -> None:
        grid = torch.tensor([1, 2, 2])
        centered = rasterize_region_patch_weights(
            {"boxes": [[0.25, 0.25, 0.75, 0.75]], "points": []},
            grid,
            spatial_merge_size=2,
        )
        tiny = rasterize_region_patch_weights(
            {"boxes": [[0.0, 0.0, 0.1, 0.1]], "points": []},
            grid,
            spatial_merge_size=2,
        )
        duplicate = rasterize_region_patch_weights(
            {"boxes": [[0.25, 0.25, 0.75, 0.75]] * 2, "points": []},
            grid,
            spatial_merge_size=2,
        )
        self.assertAlmostEqual(centered.item(), 0.25, places=6)
        self.assertAlmostEqual(tiny.item(), 0.01, places=6)
        self.assertAlmostEqual(duplicate.item(), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
