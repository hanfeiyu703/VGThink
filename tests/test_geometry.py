from __future__ import annotations

import unittest

from e2egmner.evaluation.geometry import (
    rescale_assistant_text_bboxes,
    restore_assistant_text_bboxes_to_original,
    smart_resize,
)


class GeometryContractTest(unittest.TestCase):
    def test_smart_resize_matches_factor_aligned_qwen_geometry(self) -> None:
        self.assertEqual(smart_resize(24, 62, factor=28, min_pixels=56 * 56), (56, 112))
        with self.assertRaisesRegex(ValueError, "positive"):
            smart_resize(0, 62)

    def test_trace_points_and_answer_boxes_share_one_round_trip(self) -> None:
        original = "<think><obj>entity|[27,12]</obj></think><answer>entity|PER|[4,3,50,21]</answer>"
        options = {
            "orig_w": 62,
            "orig_h": 24,
            "min_pixels": 56 * 56,
            "max_pixels": 14 * 14 * 4 * 1280,
            "factor": 28,
        }
        scaled = rescale_assistant_text_bboxes(original, **options)
        self.assertIsNotNone(scaled)
        self.assertIn("<obj>entity|[49,28]</obj>", scaled)
        self.assertIn("<answer>entity|PER|[7,7,90,49]</answer>", scaled)

        restored = restore_assistant_text_bboxes_to_original(scaled, **options)
        self.assertIsNotNone(restored)
        self.assertIn("<obj>entity|[27,12]</obj>", restored)
        self.assertIn("<answer>entity|PER|[4,3,50,21]</answer>", restored)

    def test_legacy_mixed_coordinate_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "one pixel coordinate space"):
            rescale_assistant_text_bboxes(
                "<answer>None</answer>",
                orig_w=10,
                orig_h=10,
                min_pixels=28 * 28,
                max_pixels=28 * 28,
                transform_obj_points=False,
            )


if __name__ == "__main__":
    unittest.main()
