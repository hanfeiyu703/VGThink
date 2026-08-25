from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from e2egmner.training.entity_patch_interactor import (
    EntityPatchInteractor,
    EntityPatchInteractorConfig,
    build_2d_sincos_patch_positions,
)


def _config(**overrides) -> EntityPatchInteractorConfig:
    values = {
        "hidden_size": 16,
        "hidden_dim": 8,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "spatial_merge_size": 1,
        "max_entities_per_sample": 16,
    }
    values.update(overrides)
    return EntityPatchInteractorConfig(**values)


def _inputs():
    torch.manual_seed(7)
    embeds = torch.randn(1, 7, 16)
    image_mask = torch.tensor([[True, True, False, False, False, False, False]])
    tweet_mask = torch.tensor([[False, False, True, True, True, False, False]])
    grid = torch.tensor([[1, 1, 2]])
    return embeds, tweet_mask, image_mask, grid


class _FixedAttentionLayer(nn.Module):
    def __init__(self, probabilities: torch.Tensor, num_heads: int):
        super().__init__()
        self.register_buffer("probabilities", probabilities)
        self.num_heads = num_heads

    def forward(self, tweet_states, patch_none_memory):
        attention = self.probabilities.view(
            1, 1, self.probabilities.shape[0], self.probabilities.shape[1]
        ).expand(1, self.num_heads, -1, -1)
        return tweet_states, attention


class EntityPatchInteractorTest(unittest.TestCase):
    def test_positions_and_config_validation(self) -> None:
        positions = build_2d_sincos_patch_positions([2, 2, 4], 8, spatial_merge_size=2)
        self.assertEqual(tuple(positions.shape), (4, 8))
        self.assertTrue(torch.equal(positions[:2], positions[2:]))
        self.assertFalse(torch.equal(positions[0], positions[1]))
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            _config(hidden_dim=12, num_heads=8)

    def test_zero_gate_is_identity_and_nonzero_gate_is_tweet_only(self) -> None:
        module = EntityPatchInteractor(_config())
        embeds, tweet_mask, image_mask, grid = _inputs()
        fused, loss, stats = module(embeds, tweet_mask, image_mask, grid)
        self.assertTrue(torch.equal(fused, embeds))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["eceb/alpha"], 0.0)

        module.alpha.data.fill_(0.5)
        fused, _, _ = module(embeds, tweet_mask, image_mask, grid)
        self.assertFalse(torch.equal(fused[tweet_mask], embeds[tweet_mask]))
        self.assertTrue(torch.equal(fused[~tweet_mask], embeds[~tweet_mask]))

    def test_visible_and_none_set_mass_are_class_balanced(self) -> None:
        module = EntityPatchInteractor(_config())
        probabilities = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.3, 0.5]])
        module.layers = nn.ModuleList(
            [_FixedAttentionLayer(probabilities, module.config.num_heads) for _ in range(2)]
        )
        embeds, tweet_mask, image_mask, grid = _inputs()
        targets = [
            [
                {
                    "visible": True,
                    "loss_enabled": True,
                    "mention_positions": [2],
                    "patch_weights": torch.tensor([1.0, 0.0]),
                },
                {
                    "visible": True,
                    "loss_enabled": True,
                    "mention_positions": [3],
                    "patch_weights": torch.tensor([1.0, 0.0]),
                },
                {"visible": False, "loss_enabled": True, "mention_positions": [4]},
            ]
        ]
        _, loss, stats = module(embeds, tweet_mask, image_mask, grid, targets=targets)
        visible_mean = (-math.log(0.8) - math.log(0.2)) / 2.0
        expected = (visible_mean - math.log(0.5)) / 2.0
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertEqual(stats["eceb/entities"], 3.0)
        self.assertAlmostEqual(stats["eceb/inside_mass"], 0.5, places=6)

    def test_bridge_and_region_loss_backpropagate(self) -> None:
        module = EntityPatchInteractor(_config())
        embeds, tweet_mask, image_mask, grid = _inputs()
        embeds.requires_grad_(True)
        targets = [
            [
                {
                    "visible": True,
                    "loss_enabled": True,
                    "mention_positions": [2, 3],
                    "patch_weights": torch.tensor([1.0, 0.0]),
                },
                {"visible": False, "loss_enabled": True, "mention_positions": [4]},
            ]
        ]
        fused, region_loss, _ = module(embeds, tweet_mask, image_mask, grid, targets=targets)
        (fused.sum() + region_loss).backward()
        self.assertTrue(torch.isfinite(region_loss))
        self.assertIsNotNone(module.alpha.grad)
        cross_attention = module.layers[-1].cross_attention
        self.assertGreater(float(cross_attention.in_proj_weight.grad.abs().sum()), 0.0)

    def test_grid_and_mask_mismatch_fails_closed(self) -> None:
        module = EntityPatchInteractor(_config())
        embeds, tweet_mask, image_mask, grid = _inputs()
        image_mask[0, 1] = False
        with self.assertRaisesRegex(ValueError, "expected 2 image tokens"):
            module(embeds, tweet_mask, image_mask, grid)


if __name__ == "__main__":
    unittest.main()
