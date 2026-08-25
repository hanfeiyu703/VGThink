"""Entity-conditioned evidence bridge between tweet tokens and image patches.

The :class:`EntityPatchInteractor` (ECEB) is intentionally independent from
the Qwen implementation that hosts it.  It consumes the multimodal token
embeddings after Qwen's visual merger has populated the image-token slots,
refines only the source-tweet positions, and returns a same-shaped residual
for the language model.

Gold regions are never inputs to the interaction path.  They are consumed only
after the forward interaction has produced attention probabilities, where they
supervise entity-to-patch/None set mass.  Consequently the exact same bridge is
usable when ``targets=None`` during autoregressive inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EntityPatchInteractorConfig:
    """JSON-serializable configuration for :class:`EntityPatchInteractor`."""

    hidden_size: int
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0
    spatial_merge_size: int = 2
    max_entities_per_sample: int = 16

    def __post_init__(self) -> None:
        if int(self.hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.hidden_dim) % int(self.num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        if int(self.hidden_dim) % 4:
            raise ValueError("hidden_dim must be a multiple of 4 for 2D sin/cos positions")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if int(self.spatial_merge_size) <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if int(self.max_entities_per_sample) <= 0:
            raise ValueError("max_entities_per_sample must be positive")


def merged_patch_grid_shape(
    grid_thw: torch.Tensor | Sequence[int],
    spatial_merge_size: int,
) -> tuple[int, int, int]:
    """Return the temporal and merged spatial visual-token grid shape."""

    if torch.is_tensor(grid_thw):
        values = [int(value) for value in grid_thw.detach().cpu().tolist()]
    else:
        values = [int(value) for value in grid_thw]
    if len(values) != 3:
        raise ValueError(f"Expected image grid [t,h,w], got {values}")
    t, h, w = values
    merge = int(spatial_merge_size)
    if t <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"Image grid dimensions must be positive, got {values}")
    if h % merge or w % merge:
        raise ValueError(f"Image grid {(t, h, w)} is not divisible by spatial merge {merge}")
    return t, h // merge, w // merge


def build_2d_sincos_patch_positions(
    grid_thw: torch.Tensor | Sequence[int],
    hidden_dim: int,
    *,
    spatial_merge_size: int = 2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build deterministic row-major 2D sin/cos positions for merged patches.

    The same spatial table is repeated for temporal frames.  ECEB currently
    targets images (normally ``t=1``), but preserving the temporal count keeps
    its memory layout aligned with the model's flattened visual tokens.
    """

    dim = int(hidden_dim)
    if dim <= 0 or dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of 4")
    t, grid_h, grid_w = merged_patch_grid_shape(grid_thw, spatial_merge_size)
    compute_device = device if device is not None else torch.device("cpu")
    output_dtype = dtype if dtype is not None else torch.float32

    rows = torch.arange(grid_h, device=compute_device, dtype=torch.float32)
    cols = torch.arange(grid_w, device=compute_device, dtype=torch.float32)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    row_coords = row_grid.reshape(-1)
    col_coords = col_grid.reshape(-1)

    frequencies = torch.arange(dim // 4, device=compute_device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * frequencies / max(1, dim // 4))

    def encode_axis(coords: torch.Tensor) -> torch.Tensor:
        phase = coords[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    spatial = torch.cat((encode_axis(row_coords), encode_axis(col_coords)), dim=-1)
    if t > 1:
        spatial = spatial.repeat(t, 1)
    return spatial.to(dtype=output_dtype)


class _SwiGLU(nn.Module):
    """Compact SwiGLU feed-forward network with a 4x inner width."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        inner_dim = 4 * int(hidden_dim)
        self.gate_value = nn.Linear(hidden_dim, 2 * inner_dim)
        self.output = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_value(hidden_states).chunk(2, dim=-1)
        return self.dropout(self.output(F.silu(gate) * value))


class _EntityPatchInteractionLayer(nn.Module):
    """One pre-LN tweet self-attention/cross-attention/SwiGLU layer."""

    def __init__(self, config: EntityPatchInteractorConfig):
        super().__init__()
        dim = int(config.hidden_dim)
        heads = int(config.num_heads)
        dropout = float(config.dropout)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = _SwiGLU(dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tweet_states: torch.Tensor,
        patch_none_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_tweet = self.self_norm(tweet_states)
        self_output, _ = self.self_attention(
            normalized_tweet,
            normalized_tweet,
            normalized_tweet,
            need_weights=False,
        )
        tweet_states = tweet_states + self.residual_dropout(self_output)

        cross_output, attention = self.cross_attention(
            self.cross_query_norm(tweet_states),
            self.cross_memory_norm(patch_none_memory),
            self.cross_memory_norm(patch_none_memory),
            need_weights=True,
            average_attn_weights=False,
        )
        tweet_states = tweet_states + self.residual_dropout(cross_output)
        tweet_states = tweet_states + self.ffn(self.ffn_norm(tweet_states))
        return tweet_states, attention


class EntityPatchInteractor(nn.Module):
    """Fuse source-tweet tokens with patch/None evidence before the decoder.

    Parameters passed to :meth:`forward` use full decoder-sequence coordinates.
    Only positions selected by ``tweet_token_mask`` are changed.  The image
    positions are read-only memory, and all remaining system/template/assistant
    positions are copied through exactly.

    ``targets`` is optional and is deliberately used only to calculate the
    auxiliary loss after the fused representations have already been produced.
    Each target follows the existing region-set contract: ``mention_positions``
    are full-sequence indices, visible targets provide ``patch_weights``, and
    invisible targets select the learned None memory.
    """

    def __init__(self, config: EntityPatchInteractorConfig):
        super().__init__()
        self.config = config
        self.tweet_projection = nn.Linear(config.hidden_size, config.hidden_dim)
        self.patch_projection = nn.Linear(config.hidden_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [_EntityPatchInteractionLayer(config) for _ in range(config.num_layers)]
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_size)
        self.none_embedding = nn.Parameter(torch.empty(config.hidden_dim))
        self.alpha = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.none_embedding, mean=0.0, std=config.hidden_dim**-0.5)

    @property
    def effective_alpha(self) -> torch.Tensor:
        """Bounded scalar residual gate used by the model path."""

        return torch.tanh(self.alpha)

    def _empty_stats(self) -> dict[str, float]:
        return {
            "eceb/raw_loss": 0.0,
            "eceb/entities": 0.0,
            "eceb/visible_entities": 0.0,
            "eceb/none_entities": 0.0,
            "eceb/point_acc": 0.0,
            "eceb/visible_point_acc": 0.0,
            "eceb/none_acc": 0.0,
            "eceb/inside_mass": 0.0,
            "eceb/p_none_visible": 0.0,
            "eceb/p_none_none": 0.0,
            "eceb/skipped_span": 0.0,
            "eceb/skipped_mask": 0.0,
            "eceb/skipped_disabled": 0.0,
            "eceb/alpha": float(self.effective_alpha.detach().item()),
        }

    @staticmethod
    def _raw_positions(target: dict[str, Any]) -> list[int]:
        raw = target.get("mention_positions")
        if raw is None:
            return []
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().flatten().tolist()
        try:
            return [int(position) for position in raw]
        except (TypeError, ValueError):
            return []

    def _set_mass_loss(
        self,
        *,
        inputs_embeds: torch.Tensor,
        tweet_positions: list[torch.Tensor],
        final_attentions: list[Optional[torch.Tensor]],
        patch_counts: list[int],
        targets: Optional[list[list[dict[str, Any]]]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        stats = self._empty_stats()
        if targets is None:
            return inputs_embeds.sum() * 0.0, stats

        visible_losses: list[torch.Tensor] = []
        none_losses: list[torch.Tensor] = []
        visible_count = none_count = 0
        point_correct = visible_correct = none_correct = 0
        inside_mass_sum = p_none_visible_sum = p_none_none_sum = 0.0
        skipped_span = skipped_mask = skipped_disabled = 0

        for batch_idx in range(int(inputs_embeds.shape[0])):
            sample_targets = targets[batch_idx] if batch_idx < len(targets) else []
            full_positions = tweet_positions[batch_idx].detach().cpu().tolist()
            full_to_local = {
                int(full_position): local_position
                for local_position, full_position in enumerate(full_positions)
            }
            attention = final_attentions[batch_idx]
            patch_count = patch_counts[batch_idx]

            for target in sample_targets[: int(self.config.max_entities_per_sample)]:
                if not bool(target.get("loss_enabled", True)):
                    skipped_disabled += 1
                    continue

                local_positions = list(
                    dict.fromkeys(
                        full_to_local[position]
                        for position in self._raw_positions(target)
                        if position in full_to_local
                    )
                )
                if attention is None or not local_positions:
                    skipped_span += 1
                    continue

                # ``attention`` is [heads, tweet_tokens, patches + None].
                selected = attention[:, local_positions, :]
                probabilities = selected.mean(dim=(0, 1)).float().clamp_min(0.0)
                normalizer = probabilities.sum()
                if not bool(torch.isfinite(normalizer)) or float(normalizer.detach().item()) <= 0.0:
                    skipped_mask += 1
                    continue
                probabilities = probabilities / normalizer
                if probabilities.numel() != patch_count + 1:
                    raise RuntimeError(
                        f"ECEB attention has {probabilities.numel()} memory positions, "
                        f"expected {patch_count + 1}"
                    )

                p_none = probabilities[-1]
                prediction = int(torch.argmax(probabilities).detach().item())

                if bool(target.get("visible")):
                    raw_weights = target.get("patch_weights")
                    if raw_weights is None:
                        skipped_mask += 1
                        continue
                    try:
                        weights = torch.as_tensor(
                            raw_weights,
                            device=probabilities.device,
                            dtype=probabilities.dtype,
                        ).flatten()
                    except (TypeError, ValueError):
                        skipped_mask += 1
                        continue
                    valid_weights = (
                        weights.numel() == patch_count
                        and bool(torch.isfinite(weights).all())
                        and bool(weights.ge(0.0).all())
                        and bool(weights.le(1.0).all())
                        and bool(weights.gt(0.0).any())
                    )
                    if not valid_weights:
                        skipped_mask += 1
                        continue

                    inside_mass = (probabilities[:-1] * weights).sum().clamp(max=1.0)
                    visible_losses.append(-inside_mass.clamp_min(1e-8).log())
                    correct = prediction < patch_count and bool(weights[prediction] > 0)
                    visible_count += 1
                    visible_correct += int(correct)
                    point_correct += int(correct)
                    inside_mass_sum += float(inside_mass.detach().item())
                    p_none_visible_sum += float(p_none.detach().item())
                else:
                    none_losses.append(-p_none.clamp_min(1e-8).log())
                    correct = prediction == patch_count
                    none_count += 1
                    none_correct += int(correct)
                    point_correct += int(correct)
                    p_none_none_sum += float(p_none.detach().item())

        # Equal class contribution prevents the more frequent None targets from
        # dominating localization.  If a batch has only one class, retain its
        # ordinary mean instead of manufacturing a missing-class loss.
        class_losses: list[torch.Tensor] = []
        if visible_losses:
            class_losses.append(torch.stack(visible_losses).mean())
        if none_losses:
            class_losses.append(torch.stack(none_losses).mean())
        if class_losses:
            loss = torch.stack(class_losses).mean()
        else:
            loss = inputs_embeds.sum() * 0.0

        entity_count = visible_count + none_count
        stats.update(
            {
                "eceb/raw_loss": float(loss.detach().item()),
                "eceb/entities": float(entity_count),
                "eceb/visible_entities": float(visible_count),
                "eceb/none_entities": float(none_count),
                "eceb/point_acc": float(point_correct / max(1, entity_count)),
                "eceb/visible_point_acc": float(visible_correct / max(1, visible_count)),
                "eceb/none_acc": float(none_correct / max(1, none_count)),
                "eceb/inside_mass": float(inside_mass_sum / max(1, visible_count)),
                "eceb/p_none_visible": float(p_none_visible_sum / max(1, visible_count)),
                "eceb/p_none_none": float(p_none_none_sum / max(1, none_count)),
                "eceb/skipped_span": float(skipped_span),
                "eceb/skipped_mask": float(skipped_mask),
                "eceb/skipped_disabled": float(skipped_disabled),
            }
        )
        return loss, stats

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        tweet_token_mask: torch.Tensor,
        image_token_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        targets: Optional[list[list[dict[str, Any]]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Run ECEB and optionally compute entity-to-region supervision.

        Args:
        inputs_embeds: Full multimodal embedding sequence
            ``[batch, sequence, hidden]``.
            tweet_token_mask: Boolean ``[batch, sequence]`` source-tweet mask.
            image_token_mask: Boolean ``[batch, sequence]`` image-patch mask.
            image_grid_thw: Per-sample pre-merge image grid ``[batch, 3]``.
            targets: Optional auxiliary region targets.  Omit at inference.
        """

        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [batch, sequence, hidden]")
        batch_size, sequence_length, hidden_size = inputs_embeds.shape
        if hidden_size != int(self.config.hidden_size):
            raise ValueError(f"Expected hidden size {self.config.hidden_size}, got {hidden_size}")
        expected_mask_shape = (batch_size, sequence_length)
        if tuple(tweet_token_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"tweet_token_mask must have shape {expected_mask_shape}, "
                f"got {tuple(tweet_token_mask.shape)}"
            )
        if tuple(image_token_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"image_token_mask must have shape {expected_mask_shape}, "
                f"got {tuple(image_token_mask.shape)}"
            )
        if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape) != (batch_size, 3):
            raise ValueError(
                f"image_grid_thw must have shape {(batch_size, 3)}, "
                f"got {tuple(image_grid_thw.shape)}"
            )

        tweet_mask = tweet_token_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        image_mask = image_token_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        if bool((tweet_mask & image_mask).any()):
            raise ValueError("tweet_token_mask and image_token_mask must not overlap")

        sample_deltas: list[torch.Tensor] = []
        tweet_positions: list[torch.Tensor] = []
        final_attentions: list[Optional[torch.Tensor]] = []
        patch_counts: list[int] = []

        for batch_idx in range(batch_size):
            sample_tweet_positions = torch.nonzero(tweet_mask[batch_idx], as_tuple=False).flatten()
            sample_image_positions = torch.nonzero(image_mask[batch_idx], as_tuple=False).flatten()
            t, grid_h, grid_w = merged_patch_grid_shape(
                image_grid_thw[batch_idx], self.config.spatial_merge_size
            )
            expected_patch_count = t * grid_h * grid_w
            if sample_image_positions.numel() != expected_patch_count:
                raise ValueError(
                    f"Sample {batch_idx}: expected {expected_patch_count} image tokens "
                    f"from grid, found {sample_image_positions.numel()}"
                )

            patch_hidden = inputs_embeds[batch_idx].index_select(0, sample_image_positions)
            patch_memory = self.patch_projection(patch_hidden)
            positions_2d = build_2d_sincos_patch_positions(
                image_grid_thw[batch_idx],
                self.config.hidden_dim,
                spatial_merge_size=self.config.spatial_merge_size,
                device=patch_memory.device,
                dtype=patch_memory.dtype,
            )
            patch_memory = patch_memory + positions_2d
            none_memory = self.none_embedding.to(
                device=patch_memory.device, dtype=patch_memory.dtype
            ).view(1, -1)
            patch_none_memory = torch.cat((patch_memory, none_memory), dim=0).unsqueeze(0)

            sample_delta = torch.zeros_like(inputs_embeds[batch_idx])
            final_attention: Optional[torch.Tensor] = None
            if sample_tweet_positions.numel() > 0:
                tweet_hidden = inputs_embeds[batch_idx].index_select(0, sample_tweet_positions)
                tweet_states = self.tweet_projection(tweet_hidden).unsqueeze(0)
                for layer in self.layers:
                    tweet_states, layer_attention = layer(tweet_states, patch_none_memory)
                projected_delta = self.output_projection(tweet_states.squeeze(0))
                sample_delta = sample_delta.index_copy(0, sample_tweet_positions, projected_delta)
                final_attention = layer_attention.squeeze(0)

            sample_deltas.append(sample_delta)
            tweet_positions.append(sample_tweet_positions)
            final_attentions.append(final_attention)
            patch_counts.append(expected_patch_count)

        delta = torch.stack(sample_deltas, dim=0)
        fused_embeds = (
            inputs_embeds + self.effective_alpha.to(dtype=delta.dtype, device=delta.device) * delta
        )
        auxiliary_loss, stats = self._set_mass_loss(
            inputs_embeds=inputs_embeds,
            tweet_positions=tweet_positions,
            final_attentions=final_attentions,
            patch_counts=patch_counts,
            targets=targets,
        )
        return fused_embeds, auxiliary_loss, stats


__all__ = [
    "EntityPatchInteractor",
    "EntityPatchInteractorConfig",
    "build_2d_sincos_patch_positions",
    "merged_patch_grid_shape",
]
