"""
Trainer v3 — identical training loop to TrainerV2, with the model
swapped for WorldModelV3 (IMU-anchored rotation, see
src/models/world_model_v2/model_v3.py for the full rationale).

Implemented as a thin subclass that overrides ONLY `_build_model`.
Every loss term, logging line, checkpointing, EMA, optimizer,
scheduler, and eval path in TrainerV2 has already been debugged
across 9 patch cycles (see CHANGES.md / INTEGRATION_PATCH*.txt) and
is left completely untouched here, in the exact order TrainerV2's
__init__ already builds it (model -> loss -> optimizer -> dataloaders
-> scheduler -> EMA). This guarantees any IoU difference between a
v2 run and a v3 run is attributable to the model change alone, not to
an accidental trainer discrepancy — the same kind of ambiguity that
cost multiple patch cycles during v1/v2 debugging.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from trainer_v2 import TrainConfigV2, TrainerV2
from src.models.world_model_v2 import WorldModelV3

logger = logging.getLogger(__name__)


# Reuse the exact same config dataclass — no new fields needed yet.
# (Any config needed for a future accelerometer scale anchor belongs
# here once that follow-up is actually justified by results — kept
# out of scope for this change.)
TrainConfigV3 = TrainConfigV2


class TrainerV3(TrainerV2):
    def _build_model(self, cfg: TrainConfigV3) -> nn.Module:
        return WorldModelV3(
            num_bins=cfg.num_bins,
            event_channels=cfg.event_channels,
            imu_hidden=cfg.imu_hidden,
            imu_embedding=cfg.imu_embedding,
            memory_type=cfg.memory_type,
        )
