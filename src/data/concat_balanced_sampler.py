"""
ConcatBalancedBatchSampler — same guarantee as DynamicBalancedBatchSampler
(every batch contains at least `min_dynamic_per_batch` windows whose
target frame has a non-empty ground-truth moving-object mask), but
works over a torch ConcatDataset of several TemporalEVIMO2Dataset
objects (e.g. several subsets: imo + imo_II), which the original
sampler cannot do.

The dynamic/static flag for each window is computed with the original,
already-debugged `build_dynamic_window_index` (including its fix for the
worker zip-handle race), once per component dataset, then offset into
ConcatDataset index space. Batch construction mirrors the original
sampler exactly (dynamic windows drawn with replacement, static
windows enumerated once per epoch).
"""

from __future__ import annotations

import logging
import random

from torch.utils.data import Sampler

from src.data.dynamic_balanced_sampler import build_dynamic_window_index

logger = logging.getLogger(__name__)


class ConcatBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        temporal_datasets: list,
        batch_size: int,
        min_dynamic_per_batch: int = 1,
        drop_last: bool = True,
        seed: int = 42,
        is_dynamic: list[bool] | None = None,
    ):
        self.batch_size = batch_size
        self.min_dynamic_per_batch = min(min_dynamic_per_batch, batch_size)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        if is_dynamic is None:
            is_dynamic = []
            for ds in temporal_datasets:
                is_dynamic.extend(build_dynamic_window_index(ds))

        self.dynamic_indices = [i for i, d in enumerate(is_dynamic) if d]
        self.static_indices = [i for i, d in enumerate(is_dynamic) if not d]

        n_dyn, n_total = len(self.dynamic_indices), len(is_dynamic)
        logger.info(
            f"Balanced sampler: {n_dyn}/{n_total} training windows contain a moving "
            f"object ({100 * n_dyn / max(1, n_total):.1f}%)."
        )
        if n_dyn == 0:
            logger.warning(
                "Balanced sampler: NO training window contains a moving object above "
                "the speed threshold. The mask task has nothing to learn — check the "
                "sequence selection."
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        static_pool = self.static_indices[:]
        rng.shuffle(static_pool)
        all_pool = self.static_indices + self.dynamic_indices
        ptr = 0
        for _ in range(len(self)):
            batch = []
            if self.dynamic_indices:
                batch.extend(rng.choices(self.dynamic_indices, k=self.min_dynamic_per_batch))
            while len(batch) < self.batch_size:
                if ptr < len(static_pool):
                    batch.append(static_pool[ptr])
                    ptr += 1
                else:
                    batch.append(rng.choice(all_pool))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        n_total = len(self.static_indices) + len(self.dynamic_indices)
        if self.drop_last:
            return max(1, n_total // self.batch_size)
        return (n_total + self.batch_size - 1) // self.batch_size
