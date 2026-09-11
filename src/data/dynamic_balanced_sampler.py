"""
DynamicBalancedBatchSampler.

Problem
-------
EVIMO2 sequences are mostly static: in a typical clip, only a small
contiguous fraction of frames have any object moving fast enough to
count as "dynamic" GT (see MOTION_THRESHOLD_SPEED in
src/utils/metrics.py). With a plain shuffled DataLoader, a mask-
prediction model can go for many consecutive batches without ever
seeing a positive (dynamic) pixel, which makes training progress
very hard to read from the loss/dr curves and can slow convergence.

This sampler fixes that by guaranteeing EVERY batch contains at
least `min_dynamic_per_batch` windows whose TARGET frame has a
non-empty GT dynamic mask, with the rest of the batch filled by
random draws from the full pool (including more dynamic windows,
since sampling is with replacement across epochs by design -- see
Note below). This does not change what the GT mask IS (still the
real speed-thresholded EVIMO2 mask, computed identically to
src/utils/metrics.py) -- it only changes which windows get grouped
into the same batch, so every logged training step carries a
meaningful gt_dr signal to watch.

How the dynamic/static split is computed
-----------------------------------------
For each window in `TemporalEVIMO2Dataset.valid_windows`, we look at
the window's TARGET frame (the one whose mask GTMaskModel is
actually supervised against -- window_indices[-1]) and check whether
it has ANY dynamic pixel, using the exact same
`get_dynamic_object_ids` + `evimo2_mask_to_binary_dynamic` functions
used everywhere else in this repo. This only touches the mask
reader (cheap) and the cached frame_motion (already in memory) --
it does NOT load events, depth, IMU, or RGB, so building the index
is fast even for large datasets.

Note on sampling semantics
---------------------------
This sampler enumerates one pass over "static" windows per epoch
(so every static window is still seen exactly once per epoch, same
as a normal DataLoader), but draws dynamic windows WITH replacement
to fill each batch's guaranteed dynamic slot(s) -- because there are
usually far fewer dynamic windows than batches per epoch. If you'd
rather each dynamic window also appear exactly once per epoch (and
are fine with fewer batches actually containing one, once the
dynamic pool is exhausted), set `exhaust_dynamic_pool=True`.
"""

from __future__ import annotations

import logging
import random

import numpy as np
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


def build_dynamic_window_index(temporal_dataset) -> list[bool]:
    """
    Returns a list of length len(temporal_dataset): True at index i
    iff temporal_dataset[i]'s TARGET frame has a non-empty GT dynamic
    mask.

    IMPORTANT: this opens its OWN independent np.load() handle per
    sequence's dataset_mask.npz, separate from
    frame_dataset.readers[sid]. Do NOT call reader.load_mask() (the
    live EVIMO2Reader instance shared with the dataset) here: that
    lazily opens and CACHES an npz zip handle on `reader._mask`, and
    since this function runs in the main process before the
    DataLoader forks/spawns its workers, that live reader object
    (and its now-open, partially-read zip handle) gets duplicated
    into every worker process. Workers then race on the same
    zip file offset, corrupting reads and raising
    `zipfile.BadZipFile: Bad magic number for file header` --
    nondeterministically, on whichever sequence/worker loses the
    race. Opening a throwaway np.load() here (like
    EVIMO2Reader._build_index does for its own one-off index scan)
    and closing it before returning avoids touching any object that
    later gets forked into a worker.
    """
    from src.utils.metrics import get_dynamic_object_ids, evimo2_mask_to_binary_dynamic

    frame_dataset = temporal_dataset.frame_dataset
    references = frame_dataset.index.references

    # One independent, throwaway np.load() per sequence (not the
    # shared reader's cached self._mask), closed via `with` before we
    # return -- so nothing forked into DataLoader workers is touched.
    mask_archives: dict = {}

    def _get_mask_archive(sequence_id, mask_file):
        if sequence_id not in mask_archives:
            mask_archives[sequence_id] = (
                np.load(mask_file) if mask_file.exists() else None
            )
        return mask_archives[sequence_id]

    is_dynamic = []
    try:
        for window_indices in temporal_dataset.valid_windows:
            target_global_idx = window_indices[-1]
            ref = references[target_global_idx]
            sequence = frame_dataset.index.sequences[ref.sequence_id]
            parser = frame_dataset.parsers[sequence.sequence_id]
            reader = frame_dataset.readers[sequence.sequence_id]
            frame = parser.frames[ref.local_frame_index]

            mask = None
            if frame_dataset.load_mask:
                archive = _get_mask_archive(sequence.sequence_id, reader.mask_file)
                if archive is not None:
                    key = f"mask_{frame.frame_id:010d}"
                    if key in archive.files:
                        mask = archive[key]

            if mask is None:
                is_dynamic.append(False)
                continue

            motion_cache = frame_dataset.frame_motion[sequence.sequence_id]

            class _FM:
                pass
            fm = _FM()
            fm.object_ids = motion_cache.object_ids
            fm.speed = motion_cache.speed[ref.local_frame_index]

            ids = get_dynamic_object_ids(fm)
            gt = evimo2_mask_to_binary_dynamic(mask, ids)
            is_dynamic.append(bool(gt.any()))
    finally:
        for archive in mask_archives.values():
            if archive is not None:
                archive.close()

    return is_dynamic


class DynamicBalancedBatchSampler(Sampler[list[int]]):
    """
    Yields batches of dataset indices where every batch contains at
    least `min_dynamic_per_batch` windows with a non-empty GT dynamic
    mask (subject to at least that many dynamic windows existing at
    all in the dataset -- if the dataset has zero dynamic windows,
    this degrades to plain random batching, same as before).

    Parameters
    ----------
    temporal_dataset : TemporalEVIMO2Dataset
    batch_size : int
    min_dynamic_per_batch : int
        How many dynamic windows to guarantee per batch. Default 1.
    drop_last : bool
        Same semantics as DataLoader's drop_last.
    exhaust_dynamic_pool : bool
        If True, each dynamic window is used at most once per epoch
        (sampling without replacement) -- once the pool runs out,
        remaining batches fall back to plain random draws from
        whatever's left. If False (default), dynamic windows are
        drawn WITH replacement each batch, so every batch keeps its
        guarantee even if there are very few dynamic windows total
        (the common case for a single short sequence).
    seed : int
        Reshuffled every epoch via `set_epoch`, same convention as
        torch's DistributedSampler, so you get a different mix each
        epoch but reproducible runs given the same seed.
    """

    def __init__(
        self,
        temporal_dataset,
        batch_size: int,
        min_dynamic_per_batch: int = 1,
        drop_last: bool = True,
        exhaust_dynamic_pool: bool = False,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.min_dynamic_per_batch = min(min_dynamic_per_batch, batch_size)
        self.drop_last = drop_last
        self.exhaust_dynamic_pool = exhaust_dynamic_pool
        self.seed = seed
        self.epoch = 0

        is_dynamic = build_dynamic_window_index(temporal_dataset)
        self.dynamic_indices = [i for i, d in enumerate(is_dynamic) if d]
        self.static_indices = [i for i, d in enumerate(is_dynamic) if not d]

        n_dyn, n_total = len(self.dynamic_indices), len(is_dynamic)
        logger.info(
            f"DynamicBalancedBatchSampler: {n_dyn}/{n_total} windows have "
            f"non-empty GT dynamic mask ({100*n_dyn/max(1,n_total):.1f}%)."
        )
        if n_dyn == 0:
            logger.warning(
                "DynamicBalancedBatchSampler: NO windows have any dynamic "
                "GT pixels -- the guarantee cannot be met, falling back to "
                "plain random batching. Check MOTION_THRESHOLD_SPEED / "
                "your sequence selection if this is unexpected."
            )

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch for a different (but
        reproducible) shuffle -- mirrors DistributedSampler's API."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        static_pool = self.static_indices[:]
        rng.shuffle(static_pool)

        if self.exhaust_dynamic_pool:
            dynamic_pool = self.dynamic_indices[:]
            rng.shuffle(dynamic_pool)
        else:
            dynamic_pool = None  # sampled with replacement on demand

        n_batches = len(self) if not self.drop_last else len(self)
        static_ptr = 0
        dynamic_ptr = 0

        for _ in range(n_batches):
            batch = []

            if self.dynamic_indices:
                n_dyn_this_batch = self.min_dynamic_per_batch
                if self.exhaust_dynamic_pool:
                    remaining = len(dynamic_pool) - dynamic_ptr
                    n_dyn_this_batch = min(n_dyn_this_batch, max(0, remaining))
                    for _ in range(n_dyn_this_batch):
                        batch.append(dynamic_pool[dynamic_ptr])
                        dynamic_ptr += 1
                else:
                    batch.extend(rng.choices(self.dynamic_indices, k=n_dyn_this_batch))

            n_remaining = self.batch_size - len(batch)
            for _ in range(n_remaining):
                if static_ptr >= len(static_pool):
                    # Ran out of fresh static windows this epoch --
                    # top up from the full pool (static+dynamic) with
                    # replacement rather than under-filling the batch.
                    all_pool = self.static_indices + self.dynamic_indices
                    batch.append(rng.choice(all_pool))
                else:
                    batch.append(static_pool[static_ptr])
                    static_ptr += 1

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        n_total = len(self.static_indices) + len(self.dynamic_indices)
        if self.drop_last:
            return n_total // self.batch_size
        return (n_total + self.batch_size - 1) // self.batch_size