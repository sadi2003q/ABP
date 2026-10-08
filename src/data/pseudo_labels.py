"""
Pseudo-label storage for training without human mask labels.

Layout (one file per sequence):
    <pseudo_dir>/<sensor>/<sequence_name>.npz
        local_frame_indices : (N,) int     frames that have a pseudo-label
        packed              : (N, B) uint8 bit-packed boolean masks
        shape               : (2,)  int    (H, W)
    <pseudo_dir>/pseudo_label_report.json  quality vs. true masks + settings

A frame without an entry has NO pseudo-label (e.g. no depth/pose was
available); the trainer treats it as unlabelled, never as "empty".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_sequence_labels(pseudo_dir, sensor: str, sequence_name: str, labels: dict[int, np.ndarray]):
    out = Path(pseudo_dir) / sensor
    out.mkdir(parents=True, exist_ok=True)
    if not labels:
        return
    idx = np.array(sorted(labels), dtype=np.int64)
    shape = np.array(labels[int(idx[0])].shape, dtype=np.int64)
    packed = np.stack([np.packbits(labels[int(i)].astype(bool).ravel()) for i in idx])
    np.savez_compressed(out / f"{sequence_name}.npz",
                        local_frame_indices=idx, packed=packed, shape=shape)


class PseudoLabelStore:
    """Read-only lookup: (sensor, sequence_name, local_frame_index) -> bool mask or None."""

    def __init__(self, pseudo_dir):
        self.root = Path(pseudo_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Pseudo-label folder not found: {self.root}")
        self._entries: dict[tuple, np.ndarray] = {}
        self._shapes: dict[tuple, tuple] = {}
        n_files = 0
        for f in sorted(self.root.glob("*/*.npz")):
            sensor, seq = f.parent.name, f.stem
            d = np.load(f)
            shape = tuple(int(v) for v in d["shape"])
            for i, row in zip(d["local_frame_indices"], d["packed"]):
                self._entries[(sensor, seq, int(i))] = row
                self._shapes[(sensor, seq, int(i))] = shape
            n_files += 1
        if not self._entries:
            raise RuntimeError(f"No pseudo-labels found under {self.root}")
        logger.info(f"Pseudo-labels: {len(self._entries)} frames from {n_files} sequence file(s) in {self.root}")

    def __len__(self):
        return len(self._entries)

    def get(self, sensor: str, sequence_name: str, local_frame_index: int):
        key = (sensor, sequence_name, int(local_frame_index))
        row = self._entries.get(key)
        if row is None:
            return None
        H, W = self._shapes[key]
        return np.unpackbits(row, count=H * W).reshape(H, W).astype(bool)

    def has_positive(self, sensor, sequence_name, local_frame_index) -> bool:
        row = self._entries.get((sensor, sequence_name, int(local_frame_index)))
        return row is not None and bool(row.any())


def target_frame_keys(temporal_dataset):
    """For each window of a TemporalEVIMO2Dataset: (sensor, sequence_name, local_frame_index)
    of the frame the model predicts (offset 0, else the last frame)."""
    offsets = list(temporal_dataset.history_offsets)
    pos = offsets.index(0) if 0 in offsets else len(offsets) - 1
    frame_ds = temporal_dataset.frame_dataset
    refs = frame_ds.index.references
    seqs = {s.sequence_id: s for s in frame_ds.index.sequences}
    keys = []
    for window in temporal_dataset.valid_windows:
        ref = refs[window[pos]]
        s = seqs[ref.sequence_id]
        keys.append((s.sensor, s.sequence_name, int(ref.local_frame_index)))
    return keys
