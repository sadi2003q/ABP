"""
Stage A tests: pseudo-label storage, generation, and LEAK-FREE training.

    python tests/test_pseudo_labels.py

The leak test trains twice on synthetic batches whose true masks are
identical. Run 1 uses correct pseudo-labels, run 2 uses pseudo-labels
shifted 20 pixels sideways. If the true masks leaked into training,
both runs would match the truth; without leaks, run 2 must not.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.pseudo_labels import save_sequence_labels, PseudoLabelStore
from generate_pseudo_labels import pseudo_score, score_to_label
from test_selfsup_signal import make_scene, K, DIST
from test_supervised_model import make_batch, FakeLoader, make_transform, DYNAMIC_ID
from trainer_supervised import TrainConfigSupervised, TrainerSupervised


def test_store_roundtrip(tmp):
    rng = np.random.default_rng(0)
    labels = {3: rng.random((48, 64)) > 0.7, 7: np.zeros((48, 64), bool)}
    save_sequence_labels(tmp / "store", "left_camera", "seqA", labels)
    store = PseudoLabelStore(tmp / "store")
    assert np.array_equal(store.get("left_camera", "seqA", 3), labels[3])
    assert store.get("left_camera", "seqA", 5) is None          # unlabelled, not "empty"
    assert store.has_positive("left_camera", "seqA", 3) and not store.has_positive("left_camera", "seqA", 7)
    print("  store save / load ................... ok")


def test_generator_on_synthetic_scene():
    xy, tt, R, t, depth, obj = make_scene()
    score = pseudo_score(xy, tt, 0.0, 1.0, K, DIST, R, t, depth)
    label = score_to_label(score, 0.3, 0.003)
    iou = (label & obj).sum() / max(1, (label | obj).sum())
    print(f"  generator pseudo-label IoU on synthetic scene = {iou:.3f}")
    assert iou > 0.6
    print("  generator ............................ ok")


def _attach_ids(pairs, B):
    """Give every sample a unique (sensor, sequence, local frame) identity."""
    for n, (raw, _) in enumerate(pairs):
        last = raw.frames[-1]
        last.sensors = ["left_camera"] * B
        last.sequence_names = ["synthetic_seq"] * B
        last.local_frame_indices = np.arange(n * B, (n + 1) * B)


def _true_binary(raw_mask):
    return raw_mask == DYNAMIC_ID * 1000


def _train(tmp, name, train, val, shift_px):
    labels = {}
    for raw, _ in train.pairs:
        last = raw.frames[-1]
        for k, m in enumerate(last.mask):
            lab = _true_binary(m)
            labels[int(last.local_frame_indices[k])] = np.roll(lab, shift_px, axis=1) if shift_px else lab
    save_sequence_labels(tmp / name, "left_camera", "synthetic_seq", labels)

    cfg = TrainConfigSupervised(
        dataset_root="unused", history_offsets=(-2, -1, 0), num_bins=5,
        label_source="pseudo", pseudo_label_dir=str(tmp / name),
        depth_weight=0.0, pose_weight=0.0,
        epochs=10, learning_rate=1e-3, eval_every_n_epochs=2,
        log_every_n_steps=1000, viz_every_n_steps=0,
        save_dir=str(tmp / f"run_{name}"), num_workers=0, mixed_precision="none",
    )
    trainer = TrainerSupervised(cfg, train_loader=train, eval_loader=val,
                                transform=make_transform([train, val]), eval_tag="synthetic_val")
    trainer.train()
    return trainer.best_metric


def test_training_uses_pseudo_labels_only(tmp):
    rng = np.random.default_rng(3)
    B = 4
    train = FakeLoader([make_batch(rng, B=B) for _ in range(10)])
    val = FakeLoader([make_batch(rng, B=B) for _ in range(3)])
    _attach_ids(train.pairs, B)
    _attach_ids(val.pairs, B)

    correct = _train(tmp, "correct", train, val, shift_px=0)
    shifted = _train(tmp, "shifted", train, val, shift_px=20)
    print(f"  IoU vs TRUE masks: trained on correct pseudo-labels = {correct:.3f}, "
          f"on shifted pseudo-labels = {shifted:.3f}")
    assert correct > 0.85, "should learn from correct pseudo-labels"
    assert shifted < 0.3, "true masks leaked into training"
    print("  training reads pseudo-labels only .... ok")


def test_ignore_band_weight():
    from src.models.supervised_model.targets import ignore_band_weight
    lab = np.zeros((40, 40), bool)
    lab[15:25, 15:25] = True
    w = ignore_band_weight(lab, 3)
    assert (w[lab] == 1).all(), "labelled pixels must keep weight 1"
    assert w[15, 12] == 0 and w[15, 11] == 1, "ring must be exactly 3 px wide"
    assert w[0, 0] == 1, "far background keeps weight 1"
    assert (ignore_band_weight(lab, 0) == 1).all()
    assert (ignore_band_weight(np.zeros((40, 40), bool), 3) == 1).all(), "empty label: nothing ignored"
    print("  ignore-band weights ................. ok")


def test_loss_ignores_band():
    import torch
    from src.models.supervised_model.loss import SupervisedLoss
    from src.models.supervised_model.targets import ignore_band_weight
    lab = np.zeros((40, 40), bool)
    lab[15:25, 15:25] = True
    w = torch.from_numpy(ignore_band_weight(lab, 4))[None, None]
    targets = {"gt_mask": torch.from_numpy(lab.astype(np.float32))[None, None],
               "mask_valid": torch.tensor([True]), "mask_weight": w}
    loss_fn = SupervisedLoss(pos_weight=5.0, depth_weight=0.0, pose_weight=0.0)
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 40, 40)
    base = loss_fn({"mask": logits}, targets)["loss"]

    in_ring = logits.clone(); in_ring[w == 0] += 5.0
    assert torch.allclose(loss_fn({"mask": in_ring}, targets)["loss"], base), "ring pixels must not affect the loss"

    outside = logits.clone(); outside[0, 0, 0, 0] += 5.0
    assert not torch.allclose(loss_fn({"mask": outside}, targets)["loss"], base), "other pixels must still count"

    # Without a weight map the loss is unchanged from the ground-truth version
    plain = dict(targets); plain.pop("mask_weight")
    ones = dict(targets); ones["mask_weight"] = torch.ones_like(w)
    assert torch.allclose(loss_fn({"mask": logits}, plain)["loss"], loss_fn({"mask": logits}, ones)["loss"], atol=1e-6)
    print("  loss ignores the band only .......... ok")


def _train_partial_labels(tmp, name, train, val, erode_px, band):
    """Pseudo-labels that cover only the CENTRE of each object (eroded true mask)."""
    from scipy import ndimage
    labels = {}
    for raw, _ in train.pairs:
        last = raw.frames[-1]
        for k, m in enumerate(last.mask):
            lab = _true_binary(m)
            if lab.any():
                lab = ndimage.binary_erosion(lab, iterations=erode_px)
            labels[int(last.local_frame_indices[k])] = lab
    save_sequence_labels(tmp / name, "left_camera", "synthetic_seq", labels)
    cfg = TrainConfigSupervised(
        dataset_root="unused", history_offsets=(-2, -1, 0), num_bins=5,
        label_source="pseudo", pseudo_label_dir=str(tmp / name), pseudo_ignore_band=band,
        depth_weight=0.0, pose_weight=0.0,
        epochs=12, learning_rate=1e-3, eval_every_n_epochs=2,
        log_every_n_steps=1000, viz_every_n_steps=0,
        save_dir=str(tmp / f"run_{name}"), num_workers=0, mixed_precision="none",
    )
    trainer = TrainerSupervised(cfg, train_loader=train, eval_loader=val,
                                transform=make_transform([train, val]), eval_tag="synthetic_val")
    trainer.train()
    return trainer.best_metric


def test_band_helps_with_partial_labels(tmp):
    """Informational: labels cover only the centre 4x4 of each 10x10 object."""
    rng = np.random.default_rng(5)
    B = 4
    train = FakeLoader([make_batch(rng, B=B) for _ in range(10)])
    val = FakeLoader([make_batch(rng, B=B) for _ in range(3)])
    _attach_ids(train.pairs, B)
    _attach_ids(val.pairs, B)
    label_iou = (4 * 4) / (10 * 10)
    without = _train_partial_labels(tmp, "partial_noband", train, val, erode_px=3, band=0)
    with_band = _train_partial_labels(tmp, "partial_band", train, val, erode_px=3, band=4)
    print(f"  partial labels (IoU vs truth {label_iou:.2f}): model without band = {without:.3f}, "
          f"with band = {with_band:.3f}")
    if with_band <= without:
        print("  NOTE: no gain from the band on this synthetic case (not an error; report it)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    tmp = Path(tempfile.mkdtemp())
    try:
        print("Stage A (pseudo-labels) self-test")
        test_store_roundtrip(tmp)
        test_generator_on_synthetic_scene()
        test_training_uses_pseudo_labels_only(tmp)
        test_ignore_band_weight()
        test_loss_ignores_band()
        test_band_helps_with_partial_labels(tmp)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
