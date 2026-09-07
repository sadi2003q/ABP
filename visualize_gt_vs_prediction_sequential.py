
"""
visualize_gt_vs_prediction_sequential.py

Same 5-panel comparison figure as visualize_gt_vs_prediction.py
    [ Event input ] [ Residual ] [ Ground Truth ] [ Prediction ] [ Overlay ]

but instead of ranking/selecting a subset of frames (most-GT-active
first, then a couple of empty ones, capped at --num-images), this
script saves ONE image for EVERY valid frame it collects, in the
same order the frames occur in the sequence — frame by frame.

Use this when you want to page through an entire sequence
(e.g. after filtering to one --sequence) rather than a curated
highlight reel.

This script does not duplicate logic — it imports the event-image
rendering, overlay, metrics, prediction-collection, and single-figure
saving code directly from visualize_gt_vs_prediction.py, so both
scripts always draw the figure identically and any fix made there
also applies here.
"""

import argparse
from pathlib import Path

import torch

from trainer_v2 import TrainerV2, TrainConfigV2
from visualize_gt_vs_prediction import (
    THRESHOLDS,
    collect_predictions,
    find_best_threshold,
    plot_threshold_sweep,
    save_combined_figure,
)


# -------------------------------------------------------------------------
# Sequential (frame-by-frame) saving — no ranking, no cap.
# -------------------------------------------------------------------------

def save_visualizations_sequential(samples, output_dir, threshold):
    """
    Save one comparison figure per collected sample, in the order the
    samples were collected (which follows dataset/frame order, since
    collect_predictions increments global_index monotonically as it
    walks the loader).

    Folder names use the sample's own `sample_index` (its position in
    the underlying dataset), NOT its position among the samples list.
    Frames without a GT mask are skipped upstream by
    collect_predictions, so folder numbers can have gaps — that's
    expected and lets you match a folder back to its real frame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Already in frame/sequence order — collect_predictions appends in
    # the order it walks the (non-shuffled, per this script's use of
    # shuffle=False loaders — see note in main()) dataset.
    ordered = sorted(samples, key=lambda s: s["sample_index"])

    for sample in ordered:
        frame_dir = output_dir / f"frame_{sample['sample_index']:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        metrics = save_combined_figure(
            sample,
            threshold,
            frame_dir / "comparison.png",
        )

        with open(frame_dir / "metrics.txt", "w") as f:
            f.write(f"Sample: {sample['sample_index']}\n")
            f.write(f"Threshold: {threshold:.2f}\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    print(f"Saved {len(ordered)} frame-by-frame comparison images "
          f"to {output_dir.resolve()}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize EVIMO2 Ground Truth versus the exact V2 model "
            "prediction used for F1/IoU — one image per frame, in "
            "sequence order (no ranking/selection/cap)."
        )
    )

    parser.add_argument("--dataset-root", type=str, required=True,
                         help="Path to EVIMO2 dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to trained V2 checkpoint.")
    parser.add_argument("--output-dir", type=str,
                         default="./gt_vs_prediction_sequential",
                         help="Directory for visualization output.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sensors", nargs="+", default=["left_camera"])
    parser.add_argument("--num-workers", type=int, default=4,
                         help="Number of dataloader worker processes.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default="imo",
                         help="Subset folder name under each sensor, e.g. "
                              "'imo', 'imo_II', 'sanity', 'sanity_II', "
                              "'sfm', 'sfm_II'.")
    parser.add_argument("--sequence", type=str, nargs="+", default=None,
                         help="Optional sequence name(s) (the sequence "
                              "directory name under "
                              "<sensor>/<subset>/<split>/) to restrict "
                              "visualization to. If omitted, all "
                              "sequences found under that folder are "
                              "used — but with many sequences, 'frame "
                              "by frame' output stops being one "
                              "coherent walkthrough, so passing exactly "
                              "one --sequence is recommended here.")
    parser.add_argument("--overfit", action="store_true",
                         help="Use the project's overfit dataset behavior.")
    parser.add_argument("--no-ema", action="store_true",
                         help="Evaluate current model instead of EMA weights.")
    parser.add_argument("--max-batches", type=int, default=None,
                         help="Optional cap on number of batches to run "
                              "(useful for a quick sanity check).")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Fixed decision threshold to use for every "
                              "frame. If omitted (default), the script "
                              "sweeps THRESHOLDS and picks the one with "
                              "the best global IoU, same as "
                              "visualize_gt_vs_prediction.py.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("EVIMO2 — GROUND TRUTH vs MODEL PREDICTION (frame by frame)")
    print("=" * 70)
    print()
    print("Dataset root :", args.dataset_root)
    print("Checkpoint   :", args.checkpoint)
    print("Output       :", args.output_dir)
    print("Sequence     :", args.sequence if args.sequence else "(all sequences in split)")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    cfg = TrainConfigV2(
        dataset_root=args.dataset_root,
        sensors=tuple(args.sensors),
        split=args.split,
        subset=args.subset,
        sequence=tuple(args.sequence) if args.sequence else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overfit_mode=args.overfit,
        use_ema=not args.no_ema,
    )

    trainer = TrainerV2(cfg)

    print()
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    load_result = trainer.model.load_state_dict(state_dict, strict=False)

    if load_result.missing_keys or load_result.unexpected_keys:
        n_model_params = len(list(trainer.model.state_dict().keys()))
        n_loaded = n_model_params - len(load_result.missing_keys)
        print(f"  Loaded {n_loaded}/{n_model_params} parameter tensors. "
              f"If this is far below {n_model_params}, the checkpoint is "
              f"NOT being applied correctly — treat downstream output "
              f"as untrustworthy until this is fixed.")
    else:
        print("  All checkpoint weights matched model parameters exactly.")

    trainer.model.to(device)
    trainer.model.eval()
    print("Checkpoint loaded.")

    print()
    print("Using EVIMO2 dataloader built by TrainerV2...")

    if args.split == "val":
        if trainer.val_loader is None:
            raise RuntimeError(
                "No validation loader was built (val split may be empty "
                "or overfit_mode is enabled). Use --split train instead."
            )
        loader = trainer.val_loader
    else:
        loader = trainer.train_loader

    # NOTE on ordering: TrainerV2._build_dataloaders() builds the TRAIN
    # loader with shuffle=True. For a true frame-by-frame walkthrough
    # that reads in sequence order, that shuffling would scramble the
    # output order (the folder numbers — sample_index — would still be
    # correct, but the loader would visit them out of order, which is
    # slower to reason about and irrelevant here since we sort by
    # sample_index before saving anyway). Sorting in
    # save_visualizations_sequential() makes the final saved order
    # correct regardless of loader shuffling, so no change to the
    # trainer's dataloader construction is required.
    print(f"Dataloader ready ({len(loader)} batches).")

    print()
    print("Running model...")
    print()
    print("Prediction used:")
    print("    sigmoid(model_output['mask'])")
    print()

    samples = collect_predictions(trainer, loader, max_batches=args.max_batches)

    if len(samples) == 0:
        raise RuntimeError(
            "No valid EVIMO2 samples with Ground Truth masks were found."
        )

    print(f"Collected {len(samples)} valid samples.")

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using fixed threshold: {threshold:.2f}")
    else:
        best, all_results = find_best_threshold(samples)

        print()
        print("=" * 70)
        print("THRESHOLD RESULTS")
        print("=" * 70)
        for result in all_results:
            print(
                f"Threshold {result['threshold']:.2f} | "
                f"IoU {result['iou']:.4f} | "
                f"F1 {result['f1']:.4f} | "
                f"Precision {result['precision']:.4f} | "
                f"Recall {result['recall']:.4f}"
            )

        print()
        print(f"BEST THRESHOLD: {best['threshold']:.2f}")
        print(f"BEST IoU      : {best['iou']:.4f}")
        print(f"BEST F1       : {best['f1']:.4f}")

        threshold = best["threshold"]

    print()
    print("Saving frame-by-frame visualizations...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_visualizations_sequential(
        samples=samples,
        output_dir=output_dir,
        threshold=threshold,
    )

    if args.threshold is None:
        plot_threshold_sweep(all_results, output_dir)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Visualizations saved to:", output_dir.resolve())
    print()
    print("Each frame_XXXXX folder's comparison.png shows input / ground")
    print("truth / prediction / overlay for that specific frame index.")


if __name__ == "__main__":
    main()