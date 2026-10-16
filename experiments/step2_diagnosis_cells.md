# Step 2: why does the SUPERVISED network not generalise to unseen scenes?

Purpose: find which part of the shared training setup (input, sensor input,
loss, augmentation) makes the network collapse on unseen scenes. Every fix
found here also applies to the self-supervised version (same trainer).

Observed problem (scene15 held out, 15 Oct 2026): supervised reached only
0.056 IoU at 0.5 and predicted about 12x too few moving pixels
(predicted fraction 0.005, true 0.064). Validation IoU was best at epoch 1
and fell afterwards while the training loss kept dropping (over-fitting to
the training scenes).

## Honesty rule for this step

All choices are made on a VALIDATION scene (scene14). Scene13 is never
loaded in this step, so it stays a clean final test. Scene15 is used for
training here (it is a test scene only in split 2; caveat already noted).

| Role | Sequences | Frames |
|---|---|---|
| TRAIN (5) | scene15_dyn_test_00/03/04/06, scene6_dyn_train_03 | 1498 |
| VALIDATION (3) | scene14_dyn_test_00/01/02 | 1220 |
| NOT USED | scene13 (all) | – |

## What each run changes (one change at a time vs. R0)

| Run | Change | Question |
|---|---|---|
| R0 | none (current setup) | baseline on this split |
| R1 | `--normalize-voxels` | does scene-dependent event density break it? |
| R2 | `--no-imu` | does the rotation-sensor input over-fit to the training trajectories? |
| R3 | `--dice-weight 0` | does batch Dice on tiny objects push it to predict nothing? |
| R4 | `--no-imu --augment --event-dropout 0.1` | does mirror + zoom (object size) augmentation help? |

New score in every log: "event-pixel IoU" = overlap counted only on pixels
with events, in frames with a moving object (the style of published EVIMO2
numbers). Reported next to the usual all-pixel IoU; never used for training.

Decide with: LAST EPOCH, IoU at 0.50 (all pixels) AND event-pixel per-frame.
A change counts as real only if it beats R0 by clearly more than 0.01
(run-to-run noise seen earlier).

## Cells

```python
# cell 01: clone the code (selfsup branch)
import os
os.chdir("/kaggle/working")
!rm -rf /kaggle/working/ABP
!git clone --branch selfsup --single-branch https://github.com/sadi2003q/ABP.git /kaggle/working/ABP
```

```python
# cell 02: link the data folder
!mkdir -p /kaggle/working/evimo_root/right_camera/imo
!ln -sfn /kaggle/input/datasets/makkisakib1/evimo2-right-camera-imo-train/train \
    /kaggle/working/evimo_root/right_camera/imo/train
```

```python
# cell 03: sequence names for the diagnosis split (scene13 is NOT used)
TRAIN = "scene15_dyn_test_00_000000 scene15_dyn_test_03_000000 scene15_dyn_test_04_000000 scene15_dyn_test_06_000000 scene6_dyn_train_03_000000"
VAL = "scene14_dyn_test_00_000000 scene14_dyn_test_01_000000 scene14_dyn_test_02_000000"
```

### Part A: small code check (about 5 minutes)

All new code paths in one tiny run (1 short sequence, 1 epoch). It only
checks that nothing crashes; the numbers mean nothing.

```python
# cell A1: tiny run with every new option switched on
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence scene15_dyn_test_06_000000 \
    --val-split train \
    --val-sequence scene14_dyn_test_02_000000 \
    --depth-weight 0 \
    --pose-weight 0 \
    --no-imu \
    --normalize-voxels \
    --augment \
    --event-dropout 0.1 \
    --epochs 1 \
    --save-dir /kaggle/working/runs/step2_codecheck
```

Pass = it finishes and prints a FINAL block with an "Event-pixel IoU" line.

### Part B: the five diagnosis runs (about 20 minutes each, about 1 h 45 min total)

```python
# cell B1: R0 baseline (current setup)
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {VAL} \
    --depth-weight 0 \
    --pose-weight 0 \
    --epochs 4 \
    --save-dir /kaggle/working/runs/step2_R0_baseline
```

```python
# cell B2: R1 voxel normalisation
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {VAL} \
    --depth-weight 0 \
    --pose-weight 0 \
    --normalize-voxels \
    --epochs 4 \
    --save-dir /kaggle/working/runs/step2_R1_normalize
```

```python
# cell B3: R2 no rotation-sensor input
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {VAL} \
    --depth-weight 0 \
    --pose-weight 0 \
    --no-imu \
    --epochs 4 \
    --save-dir /kaggle/working/runs/step2_R2_noimu
```

```python
# cell B4: R3 no Dice loss (cross-entropy only)
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {VAL} \
    --depth-weight 0 \
    --pose-weight 0 \
    --dice-weight 0 \
    --epochs 4 \
    --save-dir /kaggle/working/runs/step2_R3_nodice
```

```python
# cell B5: R4 mirror + zoom augmentation (needs no rotation sensor)
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {VAL} \
    --depth-weight 0 \
    --pose-weight 0 \
    --no-imu \
    --augment \
    --event-dropout 0.1 \
    --epochs 4 \
    --save-dir /kaggle/working/runs/step2_R4_augment
```

```python
# cell B6: one table with the LAST EPOCH of every run
import json

# Step 1: the run folders, in order
run_names = [
    "step2_R0_baseline",
    "step2_R1_normalize",
    "step2_R2_noimu",
    "step2_R3_nodice",
    "step2_R4_augment",
]

# Step 2: print the table header
print(f"{'run':22s} {'IoU@0.5':>8s} {'P':>6s} {'R':>6s} {'event pooled':>13s} {'event per-frame':>16s}")

# Step 3: one line per run
for name in run_names:
    path = f"/kaggle/working/runs/{name}/final_metrics.json"

    # Step 3a: skip runs that did not finish
    if not os.path.exists(path):
        print(f"{name:22s} (no final_metrics.json)")
        continue

    # Step 3b: read the LAST EPOCH block (no choice made with validation masks)
    with open(path) as f:
        report = json.load(f)
    last = report["last_epoch"]

    # Step 3c: print the numbers
    print(f"{name:22s} {last['iou_at_0.5']:8.4f} {last['precision']:6.3f} {last['recall']:6.3f} "
          f"{last['event_iou_pooled']:13.4f} {last['event_iou_mean_per_frame']:16.4f}")
```

```python
# cell B7: keep the results (small files only) so they survive the session
!cd /kaggle/working && zip -r step2_results.zip runs/step2_*/final_metrics.json runs/step2_*/train.log
```

## Needed back

The table from cell B6, plus the `[train | epoch N]` lines of R0 and of the
best run (to see whether validation still falls after epoch 1).

## What happens next

- One or more changes clearly beat R0 -> combine the winners in one run (R5)
  and check the combination on the same validation scene.
- Supervised with the winning setup reaches about 0.45 or more on scene14
  -> rerun both splits (scene13 and scene15 held out), ours and supervised.
- Nothing helps -> the network itself is the limit; decide on a different
  network or reframe the paper.
