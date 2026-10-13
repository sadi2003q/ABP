# Second split: scene15 held out (frozen pipeline, no self-training)

Purpose: check that the main result (self-supervised reaches about 75-78% of
supervised on unseen scene13) also holds on a second unseen scene.

Method is FROZEN (decided 13 Oct 2026). No setting below may be changed after
seeing the scene15 numbers:
dense-flow labels ("smoothed" rule) -> refine (window 1, min support 2)
-> train 8 epochs -> report the LAST epoch, IoU at threshold 0.50.

Honest note for the paper: scene15 sequences were among the training
sequences of the first split, and two of them (15_00, 15_03) were used in the
small label tests while the label rules were developed. So scene15 is unseen
by the NETWORK here, but not fully unseen by the method design. Scene13 is
the cleaner unseen test.

## Split

| Role | Sequences | Frames |
|---|---|---|
| TRAIN (7) | scene13_dyn_test_01/02/03, scene14_dyn_test_00/01/02, scene6_dyn_train_03 | 3282 |
| TEST (4) | scene15_dyn_test_00/03/04/06 | 1394 |

## Part A: small test first (about 10 minutes, interactive session)

Run cells 01, 02, 03, then A1. Check the "ALL sequences pooled" block: the
`smoothed` line should give an IoU clearly above zero (first split: 0.259).
If it is near zero, stop and send me the output.

```
# cell A1: small test, labels on every 10th frame of the 3 new training sequences
!cd /kaggle/working/ABP && python flow_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence scene13_dyn_test_01_000000 scene13_dyn_test_02_000000 scene13_dyn_test_03_000000 \
    --frame-step 10 \
    --out-dir /kaggle/working/small_test_scene13
```

## Part B: full job (Save Version -> Save & Run All, about 3.5 hours)

Start a fresh session and run cells 01-08 as one committed job.
Everything in /kaggle/working is kept in the version's Output tab,
so the checkpoints are not lost this time.

```
# cell 01: clone the code
import os
os.chdir("/kaggle/working")
!rm -rf /kaggle/working/ABP
!git clone --branch selfsup --single-branch https://github.com/sadi2003q/ABP.git /kaggle/working/ABP
```

```
# cell 02: link the data folder
!mkdir -p /kaggle/working/evimo_root/right_camera/imo
!ln -sfn /kaggle/input/datasets/makkisakib1/evimo2-right-camera-imo-train/train \
    /kaggle/working/evimo_root/right_camera/imo/train
```

```
# cell 03: sequence names for the second split (scene15 held out)
TRAIN = "scene13_dyn_test_01_000000 scene13_dyn_test_02_000000 scene13_dyn_test_03_000000 scene14_dyn_test_00_000000 scene14_dyn_test_01_000000 scene14_dyn_test_02_000000 scene6_dyn_train_03_000000"
TEST = "scene15_dyn_test_00_000000 scene15_dyn_test_03_000000 scene15_dyn_test_04_000000 scene15_dyn_test_06_000000"
```

```
# cell 04: flow labels for every frame of the 7 training sequences (about 70 min)
!cd /kaggle/working/ABP && python flow_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence {TRAIN} \
    --frame-step 1 \
    --out-dir /kaggle/working/flow_labels_split15
```

```
# cell 05: refine the "smoothed" labels (about 2 min)
!cd /kaggle/working/ABP && python refine_pseudo_labels.py \
    --pseudo-label-dir /kaggle/working/flow_labels_split15/smoothed \
    --out-dir /kaggle/working/flow_labels_split15/smoothed_refined \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subset imo \
    --split train \
    --window 1 \
    --min-support 2
```

```
# cell 06: train on the automatic labels, score on scene15 (about 70 min)
!cd /kaggle/working/ABP && python train_selfsup.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {TEST} \
    --pseudo-label-dir /kaggle/working/flow_labels_split15/smoothed_refined \
    --pseudo-ignore-band 20 \
    --depth-weight 0 \
    --pose-weight 0 \
    --epochs 8 \
    --save-dir /kaggle/working/runs/flow_split15
```

```
# cell 07: SUPERVISED upper bound on the same split (true masks, about 70 min)
# Same network, same flags, only the labels differ.
!cd /kaggle/working/ABP && python train_supervised.py \
    --dataset-root /kaggle/working/evimo_root \
    --sensors right_camera \
    --subsets imo \
    --split train \
    --sequence {TRAIN} \
    --val-split train \
    --val-sequence {TEST} \
    --depth-weight 0 \
    --pose-weight 0 \
    --epochs 8 \
    --save-dir /kaggle/working/runs/supervised_split15
```

```
# cell 08: print both final results again in one place
!cat /kaggle/working/runs/flow_split15/final_metrics.json
!cat /kaggle/working/runs/supervised_split15/final_metrics.json
```

## What to send back

1. The "ALL sequences pooled" block of cell 04 (label quality, report only).
2. The table printed by cell 05.
3. The FINAL block of cell 06 (ours).
4. The FINAL block of cell 07 (supervised).

Report only the LAST EPOCH lines. The "BEST CHECKPOINT" line is chosen with
the scene15 masks and must not be reported.
