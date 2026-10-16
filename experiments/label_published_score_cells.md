# Quick check: the label generator scored the PUBLISHED way (about 30 minutes, no training)

Why: Step 2 showed that no training fix helps the network, but the label
generator (geometry only) beats the trained network on the hard parallax
sequences. The best recent event-only EVIMO2 method (normal flow + planar
background + clustering, arXiv 2507.14500) reports about 0.77 IoU, scored
only on EVENT PIXELS and only in frames that contain a moving object. We
have never scored our labels that way.

What is scored: the FROZEN method ("smoothed" rule, unchanged since 12 Oct).
The other rules are printed for reference only and must not be used to pick
a new rule (that would be choosing with the true masks).

Honest note: scene13 is the cleanest (not used to design the label rules);
scene14_00/01/02, scene15_00/03 and scene6_03 were used in the small tests
while the rules were developed. The fully clean test is the official
`imo/eval` split, which we do not have on Kaggle yet.

## Decision rule (fixed BEFORE running)

| Frozen method, average over sequences (event pixels, per frame) | Meaning |
|---|---|
| about 0.6 or more | competitive with published optimisation methods (0.61-0.77) -> get the official eval split and build the paper on it |
| about 0.4 to 0.6 | promising; improve the parallax cases (segment-level decisions, merging over time) |
| below about 0.4 | the geometry route is also too weak -> rethink |

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
# cell 03: all 11 right-camera sequences we have
ALL_SEQUENCES = "scene13_dyn_test_01_000000 scene13_dyn_test_02_000000 scene13_dyn_test_03_000000 scene14_dyn_test_00_000000 scene14_dyn_test_01_000000 scene14_dyn_test_02_000000 scene15_dyn_test_00_000000 scene15_dyn_test_03_000000 scene15_dyn_test_04_000000 scene15_dyn_test_06_000000 scene6_dyn_train_03_000000"
```

```python
# cell 04: labels on every 5th frame, scored both ways (about 25-30 minutes)
!cd /kaggle/working/ABP && python flow_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence {ALL_SEQUENCES} \
    --frame-step 5 \
    --out-dir /kaggle/working/published_score_check
```

## Needed back

Everything from the line `PUBLISHED-STYLE SCORE` to the end of the output
(the per-sequence blocks, the ALL block and the "FROZEN method ... average"
line).
