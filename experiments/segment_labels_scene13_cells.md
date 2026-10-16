# Plan A, small test: segment-level labels with tracking on scene13 (about 15-20 minutes)

New label maker `segment_labels.py`: K-means segments (30 per frame, paper
setting), one decision per segment, a lower bar for segments where an object
was found in the previous labelled frame, smallest kept blob 0.1 % (was 1 %,
which removed every scene13 object). All settings fixed before running.

Stop rule (fixed before running): continue plan A only if the NEW average
over the three scene13 sequences (event pixels, mean per frame) is >= 0.50.
Frozen labels on the same frames: 0.000 / 0.100 / 0.307 (average 0.136).

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
# cell 03: segment labels on every 5th frame of scene13 (about 15-20 minutes)
!cd /kaggle/working/ABP && python segment_labels.py \
    --dataset-root /kaggle/working/evimo_root \
    --sequence scene13_dyn_test_01_000000 scene13_dyn_test_02_000000 scene13_dyn_test_03_000000 \
    --frame-step 5 \
    --out-dir /kaggle/working/segment_test_scene13
```

Needed back: everything from the line `SEGMENT LABELS` to the end.
