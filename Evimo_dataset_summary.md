# EVIMO2v2 Dataset Summary

This document summarizes the EVIMO2v2 dataset organization, file formats, timing conventions, coordinate systems, and implementation notes for this project.

---

# 1. Dataset Overview

EVIMO2 (Event-based VIdeo Motion dataset) is an event-camera dataset containing

- Event streams
- Depth maps
- Object masks
- Classical images
- Camera intrinsics
- Object poses
- Camera poses
- IMU measurements (EVIMO2 only)

Unlike many event-camera datasets, EVIMO2 provides dense synchronized ground truth generated from Vicon motion capture.

---

# 2. Dataset Versions

The official release provides several representations.

## EVIMO2v2 (Recommended)

Each sequence is stored as a folder.

```
camera/
    category/
        subcategory/
            sequence/
```

Files are separated by modality.

Advantages

- memory mapping
- faster loading
- individual files
- scalable
- easier preprocessing

This project uses **EVIMO2v2 exclusively**.

---

## EVIMO2v1 / EVIMO

Everything is compressed into one NPZ file.

```
sequence.npz
```

This format is not used.

---

## TXT

Contains png images and text files.

Mainly intended for debugging and visualization.

---

# 3. Directory Structure

Each sequence is located at

```
<camera>/<category>/<subcategory>/<sequence_name>/
```

Example

```
left_camera/
    tabletop/
        eval/
            scene_01/
```

---

# 4. Sequence Contents

Every sequence contains

```
dataset_classical.npz
dataset_depth.npz
dataset_mask.npz

dataset_events_t.npy
dataset_events_xy.npy
dataset_events_p.npy

dataset_info.npz
```

Optionally

```
dataset_extrinsics.npz
```

if generated using

```
generate_extrinsics_npz.py
```

---

# 5. Event Representation

Events are split into three files.

## dataset_events_t.npy

Shape

```
(N,)
```

Type

```
float64
```

Contains

```
event timestamps (seconds)
```

Example

```
0.016667463
0.016667611
0.016667741
...
```

Properties

- sorted
- monotonic
- memory mappable

---

## dataset_events_xy.npy

Shape

```
(N,2)
```

Columns

```
x
y
```

Example

```
[142  81]
```

---

## dataset_events_p.npy

Shape

```
(N,)
```

Contains

```
event polarity
```

Values

```
0
1
```

Note

Samsung polarity is inverted relative to Prophesee.

---

# 6. Classical Images

```
dataset_classical.npz
```

Dictionary

```
classical_000000
classical_000001
...
```

Each entry

```
(H,W)
```

uint8 image.

---

# 7. Depth Maps

```
dataset_depth.npz
```

Dictionary

```
depth_000000
depth_000001
...
```

Each frame

```
(H,W)
```

Depth unit

```
millimeters
```

Ground truth is **not uniformly sampled in time** because Vicon occasionally loses tracking.

---

# 8. Object Masks

```
dataset_mask.npz
```

Dictionary

```
mask_000000
mask_000001
...
```

Each pixel

```
object_id × 1000
```

Example

```
5000
12000
```

Object IDs should therefore be divided by 1000.

---

# 9. dataset_info.npz

This file contains

```
meta
K
D
index
discretization
```

---

## 9.1 meta

meta is a Python dictionary.

Important entry

```
meta["frames"]
```

This is a list.

Each element corresponds to one available ground-truth frame.

Each frame contains

```
timestamp
camera pose
object poses
frame id
image names
```

Typical access

```python
frames = meta["frames"]

timestamps = np.array([
    frame["ts"]
    for frame in frames
])
```

---

## 9.2 Camera Intrinsics

Contained inside

```
meta["meta"]
```

Fields

```
fx
fy
cx
cy

k1
k2
k3
k4

p1
p2

resolution
```

Equivalent copies exist in

```
K
D
```

---

## 9.3 index

The lookup table used for timestamp search.

Example

```
index.shape

(1301,)
```

---

## 9.4 discretization

Usually

```
0.01
```

seconds.

This is related to the lookup table.

---

# 10. Extrinsics

Not distributed in EVIMO2v2.

Generated using

```
generate_extrinsics_npz.py
```

Produces

```
dataset_extrinsics.npz
```

Contents

```
translation

rotation quaternion
```

representing

```
rig → camera
```

transform.

---

# 11. Coordinate Frames

According to EVIMO documentation

Object poses

```
object → camera
```

Camera poses

```
camera → world
```

Extrinsics

```
rig → camera
```

---

# 12. Ground Truth Sampling

Depth and masks are generated from Vicon.

Therefore

Ground truth timestamps are

- irregular
- sparse
- approximately 60 Hz

Typical sequence

```
0.016667
0.033333
0.050000
...
```

---

# 13. Event Stream

Typical sequence

```
62 million events

13 seconds
```

Example statistics

```
Events

62,915,815

Frames

782

Duration

12.997 s

Event rate

≈4.84 million events/sec
```

---

# 14. Lookup Table Investigation

The official documentation states

```
index

contains event indices every discretization seconds
```

However, inspection shows more subtle behavior.

Observed

```
dt = 0.01 s
```

```
bucket 0

event time ≈0.016667
```

```
bucket 1

event time ≈0.026668
```

etc.

The lookup entry consistently points to an event approximately

```
6.67 ms

after

(bucket+1)×dt
```

rather than exactly at

```
bucket×dt
```

This appears to be intentional and likely reflects the dataset generation pipeline.

Consequences

Using the lookup table alone yields

```
Mean timestamp error

≈5–11 ms
```

whereas a full binary search gives

```
≈30 μs
```

Therefore

**the lookup table should not be interpreted as the exact event index for a timestamp.**

---

# 15. Recommended Timestamp Search

Preferred implementation

```python
idx = np.searchsorted(event_t, timestamp)
```

Complexity

```
O(log N)
```

Accuracy

```
≈30 μs
```

The lookup table should only be used if reproducing the official pipeline.

---

# 16. Memory Mapping

Recommended

```python
event_t = np.load(
    "dataset_events_t.npy",
    mmap_mode="r"
)

event_xy = np.load(
    "dataset_events_xy.npy",
    mmap_mode="r"
)

event_p = np.load(
    "dataset_events_p.npy",
    mmap_mode="r"
)
```

This avoids loading tens of millions of events into RAM.

---

# 17. Recommended Loader Order

```
Load dataset_info

↓

Load event arrays (memory mapped)

↓

Read timestamps from meta["frames"]

↓

Build frame index

↓

Binary search event indices

↓

Load depth/classical/mask lazily

↓

Return training sample
```

---

# 18. Notes for This Project

This project will

- use EVIMO2v2 format only
- ignore EVIMO2v1
- ignore TXT format
- use memory-mapped event arrays
- use binary search (`np.searchsorted`) for timestamp lookup
- load image modalities lazily
- optionally load generated camera extrinsics
- avoid relying on the provided lookup table for precise synchronization

This design provides the highest accuracy while remaining memory efficient and scalable to long event sequences.