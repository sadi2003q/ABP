# EVIMO2 Preprocessing Pipeline

## Overview

The preprocessing stage converts the raw EVIMO2 dataset into a compact set of deterministic cache files that can be loaded efficiently during training.

The guiding philosophy of this project is:

> **Precompute only deterministic dataset information.**
>
> Anything that depends on the training configuration, model architecture, temporal sampling strategy, or loss function is computed online inside the Dataset.

This design keeps preprocessing

- deterministic
- reproducible
- lightweight
- reusable across experiments
- independent of future model changes

---

# Pipeline Overview

```
Raw EVIMO2 Sequence
│
├── dataset_info.npz
├── dataset_events_xy.npy
├── dataset_events_t.npy
├── dataset_events_p.npy
├── dataset_depth.npz
├── dataset_mask.npz
│
▼
Metadata Parsing
│
▼
Preprocessing
│
├── event_index.npz
├── frame_motion.npz
└── camera_motion.npz
│
▼
Dataset Loader
│
├── temporal window selection
├── event slicing
├── augmentations
├── target generation
└── training sample
```

The Dataset class never reparses the raw metadata or scans event timestamps during training.

Instead, it loads lightweight cache files generated once during preprocessing.

---

# Dataset Structure

A processed sequence has the following structure.

```
sequence/

├── dataset_info.npz
├── dataset_events_xy.npy
├── dataset_events_t.npy
├── dataset_events_p.npy
├── dataset_depth.npz
├── dataset_mask.npz
│
└── cache/
    ├── event_index.npz
    ├── frame_motion.npz
    └── camera_motion.npz
```

---

# Generated Cache Files

## 1. Event Index Cache

File

```
cache/event_index.npz
```

### Purpose

Provides a fast lookup from timestamps to event indices.

Without this cache, every training sample would require binary searching tens of millions of timestamps.

### Stored information

- frame timestamps
- event index for every frame
- discretization metadata
- camera intrinsics

### Used for

- event slicing
- temporal window extraction
- online event accumulation

---

## 2. Frame Motion Cache

File

```
cache/frame_motion.npz
```

### Purpose

Stores the motion of every annotated object between consecutive frames.

Object trajectories are reconstructed using

- camera pose
- object pose
- rigid-body transformations

The preprocessing computes

- world trajectories
- per-frame displacement
- linear speed
- dynamic/static classification

No image processing is involved.

### Stored information

Typical cached quantities include

- object IDs
- visibility
- object positions
- displacement
- linear speed
- dynamic flags

This cache will later provide self-supervised motion targets.

---

## 3. Camera Motion Cache

File

```
cache/camera_motion.npz
```

### Purpose

Stores the ego-motion of the event camera.

Unlike `frame_motion`, this describes how the sensor itself moves between frames.

### Stored information

- frame IDs
- timestamps
- frame time interval (`dt`)
- pose availability
- camera translation
- camera quaternion
- translation difference
- linear speed
- relative quaternion
- angular speed

---

# Metadata Parsing

The parser converts the raw EVIMO2 metadata into project datatypes.

The parser intentionally ignores

```
flea3_7
```

metadata because only the event-camera streams are used for training.

The RGB camera remains available later for visualization.

---

# Camera Pose Parsing

Each frame may contain

```
camera_pose
```

or

```
camera_pose = None
```

when no pose is available.

The parser converts every valid pose into

```
Pose
├── translation
└── quaternion
```

using

```
translation
quaternion (x,y,z,w)
```

---

# Object Parsing

Unlike earlier versions of the parser, object metadata is **not** assumed to be globally declared.

Instead,

every frame is scanned independently.

Whenever an object appears inside a frame,

```
frame
    └── object_id
```

that object is added to the global object dictionary.

Consequently,

```
objects
```

becomes the union of

```
all object IDs appearing anywhere
```

rather than assuming every object is visible in every frame.

This matches the actual EVIMO2 annotation format.

---

# Availability of Ground Truth

Depth and masks are not available for every frame.

During metadata parsing, the parser first scans

```
dataset_depth.npz
dataset_mask.npz
```

to determine which frame IDs actually exist.

Each frame stores

```
depth_available
mask_available
```

allowing the Dataset to know immediately whether a particular frame has supervision.

---

# Camera Pose Availability

Some EVIMO2 sequences contain missing camera poses.

Example

```
Frames : 404

Missing poses

61
62
63
...
70

289
290
```

For these frames

```
camera_pose = None
```

The preprocessing intentionally

- stores zero translation
- stores identity quaternion
- stores zero translation delta
- stores zero linear speed
- stores identity delta quaternion
- stores zero angular speed

while recording

```
pose_available = False
```

This preserves

- one cache entry per frame
- frame indexing consistency
- temporal alignment

without fabricating motion.

---

# Frame Time Interval

Camera motion additionally stores

```
dt
```

for every frame.

```
dt[i] =
timestamp[i] - timestamp[i-1]
```

The first frame stores

```
dt = 0
```

This avoids recomputing frame intervals during training and supports velocity-based models directly.

---

# Cache Generation Policy

Each cache is generated independently.

If a cache already exists,

it is skipped unless

```
--overwrite
```

is specified.

Example

```
event_index.npz      ✓ existing
frame_motion.npz     ✓ existing
camera_motion.npz    missing
```

Running

```
--camera-motion
```

will generate only

```
camera_motion.npz
```

without touching the other caches.

---

# Command Line Interface

Generate everything

```bash
python -m tools.preprocessing.preprocess \
    ~/HDD/EventDatasets/EVIMO2_official
```

Overwrite everything

```bash
python -m tools.preprocessing.preprocess \
    ~/HDD/EventDatasets/EVIMO2_official \
    --overwrite
```

Generate only event index

```bash
python -m tools.preprocessing.preprocess \
    DATASET \
    --event-index
```

Generate only frame motion

```bash
python -m tools.preprocessing.preprocess \
    DATASET \
    --frame-motion
```

Generate only camera motion

```bash
python -m tools.preprocessing.preprocess \
    DATASET \
    --camera-motion
```

Generate a single sequence

```bash
python -m tools.preprocessing.preprocess \
    DATASET \
    --camera-motion \
    --sequence "left_camera/imo/train/scene13_dyn_test_01_000000"
```

---

# Multi-Sensor Organization

The official EVIMO2 dataset contains four camera folders.

```
flea3_7/
left_camera/
right_camera/
samsung_mono/
```

## flea3_7

RGB camera only.

Used later for

- visualization
- debugging
- qualitative inspection

Not used for event training.

---

## left_camera

Event camera.

---

## right_camera

Event camera.

---

## samsung_mono

Event camera.

---

The preprocessing is sensor-independent.

Every sequence is processed identically regardless of which event camera produced it.

Later, the Dataset class can be configured to use

- one event sensor
- any subset of event sensors
- all event sensors

while still returning synchronized

- RGB image
- depth map
- segmentation mask

for visualization and verification.

---

# Robustness Improvements

Several EVIMO2 dataset inconsistencies were discovered during development.

## 1. Missing Camera Poses

Some frames legitimately contain

```
camera_pose = None
```

These are handled by

- pose availability flag
- zero motion
- identity rotation

instead of removing frames.

---

## 2. Object Visibility

An object is **not guaranteed** to appear in every frame.

Earlier parser versions incorrectly assumed the metadata contained a complete object list.

This occasionally produced preprocessing failures.

The parser now constructs the object dictionary dynamically by scanning every frame.

Therefore,

```
dataset.objects
```

always equals

```
Union of all object IDs appearing anywhere in the sequence.
```

This correctly reflects the EVIMO2 annotation format and prevents failures when objects enter or leave the camera field of view.

---

# Design Philosophy

The preprocessing stage intentionally avoids computing anything that depends on training configuration.

Examples intentionally **not** cached include

- event voxel grids
- event images
- temporal windows
- frame pairs
- supervision targets
- augmentations
- normalization
- feature extraction

These belong inside the Dataset class so they remain experiment-dependent.

---

# Current Preprocessing Outputs

Every processed sequence now contains

```
cache/

event_index.npz
frame_motion.npz
camera_motion.npz
```

These caches provide all deterministic information required by the Dataset while keeping preprocessing completely independent of future research experiments.