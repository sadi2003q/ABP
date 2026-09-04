# EventCameraProject

**Self-Supervised Dynamic Object Detection from Event Cameras**

This repository contains the implementation of a self-supervised framework for learning dynamic object detection from event camera data. The long-term objective is to build a geometry-aware and temporally consistent perception system capable of separating ego-motion-induced events from independently moving objects without requiring dense pixel-level annotations.

---

# Project Goals

The project is divided into several stages.

### Stage 1 – Data Pipeline

* Dataset management
* Dataset inspection
* Event preprocessing
* Event voxelization
* Visualization
* PyTorch Dataset

### Stage 2 – Baseline Model

* Event Encoder
* Temporal Fusion (ConvLSTM)
* Depth Prediction
* Motion Prediction
* Dynamic Segmentation

### Stage 3 – Self-Supervised Learning

* Ego-motion consistency
* Temporal consistency
* Geometry consistency
* Multi-frame learning

### Stage 4 – Research

* Ablation studies
* Benchmark evaluation
* Dataset generalization
* Publication

---

# Repository Structure

```
EventCameraProject/

├── assets/
├── checkpoints/
├── configs/
├── docs/
├── experiments/
├── logs/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── tools/
├── train.py
├── trainer.py
├── inference.py
├── environment.yml
├── LICENSE
└── README.md
```

---

# Folder Description

## assets/

Contains images used in documentation.

Examples

* Architecture diagrams
* Pipeline figures
* Sample outputs
* GIFs
* Paper illustrations

---

## checkpoints/

Stores trained model weights.

Example

```
checkpoints/

baseline_epoch10.pth

best_model.pth
```

These files should **not** be committed to Git.

---

## configs/

Stores reusable configuration files.

Example

```
configs/

dataset.yaml

model.yaml

training.yaml

experiment.yaml
```

Typical configuration items include:

* dataset location
* training parameters
* optimizer
* scheduler
* augmentation
* voxel settings

---

## docs/

Project documentation.

Examples

* Project outline
* Design notes
* Research ideas
* Dataset documentation

---

## experiments/

Each experiment should have its own configuration.

Example

```
001_baseline.yaml

002_temporal.yaml

003_depth.yaml
```

This makes experiments reproducible.

---

## logs/

Training logs.

Examples

* TensorBoard
* Console logs
* Validation logs

---

## notebooks/

Exploratory notebooks.

Use notebooks only for

* debugging
* visualization
* quick experiments

Do **not** implement the main pipeline here.

---

## outputs/

Generated outputs.

Examples

* Predictions
* Videos
* Figures
* Evaluation results

---

## scripts/

Convenience shell scripts.

```
scripts/

dataset_manager.sh

train.sh

test.sh
```

These should only call Python programs.

Business logic should remain in Python.

---

# Source Code

All source code lives inside **src/**.

```
src/

data/

models/

losses/

utils/

registry.py
```

---

## src/data/

Contains everything related to loading data.

Examples

```
event_dataset.py

voxelizer.py

transforms.py
```

Responsibilities

* HDF5 loading
* voxel grid creation
* data augmentation
* PyTorch Dataset

---

## src/models/

Contains all neural network components.

Examples

```
Event Encoder

ConvLSTM

Geometry Head

Motion Predictor

Segmentation Head
```

No training logic should be placed here.

Models only define neural network architectures.

---

## src/losses/

Contains every loss function.

Examples

```
Geometry Loss

Temporal Loss

Consistency Loss

Total Loss
```

Each loss should remain independent.

---

## src/utils/

Reusable helper functions.

Examples

* metrics
* visualization
* logging
* miscellaneous utilities

---

## src/registry.py

Future registry for

* datasets
* models
* losses

This allows selecting components from configuration files.

---

# tools/

Contains utilities used during development.

```
tools/

dataset/

preprocessing/

visualization/
```

---

## tools/dataset/

Dataset-specific implementations.

Each dataset should implement a common interface.

Example

```
MVSEC

DSEC

EVIMO
```

Responsibilities

* download
* inspection
* metadata
* dataset-specific helpers

---

## tools/preprocessing/

Dataset preparation tools.

Responsibilities

* inspect datasets
* prepare indices
* verify processed data
* cleanup

---

## tools/visualization/

Standalone visualization utilities.

Examples

* event viewer
* voxel viewer
* depth viewer

These are intended for debugging and qualitative analysis.

---

# tests/

Unit tests.

Examples

```
test_dataset.py

test_model.py

test_losses.py
```

Every new feature should ideally include a corresponding test.

---

# Dataset Storage

Datasets are **not** stored inside this repository.

Recommended location

```
~/HDD/EventDatasets/

mvsec/

dsec/

evimo/

prophesee/
```

The repository should only contain code.

---

# Development Workflow

The recommended workflow is:

1. Download dataset
2. Inspect dataset
3. Prepare dataset
4. Verify dataset
5. Train model
6. Evaluate model

---

# Conda Environment

Create the environment

```bash
conda env create -f environment.yml
```

Activate

```bash
conda activate EventProject
```

Update the environment after installing new packages

```bash
conda env export --no-builds > environment.yml
```

---

# Coding Guidelines

* Keep functions small and modular.
* Prefer composition over large monolithic classes.
* Document public functions.
* Add type hints whenever possible.
* Use configuration files instead of hardcoded values.
* Avoid duplicate code.

---

# Git Workflow

Each milestone should end with:

* Working code
* Successful tests
* Updated documentation
* Git commit

Suggested commit style

```
Add MVSEC dataset manager

Implement voxelizer

Add ConvLSTM backbone

Implement geometry loss

Add training pipeline
```

---

# Research Roadmap

## Milestone 1

* Repository setup
* Dataset manager
* Dataset inspection

## Milestone 2

* Dataset preparation
* Event voxelization
* Visualization

## Milestone 3

* Baseline model

## Milestone 4

* Self-supervised losses

## Milestone 5

* Training pipeline

## Milestone 6

* Evaluation

## Milestone 7

* Research experiments

## Milestone 8

* Paper writing

---

# License

See the LICENSE file for licensing information.
