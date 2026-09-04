"""
Constants used throughout the EVIMO2 data pipeline.

This file contains only dataset and format constants.

Do NOT put experiment-specific hyperparameters here.
Those belong in configuration (.yaml) files.
"""

from __future__ import annotations

# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "EVIMO2"

COMPILED_FORMAT_VERSION = "1.0"


# =============================================================================
# Event Array Layout
# =============================================================================

EVENT_X = 0
EVENT_Y = 1
EVENT_T = 2
EVENT_P = 3

NUM_EVENT_COLUMNS = 4


# =============================================================================
# Translation / Rotation
# =============================================================================

NUM_TRANSLATION_COMPONENTS = 3

NUM_QUATERNION_COMPONENTS = 4

POSE_DIMENSION = (
    NUM_TRANSLATION_COMPONENTS +
    NUM_QUATERNION_COMPONENTS
)


# =============================================================================
# File Names
# =============================================================================

MANIFEST_FILENAME = "manifest.json"

METADATA_FILENAME = "metadata.json"

STATISTICS_FILENAME = "statistics.json"


FRAME_INDEX_FILENAME = "frame_index.npy"

EVENT_INDEX_FILENAME = "event_index.npy"

OBJECT_STATE_FILENAME = "object_states.npy"

MOTION_INDEX_FILENAME = "motion_index.npy"


# =============================================================================
# Verification
# =============================================================================

VERIFICATION_DIRECTORY = "verification"

REPORT_FILENAME = "report.json"

OBJECT_MOTION_FILENAME = "object_motion.csv"

FAILED_FRAMES_FILENAME = "failed_frames.json"

SUMMARY_FIGURE_FILENAME = "summary.png"


# =============================================================================
# NumPy dtypes
# =============================================================================

EVENT_DTYPE = "float32"

POSE_DTYPE = "float32"

MASK_DTYPE = "uint16"

MOTION_DTYPE = "bool"


# =============================================================================
# Manifest
# =============================================================================

MANIFEST_KEYS = (
    "dataset",
    "version",
    "sequence_name",
    "created",
)