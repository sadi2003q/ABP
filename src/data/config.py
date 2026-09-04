"""
Configuration defaults for EVIMO2 datasets.

This module centralizes dataset-wide constants used by the Dataset
implementation. Keeping them here avoids scattering hardcoded values
throughout the codebase and makes future experiments easier to
configure.

These values are only defaults.

The Dataset constructor may override any of them.
"""

from __future__ import annotations

# ============================================================
# Supported event-camera sensors
# ============================================================

EVENT_SENSORS = (
    "left_camera",
    "right_camera",
    "samsung_mono",
)

RGB_SENSOR = "flea3_7"


# ============================================================
# Dataset splits
# ============================================================

VALID_SPLITS = (
    "train",
    "eval",
)


# ============================================================
# Default Dataset behaviour
# ============================================================

DEFAULT_EVENT_WINDOW = 50_000

DEFAULT_INCLUDE_DEPTH = True

DEFAULT_INCLUDE_MASK = True

DEFAULT_INCLUDE_RGB = True

DEFAULT_VERIFY_CACHE = True


# ============================================================
# Required cache files
# ============================================================

REQUIRED_CACHE_FILES = (
    "event_index.npz",
    "frame_motion.npz",
    "camera_motion.npz",
)