"""
Utilities for working with EVIMO2 instance masks.

EVIMO2 stores object IDs as

    object_id * 1000

For example

    background = 0
    object 5  = 5000
    object 12 = 12000

These helper functions convert masks into a more convenient
representation for preprocessing and training.
"""

from __future__ import annotations

import numpy as np




# =============================================================================
# Mask decoding
# =============================================================================

def decode_instance_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert an EVIMO2 mask into object IDs.

    Parameters
    ----------
    mask
        Raw EVIMO2 mask.

    Returns
    -------
    np.ndarray

        Mask where every pixel contains the object id.

    Example
    -------

    5000 -> 5

    12000 -> 12
    """

    return mask // 1000


# =============================================================================
# Object IDs
# =============================================================================

def unique_object_ids(mask: np.ndarray) -> np.ndarray:
    """
    Return visible object IDs.

    Background (0) is removed.
    """

    ids = np.unique(
        decode_instance_mask(mask)
    )

    return ids[ids != 0]


# =============================================================================
# Binary mask
# =============================================================================

def binary_mask(
    mask: np.ndarray,
    object_id: int,
) -> np.ndarray:
    """
    Return binary mask of one object.
    """

    decoded = decode_instance_mask(mask)

    return decoded == object_id


# =============================================================================
# Area
# =============================================================================

def object_area(
    mask: np.ndarray,
    object_id: int,
) -> int:
    """
    Pixel area of one object.
    """

    return int(
        np.count_nonzero(
            binary_mask(
                mask,
                object_id,
            )
        )
    )


# =============================================================================
# Bounding box
# =============================================================================

def bounding_box(
    mask: np.ndarray,
    object_id: int,
):
    """
    Bounding box of one object.

    Returns

    (xmin, ymin, xmax, ymax)

    Returns None if object is absent.
    """

    ys, xs = np.where(
        binary_mask(
            mask,
            object_id,
        )
    )

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )