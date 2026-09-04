"""
Dataset registry for EventCameraProject.

This module provides a central registry of all supported datasets.
New datasets should be registered here so they become available
through the dataset manager.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseDataset
from .mvsec import MVSEC

# -----------------------------------------------------------------------------
# Dataset Registry
# -----------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    MVSEC.NAME.lower(): MVSEC,
}


def get_dataset_class(name: str) -> Type[BaseDataset]:
    """
    Return the dataset class associated with a dataset name.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    Type[BaseDataset]
        Dataset class.

    Raises
    ------
    ValueError
        If the dataset is not registered.
    """
    key = name.lower()

    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {available}"
        )

    return DATASET_REGISTRY[key]


def list_datasets() -> list[str]:
    """
    Return the list of supported datasets.

    Returns
    -------
    list[str]
        Sorted dataset names.
    """
    return sorted(DATASET_REGISTRY.keys())