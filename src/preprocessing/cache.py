"""
Generic cache utilities.

All preprocessing artifacts are stored inside

sequence/
    cache/

The official EVIMO2 dataset files are never modified.

This module provides small helper functions for creating,
checking and clearing cache directories.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import json

import numpy as np


# ============================================================
# Constants
# ============================================================

CACHE_DIR_NAME = "cache"


# ============================================================
# Cache directory
# ============================================================

def get_cache_dir(
    sequence_dir: str | Path,
) -> Path:
    """
    Return the cache directory for a sequence.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    Path
        Path to

            sequence/cache
    """

    return Path(sequence_dir) / CACHE_DIR_NAME


def ensure_cache_dir(
    sequence_dir: str | Path,
) -> Path:
    """
    Create the cache directory if it does not already exist.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    Path
        Cache directory path.
    """

    cache_dir = get_cache_dir(sequence_dir)

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_dir


def clear_cache(
    sequence_dir: str | Path,
) -> None:
    """
    Delete the cache directory.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.
    """

    cache_dir = get_cache_dir(sequence_dir)

    if cache_dir.exists():

        shutil.rmtree(cache_dir)


def cache_exists(
    sequence_dir: str | Path,
    filename: str,
) -> bool:
    """
    Check whether a cache file exists.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    filename
        Cache filename.

    Returns
    -------
    bool
    """

    return (
        get_cache_dir(sequence_dir) /
        filename
    ).exists()


# ============================================================
# JSON
# ============================================================

def save_json(
    sequence_dir: str | Path,
    filename: str,
    data: dict,
) -> Path:
    """
    Save a JSON file inside the cache directory.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    filename
        Cache filename (e.g. "metadata.json").

    data
        Dictionary to save.

    Returns
    -------
    Path
        Saved file path.
    """

    path = ensure_cache_dir(sequence_dir) / filename

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4,
        )

    return path


def load_json(
    sequence_dir: str | Path,
    filename: str,
) -> dict:
    """
    Load a JSON cache file.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    filename
        Cache filename.

    Returns
    -------
    dict
    """

    path = get_cache_dir(sequence_dir) / filename

    with open(path, "r") as f:

        return json.load(f)


# ============================================================
# NPZ
# ============================================================

def save_npz(
    sequence_dir: str | Path,
    filename: str,
    **arrays,
) -> Path:
    """
    Save NumPy arrays inside the cache directory.

    Example
    -------
    save_npz(
        sequence_dir,
        "motion.npz",
        positions=positions,
        timestamps=timestamps,
    )
    """

    path = ensure_cache_dir(sequence_dir) / filename

    np.savez_compressed(
        path,
        **arrays,
    )

    return path


def load_npz(
    sequence_dir: str | Path,
    filename: str,
):
    """
    Load an NPZ cache file.

    Returns
    -------
    numpy.lib.npyio.NpzFile
    """

    path = get_cache_dir(sequence_dir) / filename

    return np.load(
        path,
        allow_pickle=True,
    )


# ============================================================
# Verification
# ============================================================

if __name__ == "__main__":

    import tempfile

    print("=" * 70)
    print("CACHE VERIFICATION")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:

        sequence_dir = Path(tmp) / "sequence"
        sequence_dir.mkdir()

        print("\nCreating cache directory...")

        ensure_cache_dir(sequence_dir)

        print("✓ cache directory")

        # --------------------------------------------------

        metadata = {
            "dataset": "EVIMO2",
            "frames": 404,
        }

        save_json(
            sequence_dir,
            "metadata.json",
            metadata,
        )

        loaded = load_json(
            sequence_dir,
            "metadata.json",
        )

        assert loaded == metadata

        print("✓ json")

        # --------------------------------------------------

        save_npz(
            sequence_dir,
            "motion.npz",
            timestamps=np.arange(5),
            positions=np.random.rand(5, 3),
        )

        data = load_npz(
            sequence_dir,
            "motion.npz",
        )

        assert "timestamps" in data
        assert "positions" in data

        print("✓ npz")

        # --------------------------------------------------

        assert cache_exists(
            sequence_dir,
            "motion.npz",
        )

        print("✓ exists")

        # --------------------------------------------------

        clear_cache(sequence_dir)

        assert not get_cache_dir(sequence_dir).exists()

        print("✓ clear")

    print("\nVerification complete.")
    print("=" * 70)