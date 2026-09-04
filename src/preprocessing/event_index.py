"""
Precompute event indices for every frame in an EVIMO2 sequence.

During training, every sample needs to retrieve the events
belonging to a frame interval.

Performing binary searches over tens of millions of event
timestamps every iteration is unnecessary because the mapping
between frames and event indices never changes.

This module computes the mapping once and stores it inside

    sequence/
        cache/
            event_index.npz

The cache contains every frame, not only frames with ground
truth supervision. This is important because the project is
self-supervised and inference can be performed for every frame.

Ground-truth availability is cached separately for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path
import sys

import numpy as np

from src.data.evimo2.parser import EVIMO2Parser

from src.preprocessing.cache import (
    save_npz,
    load_npz,
    cache_exists,
)

CACHE_FILENAME = "event_index.npz"

@dataclass(slots=True)
class EventIndexCache:
    """
    Cached mapping between frames and event indices.
    """

    frame_ids: np.ndarray

    timestamps: np.ndarray

    frame_dt: np.ndarray

    event_start: np.ndarray

    event_end: np.ndarray

    event_count: np.ndarray

    has_events: np.ndarray

    depth_available: np.ndarray

    mask_available: np.ndarray


def compute_event_index(
    dataset: EVIMO2Parser,
) -> dict:
    """
    Compute the event interval corresponding to every frame.

    Parameters
    ----------
    dataset
        Parsed EVIMO2 sequence.

    Returns
    -------
    dict
        Dictionary containing

        frame_ids
        timestamps
        frame_dt
        event_start
        event_end
        event_count
        depth_available
        mask_available
    """

    frames = dataset.frames

    events_t = dataset.reader.events_t

    total_events = len(events_t)

    n = len(frames)

    frame_ids = np.empty(
        n,
        dtype=np.int32,
    )

    timestamps = np.empty(
        n,
        dtype=np.float64,
    )

    frame_dt = np.empty(
        n,
        dtype=np.float64,
    )

    event_start = np.empty(
        n,
        dtype=np.int64,
    )

    event_end = np.empty(
        n,
        dtype=np.int64,
    )

    depth_available = np.empty(
        n,
        dtype=bool,
    )

    mask_available = np.empty(
        n,
        dtype=bool,
    )

    for i, frame in enumerate(frames):

        frame_ids[i] = frame.frame_id

        timestamps[i] = frame.timestamp

        depth_available[i] = frame.depth_available

        mask_available[i] = frame.mask_available

        event_start[i] = np.searchsorted(
            events_t,
            frame.timestamp,
            side="left",
        )

        if i < n - 1:

            event_end[i] = np.searchsorted(
                events_t,
                frames[i + 1].timestamp,
                side="left",
            )

            frame_dt[i] = (
                frames[i + 1].timestamp
                - frame.timestamp
            )

        else:

            event_end[i] = total_events

            if n > 1:

                frame_dt[i] = frame_dt[i - 1]

            else:

                frame_dt[i] = 0.0

    event_count = event_end - event_start
    has_events = event_count > 0
    return {

        "frame_ids": frame_ids,

        "timestamps": timestamps,

        "frame_dt": frame_dt,

        "event_start": event_start,

        "event_end": event_end,

        "event_count": event_count,
        "has_events": has_events,

        "depth_available": depth_available,

        "mask_available": mask_available,

    }


# ============================================================
# Cache I/O
# ============================================================

def save_event_index(
    sequence_dir: str | Path,
    cache: dict,
) -> Path:
    """
    Save the computed event index cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    cache
        Dictionary returned by
        compute_event_index().

    Returns
    -------
    Path
        Saved cache file.
    """

    return save_npz(
        sequence_dir,
        CACHE_FILENAME,
        **cache,
    )


def load_event_index(
    sequence_dir: str | Path,
        ) -> EventIndexCache:
    """
    Load a previously generated event-index cache.
    """

    data = load_npz(
        sequence_dir,
        CACHE_FILENAME,
    )

    return EventIndexCache(

        frame_ids=data["frame_ids"],

        timestamps=data["timestamps"],

        frame_dt=data["frame_dt"],

        event_start=data["event_start"],

        event_end=data["event_end"],

        event_count=data["event_count"],

        has_events=data["has_events"],

        depth_available=data["depth_available"],

        mask_available=data["mask_available"],
    )



def generate_event_index(
    sequence_dir: str | Path,
) -> Path:
    """
    Generate and save the event-index cache.

    Parameters
    ----------
    sequence_dir
        EVIMO2 sequence directory.

    Returns
    -------
    Path
        Path to the generated cache.
    """

    from src.data.evimo2.parser import EVIMO2Parser

    dataset = EVIMO2Parser(sequence_dir)

    cache = compute_event_index(
        dataset,
    )

    output_path = save_event_index(
        sequence_dir,
        cache,
    )

    return output_path

# ============================================================
# Verification
# ============================================================

def verify_event_index(
    sequence_dir: str | Path,
):
    """
    Verify event index generation.
    """

    print("=" * 90)
    print("EVENT INDEX VERIFICATION")
    print("=" * 90)

    cache_path = generate_event_index(
        sequence_dir,
    )

    cache = load_event_index(
        cache_path,
    )

    dataset = EVIMO2Parser(
        sequence_dir,
    )

    print()

    print(f"Frames              : {len(cache['frame_ids'])}")
    print(f"Events              : {dataset.sequence.num_events:,}")

    print()

    print("Ground Truth")
    print("-" * 40)
    print(
        f"Depth Available     : {cache['depth_available'].sum()}"
    )
    print(
        f"Mask Available      : {cache['mask_available'].sum()}"
    )

    print()

    print("Event Statistics")
    print("-" * 40)
    counts = cache["event_count"]

    valid_counts = counts[counts > 0]

    print(
        f"Mean events/frame   : {cache['event_count'].mean():.0f}"
    )

    print(
        f"Median              : {np.median(cache['event_count']):.0f}"
    )

    print(
        f"Minimum             : {cache['event_count'].min()}"
    )

    print(
        f"Maximum             : {cache['event_count'].max()}"
    )
    print(
        f"Zero-event frames   : {(counts == 0).sum()}"
    )

    valid_counts = counts[counts > 0]

    print(
        f"Minimum (non-empty): {valid_counts.min()}"
    )
    print()

    print("First Frame")
    print("-" * 40)

    print(
        f"Frame ID            : {cache['frame_ids'][0]}"
    )

    print(
        f"Timestamp           : {cache['timestamps'][0]:.6f}"
    )

    print(
        f"Start Index         : {cache['event_start'][0]}"
    )

    print(
        f"End Index           : {cache['event_end'][0]}"
    )

    print()

    print("Last Frame")
    print("-" * 40)

    print(
        f"Frame ID            : {cache['frame_ids'][-1]}"
    )

    print(
        f"Timestamp           : {cache['timestamps'][-1]:.6f}"
    )

    print(
        f"Start Index         : {cache['event_start'][-1]}"
    )

    print(
        f"End Index           : {cache['event_end'][-1]}"
    )

    print()

    print("Sanity Checks")
    print("-" * 40)

    assert np.all(
        np.diff(cache["event_start"]) >= 0
    )

    print("✓ Start indices increasing")

    assert np.all(
        np.diff(cache["event_end"]) >= 0
    )

    print("✓ End indices increasing")

    assert np.all(
        cache["event_start"][1:]
        >=
        cache["event_end"][:-1]
    )

    print("✓ Non-overlapping intervals")

    assert (
        cache["event_end"][-1]
        ==
        dataset.sequence.num_events
    )

    print("✓ Last interval reaches final event")

    print()
    assert np.all(
        cache["has_events"]
        ==
        (cache["event_count"] > 0)
    )

    print("✓ Event availability flags verified")
    zero_event_idx = np.where(cache["event_count"] == 0)[0]

    print("\nZero-event frames")
    print("-" * 40)

    for idx in zero_event_idx:
        print(
            f"Index {idx:3d} | "
            f"Frame {cache['frame_ids'][idx]:3d} | "
            f"Timestamp {cache['timestamps'][idx]:.6f}"
        )


    events_t = dataset.reader.events_t

    print()
    print("Event Stream")
    print("-" * 40)

    print(f"First event : {events_t[0]:.6f}")
    print(f"Last event  : {events_t[-1]:.6f}")

    print()
    print("Last frame timestamps")

    for i in range(400, 404):
        print(
            i,
            cache["timestamps"][i],
            cache["event_start"][i],
        )

    gap = (
        cache["timestamps"][-1]
        - dataset.reader.events_t[-1]
    )

    print(
        f"Event stream ends  : {gap*1000:.3f} ms before final frame"
    )
    print("Verification completed successfully.")
    assert (
        cache["event_count"].sum()
        == dataset.sequence.num_events
    )

    print("✓ Event counts sum to total number of events")
    print("=" * 90)



if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python -m src.preprocessing.event_index <sequence_dir>"
        )

        sys.exit(1)

    verify_event_index(
        sys.argv[1]
    )