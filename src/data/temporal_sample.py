"""
Temporal sample returned by the TemporalEVIMO2Dataset.

A temporal sample consists of several frame-level samples
ordered in time.

The last frame is normally the prediction target.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.sample import EVIMO2Sample


@dataclass(slots=True)
class TemporalEVIMO2Sample:
    """
    One temporal training sample.

    Parameters
    ----------
    frames

        Ordered list of frame samples.

        Example
        -------

        history_offsets = [-3,-2,-1,0]

        frames[0] -> t-3

        frames[1] -> t-2

        frames[2] -> t-1

        frames[3] -> t

    anchor_index

        Global frame index of the prediction frame.

    history_offsets

        Offsets used to construct this sample.
    """

    frames: list[EVIMO2Sample]

    global_indices: list[int]

    history_offsets: tuple[int, ...]

