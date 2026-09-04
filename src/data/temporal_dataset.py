"""
Temporal wrapper around EVIMO2Dataset.

The frame dataset returns one frame.

This dataset returns an ordered temporal window
constructed from arbitrary frame offsets.
"""

from __future__ import annotations

from collections import defaultdict

from src.data.dataset import EVIMO2Dataset
from src.data.temporal_sample import TemporalEVIMO2Sample


class TemporalEVIMO2Dataset:
    """
    Wraps EVIMO2Dataset.

    Produces temporal samples.

    Example
    -------

    history_offsets=(-3,-2,-1,0)

    returns

        t-3
        t-2
        t-1
        t
    """

    def __init__(

        self,

        frame_dataset: EVIMO2Dataset,

        history_offsets=(-3,-2,-1,0)

    ):

        self.frame_dataset = frame_dataset

        self.history_offsets = tuple(history_offsets)


        #
        # Each element contains the exact global indices
        # corresponding to one temporal sample.
        #
        # Example:
        #
        # history_offsets=(-3,-2,-1,0)
        #
        # [
        #     [12,13,14,15],
        #     [13,14,15,16],
        #     ...
        # ]
        #
        self.valid_windows = []

        self._build_index()


    def __len__(self):

        return len(
                    self.valid_windows
                )


    def __getitem__(

        self,

        index,

    ):

        window_indices = self.valid_windows[index]

        frames = [

            self.frame_dataset[i]

            for i in window_indices

        ]

        return TemporalEVIMO2Sample(

            frames=frames,

            global_indices=window_indices,

            history_offsets=self.history_offsets,

        )


    def _build_index(self):

        grouped = defaultdict(list)

        references = self.frame_dataset.index.references

        #
        # Group global indices by sequence
        #

        for global_index, ref in enumerate(references):

            grouped[
                ref.sequence_id
            ].append(global_index)

        #
        # Build valid anchors
        #

        min_offset = min(
            self.history_offsets
        )

        max_offset = max(
            self.history_offsets
        )

        for sequence_indices in grouped.values():

            n = len(sequence_indices)

            start = -min_offset

            end = (
                n
                - 1
                - max_offset
            )

            for local_anchor in range(

                start,

                end + 1,

            ):

                window = [

                    sequence_indices[
                        local_anchor + offset
                    ]

                    for offset in self.history_offsets

                ]

                self.valid_windows.append(
                    window
                )