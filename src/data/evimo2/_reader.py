"""
EVIMO2v2 data reader.

Responsibilities:

- Load event stream lazily using mmap.
- Load depth and mask frames lazily.
- Handle irregular ground truth sampling.
- Handle missing depth/mask frames.

Classical camera data is intentionally not handled here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EVIMO2Reader:
    """
    Lazy reader for one EVIMO2v2 event-camera sequence.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        rgb_sequence_dir: str | Path | None = None,
    ):

        self.sequence_dir = Path(sequence_dir)

        self.rgb_sequence_dir = (
            Path(rgb_sequence_dir)
            if rgb_sequence_dir is not None
            else None
        )


        # -------------------------------------------------
        # Event stream
        # -------------------------------------------------

        self.events_t = np.load(
            self.sequence_dir /
            "dataset_events_t.npy",

            mmap_mode="r",
        )


        self.events_xy = np.load(
            self.sequence_dir /
            "dataset_events_xy.npy",

            mmap_mode="r",
        )


        self.events_p = np.load(
            self.sequence_dir /
            "dataset_events_p.npy",

            mmap_mode="r",
        )


        # -------------------------------------------------
        # Ground truth containers
        # -------------------------------------------------

        self.depth_file = (
            self.sequence_dir /
            "dataset_depth.npz"
        )

        self.mask_file = (
            self.sequence_dir /
            "dataset_mask.npz"
        )


        self._depth = None

        self._mask = None
        self._rgb = None

        # Build available frame indexes

        self.depth_ids = (
            self._build_index(
                self.depth_file,
                "depth",
            )
        )


        self.mask_ids = (
            self._build_index(
                self.mask_file,
                "mask",
            )
        )



    # =====================================================
    # Internal helpers
    # =====================================================

    @staticmethod
    def _build_index(
        path: Path,
        prefix: str,
    ) -> set[int]:
        """
        Build available frame ID index.
        """

        if not path.exists():

            return set()


        data = np.load(path)

        ids = set()


        for key in data.files:

            if key.startswith(prefix):

                ids.add(
                    int(
                        key.split("_")[1]
                    )
                )


        return ids



    def _load_depth_file(self):

        if self._depth is None:

            self._depth = np.load(
                self.depth_file
            )



    def _load_mask_file(self):

        if self._mask is None:

            self._mask = np.load(
                self.mask_file
            )

    def _load_rgb_file(
        self,
    ):
        """
        Lazily load the RGB image archive.
        """

        if self.rgb_sequence_dir is None:
            return

        if self._rgb is None:

            rgb_file = (
                self.rgb_sequence_dir /
                "dataset_classical.npz"
            )

            if rgb_file.exists():

                self._rgb = np.load(
                    rgb_file
                )

    # =====================================================
    # Events
    # =====================================================

    def get_events(
        self,
        t_start: float,
        t_end: float,
    ):
        """
        Return events inside time interval.

        Returns:

        x,y,t,p
        """

        start = np.searchsorted(
            self.events_t,
            t_start,
            side="left",
        )


        end = np.searchsorted(
            self.events_t,
            t_end,
            side="right",
        )


        return (
            self.events_xy[start:end],
            self.events_t[start:end],
            self.events_p[start:end],
        )



    def num_events(self):

        return len(self.events_t)



    # =====================================================
    # Depth
    # =====================================================

    def load_depth(
        self,
        frame_id: int,
    ):
        """
        Load depth frame.

        Returns None if unavailable.
        """

        if frame_id not in self.depth_ids:

            return None


        self._load_depth_file()


        key = (
            f"depth_{frame_id:010d}"
        )


        return self._depth[key]



    # =====================================================
    # Mask
    # =====================================================

    def load_mask(
        self,
        frame_id: int,
    ):
        """
        Load object mask.

        Returns None if unavailable.
        """

        if frame_id not in self.mask_ids:

            return None


        self._load_mask_file()


        key = (
            f"mask_{frame_id:010d}"
        )


        return self._mask[key]

    # =====================================================
    # RGB
    # =====================================================

    def load_rgb(
        self,
        frame_id: int,
    ):
        """
        Load RGB image for a frame.

        Returns
        -------
        ndarray | None
        """

        if self.rgb_sequence_dir is None:
            return None

        self._load_rgb_file()

        if self._rgb is None:
            return None

        key = (
            f"classical_{frame_id:010d}"
        )

        if key not in self._rgb.files:
            return None

        return self._rgb[key]


    # =====================================================
    # Information
    # =====================================================

    def resolution(self):

        if self._mask is None:

            self._load_mask_file()


        if len(self.mask_ids):

            first = next(
                iter(self._mask.files)
            )

            return self._mask[first].shape


        return None