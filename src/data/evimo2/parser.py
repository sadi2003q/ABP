"""
EVIMO2v2 parser.

Event-camera-only dataset interface.

Handles:

- metadata
- events
- depth
- masks

Classical camera is intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from ._metadata import parse_sequence_metadata
from ._reader import EVIMO2Reader
from .datatypes.raw import RawSequence
from .datatypes.common import CameraIntrinsics



class EVIMO2Parser:
    """
    High-level EVIMO2v2 dataset interface.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
    ):

        self.sequence_dir = Path(sequence_dir)


        if not self.sequence_dir.exists():

            raise FileNotFoundError(
                self.sequence_dir
            )


        self.reader = EVIMO2Reader(
            self.sequence_dir
        )


        self.frames, self.objects, self.imu, = (
            parse_sequence_metadata(
                self.sequence_dir
            )
        )


        self.sequence = (
            self._build_sequence()
        )


    # =====================================================
    # Sequence metadata
    # =====================================================

    def _build_sequence(self):

        info = np.load(
            self.sequence_dir /
            "dataset_info.npz",
            allow_pickle=True,
        )


        K = info["K"].astype(np.float32)

        D = info["D"].astype(np.float32)


        meta = info["meta"].item()["meta"]


        camera = CameraIntrinsics(
            fx=float(meta["fx"]),
            fy=float(meta["fy"]),

            cx=float(meta["cx"]),
            cy=float(meta["cy"]),

            width=int(meta["res_x"]),
            height=int(meta["res_y"]),
            distortion=D,
        )


        return RawSequence(

            sequence_name=
                self.sequence_dir.name,

            camera=camera,

            num_events=
                self.reader.num_events(),

            num_frames=
                len(self.frames),

            start_time=
                self.frames[0].timestamp,

            end_time=
                self.frames[-1].timestamp,

            objects=self.objects,
        )


    # =====================================================
    # Public API
    # =====================================================

    def get_frame(
        self,
        frame_id: int,
    ):

        return self.frames[frame_id]


    def get_events(
        self,
        t_start,
        t_end,
    ):

        return self.reader.get_events(
            t_start,
            t_end,
        )


    def load_depth(
        self,
        frame_id,
    ):

        return self.reader.load_depth(
            frame_id
        )


    def load_mask(
        self,
        frame_id,
    ):

        return self.reader.load_mask(
            frame_id
        )



# =========================================================
# Verification
# =========================================================

def verify_parser(sequence_dir):

    print("=" * 70)
    print("EVIMO2v2 PARSER VERIFICATION")
    print("=" * 70)


    dataset = EVIMO2Parser(
        sequence_dir
    )


    seq = dataset.sequence


    print("\n[Sequence]")
    print(
        "name:",
        seq.sequence_name
    )

    print(
        "frames:",
        seq.num_frames
    )

    print(
        "events:",
        seq.num_events
    )

    print(
        "resolution:",
        seq.camera.width,
        "x",
        seq.camera.height,
    )


    frame = dataset.get_frame(0)


    print("\n[Frame 0]")

    print(
        "timestamp:",
        f"{frame.timestamp:.6f}"
    )

    print(
        "objects:",
        len(frame.object_states)
    )

    print(
        "depth:",
        frame.depth_available
    )

    print(
        "mask:",
        frame.mask_available
    )


    print("\n[Events 10ms]")


    xy,t,p = dataset.get_events(
        frame.timestamp,
        frame.timestamp + 0.01
    )


    print(
        "count:",
        len(t)
    )


    if len(t):

        print(
            "first:",
            xy[0].tolist(),
            float(t[0]),
            int(p[0])
        )


    print("\n[Mask]")


    mask = dataset.load_mask(
        frame.frame_id
    )


    if mask is None:

        print(
            "not available"
        )

    else:

        print(
            "shape:",
            mask.shape
        )

        print(
            "unique:",
            len(np.unique(mask))
        )


    print("\n[Depth]")


    depth = dataset.load_depth(
        frame.frame_id
    )


    if depth is None:

        print(
            "not available"
        )

    else:

        print(
            "shape:",
            depth.shape
        )


    print("\nVerification complete.")
    print("=" * 70)



if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.data.evimo2.parser <sequence_dir>"
        )

        sys.exit(1)


    verify_parser(
        sys.argv[1]
    )