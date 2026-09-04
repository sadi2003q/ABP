"""
PyTorch Dataset for EVIMO2.

This dataset provides one sample per frame.

Each sample contains

    • event window
    • camera motion
    • object motion
    • optional depth
    • optional instance mask

Heavy data (events, depth, masks) are loaded lazily.

Sequence metadata, parsers and preprocessing caches are loaded once
during Dataset initialization.

The Dataset itself performs no batching. That is handled by the
PyTorch DataLoader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.data.sequence_index import SequenceIndex
from src.data.evimo2.parser import EVIMO2Parser
from src.data.evimo2._reader import EVIMO2Reader

from src.preprocessing.imu_index import (
    load_imu_index,
)
from src.data.sample import (
    EVIMO2Sample,
    CameraMotion,
    FrameMotion,
    IMUWindow,
)

from src.preprocessing.event_index import (
    load_event_index,
)

from src.preprocessing.frame_motion import (
    load_frame_motion,
)

from src.preprocessing.camera_motion import (
    load_camera_motion,
)


class EVIMO2Dataset(Dataset):
    """
    EVIMO2 Dataset.

    One dataset item corresponds to one frame.

    Parameters
    ----------
    dataset_root
        EVIMO2 root directory.

    sensors
        Sensors to include.

    split
        Dataset split.

    load_depth
        Load depth maps.

    load_mask
        Load instance masks.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        sensors: tuple[str, ...] | list[str] | None = None,
        split: str = "train",
        load_depth: bool = True,
        load_mask: bool = True,
    ):

        super().__init__()

        self.load_depth = load_depth
        self.load_mask = load_mask

        #
        # Global frame index
        #

        self.index = SequenceIndex(
            dataset_root=dataset_root,
            sensors=sensors,
            split=split,
        )

        #
        # One parser/reader/cache per sequence
        #

        self.parsers = {}
        self.readers = {}

        self.event_indices = {}
        self.frame_motion = {}
        self.camera_motion = {}
        self.imu_indices = {}

        for sequence in self.index.sequences:

            sid = sequence.sequence_id

            self.parsers[sid] = EVIMO2Parser(
                sequence.sequence_dir
            )

            self.readers[sid] = EVIMO2Reader(
                sequence.sequence_dir,
                sequence.rgb_sequence_dir,
            )

            self.event_indices[sid] = load_event_index(
                sequence.sequence_dir
            )

            self.frame_motion[sid] = load_frame_motion(
                sequence.sequence_dir
            )

            self.camera_motion[sid] = load_camera_motion(
                sequence.sequence_dir
            )

            self.imu_indices[sid] = load_imu_index(
                sequence.sequence_dir
            )

    # ======================================================
    # Dataset interface
    # ======================================================

    def __len__(
        self,
    ) -> int:

        return len(self.index)

    def __getitem__(
        self,
        index: int,
    ) -> EVIMO2Sample:

        ref = self.index[index]

        sequence = self.index.sequences[
            ref.sequence_id
        ]

        parser = self.parsers[sequence.sequence_id]

        reader = self.readers[sequence.sequence_id]

        camera = parser.sequence.camera

        frame = parser.frames[
            ref.local_frame_index
        ]

        #
        # Event window
        #

        event_cache = self.event_indices[sequence.sequence_id]

        start = event_cache.event_start[
            ref.local_frame_index
        ]

        end = event_cache.event_end[
            ref.local_frame_index
        ]

        events_xy = reader.events_xy[start:end]
        events_t = reader.events_t[start:end]
        events_p = reader.events_p[start:end]


        #
        # IMU window
        #

        imu_cache = self.imu_indices[sequence.sequence_id]

        imu_start = imu_cache.imu_start[
            ref.local_frame_index
        ]

        imu_end = imu_cache.imu_end[
            ref.local_frame_index
        ]

        sensor_imu = parser.imu.get(sequence.sensor)

        if sensor_imu is None:

            imu = IMUWindow(

                timestamps=np.empty(
                    0,
                    dtype=np.float64,
                ),

                angular_velocity=np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),

                linear_acceleration=np.empty(
                    (0, 3),
                    dtype=np.float32,
                ),
            )

        else:

            imu = IMUWindow(

                timestamps=sensor_imu["timestamps"][
                    imu_start:imu_end
                ],

                angular_velocity=sensor_imu["gyro"][
                    imu_start:imu_end
                ],

                linear_acceleration=sensor_imu["acceleration"][
                    imu_start:imu_end
                ],
            )

        #
        # Camera motion
        #

        cam_cache = self.camera_motion[sequence.sequence_id]

        camera_motion = CameraMotion(

            translation=
                cam_cache.translation[
                    ref.local_frame_index
                ],

            quaternion=
                cam_cache.quaternion[
                    ref.local_frame_index
                ],

            delta_translation=
                cam_cache.delta_translation[
                    ref.local_frame_index
                ],

            delta_quaternion=
                cam_cache.delta_quaternion[
                    ref.local_frame_index
                ],

            linear_speed=float(
                cam_cache.linear_speed[
                    ref.local_frame_index
                ]
            ),

            angular_speed=float(
                cam_cache.angular_speed[
                    ref.local_frame_index
                ]
            ),

            pose_available=bool(
                cam_cache.pose_available[
                    ref.local_frame_index
                ]
            ),

            dt=float(
                cam_cache.frame_dt[
                    ref.local_frame_index
                ]
            ),
        )

        #
        # Object motion
        #

        motion_cache = self.frame_motion[sequence.sequence_id]

        frame_motion = FrameMotion(

            object_ids=motion_cache.object_ids,

            delta_position=
                motion_cache.delta_position[
                    ref.local_frame_index
                ],

            speed=
                motion_cache.speed[
                    ref.local_frame_index
                ],
        )

        #
        # Ground truth
        #

        depth = None

        if self.load_depth:

            depth = reader.load_depth(
                frame.frame_id
            )

        mask = None

        if self.load_mask:

            mask = reader.load_mask(
                frame.frame_id
            )

        rgb = reader.load_rgb(
            frame.frame_id
        )

        camera_intrinsics = camera.matrix()

        camera_distortion = (
            camera.distortion_coefficients()
            .astype(np.float32)
        )

        #
        # Sample
        #



        return EVIMO2Sample(

            sequence_name=sequence.sequence_name,

            sensor=sequence.sensor,

            local_frame_index=ref.local_frame_index,

            frame_id=frame.frame_id,

            timestamp=frame.timestamp,

            camera_intrinsics=camera_intrinsics,

            camera_distortion=camera_distortion,

            events_xy=events_xy,

            events_t=events_t,

            events_p=events_p,

            camera_motion=camera_motion,

            imu=imu,

            frame_motion=frame_motion,

            depth=depth,

            mask=mask,

            rgb=rgb,
        )