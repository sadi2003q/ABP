"""
Visualize EVIMO2 samples produced by the Dataset.

Creates a side-by-side verification video containing

    RGB
    Events
    Motion-colored mask
    Depth

This tool is intended for debugging the complete data pipeline.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from src.data.dataset import EVIMO2Dataset


MOTION_THRESHOLD_SPEED = 0.05

def render_events(
    events_xy: np.ndarray,
    events_p: np.ndarray,
    resolution: tuple[int, int] = (480, 640),
) -> np.ndarray:

    height, width = resolution

    image = np.full(
        (height, width, 3),
        127,
        dtype=np.uint8,
    )

    if len(events_xy) == 0:
        return image

    x = events_xy[:, 0]
    y = events_xy[:, 1]

    positive = events_p.astype(bool)

    image[y[positive], x[positive]] = (255, 255, 255)
    image[y[~positive], x[~positive]] = (0, 0, 0)

    return image


def render_depth(
    depth: np.ndarray | None,
) -> np.ndarray:

    if depth is None:

        return np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

    depth = depth.astype(np.float32)

    valid = depth > 0

    image = np.zeros_like(
        depth,
        dtype=np.uint8,
    )

    if valid.any():

        dmin = depth[valid].min()
        dmax = depth[valid].max()

        if dmax > dmin:

            image[valid] = (
                (depth[valid] - dmin)
                /
                (dmax - dmin)
                * 255
            ).astype(np.uint8)

    image = cv2.applyColorMap(
        image,
        cv2.COLORMAP_TURBO,
    )

    return image

def render_rgb(
    rgb: np.ndarray | None,
):

    if rgb is None:

        return np.zeros(
            (480,640,3),
            dtype=np.uint8,
        )

    return cv2.resize(
        rgb,
        (640,480),
        interpolation=cv2.INTER_AREA,
    )


def render_mask(
    mask: np.ndarray | None,
) -> np.ndarray:
    """
    Render an instance mask using stable random colors.

    Background is black.

    Each instance receives a deterministic pseudo-random color
    based on its ID.

    Motion colouring will be added later after object-ID mapping
    is verified.
    """

    if mask is None:

        return np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

    image = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    instance_ids = np.unique(mask)

    for instance_id in instance_ids:

        if instance_id == 0:
            continue

        rng = np.random.default_rng(
            int(instance_id)
        )

        color = rng.integers(
            50,
            255,
            size=3,
            dtype=np.uint8,
        )

        image[mask == instance_id] = color

    return image


def render_motion_mask(
    mask: np.ndarray | None,
    frame_motion,
) -> np.ndarray:
    """
    Render instance mask using motion information.

    Static objects
        Green

    Moving objects
        Red

    Background
        Black

    White contour
        Object boundary
    """

    if mask is None:

        return np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

    image = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    #
    # EVIMO stores instance IDs as
    #
    #     mask_id = object_id * 1000
    #
    # Example
    #
    #     object 22
    #
    # becomes
    #
    #     22000
    #

    motion_lookup = {}

    for object_id, speed in zip(
        frame_motion.object_ids,
        frame_motion.speed,
    ):

        motion_lookup[
            int(object_id) * 1000
        ] = speed > MOTION_THRESHOLD_SPEED

    instance_ids = np.unique(mask)

    for instance_id in instance_ids:

        if instance_id == 0:
            continue

        moving = motion_lookup.get(
            int(instance_id),
            False,
        )

        #
        # Moving → Red
        #
        if moving:

            color = (0, 0, 255)

        #
        # Static → Green
        #
        else:

            color = (0, 255, 0)

        image[
            mask == instance_id
        ] = color

    #
    # Draw white object boundaries
    #

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        (255, 255, 255),
        1,
    )

    # for oid, speed in zip(
    #     frame_motion.object_ids,
    #     frame_motion.speed,
    # ):
    #     print(
    #         oid,
    #         f"{speed:.8f}",
    #     )

    return image


def compose_panel(
    rgb: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """
    Create the visualization panel.

        RGB | Events
        ----------
        Mask| Depth
    """

    top = np.hstack(
        (
            rgb,
            events,
        )
    )

    bottom = np.hstack(
        (
            mask,
            depth,
        )
    )

    return np.vstack(
        (
            top,
            bottom,
        )
    )


def draw_labels(
    panel: np.ndarray,
) -> np.ndarray:
    """
    Draw titles on each visualization panel.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX

    color = (255, 255, 255)

    thickness = 2

    cv2.putText(
        panel,
        "RGB",
        (20, 35),
        font,
        1.0,
        color,
        thickness,
    )

    cv2.putText(
        panel,
        "Events",
        (660, 35),
        font,
        1.0,
        color,
        thickness,
    )

    cv2.putText(
        panel,
        "Mask",
        (20, 515),
        font,
        1.0,
        color,
        thickness,
    )

    cv2.putText(
        panel,
        "Depth",
        (660, 515),
        font,
        1.0,
        color,
        thickness,
    )

    return panel


def draw_sample_info(
    panel: np.ndarray,
    sample,
) -> np.ndarray:
    """
    Overlay useful debugging information.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX

    color = (0, 255, 255)

    thickness = 2

    lines = [

        f"Sequence : {sample.sequence_name}",

        f"Frame    : {sample.frame_id}",

        f"Timestamp: {sample.timestamp:.6f}",

        f"Events   : {len(sample.events_t):,}",

        f"Objects  : {len(sample.frame_motion.object_ids)}",

        (
            f"Camera Speed : "
            f"{sample.camera_motion.linear_speed:.3f} m/s"
        ),

        (
            f"Angular Speed: "
            f"{sample.camera_motion.angular_speed:.3f} rad/s"
        ),
    ]

    y = 900

    for line in lines:

        cv2.putText(
            panel,
            line,
            (20, y),
            font,
            0.7,
            color,
            thickness,
        )

        y += 28

    return panel


def generate_video(
    dataset_root: str | Path,
    output_path: str | Path,
    split: str = "train",
    sensors: tuple[str, ...] | None = None,
    fps: float = 30.0,
    max_frames: int | None = None,
):
    """
    Generate a visualization video for the EVIMO2 dataset.

    Parameters
    ----------
    dataset_root
        EVIMO2 root directory.

    output_path
        Output video path.

    split
        Dataset split.

    sensors
        Sensors to include.

    fps
        Output video frame rate.

    max_frames
        Optional limit for debugging.
    """

    dataset = EVIMO2Dataset(
        dataset_root=dataset_root,
        split=split,
        sensors=sensors,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(

        str(output_path),

        cv2.VideoWriter_fourcc(*"mp4v"),

        fps,

        (
            1280,
            960,
        ),
    )

    total_frames = len(dataset)

    if max_frames is not None:
        total_frames = min(
            total_frames,
            max_frames,
        )

    print()
    print("=" * 80)
    print("Generating visualization video")
    print("=" * 80)
    print(f"Frames : {total_frames}")
    print()

    for i in tqdm(range(total_frames)):

        sample = dataset[i]

        rgb = render_rgb(
            sample.rgb
        )

        events = render_events(
            sample.events_xy,
            sample.events_p,
        )

        depth = render_depth(
            sample.depth
        )

        mask = render_mask(
            sample.mask
        )

        panel = compose_panel(
            rgb,
            events,
            mask,
            depth,
        )

        panel = draw_labels(
            panel,
        )

        panel = draw_sample_info(
            panel,
            sample,
        )

        writer.write(panel)

    writer.release()

    print()
    print(f"Saved video to")
    print(output_path)


from collections import defaultdict



def build_sequence_index(dataset: EVIMO2Dataset):
    """
    Group global dataset indices by sequence.

    Returns
    -------
    dict
        {
            (sensor, sequence_name):
                [global indices...]
        }
    """

    sequences = defaultdict(list)

    for global_idx, ref in enumerate(dataset.index.references):

        sequence = dataset.index.sequences[
            ref.sequence_id
        ]

        key = (
            sequence.sensor,
            sequence.sequence_name,
        )

        sequences[key].append(global_idx)

    return sequences


def create_sequence_video(
    dataset: EVIMO2Dataset,
    indices: list[int],
    output_path: Path,
    fps: int = 30,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 960),
    )

    for idx in tqdm(indices):

        sample = dataset[idx]

        rgb = render_rgb(sample.rgb)

        events = render_events(
            sample.events_xy,
            sample.events_p,
        )

        depth = render_depth(sample.depth)

        mask = render_motion_mask(
            sample.mask,
            sample.frame_motion,
        )

        top = np.hstack([
            rgb,
            events,
        ])

        bottom = np.hstack([
            mask,
            depth,
        ])

        canvas = np.vstack([
            top,
            bottom,
        ])

        writer.write(canvas)

    writer.release()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_root",
        type=Path,
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("verification"),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--sequence",
        default=None,
        help="Only generate one sequence.",
    )

    args = parser.parse_args()

    dataset = EVIMO2Dataset(
        dataset_root=args.dataset_root,
        split=args.split,
    )

    sequence_map = build_sequence_index(dataset)

    for (sensor, sequence_name), indices in sequence_map.items():

        if (
            args.sequence is not None
            and sequence_name != args.sequence
        ):
            continue

        output = (
            args.output
            / sensor
            / f"{sequence_name}.mp4"
        )

        print(f"\nGenerating {output}")

        create_sequence_video(
            dataset,
            indices,
            output,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()