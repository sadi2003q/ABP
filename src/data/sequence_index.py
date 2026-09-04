"""
Global sequence indexing for the EVIMO2 dataset.

This module discovers all valid EVIMO2 sequences under the dataset
root and constructs a global frame index used by the Dataset.

The goal of this module is purely indexing.

It does NOT

    • load event windows
    • load depth images
    • load masks
    • load RGB images
    • load cached preprocessing files

Those responsibilities belong to EVIMO2Dataset.

Instead this module answers one question:

    Given a global dataset index,
    which sequence and which frame does it correspond to?

The resulting lookup table allows constant-time mapping

    dataset index
            ↓
    (sequence, frame)

without repeatedly traversing the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.evimo2.parser import EVIMO2Parser
from src.data.sample import SequenceReference


# ==========================================================
# Supported sensors
# ==========================================================

DEFAULT_EVENT_SENSORS = (
    "left_camera",
    "right_camera",
    "samsung_mono",
)


# ==========================================================
# Sequence information
# ==========================================================

@dataclass(slots=True)
class SequenceInfo:
    """
    Metadata describing one EVIMO2 sequence.

    This class stores lightweight information needed by the Dataset
    before any sample is loaded.

    Parameters
    ----------
    sequence_id
        Integer identifier assigned during dataset discovery.

    sequence_dir
        Absolute path to the sequence directory.

    sensor
        Event camera sensor name.

    split
        Dataset split.

    sequence_name
        Name of the sequence directory.

    num_frames
        Number of parsed frames.
    """

    sequence_id: int

    sequence_dir: Path

    rgb_sequence_dir: Path | None

    sensor: str

    split: str

    sequence_name: str

    num_frames: int


# ==========================================================
# Sequence index
# ==========================================================

class SequenceIndex:
    """
    Global EVIMO2 sequence index.

    Discovers all requested sequences and builds a global lookup table
    mapping

        dataset index
                ↓
        SequenceReference

    The index is constructed once during Dataset initialization.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        sensors: tuple[str, ...] | list[str] | None = None,
        split: str = "train",
    ):

        self.dataset_root = Path(dataset_root).expanduser()

        self.sensors = tuple(
            sensors
            if sensors is not None
            else DEFAULT_EVENT_SENSORS
        )

        self.split = split

        self.sequences: list[SequenceInfo] = (
            self._discover_sequences()
        )

        self.references: list[SequenceReference] = (
            self._build_sample_index()
        )

        self._print_summary()

    # ======================================================
    # Discovery
    # ======================================================

    def _discover_sequences(
        self,
    ) -> list[SequenceInfo]:
        """
        Discover all valid EVIMO2 sequences.

        Returns
        -------
        list[SequenceInfo]
        """

        sequences = []

        sequence_id = 0

        for sensor in self.sensors:

            sensor_root = (
                self.dataset_root /
                sensor
            )

            if not sensor_root.exists():
                continue

            split_root = (
                sensor_root /
                "imo" /
                self.split
            )

            if not split_root.exists():
                continue

            for sequence_dir in sorted(split_root.iterdir()):

                if not sequence_dir.is_dir():
                    continue

                if not (
                    sequence_dir /
                    "dataset_info.npz"
                ).exists():
                    continue


                #
                # Matching RGB sequence.
                #
                rgb_sequence_dir = (
                    self.dataset_root
                    / "flea3_7"
                    / "imo"
                    / self.split
                    / sequence_dir.name
                )

                if not rgb_sequence_dir.exists():

                    rgb_sequence_dir = None
                
                parser = EVIMO2Parser(
                    sequence_dir
                )

                sequences.append(

                    SequenceInfo(

                        sequence_id=sequence_id,

                        sequence_dir=sequence_dir,

                        rgb_sequence_dir=rgb_sequence_dir,

                        sensor=sensor,

                        split=self.split,

                        sequence_name=sequence_dir.name,

                        num_frames=len(parser.frames),
                    )

                )

                sequence_id += 1

        return sequences

    # ======================================================
    # Global frame index
    # ======================================================

    def _build_sample_index(
        self,
    ) -> list[SequenceReference]:
        """
        Construct the global dataset index.

        Returns
        -------
        list[SequenceReference]
        """

        references = []

        for sequence in self.sequences:

            for local_frame_index in range(
                sequence.num_frames
            ):

                references.append(

                    SequenceReference(

                        sequence_id=sequence.sequence_id,

                        local_frame_index=local_frame_index,
                    )

                )

        return references

    # ======================================================
    # Summary
    # ======================================================

    def _print_summary(
        self,
    ) -> None:
        """
        Print a summary of the discovered dataset.
        """

        print("=" * 80)
        print("EVIMO2 Sequence Index")
        print("=" * 80)

        print(f"Sequences : {len(self.sequences)}")
        print(f"Frames    : {len(self.references)}")
        print(f"Sensors   : {', '.join(self.sensors)}")
        print(f"Split     : {self.split}")

        print("=" * 80)

    # ======================================================
    # Standard container interface
    # ======================================================

    def __len__(
        self,
    ) -> int:
        """
        Return the total number of indexed frames.
        """

        return len(
            self.references
        )

    def __getitem__(
        self,
        index: int,
    ) -> SequenceReference:
        """
        Return the sequence reference corresponding to a
        global dataset index.
        """

        return self.references[index]

    def __iter__(
        self,
    ):
        """
        Iterate over all frame references.
        """

        return iter(
            self.references
        )

    # ======================================================
    # Convenience helpers
    # ======================================================

    def get_sequence(
        self,
        sequence_id: int,
    ) -> SequenceInfo:
        """
        Return metadata for one sequence.
        """

        return self.sequences[
            sequence_id
        ]

    def num_sequences(
        self,
    ) -> int:
        """
        Return the number of discovered sequences.
        """

        return len(
            self.sequences
        )

    def num_frames(
        self,
        sequence_id: int,
    ) -> int:
        """
        Return the number of frames in a sequence.
        """

        return self.sequences[
            sequence_id
        ].num_frames