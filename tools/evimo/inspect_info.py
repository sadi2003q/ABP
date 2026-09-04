#!/usr/bin/env python3

"""
Inspect EVIMO2 dataset_info.npz

This script recursively prints the contents of dataset_info.npz,
including nested dictionaries and object arrays.
"""

from pathlib import Path
import argparse
import numpy as np


# -------------------------------------------------------------

def print_line():
    print("=" * 90)


def inspect_object(obj, indent=0):

    prefix = " " * indent

    # ---------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------

    if isinstance(obj, dict):

        print(f"{prefix}Dictionary ({len(obj)} keys)")

        for key, value in obj.items():

            print(f"{prefix}- {key}")

            inspect_object(value, indent + 4)

        return

    # ---------------------------------------------------------
    # ndarray
    # ---------------------------------------------------------

    if isinstance(obj, np.ndarray):

        print(f"{prefix}shape : {obj.shape}")
        print(f"{prefix}dtype : {obj.dtype}")

        if obj.dtype == object:

            print(f"{prefix}Object Array")

            if obj.size:

                inspect_object(obj.flat[0], indent + 4)

            return

        if obj.size == 0:
            return

        print(f"{prefix}min   : {obj.min()}")
        print(f"{prefix}max   : {obj.max()}")

        if obj.ndim == 1:

            print(f"{prefix}first five")

            print(obj[:5])

        else:

            print(f"{prefix}first element")

            print(obj[0])

        return

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    if isinstance(obj, list):

        print(f"{prefix}List ({len(obj)})")

        if len(obj):

            inspect_object(obj[0], indent + 4)

        return

    # ---------------------------------------------------------
    # Scalar
    # ---------------------------------------------------------

    print(f"{prefix}{type(obj)}")

    print(f"{prefix}{obj}")


# -------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        type=str,
        help="dataset_info.npz",
    )

    args = parser.parse_args()

    file = Path(args.file)

    print_line()

    print(file)

    print_line()

    data = np.load(file, allow_pickle=True)

    print("\nKeys")

    print_line()


    for key in data.files:

        print("\n")

        print(key)

        print("-" * 60)

        inspect_object(data[key])


if __name__ == "__main__":

    main()