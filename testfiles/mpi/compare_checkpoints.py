#!/usr/bin/env python3
"""Compare the restart checkpoints of two runs of the test cases of ``testfiles/mpi``, dataset by dataset.

A restart checkpoint holds everything a run carries on with: the state of every element and every
constraint, ``U``, ``V`` and ``P`` of every node field, and the solver's external work. Two runs of the
same test cases -- the serial one with ``NED``, and one over several processes with ``NEDMPI`` -- are
the same, bit for bit, if every dataset of every checkpoint is: the same names, shapes, types and bytes
(signed zeros included).

Every ``restart*.h5`` below the first directory is compared with the file at the same place below the
second::

    python testfiles/mpi/compare_checkpoints.py SERIAL DECOMPOSED

It prints one line per checkpoint and every dataset that differs, and exits with 1 if any differs or
is missing -- or if there is no checkpoint at all, which would compare nothing.
"""

import glob
import os
import sys

import h5py
import numpy as np


def datasetsOf(path: str) -> dict:
    """Every dataset of an HDF5 file, by name.

    Parameters
    ----------
    path
        The file.

    Returns
    -------
    dict
        The datasets, as arrays.
    """

    datasets = {}
    with h5py.File(path, "r") as checkpoint:

        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                datasets[name] = np.asarray(item[()])

        checkpoint.visititems(collect)
    return datasets


def differences(expected: dict, actual: dict) -> list[str]:
    """The names of the datasets that differ between two checkpoints, or are in one only.

    Parameters
    ----------
    expected
        The datasets of the one, by name.
    actual
        The datasets of the other, by name.

    Returns
    -------
    list[str]
        The names, each with how it differs.
    """

    found = ["{:} (missing)".format(name) for name in expected.keys() - actual.keys()]
    found += ["{:} (unexpected)".format(name) for name in actual.keys() - expected.keys()]
    for name in sorted(expected.keys() & actual.keys()):
        a, b = expected[name], actual[name]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            found.append(name)
    return sorted(found)


def main() -> int:
    serial, decomposed = sys.argv[1], sys.argv[2]
    checkpoints = sorted(glob.glob(os.path.join(serial, "**", "restart*.h5"), recursive=True))
    if not checkpoints:
        print("no checkpoint below {:}".format(serial))
        return 1

    failed = False
    for checkpoint in checkpoints:
        place = os.path.relpath(checkpoint, serial)
        other = os.path.join(decomposed, place)
        if not os.path.exists(other):
            print("{:}: MISSING".format(place))
            failed = True
            continue
        expected = datasetsOf(checkpoint)
        differing = differences(expected, datasetsOf(other))
        print("{:}: {:}".format(place, "DIFFERS" if differing else "SAME ({:} datasets)".format(len(expected))))
        for name in differing:
            print("    " + name)
        failed |= bool(differing)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
