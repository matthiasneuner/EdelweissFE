#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  Alexander Dummer alexander.dummer@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""A minimal binding of the METIS graph partitioner, through ``ctypes``.

Only ``METIS_PartMeshDual`` is bound: it partitions the *elements* of a mesh, given as element-node
connectivity, by building the dual graph (elements adjacent when they share at least ``nCommon``
nodes) and partitioning that. The objective is the total communication volume rather than the edge
cut, since what an explicit increment exchanges is the interface degrees of freedom, not the cut
faces.

The shared library is ``libmetis.so`` from the active environment (conda-forge's ``metis``, also a
dependency of SuiteSparse), or the one ``EDELWEISSFE_METIS_LIBRARY`` names. Its index type must be
32-bit (``IDXTYPEWIDTH 32``), which is what conda-forge builds.
"""

import ctypes
import ctypes.util
import os
import sys
from functools import cache

import numpy as np

_METIS_OK = 1
_METIS_NOPTIONS = 40
_METIS_OPTION_OBJTYPE = 1
_METIS_OPTION_SEED = 8
_METIS_OPTION_NUMBERING = 17
_METIS_OBJTYPE_VOL = 1

_idx = ctypes.c_int32
_idxPointer = ctypes.POINTER(_idx)
_realPointer = ctypes.POINTER(ctypes.c_float)


@cache
def _metisLibrary() -> ctypes.CDLL:
    """Load ``libmetis`` and declare the two functions used.

    Returns
    -------
    ctypes.CDLL
        The loaded library.

    Raises
    ------
    ImportError
        If no METIS library can be found.
    """

    candidates = [
        os.environ.get("EDELWEISSFE_METIS_LIBRARY"),
        os.path.join(sys.prefix, "lib", "libmetis.so"),
        ctypes.util.find_library("metis"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        library.METIS_SetDefaultOptions.argtypes = [_idxPointer]
        library.METIS_SetDefaultOptions.restype = ctypes.c_int
        library.METIS_PartMeshDual.argtypes = [
            _idxPointer,  # ne
            _idxPointer,  # nn
            _idxPointer,  # eptr
            _idxPointer,  # eind
            _idxPointer,  # vwgt
            _idxPointer,  # vsize
            _idxPointer,  # ncommon
            _idxPointer,  # nparts
            _realPointer,  # tpwgts
            _idxPointer,  # options
            _idxPointer,  # objval
            _idxPointer,  # epart
            _idxPointer,  # npart
        ]
        library.METIS_PartMeshDual.restype = ctypes.c_int
        return library

    raise ImportError(
        "No METIS library found (tried $EDELWEISSFE_METIS_LIBRARY, {:}, and the system search path). "
        "Install conda-forge's 'metis' into the environment.".format(os.path.join(sys.prefix, "lib", "libmetis.so"))
    )


def _asIndexArray(values) -> np.ndarray:
    """A contiguous 32-bit copy of an integer array, checked to fit."""

    values = np.asarray(values)
    if values.size and (values.max() > np.iinfo(np.int32).max or values.min() < 0):
        raise OverflowError("METIS is bound with 32-bit indices; the mesh does not fit.")
    return np.ascontiguousarray(values, dtype=np.int32)


def _pointer(array: np.ndarray, pointerType=_idxPointer):
    return array.ctypes.data_as(pointerType)


def partitionMeshDual(
    elementNodeOffsets: np.ndarray,
    elementNodes: np.ndarray,
    nNodes: int,
    nParts: int,
    elementWeights: np.ndarray,
    nCommon: int,
    partShares: np.ndarray | None = None,
) -> np.ndarray:
    """Partition the elements of a mesh into ``nParts`` parts of balanced weight -- of equal weight, or
    of the given shares of the total weight.

    Parameters
    ----------
    elementNodeOffsets
        CSR row pointer of the element-node connectivity, of length ``nElements + 1``.
    elementNodes
        The node indices of all elements, concatenated; each in ``[0, nNodes)``.
    nNodes
        The number of distinct nodes.
    nParts
        The number of parts; at least 2.
    elementWeights
        A positive integer weight per element: its expected computing cost.
    nCommon
        How many nodes two elements must share to be adjacent in the dual graph.
    partShares
        The share of the total weight each part is to receive (METIS' target part weights,
        ``tpwgts``): ``nParts`` positive numbers summing to 1. None for equal shares.

    Returns
    -------
    np.ndarray
        The part of each element, in ``[0, nParts)``.
    """

    library = _metisLibrary()

    elementNodeOffsets = _asIndexArray(elementNodeOffsets)
    elementNodes = _asIndexArray(elementNodes)
    elementWeights = _asIndexArray(elementWeights)
    nElements = elementNodeOffsets.shape[0] - 1

    if nParts < 2:
        raise ValueError("METIS partitions into at least 2 parts, not {:}.".format(nParts))
    if elementWeights.shape[0] != nElements or np.any(elementWeights < 1):
        raise ValueError("One positive integer weight per element is required.")
    if partShares is not None:
        partShares = np.ascontiguousarray(partShares, dtype=np.float32)
        if partShares.shape != (nParts,) or np.any(partShares <= 0.0) or abs(float(partShares.sum()) - 1.0) > 1e-4:
            raise ValueError("One positive share per part, summing to 1, is required.")

    options = np.zeros(_METIS_NOPTIONS, dtype=np.int32)
    library.METIS_SetDefaultOptions(_pointer(options))
    options[_METIS_OPTION_OBJTYPE] = _METIS_OBJTYPE_VOL
    options[_METIS_OPTION_NUMBERING] = 0
    # A fixed seed, so that a partition is reproducible from one run to the next.
    options[_METIS_OPTION_SEED] = 1

    ne = _idx(nElements)
    nn = _idx(nNodes)
    ncommon = _idx(nCommon)
    nparts = _idx(nParts)
    objval = _idx(0)
    elementParts = np.zeros(nElements, dtype=np.int32)
    nodeParts = np.zeros(nNodes, dtype=np.int32)

    status = library.METIS_PartMeshDual(
        ctypes.byref(ne),
        ctypes.byref(nn),
        _pointer(elementNodeOffsets),
        _pointer(elementNodes),
        _pointer(elementWeights),
        None,
        ctypes.byref(ncommon),
        ctypes.byref(nparts),
        None if partShares is None else _pointer(partShares, _realPointer),
        _pointer(options),
        ctypes.byref(objval),
        _pointer(elementParts),
        _pointer(nodeParts),
    )

    if status != _METIS_OK:
        raise RuntimeError("METIS_PartMeshDual failed with status {:}.".format(status))
    if nElements and (elementParts.min() < 0 or elementParts.max() >= nParts):
        raise RuntimeError(
            "METIS returned parts outside [0, {:}); is the library built with 64-bit indices?".format(nParts)
        )

    return elementParts.astype(int)
