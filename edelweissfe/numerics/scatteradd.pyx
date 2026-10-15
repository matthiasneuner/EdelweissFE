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
"""The scatter-add of an assembly: the sum of values at each of a set of positions, each sum formed in
the order the values are given.

Floating-point addition is not associative, so a sum of the same values in another order may differ
in its last bits. The bit-identity of the threaded and the domain-decomposed explicit element loops
with the plain one rests on every sum being formed in one order -- element after element, from left to
right -- wherever the contributions are summed. That order is this function's contract, not an
implementation detail of a library: a sequential loop, compiled without reassociating floating-point
operations.
"""

import numpy as np

from libc.stdint cimport int64_t


def sumAtPositions(const int64_t[::1] positions, const double[::1] values, Py_ssize_t nPositions):
    """The sum of the values at each position: ``sums[positions[i]] += values[i]`` for ``i = 0, 1, ...``
    in that order, from zero -- a left fold per position, the sum an element loop forms adding one
    element's contribution after another.

    Parameters
    ----------
    positions
        The position of every value, each in ``[0, nPositions)``.
    values
        The values, in the order they are added.
    nPositions
        The number of positions.

    Returns
    -------
    np.ndarray
        The sum at every position; 0.0 where no value is.

    Raises
    ------
    ValueError
        If the values do not match the positions, or a position is out of range.
    """

    cdef Py_ssize_t n = positions.shape[0]
    if values.shape[0] != n:
        raise ValueError("{:} values for {:} positions".format(values.shape[0], n))

    sums = np.zeros(nPositions)
    cdef double[::1] sumsView = sums
    cdef Py_ssize_t i
    cdef int64_t position
    cdef bint outOfRange = False
    with nogil:
        for i in range(n):
            position = positions[i]
            if position < 0 or position >= nPositions:
                outOfRange = True
                break
            sumsView[position] += values[i]
    if outOfRange:
        raise ValueError("A position lies outside [0, {:}).".format(nPositions))
    return sums
