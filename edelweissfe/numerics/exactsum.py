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
"""The exact sum of floating-point numbers, as one integer, so that sums of parts can be added.

Every finite double is an integer multiple of :math:`2^{-1074}`, the smallest subnormal. A sum of
doubles is therefore an integer multiple of it too, and Python's integers hold such a sum exactly,
however many terms it has and however their magnitudes differ. Two exact sums -- of the numbers one
process holds and of those another holds -- add exactly, in any order; rounded once at the end, the
result is the correctly rounded sum of all the numbers, which is what :func:`math.fsum` returns: the
same bits, however the numbers were split.

The terms are not summed one by one in Python: each is split into its binary exponent and two integer
halves of its significand, and the halves are summed per exponent by :func:`numpy.bincount`, exactly,
since every partial sum stays below :math:`2^{53}`. Only the few exponents that occur are then
combined as Python integers.
"""

import math

import numpy as np

#: The exponent of the integer an :class:`ExactSum` counts in: the significand of a double has 53
#: bits, and the smallest binary exponent :func:`numpy.frexp` returns is -1073, so a term
#: ``m * 2**e`` (``0.5 <= |m| < 1``) is the integer ``(m * 2**53) << (e + 1073)`` times ``2**-_SCALE``.
_SCALE = 53 + 1073

#: The low half of a 53-bit significand: summed apart from the high half, each half of up to
#: :math:`2^{26}` terms sums exactly in a double.
_LOW_BITS = 26

#: The most terms whose halves are summed in one pass of :func:`numpy.bincount`, so that every sum
#: of a half stays below :math:`2^{53}`.
_TERMS_PER_PASS = 1 << 26


class ExactSum:
    """The exact sum of finite floating-point numbers, as an integer multiple of :math:`2^{-1126}`, and
    the non-finite numbers among them.

    Parameters
    ----------
    scaled
        The sum of the finite numbers, times :math:`2^{1126}`.
    nonFinite
        The infinities and NaNs among the numbers.
    """

    __slots__ = ("scaled", "nonFinite")

    def __init__(self, scaled: int = 0, nonFinite: tuple = ()):
        self.scaled = scaled
        self.nonFinite = nonFinite

    @classmethod
    def of(cls, values: np.ndarray) -> "ExactSum":
        """The exact sum of an array of numbers.

        Parameters
        ----------
        values
            The numbers, a one-dimensional array of doubles.

        Returns
        -------
        ExactSum
            Their exact sum.
        """

        values = np.asarray(values, dtype=float)
        values = values[values != 0.0]  # zeros add nothing; and NaNs are kept
        if not values.size:
            return cls()
        finite = np.isfinite(values)
        nonFinite = ()
        if not finite.all():
            nonFinite = tuple(values[~finite].tolist())
            values = values[finite]
            if not values.size:
                return cls(0, nonFinite)

        significands, exponents = np.frexp(values)
        integers = (significands * float(1 << 53)).astype(np.int64)
        high = integers >> _LOW_BITS
        low = integers - (high << _LOW_BITS)
        shifts = exponents + 1073
        lowest = int(shifts.min())
        bins = shifts - lowest

        scaled = 0
        for begin in range(0, values.shape[0], _TERMS_PER_PASS):
            part = slice(begin, begin + _TERMS_PER_PASS)
            highSums = np.bincount(bins[part], weights=high[part])
            lowSums = np.bincount(bins[part], weights=low[part])
            for shift in np.flatnonzero((highSums != 0.0) | (lowSums != 0.0)).tolist():
                scaled += ((int(highSums[shift]) << _LOW_BITS) + int(lowSums[shift])) << (shift + lowest)
        return cls(scaled, nonFinite)

    def __add__(self, other: "ExactSum") -> "ExactSum":
        """The exact sum of both sums.

        Parameters
        ----------
        other
            The other sum.

        Returns
        -------
        ExactSum
            The sum.
        """

        return ExactSum(self.scaled + other.scaled, self.nonFinite + other.nonFinite)

    def rounded(self) -> float:
        """The sum, correctly rounded to the nearest double (ties to even): the bits :func:`math.fsum`
        returns for the same numbers, in whatever order and split. Zero is +0.0, as from
        :func:`math.fsum`. With non-finite numbers among them, the sum :func:`math.fsum` forms of those
        alone (an infinity, or NaN); a finite sum too large for a double is an infinity, where
        :func:`math.fsum` raises.

        Returns
        -------
        float
            The rounded sum.
        """

        if self.nonFinite:
            if any(math.isnan(value) for value in self.nonFinite) or len(set(self.nonFinite)) > 1:
                return math.nan
            return self.nonFinite[0]
        try:
            # true division of integers is correctly rounded, ties to even
            return self.scaled / (1 << _SCALE)
        except OverflowError:
            return math.inf if self.scaled > 0 else -math.inf
