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
"""The exact sum of floating-point numbers: split anyhow and added, rounded once, it is the bits of
:func:`math.fsum` of all of them."""

import math

import numpy as np
import pytest

from edelweissfe.numerics.exactsum import ExactSum


def _adversarial(rng, n: int) -> np.ndarray:
    """Numbers whose rounded sums differ in almost any order: magnitudes over the whole range of
    doubles, subnormals, cancelling pairs, signed zeros and ties."""

    values = rng.standard_normal(n) * 10.0 ** rng.integers(-300, 300, n)
    values[rng.integers(0, n, n // 10)] = 0.0
    values[rng.integers(0, n, n // 10)] = -0.0
    values[rng.integers(0, n, n // 20)] = 5e-324 * rng.integers(-1000, 1000, n // 20)
    cancelling = rng.integers(0, n, n // 10)
    values = np.concatenate([values, -values[cancelling], [2.0**53, 1.0, 1.0, -(2.0**-1074)]])
    rng.shuffle(values)
    return values


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("parts", [1, 2, 3, 7])
def test_parts_added_and_rounded_once_are_the_bits_of_fsum(seed, parts):
    rng = np.random.default_rng(seed)
    values = _adversarial(rng, int(rng.integers(1, 3000)))
    # a sum dominated by numbers of ordinary magnitude, too, such as the work of an increment
    if seed % 2:
        values = values[np.abs(values) < 1e10]
    cuts = np.sort(rng.integers(0, values.shape[0] + 1, parts - 1))
    total = ExactSum()
    for part in np.split(values, cuts):
        total = total + ExactSum.of(part)

    expected = math.fsum(values.tolist())
    assert np.float64(total.rounded()).tobytes() == np.float64(expected).tobytes()


def test_zeros_and_nothing_sum_to_positive_zero_as_in_fsum():
    for values in ([], [-0.0], [-0.0, -0.0], [0.0, -0.0], [1.5, -1.5]):
        rounded = (ExactSum.of(np.array(values)) + ExactSum()).rounded()
        assert np.float64(rounded).tobytes() == np.float64(math.fsum(values)).tobytes()


def test_non_finite_numbers_dominate():
    assert ExactSum.of(np.array([1.0, math.inf])).rounded() == math.inf
    assert (ExactSum.of(np.array([-math.inf])) + ExactSum.of(np.array([1.0]))).rounded() == -math.inf
    assert math.isnan(ExactSum.of(np.array([math.nan, 1.0])).rounded())
    assert math.isnan((ExactSum.of(np.array([math.inf])) + ExactSum.of(np.array([-math.inf]))).rounded())
    assert ExactSum.of(np.array([1e308, 1e308])).rounded() == math.inf
