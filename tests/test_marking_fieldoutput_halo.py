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
"""Unit tests for FieldOutputMarker's ``halo`` option (mirrors RecoveryErrorMarker's own halo,
via the shared edelweissfe.adaptivity.marking._growByNeighbors helper).

Builds a tiny 5-element chain (elements 0-1-2-3-4, each pair sharing one node) as a mesh rather than
a real .inp model: FieldOutputMarker.mark() only needs `_perElementFieldOutputResult(model, name)` ->
(element numbers, values), which is monkeypatched directly, and a `refineElements` candidate pool for
the halo to grow within -- nothing else about the model is touched.
"""

import numpy as np

import edelweissfe.adaptivity.marking as marking
from edelweissfe.adaptivity.marking import (
    FieldOutputMarker,
    FieldOutputMarkerSchema,
    RefineableElements,
)
from edelweissfe.models.mesh import Mesh
from edelweissfe.utils.schema import buildSchemaFromOptions

#: The mesh of the chain: element i has the nodes i and i + 1.
_CHAIN = Mesh()
for _number in range(5):
    _CHAIN.addElement(_number, "T2D2", None, [_number, _number + 1])


def _chainOfFive():
    """5 elements in a row, each sharing exactly one node with its immediate neighbor(s): their
    numbers."""
    return list(range(5))


def _pool(numbers):
    """The refineable elements of the chain with the given numbers."""
    return RefineableElements(numbers, _CHAIN)


def _markMiddleOnly(monkeypatch, elements, halo):
    """Mark only the middle element (index 2) via a value of 5.0 against threshold 1.0, everything
    else 0.0, and return the marked set for the given halo."""

    def fakeResult(model, fieldOutputName):
        values = np.zeros((len(elements), 1))
        values[2, 0] = 5.0
        return elements, values

    monkeypatch.setattr(marking, "_perElementFieldOutputResult", fakeResult)
    m = FieldOutputMarker("dummy", threshold=1.0, operator=">=", halo=halo)
    return m.mark(model=None, refineElements=_pool(elements), mesh=None)


def test_halo_zero_is_the_bare_threshold_set_backward_compatible(monkeypatch):
    elements = _chainOfFive()
    marked = _markMiddleOnly(monkeypatch, elements, halo=0)
    assert marked == {elements[2]}


def test_halo_one_bridges_the_immediate_neighbors(monkeypatch):
    elements = _chainOfFive()
    marked = _markMiddleOnly(monkeypatch, elements, halo=1)
    assert marked == {elements[1], elements[2], elements[3]}


def test_halo_two_reaches_the_whole_chain(monkeypatch):
    elements = _chainOfFive()
    marked = _markMiddleOnly(monkeypatch, elements, halo=2)
    assert marked == set(elements)


def test_halo_never_leaks_outside_the_refineable_candidate_pool(monkeypatch):
    """The halo only grows within `refineElements` -- an element sharing a node with the marked one
    but absent from the candidate pool must never appear in the result."""
    elements = _chainOfFive()

    def fakeResult(model, fieldOutputName):
        values = np.zeros((len(elements), 1))
        values[2, 0] = 5.0
        return elements, values

    monkeypatch.setattr(marking, "_perElementFieldOutputResult", fakeResult)
    m = FieldOutputMarker("dummy", threshold=1.0, operator=">=", halo=5)
    # candidate pool excludes element 4 entirely
    restrictedPool = elements[:4]
    marked = m.mark(model=None, refineElements=_pool(restrictedPool), mesh=None)
    assert elements[4] not in marked
    assert marked == {elements[0], elements[1], elements[2], elements[3]}


def test_halo_is_a_noop_when_nothing_is_marked(monkeypatch):
    elements = _chainOfFive()

    def fakeResult(model, fieldOutputName):
        return elements, np.zeros((len(elements), 1))

    monkeypatch.setattr(marking, "_perElementFieldOutputResult", fakeResult)
    m = FieldOutputMarker("dummy", threshold=1.0, operator=">=", halo=3)
    assert m.mark(model=None, refineElements=_pool(elements), mesh=None) == set()


def test_schema_default_halo_is_zero():
    opts = buildSchemaFromOptions(
        FieldOutputMarkerSchema, {"fieldOutput": "alphaP", "threshold": "0.95", "operator": ">"}
    )
    assert opts.halo == 0


def test_from_options_threads_halo_through():
    marker = FieldOutputMarker.fromOptions({"fieldOutput": "alphaP", "threshold": "0.95", "operator": ">", "halo": "2"})
    assert marker.halo == 2
