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

"""P1 of PLAN_TOPOLOGY_PIPELINE.md: the element number allocator and the topology mutation window.

These pin the two properties the restart replay is built on (plan §2.6): element numbering is a pure
function of the ordered *creation* sequence -- never of the deletion history, never of
``max(model.elements)`` -- and entities may only be created or deleted inside a topology change.
"""

import pytest

from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.models.modelchangeobserver import ModelChangeType as _MCT
from edelweissfe.utils.exceptions import TopologyError


class _StubElement:
    """A bare element stand-in: FEModel's allocator and window only ever look at ``elNumber``."""

    def __init__(self, elNumber: int):
        self.elNumber = elNumber


def _modelWithSetupElements(*labels: int) -> FEModel:
    """A model carrying elements as the base mesh generators leave them -- placed directly, with
    the allocator not yet raised above them."""

    model = FEModel(3)
    for label in labels:
        model.elements[label] = _StubElement(label)
    return model


def test_reserved_numbers_are_consecutive_and_monotonic():
    model = _modelWithSetupElements()
    with model.topologyChanges():
        first = model.reserveElementNumbers(3)
        second = model.reserveElementNumbers(2)

    assert list(first) == [1, 2, 3]
    assert list(second) == [4, 5]


def test_numbers_are_never_recycled_after_removal():
    """The heart of the allocator: a rebuilt contact facet set must not be handed the numbers of the
    facets it just replaced, or numbering would depend on the deletion history and a replay would
    have to reproduce that too."""

    model = _modelWithSetupElements()
    with model.topologyChanges():
        (number,) = model.reserveElementNumbers(1)
        model.createElement(_StubElement(number))
        model.removeElement(number)

        (afterRemoval,) = model.reserveElementNumbers(1)

    assert afterRemoval != number
    assert afterRemoval == number + 1


def test_allocator_ignores_the_current_maximum():
    """max(model.elements) drops when the highest-numbered elements are deleted; the allocator must
    not follow it down."""

    model = _modelWithSetupElements()
    with model.topologyChanges():
        numbers = model.reserveElementNumbers(4)
        for number in numbers:
            model.createElement(_StubElement(number))
        for number in numbers:
            model.removeElement(number)

        assert not model.elements  # max(model.elements, default=0) + 1 would restart at 1 here
        (afterEmptying,) = model.reserveElementNumbers(1)

    assert afterEmptying == numbers[-1] + 1


def test_adopt_setup_element_numbers_clears_the_base_mesh():
    model = _modelWithSetupElements(1, 2, 17)
    model.adoptSetupElementNumbers()

    with model.topologyChanges():
        (number,) = model.reserveElementNumbers(1)

    assert number == 18


def test_adopt_setup_element_numbers_never_lowers_the_mark():
    model = _modelWithSetupElements(1, 2)
    with model.topologyChanges():
        model.reserveElementNumbers(50)

    model.adoptSetupElementNumbers()

    with model.topologyChanges():
        (number,) = model.reserveElementNumbers(1)

    assert number == 51


def test_reserve_outside_a_topology_change_raises():
    model = _modelWithSetupElements()
    with pytest.raises(TopologyError, match="topology change"):
        model.reserveElementNumbers(1)


def test_create_outside_a_topology_change_raises():
    model = _modelWithSetupElements()
    with pytest.raises(TopologyError, match="outside a topology change"):
        model.createElement(_StubElement(1))


def test_remove_outside_a_topology_change_raises():
    model = _modelWithSetupElements(1)
    with pytest.raises(TopologyError, match="outside a topology change"):
        model.removeElement(1)


def test_creating_a_taken_number_raises():
    model = _modelWithSetupElements(1)
    with model.topologyChanges():
        with pytest.raises(TopologyError, match="already taken"):
            model.createElement(_StubElement(1))


def test_windows_nest_without_closing_early():
    """A setup helper may open a window without knowing whether its caller already did; the inner
    scope must not close the outer one."""

    model = _modelWithSetupElements()
    with model.topologyChanges():
        with model.topologyChanges():
            model.reserveElementNumbers(1)
        model.reserveElementNumbers(1)  # outer window still open

    with pytest.raises(TopologyError):
        model.reserveElementNumbers(1)


def test_window_closes_on_exception():
    model = _modelWithSetupElements()
    with pytest.raises(RuntimeError):
        with model.topologyChanges():
            raise RuntimeError("modifier blew up")

    with pytest.raises(TopologyError):
        model.reserveElementNumbers(1)


def test_parsed_element_set_keeps_its_declaration_order():
    """An ``*elSet`` must come out in the order it was declared.

    Element sets are OrderedSets, and their member order reaches element *numbering*: an
    element-based ``*surface`` is built from such a set, and ``surfaceElementGenerator`` walks it
    handing out sequential facet element labels. Elements are hashed by identity, so building the
    set through a raw ``set()`` fixed that order from object *addresses* -- reproducible only for a
    bit-identical allocation history, which a resumed run does not have (its ``.inp`` carries
    ``*restart``, and it has an extra open file). Verified to fail before the fix, with this exact
    fixture producing [1, 4, 9, 3, 8, 12, 2, 7, 11, 6, 5, 10].

    See PLAN_TOPOLOGY_PIPELINE.md determinism rule 2.
    """

    from collections import defaultdict

    from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
    from edelweissfe.journal.journal import Journal

    nElements = 12
    inputFile = defaultdict(list)
    inputFile["node"] = [
        {"nSet": None, "datalines": ["{:}, {:}, 0.0".format(i, float(i)) for i in range(1, 2 * nElements + 3)]}
    ]
    inputFile["element"] = [
        {
            "type": "CPE4",
            "provider": "edelweiss",
            "elSet": None,
            "elset": None,
            "datalines": [
                "{:}, {:}, {:}, {:}, {:}".format(e, e, e + 1, e + nElements + 2, e + nElements + 1)
                for e in range(1, nElements + 1)
            ],
        }
    ]
    inputFile["elSet"] = [{"elSet": "declared", "datalines": [", ".join(str(e) for e in range(1, nElements + 1))]}]

    model = AbqModelConstructor(Journal(verbose=False)).createGeometryFromInputFile(FEModel(2), inputFile)

    assert [el.elNumber for el in model.elementSets["declared"]] == list(range(1, nElements + 1))


class _StubModifier:
    """A modifier that plans a fixed number of times, then settles."""

    def __init__(self, name, plansLeft, log, reactsToOthers=False):
        self.name = name
        self._plansLeft = plansLeft
        self._log = log
        self._reactsToOthers = reactsToOthers

    def plan(self, model, change, step, timeStep):
        # react to another modifier's mutation once, then settle -- the contract that makes the
        # pipeline converge (see ModelModifierBase.plan)
        if change is not None and not self._reactsToOthers:
            return None
        if self._plansLeft <= 0:
            return None
        self._plansLeft -= 1
        return {"who": self.name}

    def apply(self, model, plan):
        self._log.append(plan["who"])
        (number,) = model.reserveElementNumbers(1)
        model.createElement(_StubElement(number))
        model.notifyModelChanged(_MCT.REFINEMENT)
        return None


def _modelWithModifiers(**modifiers) -> FEModel:
    model = FEModel(3)
    model.modelModifiers.update(modifiers)
    return model


def test_a_single_round_suffices_when_nobody_reacts():
    log = []
    model = _modelWithModifiers(
        amr=_StubModifier("amr", 1, log),
        printer=_StubModifier("printer", 1, log),
    )
    assert model.updateTopology(step=None, timeStep=0.0) is True
    # both planned in round 1; in round 2 each sees only the other's change and settles
    assert log == ["amr", "printer"]


def test_modifiers_run_in_declaration_order_every_round():
    log = []
    model = _modelWithModifiers(
        amr=_StubModifier("amr", 2, log, reactsToOthers=True),
        facets=_StubModifier("facets", 2, log, reactsToOthers=True),
    )
    model.updateTopology(step=None, timeStep=0.0)
    assert log == ["amr", "facets", "amr", "facets"]


def test_no_change_means_no_topology_update():
    log = []
    model = _modelWithModifiers(amr=_StubModifier("amr", 0, log))
    assert model.updateTopology(step=None, timeStep=0.0) is False
    assert log == []


def test_non_convergence_raises_naming_the_offender():
    log = []
    model = _modelWithModifiers(runaway=_StubModifier("runaway", 10**6, log, reactsToOthers=True))
    model.maxTopologyRounds = 4
    with pytest.raises(TopologyError, match="did not settle within 4 rounds.*runaway"):
        model.updateTopology(step=None, timeStep=0.0)


def test_the_window_is_closed_again_after_the_update():
    model = _modelWithModifiers(amr=_StubModifier("amr", 1, []))
    model.updateTopology(step=None, timeStep=0.0)
    with pytest.raises(TopologyError):
        model.reserveElementNumbers(1)


class _StubMeshDependent:
    """A consumer that records how many times it was refreshed, and with what."""

    _lastSeenTopologyVersion = 0

    def __init__(self, relevant=True):
        self.refreshes = []
        self._relevant = relevant

    def refresh(self, model, change):
        self.refreshes.append(change)
        return self._relevant

    refreshIfMeshChanged = MeshDependent.refreshIfMeshChanged


def test_consumers_refresh_once_on_the_net_change_of_all_rounds():
    """The point of pull: a consumer sees the settled model, not each intermediate round. Under the
    old push observer it would have been called once per mutation."""

    log = []
    model = _modelWithModifiers(
        amr=_StubModifier("amr", 2, log, reactsToOthers=True),
        facets=_StubModifier("facets", 2, log, reactsToOthers=True),
    )
    consumer = _StubMeshDependent()
    model.registerMeshDependent(consumer)

    model.updateTopology(step=None, timeStep=0.0)
    assert len(log) == 4, "four mutations across two rounds"
    assert consumer.refreshes == [], "consumers must not be refreshed during the topology update"

    assert model.refreshMeshDependents() is True
    assert len(consumer.refreshes) == 1, "one refresh, on the net change"


def test_refresh_reports_whether_any_consumer_changed_its_footprint():
    model = _modelWithModifiers(amr=_StubModifier("amr", 1, []))
    indifferent = _StubMeshDependent(relevant=False)
    model.registerMeshDependent(indifferent)
    model.updateTopology(step=None, timeStep=0.0)
    assert model.refreshMeshDependents() is False
    assert len(indifferent.refreshes) == 1


def test_registration_is_idempotent_and_a_quiet_model_refreshes_nobody():
    model = FEModel(3)
    consumer = _StubMeshDependent()
    model.registerMeshDependent(consumer)
    model.registerMeshDependent(consumer)
    assert len(model.meshDependents) == 1
    assert model.refreshMeshDependents() is False
    assert consumer.refreshes == []
