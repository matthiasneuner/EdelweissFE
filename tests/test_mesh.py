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
"""The mesh as data: describing a mesh, making its elements, and what a process that makes only
some of them sees.

A model is described first (``model.mesh``) and its elements are made from the description
(``FEModel.createElementsOfMesh``). These tests check the description (element records, sets by
number, set unions, surfaces), the resolution of sets and surfaces to the created elements, and
the property domain decomposition relies on: a predicate that skips elements leaves the fields at
the nodes, and hence the layout of the degrees of freedom, unchanged -- while every whole-set reader
fails loudly on the partial sets.
"""

import numpy as np
import pytest

from edelweissfe.config.phenomena import domainMapping
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.elementdistribution import ElementDistribution
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.mesh import SurfaceFace
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.points.node import Node
from edelweissfe.utils.exceptions import TopologyError
from edelweissfe.utils.inputfileparser import parseInputFile

DECK = """
*job, name=meshjob, domain=2d

*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.15

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=4
y0=0, h=2
elType=CPE4
elProvider=edelweiss
nX=4
nY=2

*node
101, 5.0, 0.0
102, 6.0, 0.0
103, 6.0, 1.0
104, 5.0, 1.0
105, 7.0, 0.0
106, 7.0, 1.0

*element, type=CPE8, provider=edelweiss, elset=extraQuadratic
1001, 101, 102, 103, 104, 101, 102, 103, 104

*element, type=CPE4, provider=edelweiss, elset=extra
1002, 102, 105, 106, 103

*elset, elset=odd, generate
1, 7, 2

*elset, elset=picked
6, 1002, 2, 6

*elset, elset=union
odd, picked

*surface, name=right, type=element
gen_right, S2
extra, S2

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
all
"""


class _ElementsByPredicate(ElementDistribution):
    """A distribution creating the elements for which a predicate holds, as a process of a
    domain-decomposed run creates only some."""

    def __init__(self, predicate):
        self._predicate = predicate

    def isCreatedHere(self, number: int) -> bool:
        return self._predicate(number)


def _buildModel(tmp_path, isCreatedHere=lambda number: True) -> FEModel:
    """Build the model of :data:`DECK`, creating the elements for which ``isCreatedHere`` holds."""

    deck = tmp_path / "test.inp"
    deck.write_text(DECK)
    inputFile = parseInputFile(str(deck))
    journal = Journal(verbose=False)
    model = FEModel(domainMapping[inputFile["job"][0]["domain"]])
    model.elementDistribution = _ElementsByPredicate(isCreatedHere)
    model = fillFEModelFromInputFile(model, inputFile, journal)
    model.prepareYourself(journal)
    return model


def _dofLayout(model: FEModel) -> dict:
    """The layout of the degrees of freedom: the field order, the node order per field, the number
    of DOFs and the indices of every (field, node)."""

    dofManager = DofManager(
        list(model.nodeFields.values()), list(model.scalarVariables.values()), list(model.elements.values())
    )
    return {
        "fields": list(model.nodeFields.keys()),
        "nodes": {name: [node.label for node in nodeField.nodes] for name, nodeField in model.nodeFields.items()},
        "nDof": dofManager.nDof,
        "indices": {
            (name, node.label): list(dofManager.idcsOfFieldVariablesInDofVector[node.fields[name]])
            for name, nodeField in model.nodeFields.items()
            for node in nodeField.nodes
        },
    }


def test_mesh_describes_elements_sets_and_surfaces(tmp_path):
    model = _buildModel(tmp_path)
    mesh = model.mesh

    # the generator's elements first (numbers from the allocator), then the *element blocks
    assert list(mesh.elements) == [1, 2, 3, 4, 5, 6, 7, 8, 1001, 1002]
    assert mesh.elements[1002].elType == "CPE4" and mesh.elements[1002].provider == "edelweiss"
    assert mesh.elements[1002].nodeLabels == (102, 105, 106, 103)

    # sets by number, generated, deduplicated in first-occurrence order, and unions of sets in order
    assert mesh.elementSets["odd"] == [1, 3, 5, 7]
    assert mesh.elementSets["picked"] == [6, 1002, 2]
    assert mesh.elementSets["union"] == [1, 3, 5, 7, 6, 1002, 2]
    assert mesh.elementSets["all"] == list(mesh.elements)

    # a surface is described by the element sets of its faces
    assert mesh.surfaces["right"] == {2: SurfaceFace(elementSetName="extra")}
    assert mesh.elementNumbersOfSurface("gen_bottom") == {1: [1, 3, 5, 7]}


def test_elements_are_made_from_the_mesh_in_mesh_order(tmp_path):
    model = _buildModel(tmp_path)

    assert list(model.elements) == list(model.mesh.elements)
    for number, record in model.mesh.elements.items():
        element = model.elements[number]
        assert element.elType == record.elType
        assert tuple(node.label for node in element.nodes) == record.nodeLabels

    # every set and surface of the mesh resolves to the created elements, complete
    for name, numbers in model.mesh.elementSets.items():
        elementSet = model.elementSets[name]
        assert [element.elNumber for element in elementSet] == numbers
        assert elementSet.isComplete
    assert model.surfaces["right"][2] is model.elementSets["extra"]
    model.requireCompleteMesh("the test")


def test_fields_follow_from_the_element_type(tmp_path):
    model = _buildModel(tmp_path)

    # the nodes of the CPE8's repeated corner list and the CPE4s carry displacement; the fields come
    # from the type, asked once per (type, provider)
    for record in model.mesh.elements.values():
        for label, fields in zip(record.nodeLabels, model.mesh.typeOf(record).fields):
            assert set(fields) <= set(model.nodes[label].fields)
    assert set(model.mesh._types) == {("CPE4", "edelweiss"), ("CPE8", "edelweiss")}


def test_a_predicate_that_skips_elements_keeps_the_dof_layout(tmp_path):
    whole = _buildModel(tmp_path)
    created = {2, 4, 6, 8, 1001}
    part = _buildModel(tmp_path, lambda number: number in created)

    assert set(part.elements) == created
    assert list(part.elements) == [n for n in whole.elements if n in created]
    assert part.mesh.elementSets == whole.mesh.elementSets

    # the fields at the nodes -- and with them the DOF layout -- do not depend on what was created
    assert _dofLayout(part) == _dofLayout(whole)

    # a set holds the part created here and says whether that is all of it
    assert [e.elNumber for e in part.elementSets["picked"]] == [6, 2]
    assert not part.elementSets["picked"].isComplete
    assert [e.elNumber for e in part.elementSets["gen_top"]] == [2, 4, 6, 8]
    assert part.elementSets["gen_top"].isComplete
    assert [e.elNumber for e in part.surfaces["gen_bottom"][1]] == []

    # the nodes of a set are those of the whole set, from the mesh, whatever was created here
    for name in ("picked", "gen_top", "all"):
        assert [n.label for n in part.nodesOfElementSetOfMesh(name)] == [
            n.label for n in whole.elementSets[name].extractNodeSet()
        ]
    with pytest.raises(TopologyError, match="the nodes of the element set needs the whole element set picked"):
        part.elementSets["picked"].extractNodeSet()

    # every whole-set reader fails loudly on a partial set, and on a partial model
    with pytest.raises(TopologyError, match="the test needs the whole element set picked"):
        part.elementSets["picked"].requireComplete("the test")
    part.elementSets["gen_top"].requireComplete("the test")
    with pytest.raises(TopologyError, match="the test needs every element of the model"):
        part.requireCompleteMesh("the test")


def test_an_element_can_be_created_and_removed_at_any_time(tmp_path):
    """The shape of element migration: an element described in the mesh can be created in a process
    that did not create it, later; removing an element retires it from the mesh."""

    part = _buildModel(tmp_path, lambda number: number != 3)
    with part.topology.changes():
        element = part.createElementOfMesh(3)
        part.sections["section1"].assignSectionToElement(element, part)
    assert element.hasMaterial
    part.requireCompleteMesh("the test")

    with part.topology.changes():
        part.removeElement(3)
    assert 3 not in part.elements and 3 not in part.mesh.elements
    with pytest.raises(TopologyError, match="outside a topology change"):
        part.createElementOfMesh(5)


def test_an_element_made_by_its_owner_is_described_from_the_object():
    from edelweissfe.elements.pointmass import PointMass

    model = FEModel(3)
    node = Node(1, np.zeros(3))
    model.nodes[1] = node
    with model.topology.changes():
        (number,) = model.topology.reserveElementNumbers(1)
        pointMass = PointMass(number, [node], model, 2.0, [1.0, 1.0, 1.0])
        model.createElement(pointMass)

    record = model.mesh.elements[number]
    assert record.isMadeByOwner and record.nodeLabels == (1,)
    assert model.mesh.typeOf(record).fields == [["displacement", "rotation"]]
    with pytest.raises(TopologyError, match="made by its owner"):
        model.mesh.elementClassOf(record)

    model._activateNodeFieldsFromMesh()
    assert list(node.fields) == ["displacement", "rotation"]


def test_a_number_is_described_once():
    model = FEModel(2)
    model.mesh.addElement(1, "CPE4", "edelweiss", [1, 2, 3, 4])
    with pytest.raises(TopologyError, match="already taken"):
        model.mesh.addElement(1, "CPE4", "edelweiss", [1, 2, 3, 4])
    with pytest.raises(KeyError, match="not in the mesh"):
        model.mesh.setElementSet("broken", [1, 2])


def test_elements_described_by_a_late_generator_are_created(tmp_path):
    """A generator run after the *element keywords (executeAfterManualGeneration=True) describes
    elements after the elements were made; they must be made too, not silently left out. No section
    can refer to their sets, so preparing the model then fails loudly, as it always did."""

    deck = tmp_path / "test.inp"
    deck.write_text(
        DECK
        + """
*modelGenerator, generator=planeRectQuad, name=late, executeAfterManualGeneration=True
x0=10, l=3
y0=0, h=1
elType=CPE4
elProvider=edelweiss
nX=3
nY=1
"""
    )
    inputFile = parseInputFile(str(deck))
    journal = Journal(verbose=False)
    model = fillFEModelFromInputFile(FEModel(2), inputFile, journal)

    assert list(model.elements) == list(model.mesh.elements)
    assert [e.elNumber for e in model.elementSets["late_all"]] == [1003, 1004, 1005]
    # (a Marmot element reports "No material was assigned"; a Python element has no material attribute yet)
    with pytest.raises(Exception, match="No material was assigned|_hasMaterial"):
        model.prepareYourself(journal)


def test_an_element_described_but_never_created_is_refused(tmp_path):
    model = _buildModel(tmp_path)
    model.mesh.addElement(5000, "CPE4", "edelweiss", [1, 2, 3, 4])
    with pytest.raises(TopologyError, match="described but never created"):
        model.prepareYourself(Journal(verbose=False))


def test_completeness_is_derived_from_the_mesh_and_repeated_making_changes_nothing(tmp_path):
    part = _buildModel(tmp_path, lambda number: number != 1002)
    picked = part.elementSets["picked"]
    assert not picked.isComplete

    # the set follows its members: created later and resolved again, it is whole -- no stale flag
    with part.topology.changes():
        part.createElementOfMesh(1002)
        part.resolveElementSetOfMesh("picked")
    assert part.elementSets["picked"] is picked and picked.isComplete

    # making the mesh again neither replaces nor touches sets and surfaces that did not change
    part.elementDistribution = ElementDistribution()
    with part.topology.changes():
        part.createElementsOfMesh()
    assert all(elementSet.isComplete for elementSet in part.elementSets.values())
    versions = {name: s._version for name, s in part.surfaces.items()}
    setVersions = {name: s._version for name, s in part.elementSets.items()}
    with part.topology.changes():
        part.createElementsOfMesh()
    assert {name: s._version for name, s in part.surfaces.items()} == versions
    assert {name: s._version for name, s in part.elementSets.items()} == setVersions
