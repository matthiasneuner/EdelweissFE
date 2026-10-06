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
"""The building blocks of the domain decomposition that one process can test: the METIS binding,
the element partitioning, and the subdomain interface of a single process. What only several
processes can test -- the interface exchange and the independence of the result from the
decomposition -- is covered by the decks in ``testfiles/mpi``, run under ``mpirun``."""

import numpy as np
import pytest

from edelweissfe.domaindecomposition import mpienvironment


def _quadGrid(n: int):
    """The element-node connectivity of an n x n grid of quadrilaterals."""

    connectivity = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            connectivity.append([a, a + 1, a + n + 2, a + n + 1])
    connectivity = np.array(connectivity)
    return np.arange(0, connectivity.size + 1, 4), connectivity.ravel(), (n + 1) ** 2


def _metis():
    from edelweissfe.domaindecomposition import metis

    try:
        metis._metisLibrary()
    except ImportError as exception:
        pytest.skip(str(exception))
    return metis


def test_metis_balances_a_uniform_grid():
    metis = _metis()
    offsets, nodes, nNodes = _quadGrid(8)

    parts = metis.partitionMeshDual(offsets, nodes, nNodes, 4, np.ones(64, dtype=int), 2)

    assert sorted(np.bincount(parts, minlength=4)) == [16, 16, 16, 16]
    # A fixed seed: the same mesh is partitioned the same way every time.
    assert np.array_equal(parts, metis.partitionMeshDual(offsets, nodes, nNodes, 4, np.ones(64, dtype=int), 2))


def test_metis_balances_weight_not_count():
    metis = _metis()
    offsets, nodes, nNodes = _quadGrid(8)
    weights = np.ones(64, dtype=int)
    weights[:8] = 8  # the first row of elements is eight times as expensive

    parts = metis.partitionMeshDual(offsets, nodes, nNodes, 2, weights, 2)

    weightOfPart = np.bincount(parts, weights=weights, minlength=2)
    assert abs(weightOfPart[0] - weightOfPart[1]) <= 0.05 * weights.sum()
    assert np.bincount(parts, minlength=2).min() < 32


def test_metis_rejects_nonpositive_weights():
    metis = _metis()
    offsets, nodes, nNodes = _quadGrid(2)

    with pytest.raises(ValueError):
        metis.partitionMeshDual(offsets, nodes, nNodes, 2, np.array([1, 0, 1, 1]), 2)


def test_no_launcher_means_no_mpi(monkeypatch):
    for variable in mpienvironment._LAUNCHER_VARIABLES + ("EDELWEISSFE_MPI",):
        monkeypatch.delenv(variable, raising=False)
    mpienvironment.worldCommunicator.cache_clear()
    try:
        assert mpienvironment.worldCommunicator() is None
        assert mpienvironment.numberOfProcesses() == 1
        assert mpienvironment.isRootProcess()
    finally:
        mpienvironment.worldCommunicator.cache_clear()


def test_single_subdomain_owns_and_integrates_everything():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.subdomaininterface import (
        InterfaceForceAssembly,
        SubdomainInterface,
    )

    # Degrees of freedom 3 and 7 are touched by nothing, and still integrated -- by rank 0.
    interface = SubdomainInterface(MPI.COMM_SELF, np.array([5, 0, 1, 2, 4, 6, 8, 9, 1]), 10)

    assert np.array_equal(interface.subdomainDofs, np.arange(10))
    assert interface.ownedDofMask.all()
    assert interface.neighbours == [] and interface.nInterfaceDofs == 0

    vector = np.arange(10.0)
    interface.gatherFromOwners(vector)
    assert np.array_equal(vector, np.arange(10.0))

    assembly = InterfaceForceAssembly(interface, np.array([0, 1, 1, 2]), np.array([0, 0, 1, 1]))
    assembly.assemble(np.ones(4), vector)
    assert np.array_equal(vector, np.arange(10.0))


class _SpringElement:
    """An element just complex enough for the explicit element loop: a force that depends on its
    solution, and an internal energy."""

    hasKernels = True

    def __init__(self, stiffness: float, nDof: int):
        self.stiffness = stiffness
        self.nDof = nDof

    def computeKernelsExplicit(self, Pe, Ue, dUe, time, dT):
        Pe += self.stiffness * (Ue - Ue.mean()) + np.sin(np.arange(Pe.shape[0]) + self.stiffness)

    def computeInternalEnergy(self):
        return self.stiffness / 3.0

    def computeLumpedInertia(self, Me):
        Me += self.stiffness / 7.0


def _springModel(nElements: int = 2000, nDof: int = 3001):
    """Overlapping elements, each on a few consecutive DOFs, in a scrambled element order."""

    from edelweissfe.numerics.dofvector import DofVector

    rng = np.random.default_rng(3)
    elements, indices = {}, {}
    for number in rng.permutation(nElements):
        element = _SpringElement(float(rng.uniform(0.5, 2.0)), 4)
        first = int(rng.integers(0, nDof - 4))
        elements[int(number)] = element
        indices[element] = np.arange(first, first + 4)
    U = DofVector(nDof, indices)
    U[:] = rng.standard_normal(nDof)
    return elements, indices, U


def test_element_loop_sums_in_element_order_on_any_number_of_threads():
    from edelweissfe.numerics.dofvector import DofVector
    from edelweissfe.solvers.base.parallelelementcomputation import (
        computeElementsForExplicit,
        computeLumpedDiagonalForExplicit,
        planElements,
    )
    from edelweissfe.timesteppers.timestep import TimeStep

    elements, indices, U = _springModel()
    nDof = U.shape[0]
    timeStep = TimeStep(1, 0.1, 0.1, 1e-3, 0.1, 0.1)

    # what a loop without chunks forms: element after element, P[el] += Pe
    expected = np.zeros(nDof)
    expectedEnergy = 0.0
    for element in elements.values():
        Pe = np.zeros(element.nDof)
        element.computeKernelsExplicit(Pe, U[element], U[element], 0.1, 1e-3)
        expected[indices[element]] += Pe
        expectedEnergy += element.computeInternalEnergy()

    for nThreads in (1, 4):
        plan = planElements(elements, indices, slice(None), nDof, nThreads)
        P = DofVector(nDof, indices)
        psi, contributions = computeElementsForExplicit(plan, U, U, P, timeStep)
        assert np.array_equal(np.asarray(P), expected)
        assert contributions.shape[0] == 4 * len(elements)
        if nThreads == 1:
            assert psi == expectedEnergy

    # into a subset of the DOFs, the same bits there and nothing elsewhere
    dofs = np.unique(np.concatenate(list(indices.values())))
    plan = planElements(elements, indices, dofs, nDof, 1)
    P = DofVector(nDof, indices)
    computeElementsForExplicit(plan, U, U, P, timeStep)
    assert np.array_equal(np.asarray(P), expected)

    M = DofVector(nDof, indices)
    computeLumpedDiagonalForExplicit(plan, lambda element, Me: element.computeLumpedInertia(Me), M)
    expectedM = np.zeros(nDof)
    for element in elements.values():
        expectedM[indices[element]] += element.stiffness / 7.0
    assert np.array_equal(np.asarray(M), expectedM)

    with pytest.raises(ValueError):
        planElements(elements, indices, dofs[1:], nDof, 1)


def test_the_whole_model_is_a_trivial_partition():
    from types import SimpleNamespace

    from edelweissfe.solvers.base.modelpartition import ModelPartition

    model = SimpleNamespace(elements={1: "element"}, constraints={"c": "constraint"})
    partition = ModelPartition.wholeModel(model, 5)
    vector = np.arange(5.0)

    assert partition.elements is model.elements and partition.constraints is model.constraints
    # every degree of freedom, as a view: no copy, and writing it back writes it onto itself
    assert np.shares_memory(vector[partition.dofs], vector) and vector[partition.dofs].shape == vector.shape
    assert partition.ownedDofMask.all() and partition.ownedDofMask.shape == (5,)


def test_a_single_subdomain_agrees_with_itself():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.subdomain import Subdomain
    from edelweissfe.utils.exceptions import ConditionalStop, CutbackRequest, StepFailed

    subdomain = Subdomain(MPI.COMM_SELF, journal=None, identification="test", loadBalanceTolerance=0.1)

    assert subdomain.sumAcrossParts([1.5, -2.0]) == [1.5, -2.0]
    assert subdomain.minAcrossParts(0.25) == 0.25
    assert subdomain.anyPart(True) and not subdomain.anyPart(False)
    subdomain.requireSameOnAllParts((True, False), "a verdict")

    with subdomain.agreedOnByAllParts("testing"):
        pass
    for raised, agreed in ((CutbackRequest("x", 0.25), CutbackRequest), (ConditionalStop(), ConditionalStop)):
        with pytest.raises(agreed):
            with subdomain.agreedOnByAllParts("testing"):
                raise raised
    # a cutback is agreed on with the size requested, not a fixed one
    with pytest.raises(CutbackRequest) as caught:
        with subdomain.agreedOnByAllParts("testing"):
            raise CutbackRequest("x", 0.25)
    assert caught.value.cutbackSize == 0.25
    with pytest.raises(StepFailed, match="testing failed"):
        with subdomain.agreedOnByAllParts("testing"):
            raise KeyError("a failure")


_DISTRIBUTION_DECK = """
*job, name=distributionjob, domain=2d
*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.15
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=8
y0=0, h=2
elType=CPE4
elProvider=edelweiss
nX=8
nY=2
*section, name=section1, thickness=1.0, material=linearelastic, type=plane
all
*solver, solver=NEDMPI, name=theSolver
*step, solver=theSolver
>>bodyForce, name=gravity, elSet=gen_top, forceVector='0.0, -1.0'
"""


class _SecondOfTwoProcesses:
    """What the partition asks of a communicator, as rank 1 of 2 sees it once rank 0 broadcast a
    partition: here, elements 1-8 to rank 0 and 9-16 to rank 1."""

    def Get_size(self):
        return 2

    def Get_rank(self):
        return 1

    def Bcast(self, parts, root):
        parts[:] = np.arange(parts.shape[0]) >= parts.shape[0] // 2


def test_the_rule_for_the_whole_model_names_its_reasons(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import (
        reasonsForTheWholeModel,
    )
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(_DISTRIBUTION_DECK)
    assert reasonsForTheWholeModel(parseInputFile(str(deck))) == []

    deck.write_text(
        _DISTRIBUTION_DECK
        + """
*modelModifier, type=hAdaptivity, name=amr
>>marker, type=elementSet, elSet=gen_all
*modelModifier, type=surfaceSnap, name=snap
*modelGenerator, generator=executePythonCode, name=code
print("hello")
*modelGenerator, generator=boxGen, name=late, executeAfterManualGeneration=True
nX=1
*modelGenerator, generator=surfaceElementGenerator, name=facets, executeAfterManualGeneration=True
surface=gen_top
name=top
*constraint, type=tie, name=tie
slaveSurface=top_facets, masterSurface=top_facets
*constraint, type=rigidBody, name=rigid
nSet=gen_left, referencePoint=gen_leftBottom
*fieldOutput
>>fromExpression, name=fromElements, elSet=gen_all, expression='np.zeros(len(model.elementSets["gen_all"]))'
"""
    )
    reasons = reasonsForTheWholeModel(parseInputFile(str(deck)))
    # adaptive refinement reads the mesh only and runs distributed, and so do a tie and contact facets
    # made by a late generator; the surface snap and an unverified constraint do not
    assert len(reasons) == 5
    assert "model modifier snap (surfaceSnap) changes the mesh during the run" in reasons[0]
    assert not any(name in reason for name in ("amr", "constraint tie ", "generator facets ") for reason in reasons)
    assert "constraint rigid (rigidBody) is not yet known to read only what every process holds" in reasons[1]
    assert "generator code (executePythonCode)" in reasons[2]
    assert "generator late (boxGen) runs after the mesh is partitioned" in reasons[3]
    assert "expression field output fromElements reads the elements of element set gen_all" in reasons[4]


def test_a_process_creates_its_elements_and_the_loaded_ones_touching_them(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import (
        DistributedElements,
        _stepActionDefinitions,
    )
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(_DISTRIBUTION_DECK)
    inputFile = parseInputFile(str(deck))
    model = FEModel(2)
    distribution = DistributedElements(_SecondOfTwoProcesses(), _stepActionDefinitions(inputFile))
    model.elementDistribution = distribution
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))

    own = {number for number, owner in distribution.owners.items() if owner == 1}
    assert own == set(range(9, 17))
    # The planeRectQuad grid is numbered column by column (two elements each): of the loaded top
    # row (the even numbers), element 8 shares nodes with element 9, which this process computes.
    assert set(model.elements) == own | {8}
    assert [element.elNumber for element in distribution.elementsReportedHere(model.elements.values())] == sorted(own)
    assert not model.elementSets["gen_top"].isComplete


def test_contact_facets_are_made_everywhere_and_computed_beside_their_solid_element(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import (
        DistributedElements,
        _stepActionDefinitions,
        reasonsForTheWholeModel,
    )
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(
        _DISTRIBUTION_DECK
        + """
*modelGenerator, generator=surfaceElementGenerator, name=facets, executeAfterManualGeneration=True
surface=gen_top
name=top
"""
    )
    inputFile = parseInputFile(str(deck))
    assert reasonsForTheWholeModel(inputFile) == []

    model = FEModel(2)
    distribution = DistributedElements(_SecondOfTwoProcesses(), _stepActionDefinitions(inputFile))
    model.elementDistribution = distribution
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))

    facets = model.wholeElementSet("top_facets", "this test")
    assert len(facets) == 8
    for facet in facets:
        besideElement = model.mesh.elements[facet.elNumber].besideElement
        assert distribution.owners[facet.elNumber] == distribution.owners[besideElement]
    # The facets on the top row (the even numbers) of elements 9-16 are computed here.
    reported = [element.elNumber for element in distribution.elementsReportedHere(model.elements.values())]
    assert [
        model.mesh.elements[number].besideElement
        for number in reported
        if number in model.mesh.elementSets["top_facets"]
    ] == [10, 12, 14, 16]


def test_a_node_field_output_over_an_element_set_held_nowhere_here_reads_the_whole_set(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import (
        DistributedElements,
        _stepActionDefinitions,
    )
    from edelweissfe.helpers.inputfilehelpers import (
        createFieldOutputFromInputFile,
        fillFEModelFromInputFile,
    )
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(
        _DISTRIBUTION_DECK
        + """
*fieldOutput
>>perNode, name=uLeft, elSet=gen_left, field=displacement, result=U
"""
    )
    inputFile = parseInputFile(str(deck))
    model = FEModel(2)
    model.elementDistribution = DistributedElements(_SecondOfTwoProcesses(), _stepActionDefinitions(inputFile))
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")

    # The elements of the left edge, 1 and 2, are computed by the other process: none is held here.
    assert len(model.elementSets["gen_left"]) == 0
    fieldOutput = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs["uLeft"]
    # The nodes of the two elements, not every node of the model.
    assert fieldOutput.associatedSet is model.elementSets["gen_left"]


def test_a_repartition_keeps_elements_where_they_were():
    from edelweissfe.domaindecomposition.partitioning import keepElementsWhereTheyWere

    previous = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    # METIS numbered the same parts differently, and moved element 4 from the second to the third
    repartitioned = {1: 2, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1}

    assert keepElementsWhereTheyWere(repartitioned, previous, 3) == {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}


class _SecondOfTwoProcessesExchanging(_SecondOfTwoProcesses):
    """Rank 1 of 2, as in :class:`_SecondOfTwoProcesses`, exchanging element states with rank 0: what
    rank 0 sends is the state given."""

    def __init__(self, receivedFromRankZero: dict):
        self.receivedFromRankZero = receivedFromRankZero
        self.sentToRankZero = None

    def alltoall(self, outgoing):
        self.sentToRankZero = outgoing[0]
        return [self.receivedFromRankZero, outgoing[1]]


def test_an_element_moves_with_its_state_and_its_section(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import (
        DistributedElements,
        _stepActionDefinitions,
    )
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(_DISTRIBUTION_DECK)
    inputFile = parseInputFile(str(deck))
    model = FEModel(2)
    communicator = _SecondOfTwoProcessesExchanging({})
    distribution = DistributedElements(communicator, _stepActionDefinitions(inputFile))
    model.elementDistribution = distribution
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    elementSet, element9 = model.elementSets["all"], model.elements[9]

    # Elements 7 and 8 come to this process, 9 and 10 leave it.
    stateOf7 = np.arange(model.elements[11].getStateVars().shape[0], dtype=float) + 7.0
    communicator.receivedFromRankZero = {7: stateOf7, 8: model.elements[12].getStateVars()}
    owners = dict(distribution.owners)
    owners.update({7: 1, 8: 1, 9: 0, 10: 0})
    created, dropped, received = distribution.moveElementsTo(model, owners)

    assert set(communicator.sentToRankZero) == {9, 10}
    # The grid is numbered column by column, the top row (even numbers) loaded: element 8 was held
    # before, for its load; element 6 is now held for its load, and element 10 is kept for it.
    assert (created, dropped, received) == (2, 1, 2)
    assert list(model.elements) == [6, 7, 8] + list(range(10, 17))
    assert model.elements[10] is not element9 and 9 not in model.elements
    # The sets are those references point to, updated in place.
    assert model.elementSets["all"] is elementSet and [element.elNumber for element in elementSet] == list(
        model.elements
    )
    assert model.elements[7].hasMaterial
    assert np.array_equal(model.elements[7].getStateVars(), stateOf7)
    assert distribution.ownershipVersion == 1
    assert [element.elNumber for element in distribution.elementsReportedHere(model.elements.values())] == [
        7,
        8,
    ] + list(range(11, 17))


def test_a_random_thickness_is_the_same_wherever_and_whenever_an_element_is_created(tmp_path):
    pytest.importorskip("gstools")
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(
        _DISTRIBUTION_DECK.replace(
            "*section, name=section1, thickness=1.0, material=linearelastic, type=plane",
            "*section, name=section1, thickness=1.0, material=linearelastic, type=planeRandomThickness, "
            "variance=0.1, lengthScale=2.0, seed=7",
        )
    )
    model = fillFEModelFromInputFile(FEModel(2), parseInputFile(str(deck)), Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    thicknesses = {number: element._t for number, element in model.elements.items()}
    assert len(set(thicknesses.values())) == len(thicknesses)

    # Dropped and created again, as when it moves to another process, alone and in another order.
    with model.topology.changes():
        for number in (12, 3):
            model.dropElementOfMesh(number)
        model.resolveSetsAndSurfacesOfMesh()
        created = {number: model.createElementOfMesh(number) for number in (12, 3)}
    model.putElementsInMeshOrder()
    model.resolveSetsAndSurfacesOfMesh()
    model.assignSectionsAndPropertiesToElements(created)

    assert {number: element._t for number, element in model.elements.items()} == thicknesses
