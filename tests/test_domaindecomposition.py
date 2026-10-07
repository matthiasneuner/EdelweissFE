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
decomposition -- is covered by the decks in ``testfiles/mpi``, run under ``mpirun``; a failure in one
process, which must not leave the others waiting, by a test starting ``mpirun`` itself."""

import os
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

import edelweissfe
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


def _runUnderMPI(directory, nProcesses: int = 3, timeout: float = 180.0) -> tuple[str | None, int | None]:
    """Run ``run.py`` in a directory on ``nProcesses`` processes under the MPI launcher, with this
    EdelweissFE; skip the test without ``mpi4py`` or a launcher.

    Returns
    -------
    tuple[str | None, int | None]
        The output of every process, stdout and stderr together, and the exit code of the launcher;
        both None if the run did not end within ``timeout`` seconds -- the launcher and every process
        it started are killed then, which would otherwise wait on.
    """

    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")

    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    launched = subprocess.Popen(
        [mpirun, "--bind-to", "none", "-n", str(nProcesses), sys.executable, "run.py"],
        cwd=directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = launched.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(launched.pid, signal.SIGKILL)
        launched.communicate()
        return None, None
    return output, launched.returncode


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
    interface.allgatherOwnedValues(vector)
    assert np.array_equal(vector, np.arange(10.0))

    assembly = InterfaceForceAssembly(interface, np.array([0, 1, 1, 2]), np.array([0, 0, 1, 1]))
    assembly.assemble(np.ones(4), vector)
    assert np.array_equal(vector, np.arange(10.0))


def test_loads_are_added_one_after_another_in_the_order_of_the_loads():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.subdomaininterface import (
        InterfaceLoadAssembly,
        SubdomainInterface,
    )

    rng = np.random.default_rng(7)
    nDof, nEntries = 50, 400
    interface = SubdomainInterface(MPI.COMM_SELF, np.arange(nDof), nDof)
    entryDofs = rng.integers(0, nDof, nEntries)
    # Every entry its own place in the order of the loads, given in scrambled order; values of very
    # different magnitudes and signed zeros, so that any other order of additions, or a sum started
    # from zero rather than from the vector, gives different bits.
    entryOrder = rng.permutation(nEntries)
    contributions = rng.standard_normal(nEntries) * 10.0 ** rng.integers(-12, 12, nEntries)
    contributions[::17] = -0.0
    initial = rng.standard_normal(nDof) * 1e8
    initial[::5] = -0.0

    expected = initial.copy()
    for entry in np.argsort(entryOrder):
        expected[entryDofs[entry]] += contributions[entry]

    vector = initial.copy()
    InterfaceLoadAssembly(interface, entryDofs, entryOrder).assemble(contributions, vector)

    assert np.array_equal(vector.view(np.int64), expected.view(np.int64))


def _adversarialContributions(rng, n: int) -> np.ndarray:
    """Values whose sum depends on the order it is formed in: magnitudes from 1e-12 to 1e12, and
    signed zeros."""

    values = rng.standard_normal(n) * 10.0 ** rng.integers(-12, 12, n)
    values[::13] = -0.0
    return values


def test_assembly_sums_every_degree_of_freedom_from_left_to_right_in_element_order():
    # The element loop assembles with np.bincount, which adds the weights into each bin one after
    # another, in the order they are given -- a left fold, the order a loop `P[dofs] += Pe` over the
    # elements forms. Nothing else makes a result independent of the number of threads, chunks and
    # processes, so a NumPy that summed a bin in any other order (pairwise, say) must fail here.
    # NumPy does not document that order: it is how its implementation works, verified here with
    # NumPy 2.5.2 (2026-10-07). After upgrading NumPy, this test -- and the interface test below,
    # over three processes -- is what says whether the bit-identity of a decomposed run still holds.
    from edelweissfe.numerics.dofvector import DofVector
    from edelweissfe.solvers.base.parallelelementcomputation import (
        computeLumpedDiagonalForExplicit,
        planElements,
    )

    rng = np.random.default_rng(11)
    nDof, nElements = 7, 400
    elements, indices, values = {}, {}, {}
    for number in range(nElements):
        element = _SpringElement(1.0, 3)
        elements[number] = element
        indices[element] = rng.choice(nDof, 3, replace=False)
        values[element] = _adversarialContributions(rng, 3)

    expected = np.zeros(nDof)
    for element in elements.values():
        for dof, value in zip(indices[element], values[element]):
            expected[dof] += value

    for nThreads in (1, 4):
        plan = planElements(elements, indices, slice(None), nDof, nThreads)
        vector = DofVector(nDof, indices)
        vector[:] = 0.0
        computeLumpedDiagonalForExplicit(plan, lambda element, Ve: Ve.__iadd__(values[element]), vector)
        assert np.array_equal(np.asarray(vector).view(np.int64), expected.view(np.int64))

    # the test has teeth: the same values summed pairwise give other bits
    pairwise = np.zeros(nDof)
    for dof in range(nDof):
        pairwise[dof] = np.sum([v for e in elements.values() for d, v in zip(indices[e], values[e]) if d == dof])
    assert not np.array_equal(pairwise, expected)


#: Assembles adversarial element contributions over three processes, each holding a random third of
#: the elements, and checks every degree of freedom it integrates against the serial left fold.
_INTERFACE_ASSEMBLY_SCRIPT = """
import numpy as np
from mpi4py import MPI
from edelweissfe.domaindecomposition.subdomaininterface import InterfaceForceAssembly, SubdomainInterface

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
rng = np.random.default_rng(5)
nDof, nElements = 40, 600
elementDofs = [rng.choice(nDof, 4, replace=False) for _ in range(nElements)]
values = rng.standard_normal((nElements, 4)) * 10.0 ** rng.integers(-12, 12, (nElements, 4))
values[::7, 1] = -0.0
owner = rng.integers(0, communicator.Get_size(), nElements)

expected = np.zeros(nDof)
for dofs, contribution in zip(elementDofs, values):
    for dof, value in zip(dofs, contribution):
        expected[dof] += value

mine = np.flatnonzero(owner == rank)
entryDofs = np.concatenate([elementDofs[e] for e in mine])
entryPositions = np.repeat(mine, 4)
contributions = values[mine].ravel()

interface = SubdomainInterface(communicator, entryDofs, nDof)
# the own elements, in model order, assembled as the element loop assembles them
vector = np.zeros(nDof) + np.bincount(entryDofs, weights=contributions, minlength=nDof)
InterfaceForceAssembly(interface, entryDofs, entryPositions).assemble(contributions, vector)

integrated = interface.subdomainDofs
same = np.array_equal(vector[integrated].view(np.int64), expected[integrated].view(np.int64))
print("PROCESS", rank, "BITWISE" if same else "DIFFERENT", interface.nInterfaceDofs > 0, flush=True)
"""


def test_interface_forces_are_summed_in_element_order_on_every_process(tmp_path):
    (tmp_path / "run.py").write_text(_INTERFACE_ASSEMBLY_SCRIPT)
    output, _ = _runUnderMPI(tmp_path, timeout=120)
    if output is None:
        pytest.fail("the interface exchange did not complete")

    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert reports == ["PROCESS {:} BITWISE True".format(rank) for rank in range(3)], output


#: Adds adversarial constraint forces over three processes, each owning a random part of the
#: constraints, and checks the whole net force vector against the serial sum in model order.
_CONSTRAINT_EXCHANGE_SCRIPT = """
import numpy as np
from mpi4py import MPI
from edelweissfe.domaindecomposition.subdomaininterface import ConstraintForceExchange
from edelweissfe.numerics.assembly import addNodalForces

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
rng = np.random.default_rng(9)
nDof, nConstraints = 30, 80
names = ["constraint{:}".format(i) for i in rng.permutation(nConstraints)]
constraints = {name: name for name in names}
dofs = {name: rng.integers(0, nDof, int(rng.integers(1, 7))) for name in names}  # a DOF may repeat
forces = {name: rng.standard_normal(dofs[name].shape[0]) * 10.0 ** rng.integers(-12, 12, dofs[name].shape[0])
          for name in names}
for name in names[::5]:
    forces[name][0] = -0.0
initial = rng.standard_normal(nDof) * 1e6
initial[::4] = -0.0
owner = {name: int(rng.integers(0, communicator.Get_size())) for name in names}

expected = initial.copy()
for name in names:
    addNodalForces(expected, dofs[name], forces[name], len(np.unique(dofs[name])) != len(dofs[name]))

owned = {name: constraints[name] for name in names if owner[name] == rank}
exchange = ConstraintForceExchange(communicator, constraints, owned, {constraints[name]: dofs[name] for name in names})
vector = initial.copy()
exchange.addAllConstraintForces({name: forces[name] for name in owned}, vector)
same = np.array_equal(vector.view(np.int64), expected.view(np.int64))
print("PROCESS", rank, "BITWISE" if same else "DIFFERENT", len(owned) > 0, flush=True)
"""


def test_constraint_forces_are_added_in_model_order_on_every_process(tmp_path):
    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")

    (tmp_path / "run.py").write_text(_CONSTRAINT_EXCHANGE_SCRIPT)
    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    output = subprocess.run(
        [mpirun, "--bind-to", "none", "-n", "3", sys.executable, "run.py"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert reports == ["PROCESS {:} BITWISE True".format(rank) for rank in range(3)], output


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
    partition = ModelPartition.wholeModel(model)
    vector = np.arange(5.0)

    assert partition.elements is model.elements and partition.constraints is model.constraints
    # every degree of freedom, as a view: no copy, and writing it back writes it onto itself
    assert np.shares_memory(vector[partition.dofs], vector) and vector[partition.dofs].shape == vector.shape


def test_a_single_subdomain_agrees_with_itself():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.subdomain import Subdomain
    from edelweissfe.utils.exceptions import ConditionalStop, CutbackRequest, StepFailed

    subdomain = Subdomain(Communicator(MPI.COMM_SELF), journal=None, identification="test", loadBalanceTolerance=0.1)

    assert subdomain.allreduceSum([1.5, -2.0]) == [1.5, -2.0]
    assert subdomain.allreduceMin(0.25) == 0.25
    assert subdomain.allreduceAny(True) and not subdomain.allreduceAny(False)
    subdomain.requireSameOnAllRanks((True, False), "a verdict")

    with subdomain.allRanksFailTogether("testing"):
        pass
    for raised, agreed in ((CutbackRequest("x", 0.25), CutbackRequest), (ConditionalStop(), ConditionalStop)):
        with pytest.raises(agreed):
            with subdomain.allRanksFailTogether("testing"):
                raise raised
    # a cutback is agreed on with the size requested, not a fixed one
    with pytest.raises(CutbackRequest) as caught:
        with subdomain.allRanksFailTogether("testing"):
            raise CutbackRequest("x", 0.25)
    assert caught.value.cutbackSize == 0.25
    with pytest.raises(StepFailed, match="testing failed"):
        with subdomain.allRanksFailTogether("testing"):
            raise KeyError("a failure")


def test_no_process_communicates_where_the_processes_agree_on_failures():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.mpienvironment import StepFailedOnAllRanks
    from edelweissfe.domaindecomposition.subdomain import Subdomain

    communicator = Communicator(MPI.COMM_SELF)
    subdomain = Subdomain(communicator, journal=None, identification="test", loadBalanceTolerance=0.1)

    # A communication inside the context fails, on all ranks together, and names the context.
    with pytest.raises(StepFailedOnAllRanks, match="allgather was called while gathering inside"):
        with subdomain.allRanksFailTogether("Gathering inside"):
            communicator.allgather(1.0)
    with pytest.raises(StepFailedOnAllRanks, match="Isend was called while"):
        with subdomain.allRanksFailTogether("Exchanging inside"):
            communicator.Isend(np.zeros(1), dest=0, tag=1)
    # So does one nested deeper, and the communicator is usable again after either context.
    with pytest.raises(StepFailedOnAllRanks, match="bcast was called while the inner one"):
        with subdomain.allRanksFailTogether("The outer one"):
            with communicator.withoutCommunication("The inner one"):
                pass
            with communicator.withoutCommunication("The inner one"):
                communicator.bcast(1)
    assert communicator.allgather(2.0) == [2.0]
    assert subdomain.allreduceSum([1.5]) == [1.5]


_FAILING_LOAD_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
30000.0, 0.15, 1.0
*modelGenerator, generator=planeRectQuad, name=gen
elType=CPE4
elProvider=edelweiss
nX=6
nY=6
l=100
h=100
*section, name=section1, thickness=1.0, material=linearelastic, type=plane
all
*job, name=failingloadjob, domain=2d
*solver, solver=NEDMPI, name=theSolver
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=5, maxIter=25, stepLength=1
>>dirichlet, name=bottom, nSet=gen_bottom, field=displacement, 1=0.0, 2=0.0
>>bodyForce, name=gravity, elSet=all, forceVector='0.0, -0.01'
"""

#: Runs the deck in this directory with PATCH applied -- which makes something fail in process 1
#: only --, and prints, per process, the failures the simulation reported.
_FAILING_IN_PROCESS_ONE_SCRIPT = """
import contextlib, io
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
PATCH
journal = io.StringIO()
with contextlib.redirect_stdout(journal):
    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
failures = [line.strip("> <").split("feCore")[0].strip() for line in journal.getvalue().splitlines() if "failed" in line]
print("PROCESS", rank, failures, flush=True)
"""

#: A body force kernel failing in process 1.
_FAILING_LOAD_PATCH = """
from edelweissfe.elements.base.displacementelementbase import DisplacementElementBase

computeBodyForce = DisplacementElementBase.computeBodyForce


def failingInProcessOne(self, *args):
    if rank == 1:
        raise RuntimeError("load kernel failed")
    return computeBodyForce(self, *args)


DisplacementElementBase.computeBodyForce = failingInProcessOne
"""


def test_a_load_failing_in_one_process_fails_in_every_process(tmp_path):
    (tmp_path / "test.inp").write_text(_FAILING_LOAD_DECK)
    (tmp_path / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", _FAILING_LOAD_PATCH))
    # Without the agreement, process 1 leaves the evaluation alone: uncaught, its exception aborts the
    # job with no failure reported; caught, the others wait for it in the exchange forever (the timeout).
    output, _ = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("a load failing in one process left the others waiting")

    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    expected = "['Simulation failed: Evaluating the loads failed in process 1: RuntimeError: load kernel failed']"
    assert reports == ["PROCESS {:} {:}".format(rank, expected) for rank in range(3)], output


#: _FAILING_LOAD_DECK without the load, with an element field output: a distributed model, whose
#: element results are gathered from the processes on every output increment.
_FIELD_OUTPUT_DECK = _FAILING_LOAD_DECK.replace(
    ">>bodyForce, name=gravity, elSet=all, forceVector='0.0, -0.01'",
    ">>dirichlet, name=top, nSet=gen_top, field=displacement, 2=1e-3",
).replace(
    "*solver, solver=NEDMPI, name=theSolver",
    "*fieldOutput\n>>perElement, elSet=all, result=stress, name=S, quadraturePoint=0\n"
    "*solver, solver=NEDMPI, name=theSolver\noutput-frequency=2",
)

#: Reading the element results of an increment fails in process 1, before the results are gathered
#: (the first reading, at the start of the job, succeeds).
_FAILING_FIELD_OUTPUT_PATCH = """
from edelweissfe.utils.fieldoutput import ElementFieldOutput

rebuildCollector = ElementFieldOutput._rebuildCollectorIfSetChanged
readings = []


def failingInProcessOne(self):
    readings.append(self)
    if rank == 1 and len(readings) > 1:
        raise RuntimeError("reading the element results failed")
    return rebuildCollector(self)


ElementFieldOutput._rebuildCollectorIfSetChanged = failingInProcessOne
"""


def test_a_field_output_failing_in_one_process_before_the_gather_fails_in_every_process(tmp_path):
    (tmp_path / "test.inp").write_text(_FIELD_OUTPUT_DECK)
    (tmp_path / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", _FAILING_FIELD_OUTPUT_PATCH))
    # Read inside the gather, the failure would leave process 1 agreeing on it while the others wait
    # for it in the gather: a hang (the timeout), or two different collective operations matched.
    output, _ = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("a field output failing in one process left the others waiting")

    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    expected = (
        "['Simulation failed: Reading the output failed in process 1: RuntimeError: reading the element "
        "results failed']"
    )
    assert reports == ["PROCESS {:} {:}".format(rank, expected) for rank in range(3)], output


#: A step failing in process 1 alone, outside any step the processes agree on: after the first
#: output increment, the others go on into the next increment's exchange.
_STEP_FAILING_ALONE_PATCH = """
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.exceptions import StepFailed

writeIncrementOutput = NEDMPI.writeIncrementOutput


def failingInProcessOne(self, *args):
    writeIncrementOutput(self, *args)
    if rank == 1:
        raise StepFailed("the step failed in process 1 alone")


NEDMPI.writeIncrementOutput = failingInProcessOne
"""


def test_a_step_failing_in_one_process_alone_stops_every_process(tmp_path):
    (tmp_path / "test.inp").write_text(_FIELD_OUTPUT_DECK)
    (tmp_path / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", _STEP_FAILING_ALONE_PATCH))
    # Finished like any failed step, process 1 would end its job while the others wait for it in the
    # next exchange forever (the timeout).
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("a step failing in one process alone left the others waiting")

    assert exitCode != 0, output
    assert "a step failed without the agreement of the other processes in MPI process 1 of 3" in output, output


#: _FIELD_OUTPUT_DECK repartitioned with the element numbers as costs on its output increments, so
#: that elements move between the processes.
_MIGRATING_DECK = _FIELD_OUTPUT_DECK.replace(
    "output-frequency=2", "output-frequency=2\nload-balance-costs=elementNumber"
)

#: The lumped inertia of an element depends on whether it was assembled before -- a state.
_STATE_DEPENDENT_INERTIA_PATCH = """
from edelweissfe.elements.base.displacementelementbase import DisplacementElementBase

computeLumpedInertia = DisplacementElementBase.computeLumpedInertia
assembledBefore = set()


def stateDependentInertia(self, M):
    computeLumpedInertia(self, M)
    if id(self) in assembledBefore:
        M *= 1.0 + 1e-12
    assembledBefore.add(id(self))


DisplacementElementBase.computeLumpedInertia = stateDependentInertia
"""


def test_elements_move_only_with_the_lumped_operators_a_serial_run_keeps(tmp_path):
    # Elements move, and the lumped operators assembled again afterwards are those of before.
    moving = tmp_path / "moving"
    moving.mkdir()
    (moving / "test.inp").write_text(_MIGRATING_DECK)
    (moving / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", ""))
    output, exitCode = _runUnderMPI(moving)
    assert exitCode == 0 and output.count("PROCESS") == 3 and "failed" not in output, output

    # An element whose lumped inertia depends on its state is refused at the migration, loudly.
    stateDependent = tmp_path / "stateDependent"
    stateDependent.mkdir()
    (stateDependent / "test.inp").write_text(_MIGRATING_DECK)
    (stateDependent / "run.py").write_text(
        _FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", _STATE_DEPENDENT_INERTIA_PATCH)
    )
    output, exitCode = _runUnderMPI(stateDependent)
    if output is None:
        pytest.fail("the refused migration left processes waiting")
    assert exitCode != 0, output
    assert "the lumped operators assembled again from the elements now held here differ" in output, output


_MATERIAL_PROPERTIES_DECK = """
*material, name=LinearElastic, id=linearelastic
20000.0, 0.2, 2.0e-9
*AnalyticalField, name=stiffness, type=scalarExpression
"f(x,y,z)" = "1.0 + 0.37*x + 0.11*y"
*section, name=section1, material=linearelastic, type=solid
gen_all
>>materialParameterFromField, index=0, field=stiffness, type=scale
>>writeMaterialPropertiesToFile, filename=matprops
*job, name=matpropsjob, domain=3d
*solver, solver=SOLVER, name=theSolver
*modelGenerator, generator=boxGen, name=gen
nX=9
nY=4
nZ=1
lX=90
lY=40
lZ=10
elType=C3D8
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=2, maxIter=25, stepLength=1e-6
>>dirichlet, name=fixLeft, nSet=gen_left, field=displacement, 1=0, 2=0, 3=0
"""

#: Runs the deck in this directory and prints how many elements this process created.
_RUN_DECK_SCRIPT = """
import contextlib, io
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

with contextlib.redirect_stdout(io.StringIO()):
    model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
print("CREATED", len(model.elements), "OF", len(model.mesh.elements), flush=True)
"""


def test_a_distributed_run_writes_the_material_properties_of_a_serial_one(tmp_path):
    pytest.importorskip("mpi4py.MPI")
    from edelweissfe.utils.misc import checkSuccessfulExtension

    if not checkSuccessfulExtension("edelweissfe.elements.marmotelement.element"):
        pytest.skip("the deck needs Marmot elements")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")

    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    files = {}
    for solver, launcher in (("NED", []), ("NEDMPI", [mpirun, "--bind-to", "none", "-n", "3"])):
        directory = tmp_path / solver
        directory.mkdir()
        (directory / "test.inp").write_text(_MATERIAL_PROPERTIES_DECK.replace("SOLVER", solver))
        (directory / "run.py").write_text(_RUN_DECK_SCRIPT)
        output = subprocess.run(
            launcher + [sys.executable, "run.py"],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        ).stdout
        if solver == "NEDMPI":
            # distributed: every process created only part of the 36 elements
            created = [int(line.split()[1]) for line in output.splitlines() if line.startswith("CREATED")]
            assert len(created) == 3 and max(created) < 36, output
        files[solver] = (directory / "matprops.csv").read_bytes()

    assert files["NED"].count(b"\n") == 36
    assert files["NEDMPI"] == files["NED"]


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
        reasonsToReplicateElements,
    )
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(_DISTRIBUTION_DECK)
    assert reasonsToReplicateElements(parseInputFile(str(deck))) == []

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
    reasons = reasonsToReplicateElements(parseInputFile(str(deck)))
    # adaptive refinement reads the mesh only and runs distributed, and so do a tie and contact facets
    # made by a late generator; the surface snap and an unverified constraint do not
    assert len(reasons) == 5
    assert "model modifier snap (surfaceSnap) changes the mesh during the run" in reasons[0]
    assert not any(name in reason for name in ("amr", "constraint tie ", "generator facets ") for reason in reasons)
    assert "constraint rigid (rigidBody) introduces scalar variables (Lagrange multipliers)" in reasons[1]
    assert "generator code (executePythonCode)" in reasons[2]
    assert "generator late (boxGen) runs after the mesh is partitioned" in reasons[3]
    assert "expression field output fromElements reads the elements of element set gen_all" in reasons[4]


def test_a_process_creates_only_its_own_elements_also_where_a_load_reaches_its_subdomain(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import DistributedElements
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(_DISTRIBUTION_DECK)
    inputFile = parseInputFile(str(deck))
    model = FEModel(2)
    distribution = DistributedElements(_SecondOfTwoProcesses())
    model.elementDistribution = distribution
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))

    own = {number for number, owner in distribution.owners.items() if owner == 1}
    assert own == set(range(9, 17))
    # The planeRectQuad grid is numbered column by column (two elements each): of the loaded top
    # row (the even numbers), element 8 shares nodes with element 9, which this process computes --
    # but its load is evaluated where it is computed, so it is not created here.
    assert set(model.elements) == own
    assert [element.elNumber for element in distribution.ownedElements(model.elements.values())] == sorted(own)
    assert not model.elementSets["gen_top"].isComplete


def test_contact_facets_are_made_everywhere_and_computed_with_their_host_element(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import (
        DistributedElements,
        reasonsToReplicateElements,
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
    assert reasonsToReplicateElements(inputFile) == []

    model = FEModel(2)
    distribution = DistributedElements(_SecondOfTwoProcesses())
    model.elementDistribution = distribution
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))

    facets = model.wholeElementSet("top_facets", "this test")
    assert len(facets) == 8
    for facet in facets:
        hostElement = model.mesh.elements[facet.elNumber].hostElement
        assert distribution.owners[facet.elNumber] == distribution.owners[hostElement]
    # The facets on the top row (the even numbers) of elements 9-16 are computed here.
    reported = [element.elNumber for element in distribution.ownedElements(model.elements.values())]
    assert [
        model.mesh.elements[number].hostElement for number in reported if number in model.mesh.elementSets["top_facets"]
    ] == [10, 12, 14, 16]


def test_a_node_field_output_over_an_element_set_held_nowhere_here_reads_the_whole_set(tmp_path):
    from edelweissfe.domaindecomposition.distributedelements import DistributedElements
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
    model.elementDistribution = DistributedElements(_SecondOfTwoProcesses())
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")

    # The elements of the left edge, 1 and 2, are computed by the other process: none is held here.
    assert len(model.elementSets["gen_left"].localElements()) == 0
    fieldOutput = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs["uLeft"]
    # The nodes of the two elements, read from the mesh -- not every node of the model, and not the
    # nodes of the element objects held here, of which there are none.
    assert fieldOutput.associatedSet is model.elementSets["gen_left"]
    nodesOfTheSet = {
        label for number in model.mesh.elementSets["gen_left"] for label in model.mesh.elements[number].nodeLabels
    }
    assert len(nodesOfTheSet) == 6
    fieldOutput.updateResults(model)
    assert fieldOutput.getLastResult().shape == (len(nodesOfTheSet), 2)


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
    from edelweissfe.domaindecomposition.distributedelements import DistributedElements
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "test.inp"
    deck.write_text(_DISTRIBUTION_DECK)
    inputFile = parseInputFile(str(deck))
    model = FEModel(2)
    communicator = _SecondOfTwoProcessesExchanging({})
    distribution = DistributedElements(communicator)
    model.elementDistribution = distribution
    model = fillFEModelFromInputFile(model, inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    elementSet = model.elementSets["all"]

    # Elements 7 and 8 come to this process, 9 and 10 leave it.
    stateOf7 = np.arange(model.elements[11].getStateVars().shape[0], dtype=float) + 7.0
    communicator.receivedFromRankZero = {7: stateOf7, 8: model.elements[12].getStateVars()}
    owners = dict(distribution.owners)
    owners.update({7: 1, 8: 1, 9: 0, 10: 0})
    created, dropped, received = distribution.moveElementsTo(model, owners)

    assert set(communicator.sentToRankZero) == {9, 10}
    assert (created, dropped, received) == (2, 2, 2)
    assert list(model.elements) == [7, 8] + list(range(11, 17))
    assert 9 not in model.elements and 10 not in model.elements
    # The sets are those references point to, updated in place.
    assert model.elementSets["all"] is elementSet and [
        element.elNumber for element in elementSet.localElements()
    ] == list(model.elements)
    assert model.elements[7].hasMaterial
    assert np.array_equal(model.elements[7].getStateVars(), stateOf7)
    assert distribution.ownershipVersion == 1
    assert [element.elNumber for element in distribution.ownedElements(model.elements.values())] == [
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
