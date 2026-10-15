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

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from time import sleep

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


def test_metis_gives_each_part_its_share():
    # A process busy with something else -- evaluating a large contact -- is given fewer elements.
    metis = _metis()
    offsets, nodes, nNodes = _quadGrid(16)

    parts = metis.partitionMeshDual(
        offsets, nodes, nNodes, 4, np.ones(256, dtype=int), 2, np.array([0.1, 0.3, 0.3, 0.3])
    )

    counts = np.bincount(parts, minlength=4)
    assert abs(counts[0] - 25.6) <= 0.05 * 256 and all(abs(count - 76.8) <= 0.05 * 256 for count in counts[1:])
    with pytest.raises(ValueError):
        metis.partitionMeshDual(offsets, nodes, nNodes, 4, np.ones(256, dtype=int), 2, np.array([0.5, 0.5, 0.0, 0.0]))


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
        ElementProcessesAtDofs,
        InterfaceForceAssembly,
        SubdomainInterface,
    )

    # Degrees of freedom 3 and 7 are touched by nothing, and still integrated -- by rank 0.
    elements = ElementProcessesAtDofs.ofElementDofs(np.array([0, 1, 1, 2]), np.zeros(4, dtype=np.int64), 10, 1)
    interface = SubdomainInterface(MPI.COMM_SELF, elements, np.array([5, 0, 1, 2, 4, 6, 8, 9, 1]))

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
        ElementProcessesAtDofs,
        InterfaceLoadAssembly,
        SubdomainInterface,
    )

    rng = np.random.default_rng(7)
    nDof, nEntries = 50, 400
    noElements = ElementProcessesAtDofs.ofElementDofs(np.empty(0), np.empty(0), nDof, 1)
    interface = SubdomainInterface(MPI.COMM_SELF, noElements, np.arange(nDof))
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
    # The element loop assembles with sumAtPositions, which adds the contributions at each degree of
    # freedom one after another, in the order they are given -- a left fold, the order a loop
    # `P[dofs] += Pe` over the elements forms. Nothing else makes a result independent of the number
    # of threads, chunks and processes, so an assembly summing in any other order (pairwise, say) must
    # fail here.
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


def test_the_scatter_add_is_a_left_fold_the_same_bits_as_numpy_bincount():
    # sumAtPositions replaced np.bincount(positions, weights=values, minlength=n), which summed in the
    # same order -- as NumPy implements it, without documenting it. The same bits, on values whose sum
    # differs in any other order: values of very different magnitudes, signed zeros, and positions
    # with no value, one value, and many.
    from edelweissfe.numerics.scatteradd import sumAtPositions

    rng = np.random.default_rng(3)
    for n, nPositions in ((0, 5), (1, 1), (17, 3), (5000, 40), (20000, 7000)):
        positions = rng.integers(0, nPositions, n).astype(np.int64)
        values = _adversarialContributions(rng, n)
        values[rng.random(n) < 0.05] = 0.0
        sums = sumAtPositions(positions, values, nPositions)

        leftFold = np.zeros(nPositions)
        for position, value in zip(positions.tolist(), values.tolist()):
            leftFold[position] += value
        assert np.array_equal(sums.view(np.int64), leftFold.view(np.int64))
        assert np.array_equal(
            sums.view(np.int64), np.bincount(positions, weights=values, minlength=nPositions).view(np.int64)
        )

    # a negative zero alone stays one, as in a left fold from +0.0: +0.0 + -0.0 is +0.0
    assert np.signbit(sumAtPositions(np.zeros(1, dtype=np.int64), np.array([-0.0]), 1)[0]) == np.signbit(0.0 + -0.0)
    with pytest.raises(ValueError, match="outside"):
        sumAtPositions(np.array([0, 3], dtype=np.int64), np.ones(2), 3)
    with pytest.raises(ValueError, match="outside"):
        sumAtPositions(np.array([-1], dtype=np.int64), np.ones(1), 3)
    with pytest.raises(ValueError, match="values for"):
        sumAtPositions(np.array([0], dtype=np.int64), np.ones(2), 3)


#: Assembles adversarial element contributions over three processes, each holding a random third of
#: the elements, and checks every degree of freedom it integrates against the serial left fold.
_INTERFACE_ASSEMBLY_SCRIPT = """
import numpy as np
from mpi4py import MPI
from edelweissfe.domaindecomposition.subdomaininterface import (
    ElementProcessesAtDofs, InterfaceForceAssembly, SubdomainInterface
)
from edelweissfe.numerics.scatteradd import sumAtPositions

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

# Which processes integrate a degree of freedom, derived from the element partition every process
# knows, and multi-point constraints linking 0-1-2 and 30-31 -- against the processes touching them.
links = (np.array([1, 2, 31]), np.array([0, 1, 30]))
elements = ElementProcessesAtDofs.ofElementDofs(np.concatenate(elementDofs), np.repeat(owner, 4), nDof,
                                                communicator.Get_size())
interface = SubdomainInterface(communicator, elements, entryDofs, links)
groups = [{0, 1, 2}, {30, 31}]
integrating = [set() for _ in range(nDof)]
for dofs, process in zip(elementDofs, owner):
    for dof in dofs:
        integrating[dof].add(int(process))
for group in groups:
    union = set().union(*(integrating[dof] for dof in group))
    for dof in group:
        integrating[dof] = set(union)
integrating = [processes or {0} for processes in integrating]
derived = (
    np.array_equal(interface.dofOwners, [min(processes) for processes in integrating])
    and np.array_equal(interface.subdomainDofs, [dof for dof in range(nDof) if rank in integrating[dof]])
    and interface.neighbours == sorted(set().union(*(integrating[d] for d in interface.subdomainDofs)) - {rank})
    and all(
        np.array_equal(interface.sharedDofsWith(n), [d for d in interface.subdomainDofs if n in integrating[d]])
        for n in interface.neighbours
    )
)
# the own elements, in model order, assembled as the element loop assembles them
vector = np.zeros(nDof) + sumAtPositions(entryDofs.astype(np.int64), contributions, nDof)
InterfaceForceAssembly(interface, entryDofs, entryPositions).assemble(contributions, vector)

integrated = interface.subdomainDofs
same = np.array_equal(vector[integrated].view(np.int64), expected[integrated].view(np.int64))
print("PROCESS", rank, "BITWISE" if same else "DIFFERENT", interface.nInterfaceDofs > 0,
      "DERIVED" if derived else "MISDERIVED", flush=True)
"""


def test_interface_forces_are_summed_in_element_order_on_every_process(tmp_path):
    (tmp_path / "run.py").write_text(_INTERFACE_ASSEMBLY_SCRIPT)
    output, _ = _runUnderMPI(tmp_path, timeout=120)
    if output is None:
        pytest.fail("the interface exchange did not complete")

    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert reports == ["PROCESS {:} BITWISE True DERIVED".format(rank) for rank in range(3)], output


#: Adds adversarial constraint forces over three processes, each owning a random part of the
#: constraints and integrating their degrees of freedom and a few more of its elements, and checks the
#: net force vector against the serial sum in model order: bitwise at every degree of freedom integrated
#: here, and not written anywhere else -- the forces reach only the processes integrating them.
_CONSTRAINT_EXCHANGE_SCRIPT = """
import numpy as np
from mpi4py import MPI
from edelweissfe.domaindecomposition.subdomaininterface import (
    ConstraintForceExchange, ElementProcessesAtDofs, SubdomainInterface
)
from edelweissfe.numerics.assembly import addNodalForces

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
rng = np.random.default_rng(9)
nDof, nConstraints = 60, 80
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
elementDofs = [rng.choice(nDof, 5, replace=False) for _ in range(communicator.Get_size())]

expected = initial.copy()
for name in names:
    addNodalForces(expected, dofs[name], forces[name], len(np.unique(dofs[name])) != len(dofs[name]))

owned = {name: constraints[name] for name in names if owner[name] == rank}
touched = np.concatenate([elementDofs[rank]] + [dofs[name] for name in owned])
size = communicator.Get_size()
elements = ElementProcessesAtDofs.ofElementDofs(np.concatenate(elementDofs), np.repeat(np.arange(size), 5), nDof, size)
interface = SubdomainInterface(communicator, elements, touched)
exchange = ConstraintForceExchange(interface, constraints, owned, {constraints[name]: dofs[name] for name in names})
vector = initial.copy()
exchange.addAllConstraintForces({name: forces[name] for name in owned}, vector)

integrated = np.zeros(nDof, dtype=bool)
integrated[interface.subdomainDofs] = True
same = np.array_equal(vector[integrated].view(np.int64), expected[integrated].view(np.int64))
untouched = np.array_equal(vector[~integrated].view(np.int64), initial[~integrated].view(np.int64))
print("PROCESS", rank, "BITWISE" if same else "DIFFERENT", "UNTOUCHED" if untouched else "WRITTEN",
      len(owned) > 0 and interface.nInterfaceDofs > 0 and not integrated.all(), flush=True)
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
    assert reports == ["PROCESS {:} BITWISE UNTOUCHED True".format(rank) for rank in range(3)], output


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


class _DescheduledOnceElement(_SpringElement):
    """A spring element whose process is descheduled once while its kernel is timed: its wall time
    in that increment is a stall, not its cost."""

    stalls = 1

    def computeKernelsExplicit(self, Pe, Ue, dUe, time, dT):
        super().computeKernelsExplicit(Pe, Ue, dUe, time, dT)
        if self.stalls:
            self.stalls -= 1
            sleep(0.02)


def test_a_descheduled_process_does_not_make_an_element_weigh_more_than_its_peers():
    # The load balancer weighs every element by its measured kernel time, a wall time: an element
    # whose process is descheduled while it is timed measures the whole stall -- thousands of times
    # its cost on a loaded machine --, and METIS then gives its process a part of one element. The
    # least time over the increments is what the element costs; a stall cannot inflate it.
    from edelweissfe.solvers.base.parallelelementcomputation import (
        computeElementsForExplicit,
        planElements,
    )
    from edelweissfe.timesteppers.timestep import TimeStep

    elements, indices, U = _springModel(nElements=200, nDof=301)
    stalled = next(iter(elements))
    replaced = elements[stalled]
    elements[stalled] = _DescheduledOnceElement(replaced.stiffness, replaced.nDof)
    indices[elements[stalled]] = indices.pop(replaced)
    timeStep = TimeStep(1, 0.1, 0.1, 1e-3, 0.1, 0.1)

    for nThreads in (1, 4):
        elements[stalled].stalls = 1
        plan = planElements(elements, indices, slice(None), U.shape[0], nThreads)
        costs = np.full(len(elements), np.inf)
        for _ in range(5):
            P = U.copy()
            computeElementsForExplicit(plan, U, U, P, timeStep, costs)

        assert np.all(np.isfinite(costs)) and np.all(costs > 0.0)
        position = list(plan.elements).index(stalled)
        # Its peers cost microseconds; the stall was 20 ms.
        assert costs[position] < 1e-3, costs[position]
        assert costs[position] < 20.0 * np.median(costs), (costs[position], np.median(costs))


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


def test_the_whole_model_is_a_trivial_subdomain():
    from types import SimpleNamespace

    from edelweissfe.solvers.base.subdomain import Subdomain

    model = SimpleNamespace(elements={1: "element"}, constraints={"c": "constraint"})
    subdomain = Subdomain.wholeModel(model)
    vector = np.arange(5.0)

    assert subdomain.elements is model.elements and subdomain.constraints is model.constraints
    # every degree of freedom, as a view: no copy, and writing it back writes it onto itself
    assert np.shares_memory(vector[subdomain.dofs], vector) and vector[subdomain.dofs].shape == vector.shape


def test_a_single_process_agrees_with_itself():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.mpienvironment import (
        ErrorOnAllRanks,
        StepFailedOnAllRanks,
    )
    from edelweissfe.utils.exceptions import ConditionalStop, CutbackRequest, StepFailed

    communicator = Communicator(MPI.COMM_SELF)

    assert communicator.allreduceSum([1.5, -2.0]) == [1.5, -2.0]
    assert communicator.allreduceMin(0.25) == 0.25
    assert communicator.allreduceAny(True) and not communicator.allreduceAny(False)
    communicator.requireSameOnAllRanks((True, False), "a verdict")

    with communicator.allRanksFailTogether("testing"):
        pass
    for raised, agreed in ((CutbackRequest("x", 0.25), CutbackRequest), (ConditionalStop(), ConditionalStop)):
        with pytest.raises(agreed):
            with communicator.allRanksFailTogether("testing"):
                raise raised
    # a cutback is agreed on with the size requested, not a fixed one
    with pytest.raises(CutbackRequest) as caught:
        with communicator.allRanksFailTogether("testing"):
            raise CutbackRequest("x", 0.25)
    assert caught.value.cutbackSize == 0.25
    # a step failure is agreed on as one; any other exception as an error, which is no step failure
    with pytest.raises(StepFailedOnAllRanks, match="testing failed in process 0: StepFailed: a failure"):
        with communicator.allRanksFailTogether("testing"):
            raise StepFailed("a failure")
    with pytest.raises(ErrorOnAllRanks, match="testing failed in process 0: KeyError") as caught:
        with communicator.allRanksFailTogether("testing"):
            raise KeyError("an error")
    assert not isinstance(caught.value, StepFailed)


def test_no_process_communicates_where_the_processes_agree_on_failures():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.mpienvironment import ErrorOnAllRanks

    communicator = Communicator(MPI.COMM_SELF)

    # A communication inside the context fails, on all ranks together, and names the context: an
    # error, not a step failure.
    with pytest.raises(ErrorOnAllRanks, match="allgather was called while gathering inside"):
        with communicator.allRanksFailTogether("Gathering inside"):
            communicator.allgather(1.0)
    with pytest.raises(ErrorOnAllRanks, match="Isend was called while"):
        with communicator.allRanksFailTogether("Exchanging inside"):
            communicator.Isend(np.zeros(1), dest=0, tag=1)
    # So does one nested deeper, and the communicator is usable again after either context.
    with pytest.raises(ErrorOnAllRanks, match="bcast was called while the inner one"):
        with communicator.allRanksFailTogether("The outer one"):
            with communicator.withoutCommunication("The inner one"):
                pass
            with communicator.withoutCommunication("The inner one"):
                communicator.bcast(1)
    assert communicator.allgather(2.0) == [2.0]
    assert communicator.allreduceSum([1.5]) == [1.5]


def test_the_processes_may_communicate_where_they_agree_on_failures_while_setting_up():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.mpienvironment import (
        ErrorOnAllRanks,
        StepFailedOnAllRanks,
    )
    from edelweissfe.utils.exceptions import StepFailed

    communicator = Communicator(MPI.COMM_SELF)

    # collective operations work inside, each after an agreement that no process has failed so far
    with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
        assert communicator.allgather(1.0) == [1.0]
        assert communicator.allreduceSum([2.0]) == [2.0]
    # a failure is agreed on as in allRanksFailTogether
    with pytest.raises(ErrorOnAllRanks, match="Setting up failed in process 0: KeyError"):
        with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
            communicator.allgather(1.0)
            raise KeyError("an error")
    with pytest.raises(StepFailedOnAllRanks, match="Setting up failed in process 0: StepFailed"):
        with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
            raise StepFailed("a step failure")
    # an operation not every process takes part in is refused inside
    with pytest.raises(ErrorOnAllRanks, match="Isend was called while setting up"):
        with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
            communicator.Isend(np.zeros(1), dest=0, tag=1)
    # and so is an agreement inside another
    with pytest.raises(ErrorOnAllRanks, match="cannot agree on failures inside another agreement"):
        with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
            with communicator.allRanksFailTogetherWhileCommunicating("Setting up again"):
                pass
    assert communicator.allgather(2.0) == [2.0]


def test_an_error_agreed_on_keeps_a_built_in_type_and_is_not_wrapped_again():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.mpienvironment import ErrorOnAllRanks

    communicator = Communicator(MPI.COMM_SELF)

    # an error of a built-in type is agreed on as that type too, with the message as given
    with pytest.raises(ModuleNotFoundError) as caught:
        with communicator.allRanksFailTogether("testing"):
            raise ModuleNotFoundError("no module")
    assert isinstance(caught.value, ErrorOnAllRanks)
    with pytest.raises(KeyError) as caught:
        with communicator.allRanksFailTogether("testing"):
            raise KeyError("no key")
    assert str(caught.value) == "testing failed in process 0: KeyError: 'no key'"

    # an error of another type is a plain ErrorOnAllRanks
    class _Own(Exception):
        pass

    with pytest.raises(ErrorOnAllRanks) as caught:
        with communicator.allRanksFailTogether("testing"):
            raise _Own("own")
    assert type(caught.value) is ErrorOnAllRanks

    # agreed on already, it is raised on unchanged
    agreed = ErrorOnAllRanks("inner failed in process 0: KeyError: 'x'")
    with pytest.raises(ErrorOnAllRanks) as caught:
        with communicator.allRanksFailTogether("outer"):
            raise agreed
    assert caught.value is agreed

    # no process communicates inside the agreement of the setup while it handles an exception: it
    # would agree that nothing failed
    with pytest.raises(ErrorOnAllRanks, match="Barrier was called while setting up, during the handling of KeyError"):
        with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
            try:
                raise KeyError("x")
            finally:
                communicator.Barrier()
    # but where it was entered while handling one, that one does not count
    try:
        raise ValueError("handled outside")
    except ValueError:
        with communicator.allRanksFailTogetherWhileCommunicating("Setting up"):
            assert communicator.allgather(1.0) == [1.0]


#: Runs each deck in the directories given twice -- as it is, and with every entry of the net force
#: vector at a degree of freedom not integrated in the process set to NaN after each assembly -- and
#: prints, per process and deck, whether the final solution, net force and velocity are the same bits.
_NET_FORCE_OUTSIDE_SUBDOMAIN_SCRIPT = """
import contextlib, io, os, sys
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
assembleNetForce = NEDMPI.assembleNetForce


def poisonedOutsideTheSubdomain(self, U, dU, P, stepActions, timeStep):
    P, psi = assembleNetForce(self, U, dU, P, stepActions, timeStep)
    outside = np.ones(P.shape[0], dtype=bool)
    outside[self.subdomain.dofs] = False
    P.asPlainArray()[outside] = np.nan
    return P, psi


def finalFields(directory):
    os.chdir(directory)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
    return np.hstack([f[e].flatten() for e in ("U", "P", "V") for f in model.nodeFields.values() if e in f])


for directory in sys.argv[1:]:
    plain = finalFields(directory)
    NEDMPI.assembleNetForce = poisonedOutsideTheSubdomain
    poisoned = finalFields(directory)
    NEDMPI.assembleNetForce = assembleNetForce
    same = np.array_equal(plain.view(np.int64), poisoned.view(np.int64))
    print("PROCESS", rank, os.path.basename(directory), "SAME" if same else "DIFFERENT", flush=True)
"""


def test_the_net_force_outside_the_subdomain_is_never_read(tmp_path):
    # A process holds the net force -- elements, loads, constraint forces -- only at the degrees of
    # freedom it integrates; elsewhere the vector holds whatever was left there until it is next
    # gathered from the owners. Nothing may read it there: poisoned with NaN, the result is the same.
    decks = ["NEDContact", "TieNED", "NEDSurfaceToDiscreteRigidBodyContact"]
    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    for deck in decks:
        shutil.copytree(os.path.join(testfiles, "edelweiss-only", deck), tmp_path / deck)
    (tmp_path / "run.py").write_text(_NET_FORCE_OUTSIDE_SUBDOMAIN_SCRIPT)

    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")
    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    output = subprocess.run(
        [mpirun, "--bind-to", "none", "-n", "3", sys.executable, "run.py"] + [str(tmp_path / deck) for deck in decks],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert reports == sorted("PROCESS {:} {:} SAME".format(rank, deck) for rank in range(3) for deck in decks), output


#: Runs the decks given, with vectors of the subdomain and with vectors of the whole model, and prints,
#: per process and deck, whether the final node fields are the same bits, and how many degrees of
#: freedom the DofManager of the subdomain numbered of how many of the model.
_VECTORS_OF_SUBDOMAIN_SCRIPT = """
import contextlib, io, os, sys
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
reasons = NEDMPI.__dict__["reasonsForVectorsOfWholeModel"]
sizes = []
createDofManager = NEDMPI.createDofManager


def recordingSize(self, model):
    dofManager = createDofManager(self, model)
    sizes.append(dofManager.nDof)
    return dofManager


NEDMPI.createDofManager = recordingSize


def finalFields(directory):
    os.chdir(directory)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
    return np.hstack([f[e].flatten() for e in ("U", "P", "V") for f in model.nodeFields.values() if e in f])


for directory in sys.argv[1:]:
    sizes.clear()
    ofSubdomain = finalFields(directory)
    sizeOfSubdomain = sizes[0]
    NEDMPI.reasonsForVectorsOfWholeModel = staticmethod(lambda model: ["a test"])
    sizes.clear()
    ofWholeModel = finalFields(directory)
    NEDMPI.reasonsForVectorsOfWholeModel = reasons
    same = np.array_equal(ofSubdomain.view(np.int64), ofWholeModel.view(np.int64))
    print("PROCESS", rank, os.path.basename(directory), "SAME" if same else "DIFFERENT", sizeOfSubdomain < sizes[0], flush=True)
"""


def test_vectors_of_the_subdomain_give_the_bits_of_vectors_of_the_whole_model(tmp_path):
    # A process numbering only the degrees of freedom of its subdomain -- the global numbering,
    # compressed -- computes the same bits as one holding vectors of the whole model: the elements,
    # a constraint force, Dirichlet conditions on node sets split between the processes, the output, a
    # tie (the multi-point-constraint transformation of the subdomain).
    decks = ["NED", "NEDEqualValuePenalty", "NEDNodeForcesSplit", "TieNED"]
    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    for deck in decks:
        shutil.copytree(os.path.join(testfiles, "edelweiss-only", deck), tmp_path / deck)
    (tmp_path / "run.py").write_text(_VECTORS_OF_SUBDOMAIN_SCRIPT)

    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")
    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    output = subprocess.run(
        [mpirun, "--bind-to", "none", "-n", "3", sys.executable, "run.py"] + [str(tmp_path / deck) for deck in decks],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert reports == sorted(
        "PROCESS {:} {:} SAME True".format(rank, deck) for rank in range(3) for deck in decks
    ), output


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

#: A body force kernel failing in process 1, as a step failure.
_FAILING_LOAD_PATCH = """
from edelweissfe.elements.base.displacementelementbase import DisplacementElementBase
from edelweissfe.utils.exceptions import StepFailed

computeBodyForce = DisplacementElementBase.computeBodyForce


def failingInProcessOne(self, *args):
    if rank == 1:
        raise StepFailed("load kernel failed")
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
    expected = "['Simulation failed: Evaluating the loads failed in process 1: StepFailed: load kernel failed']"
    assert reports == ["PROCESS {:} {:}".format(rank, expected) for rank in range(3)], output


#: Runs the deck in this directory with PATCH applied, as a caller catching every exception does --
#: the test runner, which goes on to the next test case --, and prints, per process, how the job
#: ended: "returned", or the exception that left it.
_CATCHING_IN_EVERY_PROCESS_SCRIPT = """
import contextlib, io
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
PATCH
journal = io.StringIO()
try:
    with contextlib.redirect_stdout(journal):
        finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
    outcome = "returned"
except Exception as exception:
    outcome = "{:}: {:}".format(type(exception).__name__, exception)
print("PROCESS", rank, outcome, flush=True)
"""

#: An element kernel failing in process 1 with an error that is no step failure: a programming error.
_ELEMENT_ERROR_PATCH = """
from edelweissfe.elements.displacementelement.element import DisplacementElement

computeKernelsExplicit = DisplacementElement.computeKernelsExplicit


def failingInProcessOne(self, *args):
    if rank == 1:
        raise KeyError("a programming error")
    return computeKernelsExplicit(self, *args)


DisplacementElement.computeKernelsExplicit = failingInProcessOne
"""


def test_an_error_in_one_process_ends_the_job_with_an_error_in_every_process(tmp_path):
    # A serial run ends with a KeyError of an element, not as after a failed step: the exception leaves
    # the job, with a non-zero exit status. Agreed on as a failed step, the job would end "normally".
    (tmp_path / "test.inp").write_text(_FAILING_LOAD_DECK)
    (tmp_path / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", _ELEMENT_ERROR_PATCH))
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("an error in one process left the others waiting")
    assert exitCode != 0, output
    assert (
        "ErrorOnAllRanks: Evaluating the elements and the constraints failed in process 1: KeyError: "
        "'a programming error'" in output
    ), output

    # Caught -- as the test runner catches it --, every process catches the same error, and none waits.
    (tmp_path / "run.py").write_text(_CATCHING_IN_EVERY_PROCESS_SCRIPT.replace("PATCH", _ELEMENT_ERROR_PATCH))
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("an error in one process, caught, left the others waiting")
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    expected = (
        "ErrorOnAllRanks: Evaluating the elements and the constraints failed in process 1: KeyError: "
        "'a programming error'"
    )
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
from edelweissfe.utils.exceptions import StepFailed
from edelweissfe.utils.fieldoutput import ElementFieldOutput

rebuildCollector = ElementFieldOutput._rebuildCollectorIfSetChanged
readings = []


def failingInProcessOne(self):
    readings.append(self)
    if rank == 1 and len(readings) > 1:
        raise StepFailed("reading the element results failed")
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
        "['Simulation failed: Reading the output failed in process 1: StepFailed: reading the element "
        "results failed']"
    )
    assert reports == ["PROCESS {:} {:}".format(rank, expected) for rank in range(3)], output


#: Setting up the model fails in one process: on rank 0 while it partitions the mesh, before the
#: partition is broadcast; on rank 1 after that, before the element results are gathered for the field
#: output at the start of the job.
_SETUP_FAILING_PATCHES = {
    0: """
from edelweissfe.domaindecomposition import partitioning


def failingInProcessZero(*args, **kwargs):
    raise RuntimeError("partitioning failed in process 0")


partitioning.partitionMeshDual = failingInProcessZero
""",
    1: """
from edelweissfe.models.femodel import FEModel

prepareYourself = FEModel.prepareYourself


def failingInProcessOne(self, journal):
    if rank == 1:
        raise RuntimeError("preparing the model failed in process 1")
    return prepareYourself(self, journal)


FEModel.prepareYourself = failingInProcessOne
""",
}


@pytest.mark.parametrize("failingRank", [0, 1])
def test_setting_up_the_model_failing_in_one_process_fails_in_every_process(tmp_path, failingRank):
    # Not agreed on, the failing process -- caught, as the test runner catches it -- would end its
    # job, while the others wait for it in the broadcast of the partition, or in the gather of the
    # element results, forever (the timeout).
    (tmp_path / "test.inp").write_text(_FIELD_OUTPUT_DECK)
    (tmp_path / "run.py").write_text(
        _CATCHING_IN_EVERY_PROCESS_SCRIPT.replace("PATCH", _SETUP_FAILING_PATCHES[failingRank])
    )
    output, _ = _runUnderMPI(tmp_path, timeout=60)
    if output is None:
        pytest.fail("setting up the model failing in one process left the others waiting")

    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    failure = "partitioning failed in process 0" if failingRank == 0 else "preparing the model failed in process 1"
    expected = "ErrorOnAllRanks: Setting up the model failed in process {:}: RuntimeError: {:}".format(
        failingRank, failure
    )
    assert reports == ["PROCESS {:} {:}".format(rank, expected) for rank in range(3)], output


#: Setting up the model fails in process 1, which communicates on its way out -- in a finally clause --
#: while the others wait in their next operation (the gather of the element results).
_SETUP_FAILING_AND_COMMUNICATING_PATCH = """
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.models.femodel import FEModel

prepareYourself = FEModel.prepareYourself


def failingInProcessOne(self, journal):
    if rank == 1:
        try:
            raise RuntimeError("preparing the model failed in process 1")
        finally:
            worldCommunicator().Barrier()
    return prepareYourself(self, journal)


FEModel.prepareYourself = failingInProcessOne
"""


def test_a_process_failing_while_setting_up_does_not_communicate_on_its_way_out(tmp_path):
    # Agreeing before the barrier that nothing failed, process 1 would meet the others' gather with a
    # barrier: a hang (the timeout). It raises instead, and every process raises that.
    (tmp_path / "test.inp").write_text(_FIELD_OUTPUT_DECK)
    (tmp_path / "run.py").write_text(
        _CATCHING_IN_EVERY_PROCESS_SCRIPT.replace("PATCH", _SETUP_FAILING_AND_COMMUNICATING_PATCH)
    )
    output, _ = _runUnderMPI(tmp_path, timeout=60)
    if output is None:
        pytest.fail("a process communicating on its way out of a failed setup left the others waiting")

    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    expected = (
        "ErrorOnAllRanks: Setting up the model failed in process 1: RuntimeError: Barrier was called while setting "
        "up the model, during the handling of RuntimeError: preparing the model failed in process 1"
    )
    assert len(reports) == 3 and all(
        report.startswith("PROCESS {:} {:}".format(rank, expected)) for rank, report in enumerate(reports)
    ), output


#: An element kernel raising NotImplementedError in process 1 -- what a missing optional module raises.
_NOT_IMPLEMENTED_PATCH = _ELEMENT_ERROR_PATCH.replace(
    'KeyError("a programming error")', 'NotImplementedError("not built")'
)


def test_an_error_on_all_ranks_is_of_the_type_every_failing_process_raised(tmp_path):
    # The test runner skips a test case on a NotImplementedError: under MPI every process must see that
    # type, not only the one that raised it.
    (tmp_path / "test.inp").write_text(_FAILING_LOAD_DECK)
    (tmp_path / "run.py").write_text(
        _CATCHING_IN_EVERY_PROCESS_SCRIPT.replace("PATCH", _NOT_IMPLEMENTED_PATCH).replace(
            'outcome = "{:}: {:}".format(type(exception).__name__, exception)',
            'outcome = "{:} {:}".format(type(exception).__name__, isinstance(exception, NotImplementedError))',
        )
    )
    output, _ = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("an error in one process left the others waiting")
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert reports == ["PROCESS {:} ErrorOnAllRanks True".format(rank) for rank in range(3)], output


#: A step failing in process 1 alone, outside any step the processes agree on: on increment 3, which
#: writes no output, so that the others go on into the next increment's exchange -- and the failing
#: process unwinds through the end of the step, whose field outputs gather element results.
_STEP_FAILING_ALONE_PATCH = """
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.exceptions import StepFailed

acceptIncrement = NEDMPI.acceptIncrement


def failingInProcessOne(self, step, model, timeStep):
    acceptIncrement(self, step, model, timeStep)
    if rank == 1 and timeStep.number == 3:
        raise StepFailed("the step failed in process 1 alone")


NEDMPI.acceptIncrement = failingInProcessOne
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
    assert (
        "the step failed without the agreement of the other processes (StepFailed: the step failed in process 1 alone)"
        in output
    ), output
    assert "in MPI process 1 of 3; aborting all processes" in output, output


#: _FIELD_OUTPUT_DECK at ten times the stable time step, over enough time to diverge.
_DIVERGING_DECK = _FIELD_OUTPUT_DECK.replace("output-frequency=2", "output-frequency=2\ncourant-number=10").replace(
    "maxNumInc=5, maxIter=25, stepLength=1", "maxNumInc=5000, maxIter=25, stepLength=1000"
)


def test_a_diverged_run_fails_on_every_process_as_a_serial_one_does(tmp_path):
    # The energy balance is summed over all processes, so every process finds it diverged in the
    # same increment: the failure is replicated, and ends the job as a failed step, not by abort.
    (tmp_path / "test.inp").write_text(_DIVERGING_DECK)
    (tmp_path / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", ""))
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("a diverged run left processes waiting")

    assert exitCode == 0, output
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert len(reports) == 3, output
    for rank, report in enumerate(reports):
        assert report.startswith(
            "PROCESS {:} ['Simulation failed: THE SOLUTION HAS DIVERGED in increment ".format(rank)
        ), output


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
    assert "the lumped operators assembled again from the elements now local differ" in output, output


#: A fourth carried vector, the acceleration ``A``, added to the one list of them -- here a copy of
#: the velocity, set at the end of every increment --, and checked to be published, synchronized,
#: migrated, poisoned and checkpointed exactly as the velocity is, in every process.
_FOURTH_CARRIED_VECTOR_SCRIPT = """
import contextlib, io
import h5py
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamic import NED, ExplicitSystem
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
ExplicitSystem.carriedEntries = ExplicitSystem.carriedEntries + ("A",)

attemptIncrement = NED.attemptIncrement


def mirroringTheVelocity(self, step, model, timeStep):
    attemptIncrement(self, step, model, timeStep)
    self._system.carried["A"].asPlainArray()[:] = self._system.V.asPlainArray()


NED.attemptIncrement = mirroringTheVelocity

differences = []


def sameAsTheVelocity(model, where):
    for nodeField in model.nodeFields.values():
        if "V" in nodeField and not np.array_equal(nodeField["A"], nodeField["V"], equal_nan=True):
            differences.append(where)


acceptIncrement = NEDMPI.acceptIncrement


def checkedAfterAccepting(self, step, model, timeStep):
    acceptIncrement(self, step, model, timeStep)
    sameAsTheVelocity(model, "increment {:}".format(timeStep.number))


NEDMPI.acceptIncrement = checkedAfterAccepting

with contextlib.redirect_stdout(io.StringIO()):
    model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
sameAsTheVelocity(model, "the end")
if rank == 0:
    with h5py.File("restart_0.h5") as checkpoint:
        names = []
        checkpoint.visit(names.append)
        accelerations = [name for name in names if name.endswith("/A")]
        for name in accelerations:
            if not np.array_equal(checkpoint[name][()], checkpoint[name[:-1] + "V"][()]):
                differences.append("checkpoint " + name)
        if not accelerations:
            differences.append("checkpoint without A")
print("PROCESS", rank, "SAME" if not differences else differences, flush=True)
"""


def test_a_carried_vector_a_refinement_would_not_carry_is_refused(monkeypatch):
    # A refinement interpolates the entries it knows onto the nodes it creates; a carried vector it does
    # not know, and the solver does not restart itself, would start from zero there -- refused instead.
    from types import SimpleNamespace

    from edelweissfe.modelmodifiers.adaptivity.hadaptivity import (
        ModelModifier as HAdaptivity,
    )
    from edelweissfe.solvers.nonlinearexplicitdynamic import NED, ExplicitSystem

    refinement = SimpleNamespace(nodeFieldEntriesOfNewNodes=HAdaptivity.nodeFieldEntriesOfNewNodes)
    creatingNoNodes = SimpleNamespace(nodeFieldEntriesOfNewNodes=None)
    NED.requireCarriedOntoNewNodes([refinement, creatingNoNodes])

    monkeypatch.setattr(ExplicitSystem, "carriedEntries", ExplicitSystem.carriedEntries + ("B",))
    with pytest.raises(ValueError, match=r"without carrying the vector\(s\) \['B'\]"):
        NED.requireCarriedOntoNewNodes([refinement])
    NED.requireCarriedOntoNewNodes([creatingNoNodes])
    # restarted by the solver itself after a topology change, it need not be carried
    monkeypatch.setattr(
        ExplicitSystem,
        "entriesRestartedAfterTopologyChange",
        ExplicitSystem.entriesRestartedAfterTopologyChange + ("B",),
    )
    NED.requireCarriedOntoNewNodes([refinement])


def test_a_vector_added_to_the_carried_vectors_reaches_every_path(tmp_path):
    # The carried vectors are listed once (ExplicitSystem.carriedEntries); every path iterates that
    # list. A vector enumerated by hand somewhere would be published only where that path ran, and the
    # copy of the velocity would differ from it -- in a migration, a synchronization, the poison, the
    # checkpoint.
    deck = _MIGRATING_DECK.replace("output-frequency=2", "output-frequency=2\ndebug-poison-stale-solution=True")
    deck += "*output, type=restart, name=restart\nwriteInterval=2, baseName=restart, numberOfFilesToKeep=1\n"
    (tmp_path / "test.inp").write_text(deck)
    (tmp_path / "run.py").write_text(_FOURTH_CARRIED_VECTOR_SCRIPT)
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("the run left processes waiting")
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert exitCode == 0 and reports == ["PROCESS {:} SAME".format(rank) for rank in range(3)], output


#: Before elements move, every process' copies of the constraints -- here a contact against a rigid
#: body, searched on its process only -- must be the same: checked at every migration, and the deck run
#: on, so that it also reaches a migration on increment 3, which follows no synchronization at all.
_CONSISTENT_COPIES_SCRIPT = """
import contextlib, hashlib, io
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.domaindecomposition.domaindecomposition import DomainDecomposition
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
verdicts = []
moveElements = DomainDecomposition.moveElements


def checkedBeforeMoving(self):
    digest = hashlib.sha1()
    for name, constraint in self._model.constraints.items():
        for entry, values in sorted(constraint.getRestartData().items()):
            digest.update(name.encode() + entry.encode() + np.ascontiguousarray(values).tobytes())
    verdicts.append(len(set(self.communicator.allgather(digest.hexdigest()))) == 1)
    return moveElements(self)


DomainDecomposition.moveElements = checkedBeforeMoving

acceptIncrement = NEDMPI.acceptIncrement


def movingElementsAfterIncrementThree(self, step, model, timeStep):
    acceptIncrement(self, step, model, timeStep)
    if timeStep.number == 3:
        self._moveElements(model, step)


NEDMPI.acceptIncrement = movingElementsAfterIncrementThree

with contextlib.redirect_stdout(io.StringIO()):
    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
print("PROCESS", rank, len(verdicts) > 1 and all(verdicts), verdicts, flush=True)
"""


def test_elements_move_only_with_the_same_copies_of_the_constraints_in_every_process(tmp_path):
    # A migration rebuilds every process' system from its copies of the constraints owned elsewhere. The
    # output synchronization reaches rank 0 only, and a contact search runs on its constraint's process
    # only: the copies elsewhere are stale until the model is synchronized to every process, which a
    # migration does first.
    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    shutil.copytree(
        os.path.join(testfiles, "edelweiss-only", "NEDSurfaceToDiscreteRigidBodyContact"), tmp_path / "deck"
    )
    deck = (tmp_path / "deck" / "test.inp").read_text()
    deck = deck.replace("output-frequency=50", "output-frequency=50\nload-balance-costs=elementNumber")
    deck = deck.replace("maxNumInc=2000", "maxNumInc=160")
    (tmp_path / "deck" / "test.inp").write_text(deck)
    tmp_path = tmp_path / "deck"
    (tmp_path / "run.py").write_text(_CONSISTENT_COPIES_SCRIPT)
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("the run left processes waiting")
    reports = sorted(line for line in output.splitlines() if line.startswith("PROCESS"))
    assert exitCode == 0 and len(reports) == 3 and all(" True " in report for report in reports), output


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
    # The facets on the top row (the even numbers) of elements 9-16 are in the subdomain of this process.
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

    # The elements of the left edge, 1 and 2, are computed by the other process: none is local.
    assert len(model.elementSets["gen_left"].localElements()) == 0
    fieldOutput = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs["uLeft"]
    # The nodes of the two elements, read from the mesh -- not every node of the model, and not the
    # nodes of the local element objects, of which there are none.
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


def test_a_part_made_for_a_share_of_its_own_stays_with_its_process():
    from edelweissfe.domaindecomposition.partitioning import keepElementsWhereTheyWere

    previous = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    repartitioned = {1: 2, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1}
    # Part 0 was made smaller, for process 0 (which evaluates a costly constraint, say): it is not
    # given to process 1, although it overlaps with process 1's elements most; parts 1 and 2, of equal
    # shares, are renumbered among themselves.
    shares = np.array([0.2, 0.4, 0.4])

    assert keepElementsWhereTheyWere(repartitioned, previous, 3, shares) == {1: 1, 2: 1, 3: 0, 4: 2, 5: 2, 6: 2}


def test_parts_of_nearly_equal_shares_are_renumbered_among_their_processes():
    from edelweissfe.domaindecomposition.partitioning import keepElementsWhereTheyWere

    # Every process evaluates some constraint, so no two shares are equal (c1_150: ten constraints on
    # eight processes). Within the tolerance they are interchangeable, and elements keep their process.
    previous = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    repartitioned = {1: 2, 2: 2, 3: 0, 4: 0, 5: 1, 6: 1}
    shares = np.array([0.335, 0.332, 0.333])

    assert keepElementsWhereTheyWere(repartitioned, previous, 3, shares) == previous
    # Without a tolerance, no part may change its process: the partition is kept as made.
    assert keepElementsWhereTheyWere(repartitioned, previous, 3, shares, 0.0) == repartitioned
    # A part of a share of its own still stays with its process, the others are renumbered.
    shares = np.array([0.2, 0.401, 0.399])
    renumbered = keepElementsWhereTheyWere(repartitioned, previous, 3, shares)
    assert renumbered == {1: 1, 2: 1, 3: 0, 4: 0, 5: 2, 6: 2}


def test_no_process_is_left_without_elements():
    from edelweissfe.domaindecomposition.partitioning import (
        MINIMUM_SHARE_OF_AN_EQUAL_SHARE,
        elementSharesBesideConstraints,
    )

    metis = _metis()
    offsets, nodes, nNodes = _quadGrid(64)
    # Process 7 evaluates a constraint costing more than all elements: it was given a share of 0.001
    # of an equal one, and METIS left its part empty (while reporting success).
    shares = elementSharesBesideConstraints(8.0, np.array([0.0] * 7 + [20.0]), 1000)
    assert shares[7] == pytest.approx(MINIMUM_SHARE_OF_AN_EQUAL_SHARE / 8)
    counts = np.bincount(metis.partitionMeshDual(offsets, nodes, nNodes, 8, np.ones(4096, dtype=int), 2, shares))
    assert counts.shape == (8,) and counts.min() > 0


def test_a_part_left_empty_is_given_an_element_and_an_empty_part_is_refused(monkeypatch):
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition import partitioning
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.models.mesh import Mesh

    mesh = Mesh()
    for number in range(1, 5):
        mesh.addElement(number, "CPE4", "edelweiss", [number, number + 1, number + 6, number + 5])
    # METIS reports success with an empty part: the part is given an element.
    monkeypatch.setattr(partitioning, "partitionMeshDual", lambda *arguments: np.array([0, 0, 1, 1]))
    owners = partitioning.partitionElementsOfMesh(mesh, 3, 2, Communicator(MPI.COMM_SELF), None, np.full(3, 1.0 / 3))
    assert owners == {1: 2, 2: 0, 3: 1, 4: 1}

    # Were it not, the partition must not be used.
    monkeypatch.setattr(partitioning, "giveEveryPartAnElement", lambda *arguments: None)
    with pytest.raises(RuntimeError, match="without elements"):
        partitioning.partitionElementsOfMesh(mesh, 3, 2, Communicator(MPI.COMM_SELF), None, np.full(3, 1.0 / 3))


def _quadGridMesh(nX: int, nY: int):
    """A mesh of an nX x nY grid of CPE4 elements, numbered from 1 row by row."""

    from edelweissfe.models.mesh import Mesh

    mesh = Mesh()
    for j in range(nY):
        for i in range(nX):
            a = j * (nX + 1) + i + 1
            mesh.addElement(j * nX + i + 1, "CPE4", "edelweiss", [a, a + 1, a + nX + 2, a + nX + 1])
    return mesh


@pytest.mark.parametrize("shares", [None, np.array([0.2502, 0.2502, 0.2495, 0.2501])])
@pytest.mark.parametrize(
    "outliers",
    [
        # A process descheduled while it timed one element: that element alone measured 1e5 times the
        # others, more than the weight of all others together (METIS left three parts empty).
        {37: 1e5},
        # Measured costs whose integer weights sum beyond METIS' 32-bit index.
        {number: 1e6 for number in range(1, 128, 4)},
    ],
)
def test_a_partition_by_measured_costs_leaves_no_process_without_elements(outliers, shares):
    MPI = pytest.importorskip("mpi4py.MPI")
    _metis()
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.partitioning import partitionElementsOfMesh

    mesh = _quadGridMesh(16, 8)
    measuredCosts = {number: 3.5e-6 * outliers.get(number, 1.0) for number in mesh.elements}

    owners = partitionElementsOfMesh(mesh, 4, 2, Communicator(MPI.COMM_SELF), measuredCosts, shares)

    assert np.bincount(list(owners.values()), minlength=4).min() > 0


def test_a_stall_in_one_increment_does_not_change_the_partition():
    # Per element the least kernel time over the increments since the last check, as the element loop
    # measures it (computeElementsForExplicit): a stall of 1e5 times an element's cost in one increment
    # partitions the mesh exactly as no stall does; the mean -- what was summed before -- does not.
    MPI = pytest.importorskip("mpi4py.MPI")
    _metis()
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.partitioning import partitionElementsOfMesh

    mesh = _quadGridMesh(16, 8)
    numbers = list(mesh.elements)
    rng = np.random.default_rng(7)
    # five increments; the elements of the left half cost twice the others, each increment a little
    # more than its least
    cost = np.where((np.arange(len(numbers)) % 16) < 8, 7e-6, 3.5e-6)
    times = cost * (1.0 + 0.05 * rng.random((5, len(numbers))))
    stalled = times.copy()
    stalled[2, 37] = 1e5 * cost[37]

    def partition(measured):
        owners = partitionElementsOfMesh(
            mesh, 4, 2, Communicator(MPI.COMM_SELF), dict(zip(numbers, measured.tolist())), None
        )
        return [owners[number] for number in numbers]

    withoutStall = partition(times.min(axis=0))
    assert partition(stalled.min(axis=0)) == withoutStall
    # the test has teeth: the mean over the increments gives the stalled element another part
    assert partition(stalled.mean(axis=0)) != partition(times.mean(axis=0))


#: Records, in process 0, the least kernel time of its elements at every check of the balance, which
#: never repartitions; from increment 3 on, every element of process 0 costs 2 ms more (a material
#: softening, say).
_COSTS_GROWING_SCRIPT = """
import contextlib, io, time
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.domaindecomposition.domaindecomposition import DomainDecomposition
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
increment = [0]
checks = []

acceptIncrement = NEDMPI.acceptIncrement


def counting(self, step, model, timeStep):
    increment[0] = timeStep.number + 1
    acceptIncrement(self, step, model, timeStep)


NEDMPI.acceptIncrement = counting

computeKernelsExplicit = DisplacementElement.computeKernelsExplicit


def growing(self, *args):
    if rank == 0 and increment[0] >= 3:
        time.sleep(2e-3)
    return computeKernelsExplicit(self, *args)


DisplacementElement.computeKernelsExplicit = growing


def recording(self, plan, load):
    if load.nIncrements:  # the zero increment computes no element
        checks.append(float(np.median(load.elementCosts)))
    return False


DomainDecomposition.rebalance = recording

with contextlib.redirect_stdout(io.StringIO()):
    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
if rank == 0:
    print("CHECKS", " ".join("{:.6f}".format(cost) for cost in checks), flush=True)
"""


def test_a_check_of_the_balance_weighs_what_elements_cost_since_the_previous_one(tmp_path):
    # Measured afresh after every check, an element that grew costlier weighs what it costs now at the
    # next check -- not the least it ever cost since its increment plan was made.
    (tmp_path / "test.inp").write_text(_FIELD_OUTPUT_DECK)
    (tmp_path / "run.py").write_text(_COSTS_GROWING_SCRIPT)
    output, exitCode = _runUnderMPI(tmp_path, nProcesses=2)
    if output is None:
        pytest.fail("the run left processes waiting")
    checks = [line for line in output.splitlines() if line.startswith("CHECKS")]
    assert exitCode == 0 and len(checks) == 1, output
    costs = [float(cost) for cost in checks[0].split()[1:]]
    # checks at increments 2 and 4: before the growth microseconds, after it at least the 2 ms
    assert len(costs) == 2 and costs[0] < 1e-3 and costs[1] >= 2e-3, output


def test_a_process_evaluating_costly_constraints_is_given_fewer_elements():
    from edelweissfe.domaindecomposition.partitioning import (
        elementSharesBesideConstraints,
    )

    assert elementSharesBesideConstraints(12.0, np.zeros(4), 1000) is None
    # Too few elements per process for unequal shares: METIS might leave a part empty.
    assert elementSharesBesideConstraints(12.0, np.array([0.0, 0.0, 0.0, 4.0]), 31) is None
    # (24 + 2) / 4 = 6.5 per process: process 3 gets 6.5 - 2 = 4.5 of the 24 of element work
    shares = elementSharesBesideConstraints(24.0, np.array([0.0, 0.0, 0.0, 2.0]), 1000)
    assert np.allclose(shares, [6.5 / 24.0, 6.5 / 24.0, 6.5 / 24.0, 4.5 / 24.0])
    # (12 + 4) / 4 = 4 per process: process 3 would be busy with its constraints alone, and keeps half
    # an equal share
    shares = elementSharesBesideConstraints(12.0, np.array([0.0, 0.0, 0.0, 4.0]), 1000)
    assert np.allclose(shares, [0.875 / 3.0, 0.875 / 3.0, 0.875 / 3.0, 0.125])
    assert np.isclose(shares.sum(), 1.0)
    # Process 0 is busier with its constraint than any process can be with the elements; process 1
    # would be filled with 1/9 of the elements, less than the minimum, too: both keep half an equal
    # share, the others share the rest.
    shares = elementSharesBesideConstraints(3.0, np.array([10.0, 1.0, 0.0, 0.0]), 1000)
    assert np.allclose(shares, [0.125, 0.125, 0.375, 0.375])


def test_the_balance_is_measured_against_what_a_partition_can_attain():
    MPI = pytest.importorskip("mpi4py.MPI")
    from edelweissfe.domaindecomposition.communicator import Communicator
    from edelweissfe.domaindecomposition.domaindecomposition import DomainDecomposition

    decomposition = DomainDecomposition(Communicator(MPI.COMM_SELF), None, "test", 0.1)
    # Busy with elements and constraints: the mean, 9, is attainable; the busiest takes 12.
    imbalance, beyond = decomposition._imbalance(np.array([8.0, 8.0, 8.0, 12.0]), np.array([0.0, 0.0, 0.0, 4.0]))
    assert imbalance == pytest.approx(12.0 / 9.0) and beyond == pytest.approx(12.0 - 1.1 * 9.0)
    # One process' constraints alone cost more than the mean: no partition of the elements does better
    # than that process' constraints, so it is within the tolerance, and not repartitioned again and again.
    imbalance, beyond = decomposition._imbalance(np.array([6.0, 6.0, 6.0, 16.5]), np.array([0.0, 0.0, 0.0, 16.0]))
    assert imbalance == pytest.approx(16.5 / 16.0) and beyond < 0.0


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


#: Runs each deck given twice -- plainly, and with ``debug-poison-stale-solution``, which sets the
#: solution, the velocity, the net force and the node fields to NaN, after every increment, outside the
#: degrees of freedom a process integrates, wherever the process does not hold the whole solution -- and
#: reports, per process, whether the two final node fields are the same bits. With
#: ``--without-fetches``, the entries a process receives from their owners point to point are not
#: received, which the poisoned run must notice.
_STALE_SOLUTION_POISONED_SCRIPT = """
import contextlib, io, os, re, sys
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.domaindecomposition.subdomaininterface import ValuesFromOwners
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

rank = worldCommunicator().Get_rank()
if "--without-fetches" in sys.argv:
    sys.argv.remove("--without-fetches")
    ValuesFromOwners.receive = lambda self, vectors: None


def finalFields(directory, poisoned):
    os.chdir(directory)
    deck = open("test.inp").read()
    if poisoned:
        deck = re.sub(r"^(\\*solver,.*NEDMPI.*)$", r"\\1\\ndebug-poison-stale-solution=True", deck, flags=re.M)
    with open("poisoned.inp" if poisoned else "plain.inp", "w") as f:
        f.write(deck)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = finiteElementSimulation(
            parseInputFile("poisoned.inp" if poisoned else "plain.inp"), verbose=False, suppressPlots=True
        )
    return np.hstack([f[e].flatten() for e in ("U", "P", "V") for f in model.nodeFields.values() if e in f])


for directory in sys.argv[1:]:
    plain = finalFields(directory, poisoned=False)
    poisoned = finalFields(directory, poisoned=True)
    same = np.array_equal(plain.view(np.int64), poisoned.view(np.int64))
    print("PROCESS", rank, os.path.basename(directory), "SAME" if same else "DIFFERENT", flush=True)
"""


def _runPoisonedDecks(tmp_path, decks: list[str], options: list[str]) -> tuple[list[str], str]:
    """Run :data:`_STALE_SOLUTION_POISONED_SCRIPT` over the edelweiss-only MPI decks given, on 3
    processes, and return its reports, sorted, and its whole output."""

    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    for deck in decks:
        shutil.copytree(os.path.join(testfiles, "edelweiss-only", deck), tmp_path / deck, dirs_exist_ok=True)
    (tmp_path / "run.py").write_text(_STALE_SOLUTION_POISONED_SCRIPT)

    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")
    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    output = subprocess.run(
        [mpirun, "--bind-to", "none", "-n", "3", sys.executable, "run.py"]
        + options
        + [str(tmp_path / deck) for deck in decks],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    ).stdout
    return sorted(line for line in output.splitlines() if line.startswith("PROCESS")), output


def test_a_process_reads_no_solution_it_did_not_receive(tmp_path):
    # Between two synchronizations of the whole model a process holds the solution, the velocity and
    # the net force only at the degrees of freedom it integrates, and an output increment gathers the
    # whole model to rank 0 only. Whatever a process reads beyond that -- the positions a contact search
    # reads, the entries it integrates from now on after a search moved a constraint -- it receives
    # from their owners. Poisoned with NaN everywhere else, the result is the same bits.
    decks = ["NEDContact", "NEDSurfaceToDiscreteRigidBodyContact", "TieNED"]
    reports, output = _runPoisonedDecks(tmp_path, decks, [])
    assert reports == sorted("PROCESS {:} {:} SAME".format(rank, deck) for rank in range(3) for deck in decks), output

    # ... and the poison bites: without receiving the positions from their owners, the contact search
    # reads NaN where the process does not integrate a node.
    reports, output = _runPoisonedDecks(tmp_path, ["NEDContact"], ["--without-fetches"])
    assert any(report.endswith("DIFFERENT") for report in reports), output


def test_a_field_output_gathered_elsewhere_refuses_to_be_read_here(tmp_path):
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
>>perNode, name=uLeft, elSet=gen_left, field=displacement, result=U, saveHistory=True
"""
    )
    inputFile = parseInputFile(str(deck))
    model = fillFEModelFromInputFile(FEModel(2), inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
    fieldOutput = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs["uLeft"]
    fieldOutput.finalizeIncrement()
    assert fieldOutput.getLastResult().shape == (6, 2)

    # gathered to the process writing the output, not here: the time is recorded, the result is not
    fieldOutput.gatherResultsOfWholeSet(toEveryProcess=False, storedHere=False)
    model.time = 1.0
    fieldOutput.finalizeIncrement()
    assert fieldOutput.getTimeHistory().tolist() == [0.0, 1.0]
    with pytest.raises(RuntimeError, match="held by the process writing the output"):
        fieldOutput.getLastResult()
    with pytest.raises(RuntimeError, match="held by the process writing the output"):
        fieldOutput.getResultHistory()

    # stored here again: the last result is readable, the history -- missing a result -- is not
    model.time = 2.0
    fieldOutput.finalizeIncrement()
    assert fieldOutput.getLastResult().shape == (6, 2)
    with pytest.raises(RuntimeError, match="held by the process writing the output"):
        fieldOutput.getResultHistory()


def test_an_element_field_output_gathered_elsewhere_reads_afresh_at_the_end_of_a_step(tmp_path):
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
>>perElement, name=sTop, elSet=gen_top, result=stress, quadraturePoint=0
"""
    )
    inputFile = parseInputFile(str(deck))
    model = fillFEModelFromInputFile(FEModel(2), inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    fieldOutput = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs["sTop"]

    # An output increment whose result was gathered to the process writing the output only. Where every
    # process holds every element, the gather leaves the whole result here all the same -- it must not
    # be kept for the next result stored here.
    fieldOutput.readResultsHere()
    fieldOutput.gatherResultsOfWholeSet(toEveryProcess=False, storedHere=False)
    model.time = 1.0
    fieldOutput.finalizeIncrement()
    with pytest.raises(RuntimeError, match="held by the process writing the output"):
        fieldOutput.getLastResult()

    # The end of a step, later: the result is read afresh from the elements, not the one of the output
    # increment.
    reads = []
    readResultsHere = fieldOutput.readResultsHere
    fieldOutput.readResultsHere = lambda: (reads.append(model.time), readResultsHere())
    model.time = 2.0
    fieldOutput.finalizeStep()
    assert reads == [2.0]
    assert fieldOutput.getLastResult().shape[0] == len(model.elementSets["gen_top"])


#: Runs the deck in this directory and writes, from rank 0 (or the one process), the final node fields
#: to the file given.
_FINAL_FIELDS_SCRIPT = """
import contextlib, io, sys
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import isRootProcess
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

with contextlib.redirect_stdout(io.StringIO()):
    model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
if isRootProcess():
    np.save(sys.argv[1], np.hstack([f[e].flatten() for e in ("U", "P", "V") for f in model.nodeFields.values() if e in f]))
"""


def test_a_rigid_body_is_moved_alike_in_every_process_when_elements_move(tmp_path):
    # An output increment gathers the solution to rank 0 only, but every process moves the surfaces of
    # the rigid bodies, which are nodes of the model: the reference nodes reach every process. A
    # migration -- forced here by element-number costs -- builds the equation system again and compares
    # the node coordinates of every process; with a surface left behind on some, the processes would
    # hold different models.
    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")
    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    for solver, launcher in (("NED", []), ("NEDMPI", [mpirun, "--bind-to", "none", "-n", "3"])):
        directory = tmp_path / solver
        shutil.copytree(os.path.join(testfiles, "edelweiss-only", "NEDNodeToDiscreteRigidBodyContact"), directory)
        deck = (directory / "test.inp").read_text().replace("solver=NEDMPI", "solver=" + solver)
        if solver == "NEDMPI":
            deck = deck.replace(
                "contact-update-frequency=10",
                "contact-update-frequency=10\nload-balance-costs=elementNumber\nload-balance-tolerance=0.01",
            )
        (directory / "test.inp").write_text(deck)
        (directory / "run.py").write_text(_FINAL_FIELDS_SCRIPT)
        output = subprocess.run(
            launcher + [sys.executable, "run.py", str(tmp_path / (solver + ".npy"))],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert (tmp_path / (solver + ".npy")).exists(), output.stdout + output.stderr
    serial, decomposed = np.load(tmp_path / "NED.npy"), np.load(tmp_path / "NEDMPI.npy")
    assert serial.view(np.int64).tobytes() == decomposed.view(np.int64).tobytes()


def test_the_end_of_a_step_on_an_output_increment_leaves_its_result_in_every_process(tmp_path):
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
>>perElement, name=sTop, elSet=gen_top, result=stress, quadraturePoint=0, saveHistory=True
"""
    )
    inputFile = parseInputFile(str(deck))
    model = fillFEModelFromInputFile(FEModel(2), inputFile, Journal(verbose=False))
    model.prepareYourself(Journal(verbose=False))
    fieldOutput = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs["sTop"]
    fieldOutput.finalizeIncrement()

    # The last output increment of the step, gathered to the process writing the output only.
    fieldOutput.readResultsHere()
    fieldOutput.gatherResultsOfWholeSet(toEveryProcess=False, storedHere=False)
    model.time = 1.0
    fieldOutput.finalizeIncrement()
    with pytest.raises(RuntimeError, match="held by the process writing the output"):
        fieldOutput.getLastResult()

    # The step ends right there: its result is read again, the time is not recorded twice, and the next
    # step may read it in this process too -- not the history, which misses a result.
    fieldOutput.finalizeStep()
    assert fieldOutput.getTimeHistory().tolist() == [0.0, 1.0]
    assert fieldOutput.getLastResult().shape[0] == len(model.elementSets["gen_top"])
    with pytest.raises(RuntimeError, match="held by the process writing the output"):
        fieldOutput.getResultHistory()


#: On process 1 only, the first element set of the mesh is moved behind the others: the same sets,
#: in another order -- which changes the place of a load contribution in the order of all of them.
_ELEMENT_SET_ORDER_PATCH = """
from edelweissfe.domaindecomposition.domaindecomposition import DomainDecomposition

requireSameLayout = DomainDecomposition._requireSameLayout


def reorderedInProcessOne(self, model, *args):
    if rank == 1:
        sets = model.mesh.elementSets
        first = next(iter(sets))
        sets[first] = sets.pop(first)
    return requireSameLayout(self, model, *args)


DomainDecomposition._requireSameLayout = reorderedInProcessOne
"""

#: On process 1 only, the first surface of the mesh is moved behind the others.
_SURFACE_ORDER_PATCH = _ELEMENT_SET_ORDER_PATCH.replace("model.mesh.elementSets", "model.mesh.surfaces")

#: Two patches bonded by a tie, a multi-point constraint.
with open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "testfiles",
        "mpi",
        "edelweiss-only",
        "TieNED",
        "test.inp",
    )
) as _deck:
    _TIE_DECK = _deck.read()

#: On process 1 only, one weight of the tie differs by an ulp, as the tie records it.
_MPC_WEIGHT_PATCH = """
import numpy as np

from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase

collect = NonlinearSolverBase._collectMultiPointConstraintRecords


def perturbedInProcessOne(self, model, dofLayout):
    records = collect(self, model, dofLayout)
    if rank == 1:
        slave, masters = records[0]
        master, weight = masters[0]
        records = [(slave, [(master, np.nextafter(weight, np.inf))] + list(masters[1:]))] + list(records[1:])
    return records


NonlinearSolverBase._collectMultiPointConstraintRecords = perturbedInProcessOne
"""


@pytest.mark.parametrize(
    "deck, patch",
    [
        (_FIELD_OUTPUT_DECK, _ELEMENT_SET_ORDER_PATCH),
        (_FIELD_OUTPUT_DECK, _SURFACE_ORDER_PATCH),
        (_TIE_DECK, _MPC_WEIGHT_PATCH),
    ],
    ids=["element set order", "surface order", "multi-point constraint weight"],
)
def test_processes_holding_different_models_are_refused(tmp_path, deck, patch):
    # The order of the element sets decides the order the loads are added in, and the weights of a
    # multi-point constraint its fold and slave motion: a process differing in either would compute
    # another model, silently.
    (tmp_path / "test.inp").write_text(deck)
    (tmp_path / "run.py").write_text(_FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", patch))
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("processes holding different models left each other waiting")

    assert exitCode != 0, output
    assert "The processes hold different models" in output, output


#: A material failing to integrate at 1.6 times the stable time step: increment 5 fails on every
#: process together (the refused cutback), after an increment that wrote no output.
_FAILING_STEP_DECK = """
*material, name=VonMises, id=myMaterial, provider=edelweiss
210000, 0.3, 550, 1000, 200, 1400, 1
*section, name=section1, material=myMaterial, type=plane, thickness=1
all
*job, name=failingstepjob, domain=2d
*fieldOutput
>>perNode, name=U, elSet=all, field=displacement, result=U
>>perElement, name=S, elSet=all, result=stress, quadraturePoint=0
*solver, solver=SOLVER, name=theSolver
output-frequency=3
courant-number=1.6
*modelGenerator, generator=planeRectQuad, name=gen
l=10
h=10
nX=6
nY=6
elType=CPE4
elProvider=edelweiss
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1e0, minInc=1e-2, maxNumInc=100000, maxIter=25, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.5
"""

#: Runs the deck in this directory, and saves on rank 0 what the job leaves: the solution, the
#: velocity and the net force of every node field, and the last result of every field output.
_FINAL_STATE_SCRIPT = """
import contextlib, io
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import processRank
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

journal = io.StringIO()
with contextlib.redirect_stdout(journal):
    model, fieldOutputs = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
failures = [line.strip("> <").split("feCore")[0].strip() for line in journal.getvalue().splitlines() if "failed" in line]
print("PROCESS", processRank(), failures, flush=True)
if processRank() == 0:
    final = {name + " " + entry: field[entry] for name, field in model.nodeFields.items() for entry in ("U", "V", "P")}
    final |= {name: np.asarray(output.getLastResult()) for name, output in fieldOutputs.fieldOutputs.items()}
    np.savez("final.npz", **final)
"""


def _finalState(directory, nProcesses: int | None) -> dict:
    """Run :data:`_FINAL_STATE_SCRIPT` in a directory, serially or under the launcher, and return
    what the job left, as bytes."""

    if nProcesses is None:
        environment = dict(
            os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
        )
        subprocess.run([sys.executable, "run.py"], cwd=directory, env=environment, check=True, timeout=180)
    else:
        output, exitCode = _runUnderMPI(directory, nProcesses)
        if output is None:
            pytest.fail("the job on {:} processes did not end".format(nProcesses))
        assert exitCode == 0, output
    with np.load(directory / "final.npz") as final:
        return {name: final[name].tobytes() for name in final.files}


@pytest.mark.parametrize("wholeModel", [False, True], ids=["distributed", "whole model"])
@pytest.mark.parametrize("nProcesses", [2, 3])
def test_a_step_failing_on_every_process_leaves_the_state_of_a_serial_run(tmp_path, nProcesses, wholeModel):
    # The failed increment changed the solution, velocity and force of each process at the degrees of
    # freedom it integrates; the job must end with the last accepted increment everywhere, as the serial
    # run does, also where the field outputs read the model a last time at the end of the step.
    serial, decomposed = tmp_path / "serial", tmp_path / "decomposed"
    for directory, solver in ((serial, "NED"), (decomposed, "NEDMPI")):
        directory.mkdir()
        deck = _FAILING_STEP_DECK.replace("SOLVER", solver)
        if wholeModel:
            # code run while the mesh is described: every process holds the whole model
            deck += "*modelGenerator, generator=executePythonCode, name=nothing\npass\n"
        (directory / "test.inp").write_text(deck)
        (directory / "run.py").write_text(_FINAL_STATE_SCRIPT)

    expected = _finalState(serial, None)
    assert expected and all(expected.values())
    result = _finalState(decomposed, nProcesses)
    assert result.keys() == expected.keys()
    assert [name for name in expected if result[name] != expected[name]] == []


#: A step action failing at the start of the step, in process 1 alone, outside any step the processes
#: agree on: the others go on into the first exchange of the step, while the failing process unwinds
#: through the end of the step, whose field outputs gather element results.
_STEP_START_FAILING_ALONE_PATCH = """
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI

applyStepActionsAtStepStart = NEDMPI.applyStepActionsAtStepStart


def failingInProcessOne(self, model, step):
    if rank == 1:
        raise RuntimeError("a step action failed in process 1 alone")
    return applyStepActionsAtStepStart(self, model, step)


NEDMPI.applyStepActionsAtStepStart = failingInProcessOne
"""


def test_a_failure_in_one_process_alone_aborts_before_the_end_of_the_step_communicates(tmp_path):
    (tmp_path / "test.inp").write_text(_FIELD_OUTPUT_DECK.replace("output-frequency=2", "output-frequency=1"))
    # The job is run as the test runner runs one: an exception leaving it is caught, and the next job
    # would follow. So the process failing alone must stop the others itself, before it unwinds through
    # anything that communicates.
    script = _FAILING_IN_PROCESS_ONE_SCRIPT.replace("PATCH", _STEP_START_FAILING_ALONE_PATCH).replace(
        '    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)',
        "    try:\n"
        '        finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)\n'
        "    except Exception as exception:\n"
        '        print("CAUGHT", exception)',
    )
    assert "CAUGHT" in script
    (tmp_path / "run.py").write_text(script)
    output, exitCode = _runUnderMPI(tmp_path, timeout=60)
    if output is None:
        pytest.fail("a failure in one process alone left the others waiting")

    assert exitCode != 0, output
    assert "a step action failed in process 1 alone" in output, output
    assert "in MPI process 1 of 3; aborting all processes" in output, output


#: Increment 3, the first after elements moved between the processes on output increment 2, fails on
#: every process together, before it changes anything.
_FAIL_AFTER_A_MIGRATION_PATCH = """
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.exceptions import CutbackRequest

solveIncrement = NED.solveIncrement


def failingInIncrementThree(self, U, dU, V, P, Minv, stepActions, model, timeStep, prevTimeStep):
    if timeStep.number == 3:
        raise CutbackRequest("the increment after the migration", 0.5)
    return solveIncrement(self, U, dU, V, P, Minv, stepActions, model, timeStep, prevTimeStep)


NED.solveIncrement = failingInIncrementThree
"""

with open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "testfiles",
        "mpi",
        "edelweiss-only",
        "NEDContact",
        "test.inp",
    )
) as _deck:
    # the upper block slides across the lower one, so that the contact searches move the constraint
    _SLIDING_CONTACT_DECK = (
        _deck.read()
        .replace("nSet=upper_front, field=displacement, 3=-0.2", "nSet=upper_front, field=displacement, 1=0.8, 3=-0.2")
        .replace(
            "nSet=upper_bottomLeftBack, field=displacement, 1=0.0, 2=0.0",
            "nSet=upper_bottomLeftBack, field=displacement, 2=0.0",
        )
    )


def test_a_step_failing_after_elements_moved_leaves_the_state_of_a_serial_run(tmp_path):
    # After a migration a process integrates degrees of freedom it did not before, completed in its
    # vectors from their previous owners; the end of a failed step reads them back from the node fields,
    # which must therefore hold them too.
    deck = _MIGRATING_DECK.replace("solver=NEDMPI", "solver=SOLVER").replace(
        "load-balance-costs=elementNumber", "LOADBALANCE"
    )
    patch = _FAIL_AFTER_A_MIGRATION_PATCH
    serial, decomposed = tmp_path / "serial", tmp_path / "decomposed"
    for directory, solver, loadBalance in (
        (serial, "NED", ""),
        (decomposed, "NEDMPI", "load-balance-costs=elementNumber\nload-balance-tolerance=0.1"),
    ):
        directory.mkdir()
        text = deck.replace("SOLVER", solver).replace("LOADBALANCE", loadBalance)
        (directory / "test.inp").write_text(text)
        (directory / "run.py").write_text(
            _FINAL_STATE_SCRIPT.replace("journal = io.StringIO()", patch + "\njournal = io.StringIO()")
        )

    expected = _finalState(serial, None)
    result = _finalState(decomposed, 3)
    assert result.keys() == expected.keys()
    assert [name for name in expected if result[name] != expected[name]] == []


#: After every rebuild of the equation system with the vectors carried over -- here a contact search
#: moving the constraint --, every process compares its node fields with its vectors at the degrees of
#: freedom it integrates, and prints how many entries differ.
_NODE_FIELDS_AFTER_A_REBUILD_SCRIPT = """
import contextlib, io
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import processRank
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamicmpi import NEDMPI
from edelweissfe.utils.inputfileparser import parseInputFile

buildEquationSystem = NEDMPI.buildEquationSystem
differing = []


def checkedRebuild(self, model, step, previous=None):
    system = buildEquationSystem(self, model, step, previous)
    if previous is not None:
        count = 0
        for slot in self._integratedNodeFieldSlots:
            for vector, entry in ((system.U, "U"), (system.V, "V"), (system.P, "P")):
                published = slot.nodeField[entry].reshape(-1)[slot.positions]
                count += int(np.count_nonzero(published != np.asarray(vector)[slot.dofs]))
        differing.append(count)
    return system


NEDMPI.buildEquationSystem = checkedRebuild
with contextlib.redirect_stdout(io.StringIO()):
    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
print("PROCESS", processRank(), len(differing), sum(differing), flush=True)
"""


def test_the_node_fields_hold_what_a_process_integrates_after_a_contact_search_moved_a_constraint(tmp_path):
    # A process newly integrating degrees of freedom after the search completes its vectors there from
    # their previous owners; the node fields must hold them too, since the end of a failed step reads
    # the last accepted increment back from them (NEDMPI._synchronizeAcceptedModel).
    deck = re.sub(r"^(\*solver,.*NEDMPI.*)$", r"\1\nload-balance-tolerance=0", _SLIDING_CONTACT_DECK, flags=re.M)
    (tmp_path / "test.inp").write_text(deck)
    (tmp_path / "run.py").write_text(_NODE_FIELDS_AFTER_A_REBUILD_SCRIPT)
    output, exitCode = _runUnderMPI(tmp_path)
    if output is None:
        pytest.fail("the job did not end")
    assert exitCode == 0, output

    reports = sorted(line.split()[1:] for line in output.splitlines() if line.startswith("PROCESS"))
    assert [rank for rank, _, _ in reports] == ["0", "1", "2"], output
    assert all(int(rebuilds) > 0 for _, rebuilds, _ in reports), output
    assert [int(differing) for _, _, differing in reports] == [0, 0, 0], output


_EXPRESSION_HISTORIES_SCRIPT = """
import contextlib, io, sys
import numpy as np
from edelweissfe.domaindecomposition.mpienvironment import isRootProcess
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

with contextlib.redirect_stdout(io.StringIO()):
    _, fieldOutputs = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
if isRootProcess():
    names = sys.argv[2:]
    np.savez(sys.argv[1], **{name: np.asarray(fieldOutputs.fieldOutputs[name].result, dtype=float) for name in names})
"""

#: Contact decks of testfiles/mpi, a node set of one node in each (an expression field output must name
#: a set, and its result is reshaped to one row per node), and expression field outputs reading the
#: results of the contact constraint's last evaluation, which only the process evaluating it computes.
_ONE_NODE_SET = {
    "NEDContact": "upper_bottomLeftBack",
    "NEDSurfaceContact": "upper_bottomLeftBack",
    "NEDSurfaceToDiscreteRigidBodyContact": "support_rp",
    "NEDNodeToDiscreteRigidBodyContact": "cylinder_rp",
}
_CONTACT_RESULTS_READ_BY_OUTPUTS = {
    "NEDContact": {
        "pressures": 'model.constraints["contact"].getNormalPressures()',
        "total": 'np.array([model.constraints["contact"].totalNormalForce])',
    },
    "NEDSurfaceContact": {
        "pressures": 'model.constraints["contact"].getNormalPressures()',
        "nodalForces": 'model.constraints["contact"].getSlaveNodalNormalForces()',
    },
    "NEDSurfaceToDiscreteRigidBodyContact": {
        "pressures": 'model.constraints["contact"].getNormalPressures()',
        "gaps": 'model.constraints["contact"].getGaps()',
        "total": 'np.array([model.constraints["contact"].totalNormalForce])',
    },
    "NEDNodeToDiscreteRigidBodyContact": {
        "total": 'np.array([model.constraints["contact"].totalNormalForce])',
    },
}


@pytest.mark.parametrize("deck", sorted(_CONTACT_RESULTS_READ_BY_OUTPUTS))
def test_an_output_reads_the_contact_results_of_the_process_evaluating_the_contact(tmp_path, deck):
    # The normal forces and gaps of a contact are no state -- a checkpoint does not carry them -- but
    # an expression field output reads them on rank 0. With 2 processes the contact is evaluated by
    # rank 1, and rank 0's copy held zeros (reproduced: every history all zeros) until the results
    # travelled with the state of the constraint (ConstraintBase.outputResults).
    pytest.importorskip("mpi4py.MPI")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no MPI launcher")
    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    environment = dict(
        os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)), OMP_NUM_THREADS="1"
    )
    expressions = _CONTACT_RESULTS_READ_BY_OUTPUTS[deck]
    outputs = "".join(
        ">>fromExpression, name={:}, nSet={:}, expression='{:}', saveHistory=True\n".format(
            name, _ONE_NODE_SET[deck], expression
        )
        for name, expression in expressions.items()
    )
    histories = {}
    for solver, launcher in (("NED", []), ("NEDMPI", [mpirun, "--bind-to", "none", "-n", "2"])):
        directory = tmp_path / solver
        shutil.copytree(os.path.join(testfiles, "edelweiss-only", deck), directory)
        text = (directory / "test.inp").read_text().replace("solver=NEDMPI", "solver=" + solver)
        text = re.sub(r"^\*fieldOutput\n", "*fieldOutput\n" + outputs, text, count=1, flags=re.M)
        (directory / "test.inp").write_text(text)
        (directory / "run.py").write_text(_EXPRESSION_HISTORIES_SCRIPT)
        result = tmp_path / (solver + ".npz")
        output = subprocess.run(
            launcher + [sys.executable, "run.py", str(result)] + list(expressions),
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.exists(), output.stdout + output.stderr
        histories[solver] = np.load(result)

    for name in expressions:
        serial, decomposed = histories["NED"][name], histories["NEDMPI"][name]
        assert np.any(serial), "{:}: the serial run reports no contact".format(name)
        assert serial.shape == decomposed.shape and serial.tobytes() == decomposed.tobytes(), name


#: Runs a deck, recording every kinetic energy and momentum the solver forms -- the numbers of the energy
#: table and of the topology-change report -- as exact hexadecimal strings, one file per process.
_RECORDING_KINETIC_ENERGY_SCRIPT = """
import contextlib, io, json
from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.inputfileparser import parseInputFile

communicator = worldCommunicator()
rank = 0 if communicator is None else communicator.Get_rank()
recorded = []
kineticEnergy, momentum = NED.kineticEnergy, NED.momentum


def recordedKineticEnergy(self, V, dofs=None):
    value = kineticEnergy(self, V, dofs)
    recorded.append(["kineticEnergy", float(value).hex()])
    return value


def recordedMomentum(self, V, model):
    value = momentum(self, V, model)
    recorded.append(["momentum", [float(component).hex() for component in value]])
    return value


NED.kineticEnergy, NED.momentum = recordedKineticEnergy, recordedMomentum
with contextlib.redirect_stdout(io.StringIO()):
    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
with open("recorded_{:}.json".format(rank), "w") as file:
    json.dump(recorded, file)
"""


@pytest.mark.parametrize("deck", ["edelweiss-only/NED", "marmot/NEDLiveAMR"])
def test_the_kinetic_energy_and_the_momentum_are_the_same_bits_on_any_number_of_processes(tmp_path, deck):
    # The energy table every output increment, and -- in the refined deck -- the topology-change report
    # before and after the refinement: exact sums, so the same bits serially and over 2 and 3 processes.
    if deck.startswith("marmot"):
        from edelweissfe.utils.misc import checkSuccessfulExtension

        if not checkSuccessfulExtension("edelweissfe.elements.marmotelement.element"):
            pytest.skip("the deck needs Marmot elements")
    testfiles = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles", "mpi")
    records = {}
    for nProcesses in (1, 2, 3):
        directory = tmp_path / str(nProcesses)
        shutil.copytree(os.path.join(testfiles, deck), directory)
        if nProcesses == 1:
            deckText = (directory / "test.inp").read_text().replace("solver=NEDMPI", "solver=NED")
            (directory / "test.inp").write_text(
                "\n".join(line for line in deckText.splitlines() if not line.startswith("load-balance-"))
            )
        (directory / "run.py").write_text(_RECORDING_KINETIC_ENERGY_SCRIPT)
        output, exitCode = _runUnderMPI(directory, nProcesses)
        assert output is not None and exitCode == 0, output
        records[nProcesses] = [
            json.loads((directory / "recorded_{:}.json".format(rank)).read_text()) for rank in range(nProcesses)
        ]

    serial = records[1][0]
    kinds = {kind for kind, _ in serial}
    assert "kineticEnergy" in kinds and (deck.endswith("AMR") <= ("momentum" in kinds)), serial
    for nProcesses in (2, 3):
        assert records[nProcesses] == [serial] * nProcesses, nProcesses


def test_a_constraint_is_evaluated_by_the_process_holding_most_of_its_nodes():
    from edelweissfe.domaindecomposition.partitioning import assignConstraintsByMajority
    from edelweissfe.domaindecomposition.subdomaininterface import (
        ElementProcessesAtNodes,
    )
    from edelweissfe.models.mesh import ElementGroup, ElementTypeInfo, MeshConnectivity

    # Four two-node elements in a row, on processes 2, 1, 1, 0; node 5 belongs to no element.
    typeInfo = ElementTypeInfo([["displacement"], ["displacement"]], "bar2", 2, True)
    connectivity = MeshConnectivity(
        np.arange(4), [ElementGroup(typeInfo, np.arange(4), np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))], 0
    )
    lowest = ElementProcessesAtNodes.ofMesh(connectivity, np.array([2, 1, 1, 0]), 6, 3).lowestProcessAtNodes()
    assert lowest.tolist() == [2, 1, 1, 0, 0, 3]

    # In name order: "a tie of 1 and 2" (candidates 1, 2), "majority 0", "most on 1", "on no element"
    # (candidates 0, 1, 2) -- ties and node-less constraints dealt from the highest candidate down.
    nodeRows = {"most on 1": [1, 2, 3], "a tie of 1 and 2": [0, 2], "on no element": [5], "majority 0": [3, 4, 5]}
    owners = assignConstraintsByMajority(nodeRows, lowest, 3)
    assert owners == {"most on 1": 1, "a tie of 1 and 2": 2, "on no element": 2, "majority 0": 0}
