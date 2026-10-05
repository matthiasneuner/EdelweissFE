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


from dataclasses import dataclass
from time import perf_counter

import numpy as np

from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    chunked_iterable,
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from edelweissfe.timesteppers.timestep import TimeStep


def computeElementsInParallel(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, K: VIJSystemMatrix, F: DofVector, timeStep: TimeStep
) -> tuple[DofVector, VIJSystemMatrix, DofVector]:
    """
    Compute the elements in parallel for quasi-static anlysis.

    Parameters
    ----------
    elements : dict
        The elements to compute.
    Un1 : DofVector
        The displacement vector.
    dU : DofVector
        The displacement increment vector.
    P : DofVector
        The internal force vector.
    K : VIJSystemMatrix
        The stiffness matrix.
    F : DofVector
        The flux vector.
    timeStep : TimeStep
        The time step.

    Returns
    -------
    P : DofVector
        The internal force vector.
    K : VIJSystemMatrix
        The stiffness matrix.
    F : DofVector
        The flux vector.
    """

    scatter_P = (
        P.createScatterVector()
    )  # make a scatter vector; which gives 1) contiguous memory access and 2) thread safety

    time = timeStep.totalTime
    dT = timeStep.timeIncrement

    # Process a CHUNK of elements per task, not just one, to keep the per-task
    # dispatch overhead negligible compared to the actual element computation.
    def computeElementsWorker(elementChunk):
        for element in elementChunk:
            Pe = scatter_P[element]
            Ue = Un1[element]
            dUe = dU[element]
            Ke = K[element]
            element.computeKernels(Ke, Pe, Ue, dUe, time, dT)

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    if numThreads == 1:
        # avoid ThreadPoolExecutor/task dispatch overhead when there is nothing to parallelize
        computeElementsWorker(elements.values())
    else:
        chunkSize = max(1, len(elements) // (numThreads * 4))
        chunks = chunked_iterable(elements.values(), chunkSize)

        executor = getThreadPool(numThreads)
        list(executor.map(computeElementsWorker, chunks))

    scatter_P.assembleInto(P)
    scatter_P.assembleInto(F, absolute=True)

    return P, K, F


#: How many chunks of elements each thread is handed in the explicit element loop, so that the
#: executor can balance unequal chunks dynamically; see :func:`planElements`. Raising it costs one
#: plan entry and one buffer allocation per chunk, and buys finer balancing, so it pays until the
#: chunks get small enough for that per-chunk cost to show.
_chunksPerThread = 16

#: The chunk size of a single-threaded loop: large enough that the per-chunk cost vanishes, small
#: enough that a chunk's gathered buffers stay in cache.
_chunkSizeOfOneThread = 4000


@dataclass(frozen=True)
class ElementChunk:
    """A run of consecutive elements of an :class:`ElementPlan`, gathered and evaluated together.

    Parameters
    ----------
    elements
        The elements of the chunk, in order.
    dofIndices
        The concatenation of the chunk's elements' degree-of-freedom indices.
    offsets
        Where each element's slice of ``dofIndices`` begins; one entry more than elements.
    begin
        Where the chunk's entries begin in the plan's contribution buffer.
    end
        Where they end.
    firstElement
        The position of the chunk's first element among all elements of the plan.
    """

    elements: tuple
    dofIndices: np.ndarray
    offsets: np.ndarray
    begin: int
    end: int
    firstElement: int


@dataclass(frozen=True)
class ElementPlan:
    """How a set of elements gathers its degrees of freedom and assembles what it contributes.

    The elements are cut into chunks; each chunk gathers its solution in one indexed read, evaluates
    its elements into a buffer of its own, and places that buffer into the plan's *contribution
    buffer* -- one entry per element and degree of freedom, element after element, in the order of
    the elements. Summing that buffer into a vector, in that order, is the assembly: the sum at every
    degree of freedom is formed in element order, whatever the number of threads or chunks.

    The plan assembles into the degrees of freedom a model partition integrates, never into more:
    a process computing one part of a model of many would otherwise pay, every increment, for the
    entities it does not compute. Valid for one equation system and one partition; built afresh
    with either.

    Parameters
    ----------
    elements
        The elements, by number, in the order they are chunked and summed.
    chunks
        The chunks.
    dofs
        The degrees of freedom assembled into, as an index into a vector of the whole model: a
        slice, or a sorted index array.
    nDofs
        How many degrees of freedom ``dofs`` selects.
    entryDofs
        The degree of freedom of every entry of the contribution buffer.
    positions
        The position of every entry's degree of freedom among ``dofs``.
    nThreads
        The number of threads the plan is evaluated with.
    """

    elements: dict
    chunks: list[ElementChunk]
    dofs: slice | np.ndarray
    nDofs: int
    entryDofs: np.ndarray
    positions: np.ndarray
    nThreads: int

    @property
    def nEntries(self) -> int:
        """The length of the contribution buffer."""

        return self.entryDofs.shape[0]

    def assembleInto(self, contributions: np.ndarray, vector: DofVector):
        """Add the contribution buffer into ``vector`` at the plan's degrees of freedom, summing the
        entries at each degree of freedom in element order.

        Parameters
        ----------
        contributions
            The contribution buffer.
        vector
            The vector to add into.
        """

        vector.asPlainArray()[self.dofs] += np.bincount(self.positions, weights=contributions, minlength=self.nDofs)


def planElements(
    elements: dict, entitiesInDofVector: dict, dofs: slice | np.ndarray, nDof: int, nThreads: int
) -> ElementPlan:
    """Build the gather and assembly plan of a set of elements.

    Parameters
    ----------
    elements
        The elements, by number.
    entitiesInDofVector
        The entity-to-DOF-index mapping of the current equation system.
    dofs
        The degrees of freedom assembled into (every element DOF must be among them), as an index
        into a vector of the whole model: ``slice(None)`` for all of them, or a sorted index array.
    nDof
        The size of the equation system.
    nThreads
        The number of threads the plan will be evaluated with; it sets the chunk size.

    Returns
    -------
    ElementPlan
        The plan.
    """

    # Many small chunks per thread rather than one large one each, because the elements are not
    # equally expensive: a quadrature point that is yielding or damaging pays for a return mapping
    # that an elastic one does not, so the elements at a propagating front cost several times what
    # the bulk costs. Those elements sit next to each other in the mesh, and therefore next to each
    # other in element order, so chunks cut from that order are systematically unequal -- and with
    # one chunk per thread the whole map waits for whichever thread drew the front. Handing the
    # executor many more chunks than it has threads lets it even that out as it goes.
    if nThreads > 1:
        chunkSize = max(1, len(elements) // (nThreads * _chunksPerThread))
    else:
        chunkSize = min(max(len(elements), 1), _chunkSizeOfOneThread)

    chunks = []
    begin = 0
    firstElement = 0
    for chunk in chunked_iterable(elements.values(), chunkSize):
        indicesPerElement = [entitiesInDofVector[element] for element in chunk]
        offsets = np.zeros(len(chunk) + 1, dtype=np.intp)
        np.cumsum([len(indices) for indices in indicesPerElement], out=offsets[1:])
        end = begin + int(offsets[-1])
        chunks.append(
            ElementChunk(
                chunk,
                np.concatenate(indicesPerElement),
                offsets,
                begin,
                end,
                firstElement,
            )
        )
        begin = end
        firstElement += len(chunk)

    entryDofs = np.concatenate([chunk.dofIndices for chunk in chunks]) if chunks else np.empty(0, dtype=np.int64)

    selected = np.arange(nDof)[dofs]
    if selected.shape[0] == nDof:
        # every degree of freedom, in order: each is its own position
        positions = entryDofs
    else:
        positions = np.searchsorted(selected, entryDofs)
    if entryDofs.size and not np.array_equal(selected[positions], entryDofs):
        raise ValueError("An element has a degree of freedom that is not among those its plan assembles into.")

    return ElementPlan(elements, chunks, dofs, selected.shape[0], entryDofs, positions, nThreads)


def _evaluateExplicitKernels(
    chunk: ElementChunk,
    Pe: np.ndarray,
    gatheredU: np.ndarray,
    gatheredDU: np.ndarray,
    time: float,
    dT: float,
    psi: float,
    costs: np.ndarray = None,
) -> float:
    """Evaluate the explicit kernels of one chunk of elements into the chunk's own force buffer.

    Parameters
    ----------
    chunk
        The chunk.
    Pe
        The chunk's force buffer, written into; zero on entry.
    gatheredU
        The chunk's solution, gathered in the same layout.
    gatheredDU
        The chunk's solution increment, gathered in the same layout.
    time
        The current time.
    dT
        The time increment.
    psi
        The internal energy to add the chunk's elements' to.
    costs
        If given, the wall-clock time of each element's kernels is added to its entry -- one entry
        per element of the chunk, in order.

    Returns
    -------
    float
        ``psi`` plus the internal energy the chunk's elements report.
    """

    offsets = chunk.offsets
    for position, element in enumerate(chunk.elements):
        begin = offsets[position]
        end = offsets[position + 1]

        if costs is not None:
            tic = perf_counter()
        element.computeKernelsExplicit(Pe[begin:end], gatheredU[begin:end], gatheredDU[begin:end], time, dT)
        psi += element.computeInternalEnergy()
        if costs is not None:
            costs[position] += perf_counter() - tic

    return psi


def computeElementsForExplicit(
    plan: ElementPlan,
    Un1: DofVector,
    dU: DofVector,
    P: DofVector,
    timeStep: TimeStep,
    elementCosts: np.ndarray = None,
) -> tuple[float, np.ndarray]:
    """Evaluate the explicit kernels of a plan's elements, on the plan's threads, and add their
    forces into ``P`` at the plan's degrees of freedom.

    Every chunk works on buffers of its own: the gathered solution and increment, and a force buffer
    its elements write into, which reaches the shared contribution buffer in one assignment once the
    chunk is done. Nothing shared is touched per element, which is what lets the loop scale with the
    GIL disabled: a view into one shared buffer holds a reference to it, and every element would make
    all the cores take turns updating its single reference count.

    Parameters
    ----------
    plan
        The plan; see :func:`planElements`.
    Un1
        The solution vector.
    dU
        The solution increment vector.
    P
        The internal force vector, assembled into.
    timeStep
        The time step.
    elementCosts
        If given, the wall-clock time of each element's kernels is added to its entry, one per
        element of the plan in order; what a load balancer weighs the elements by.

    Returns
    -------
    tuple[float, np.ndarray]
        The summed internal energy, and the contribution buffer the forces were assembled from -- one
        entry per element and degree of freedom, in plan order -- which completing the forces at the
        interface of a model partition needs.
    """

    time = timeStep.totalTime
    dT = timeStep.timeIncrement

    Un1_plain = Un1.asPlainArray()
    dU_plain = dU.asPlainArray()
    forces = np.empty(plan.nEntries)

    def compute_chunk(chunk: ElementChunk, psi: float = 0.0) -> float:
        Pe = np.zeros(chunk.end - chunk.begin)
        psi = _evaluateExplicitKernels(
            chunk,
            Pe,
            Un1_plain[chunk.dofIndices],
            dU_plain[chunk.dofIndices],
            time,
            dT,
            psi,
            (
                None
                if elementCosts is None
                else elementCosts[chunk.firstElement : chunk.firstElement + len(chunk.elements)]
            ),
        )
        forces[chunk.begin : chunk.end] = Pe
        return psi

    if plan.nThreads == 1:
        # One running sum, element after element, as a loop without chunks would form it; and no
        # executor dispatch when there is nothing to parallelize.
        psi_total = 0.0
        for chunk in plan.chunks:
            psi_total = compute_chunk(chunk, psi_total)
    else:
        psi_total = sum(getThreadPool(plan.nThreads).map(compute_chunk, plan.chunks))

    plan.assembleInto(forces, P)

    return psi_total, forces


def computeLumpedDiagonalForExplicit(plan: ElementPlan, elementContribution, vector: DofVector) -> np.ndarray:
    """Assemble a lumped operator -- the inertia or the damping -- of a plan's elements into ``vector``
    at the plan's degrees of freedom.

    Parameters
    ----------
    plan
        The plan; see :func:`planElements`.
    elementContribution
        ``elementContribution(element, Ve)`` writes an element's diagonal into the zero ``Ve``.
    vector
        The vector to add into.

    Returns
    -------
    np.ndarray
        The contribution buffer it was assembled from, as :func:`computeElementsForExplicit` returns.
    """

    contributions = np.zeros(plan.nEntries)
    for chunk in plan.chunks:
        for position, element in enumerate(chunk.elements):
            begin = chunk.begin + chunk.offsets[position]
            end = chunk.begin + chunk.offsets[position + 1]
            elementContribution(element, contributions[begin:end])

    plan.assembleInto(contributions, vector)
    return contributions
