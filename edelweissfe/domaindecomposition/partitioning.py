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
"""Which process computes which element and which constraint.

The elements are partitioned on the mesh (:class:`~edelweissfe.models.mesh.Mesh`) -- its element
numbers, connectivity and element types -- and not on element objects, so that a partition can be
made before any element exists, and every process can make it without holding every element. METIS
runs on the root process, which broadcasts the result, so every process holds the same partition
regardless of whether METIS itself would reproduce it. The partition covers every element of the
mesh. The auxiliary elements -- contact facets, the point masses of rigid bodies, which
every process makes itself -- are not given to METIS: each is computed by the process computing its
host element (the solid element a facet lies on, whose degrees of freedom are the facet's), or, if it
has none, by the process given part 0 of the partition (:func:`processOfAuxiliaryElement`). Their
owner is still unique.

Constraints are not partitioned by geometry: each is one object, evaluated by one process, and is
dealt out round-robin in name order. The assignment depends on the constraint names only, so it
survives a topology change -- a stateful constraint keeps the process that holds its authoritative
state. A process evaluating a costly constraint is given correspondingly fewer elements instead: the
elements are partitioned into parts of unequal shares (:func:`elementSharesBesideConstraints`), and
part *p* is computed by process *p*.
"""

import numpy as np

from edelweissfe.domaindecomposition.metis import partitionMeshDual
from edelweissfe.models.mesh import ElementTypeInfo, Mesh, MeshElement

#: The smallest share of the elements' work a process is given, as a fraction of an equal share
#: (:func:`elementSharesBesideConstraints`). METIS bisects recursively, and a part of a much smaller
#: share than the others comes out empty -- on a 64 x 64 grid in 32 parts already at a tenth of an
#: equal share --, while METIS still reports success; at half an equal share every part kept a
#: quarter of an equal part's elements in every test. A process without elements is never intended.
MINIMUM_SHARE_OF_AN_EQUAL_SHARE = 0.5

#: By how much, as a fraction of an equal share, the share a part was made for and the share of the
#: process computing it may differ when a repartition keeps elements where they were
#: (:func:`keepElementsWhereTheyWere`). The speed of the processes alone scatters by 5--8 % for the
#: same element (measured on rabbit, 2026-10-07), so shares closer than that are not told apart by
#: the measurement they come from.
INTERCHANGEABLE_SHARES = 0.05


def _elementWeight(typeInfo: ElementTypeInfo) -> int:
    """The expected cost of one element's increment, as a positive integer.

    Its number of degrees of freedom: a proxy for both its quadrature-point count and its kernel's
    arithmetic, and zero-cost facets (no kernels) weigh 1 so that METIS still places them.
    """

    return max(1, typeInfo.nDof) if typeInfo.hasKernels else 1


def partitionElementsOfMesh(
    mesh: Mesh,
    nParts: int,
    domainSize: int,
    communicator,
    measuredCosts: dict = None,
    processShares: np.ndarray | None = None,
) -> dict:
    """Assign every element of the mesh to one of ``nParts`` processes.

    Parameters
    ----------
    mesh
        The mesh; its elements are partitioned in mesh order.
    nParts
        The number of processes.
    domainSize
        The spatial dimension; elements are adjacent when they share at least this many nodes (a
        face in 3D, an edge in 2D).
    communicator
        The communicator; the partition is computed on rank 0 and broadcast.
    measuredCosts
        The measured cost of elements, by element number, as seen on rank 0; where given, it
        replaces the estimate. An element without a measurement -- a contact facet, a child created
        since -- is weighed by the median measured cost per degree of freedom.
    processShares
        The share of the weight of the elements each process is to receive, by rank (see
        :func:`elementSharesBesideConstraints`); None for equal shares.

    Returns
    -------
    dict
        The rank of every element, by element number.
    """

    numbers = [number for number, record in mesh.elements.items() if not record.isAuxiliary]
    parts = np.zeros(len(numbers), dtype=np.int64)

    if nParts > 1:
        if communicator.Get_rank() == 0:
            typeInfos = [mesh.typeOf(mesh.elements[number]) for number in numbers]
            nodeIndex = {}
            offsets = np.zeros(len(numbers) + 1, dtype=np.int64)
            connectivity = []
            for position, number in enumerate(numbers):
                for label in mesh.elements[number].nodeLabels:
                    connectivity.append(nodeIndex.setdefault(label, len(nodeIndex)))
                offsets[position + 1] = len(connectivity)

            if measuredCosts:
                weights = _measuredWeights(numbers, typeInfos, measuredCosts)
            else:
                weights = np.array([_elementWeight(typeInfo) for typeInfo in typeInfos], dtype=np.int64)
            parts[:] = partitionMeshDual(
                offsets,
                np.array(connectivity, dtype=np.int64),
                len(nodeIndex),
                nParts,
                weights,
                max(1, domainSize),
                processShares,
            )

        communicator.Bcast(parts, root=0)

        # Shares are given no smaller than MINIMUM_SHARE_OF_AN_EQUAL_SHARE so that no part is empty;
        # checked, the same in every process, since every process holds the broadcast partition. (Equal
        # shares of a mesh of hardly more elements than processes may leave a process without
        # elements, as they always could.)
        emptyParts = np.flatnonzero(np.bincount(parts, minlength=nParts) == 0)
        if processShares is not None and emptyParts.size:
            raise RuntimeError(
                "the partition of {:} elements into {:} parts of shares {:} left part(s) {:} without elements".format(
                    len(numbers), nParts, np.round(processShares, 4).tolist(), emptyParts.tolist()
                )
            )

    owners = dict(zip(numbers, parts.tolist()))
    for number, record in mesh.elements.items():
        if record.isAuxiliary:
            owners[number] = processOfAuxiliaryElement(record, owners)
    # in mesh order, as the elements are listed everywhere else
    return {number: owners[number] for number in mesh.elements}


def processOfAuxiliaryElement(record: MeshElement, owners: dict) -> int:
    """The process computing an auxiliary element (a contact facet, the point mass of a
    rigid body): the process computing its host element, or part 0 if it has none (a point mass).
    That is rank 0 in a first partition; a repartition renumbers its parts to keep elements where
    they were (:func:`keepElementsWhereTheyWere`), and a point mass then moves with part 0.

    Parameters
    ----------
    record
        The element, as described in the mesh.
    owners
        The rank of every element of the mesh made from the mesh, by number.

    Returns
    -------
    int
        The rank (or, inside :func:`partitionElementsOfMesh`, the part).
    """

    if record.hostElement is None:
        return 0
    return owners[record.hostElement]


def keepElementsWhereTheyWere(
    owners: dict,
    previousOwners: dict,
    nParts: int,
    processShares: np.ndarray | None = None,
    shareTolerance: float = INTERCHANGEABLE_SHARES,
) -> dict:
    """Renumber the parts of a new partition so that as many elements as possible keep their process.

    METIS numbers the parts of a partition arbitrarily: a repartition close to the previous one may
    still give every part another number, and so move every element to another process. Each new part
    is given the number of the previous part it shares the most elements with, largest overlaps first;
    the parts left over are paired in the order of their shares. The parts themselves -- which
    elements are computed together -- do not change, only which process computes them.

    A part made for a share (``processShares``) is meant for its process: it may go to another
    process only if their shares differ by no more than ``shareTolerance`` of an equal share
    (:data:`INTERCHANGEABLE_SHARES`) -- a
    process evaluating a costly constraint keeps its smaller part, while the others, whose shares
    differ a little, are renumbered among themselves. If the parts left over cannot be paired within
    the tolerance, the parts are not renumbered at all: part *p* goes to process *p*, as made.

    Parameters
    ----------
    owners
        The rank of every element, by number, in the new partition.
    previousOwners
        The rank of every element, by number, in the previous partition; the same elements.
    nParts
        The number of processes.
    processShares
        The shares the new partition was made for, by part (= rank); None for equal shares.
    shareTolerance
        By how much, as a fraction of an equal share, the shares of a part and of the process it goes
        to may differ.

    Returns
    -------
    dict
        The new partition, its parts renumbered.
    """

    numbers = list(owners.keys())
    new = np.array([owners[number] for number in numbers], dtype=np.int64)
    previous = np.array([previousOwners[number] for number in numbers], dtype=np.int64)
    overlap = np.zeros((nParts, nParts), dtype=np.int64)
    np.add.at(overlap, (new, previous), 1)
    shares = np.full(nParts, 1.0 / nParts) if processShares is None else np.asarray(processShares)
    interchangeable = np.abs(shares[:, None] - shares[None, :]) <= shareTolerance / nParts + 1e-15

    renumbered = np.full(nParts, -1, dtype=np.int64)
    taken = np.zeros(nParts, dtype=bool)
    # Largest overlap first; ties in the order of the new, then the previous part, so that every
    # process arrives at the same renumbering.
    newParts, previousParts = np.unravel_index(np.argsort(-overlap, axis=None, kind="stable"), overlap.shape)
    for newPart, previousPart in zip(newParts, previousParts):
        if (
            renumbered[newPart] < 0
            and not taken[previousPart]
            and overlap[newPart, previousPart] > 0
            and interchangeable[newPart, previousPart]
        ):
            renumbered[newPart] = previousPart
            taken[previousPart] = True
    leftOver = np.flatnonzero(renumbered < 0)
    free = np.flatnonzero(~taken)
    renumbered[leftOver[np.argsort(shares[leftOver], kind="stable")]] = free[np.argsort(shares[free], kind="stable")]
    if not np.all(interchangeable[np.arange(nParts), renumbered]):
        return dict(owners)

    return dict(zip(numbers, renumbered[new].tolist()))


def elementSharesBesideConstraints(elementCost: float, constraintCosts: np.ndarray) -> np.ndarray | None:
    """The share of the elements' work each process is to receive, so that every process is equally
    busy with its elements and the constraints it evaluates -- as far as that is possible: no process
    is given less than :data:`MINIMUM_SHARE_OF_AN_EQUAL_SHARE` of an equal share, so that every process
    keeps elements -- a process whose constraints cost that much is busier than the others --, and
    the others share the rest of the elements among them.

    Parameters
    ----------
    elementCost
        The cost of all elements, e.g. their kernel time per increment summed over all processes.
    constraintCosts
        The cost of the constraints each process evaluates, in the same unit, by rank.

    Returns
    -------
    np.ndarray | None
        The shares, by rank, summing to 1; None if no process evaluates a costly constraint (equal
        shares).
    """

    constraintCosts = np.asarray(constraintCosts, dtype=float)
    nProcesses = constraintCosts.shape[0]
    if not np.any(constraintCosts > 0.0) or elementCost <= 0.0:
        return None

    # The level every process is filled to: of the processes below it, the elements and constraints
    # divided equally; a process whose constraints exceed the level is left out, and the level computed
    # again for the others.
    below = np.ones(nProcesses, dtype=bool)
    while True:
        level = (elementCost + constraintCosts[below].sum()) / below.sum()
        exceeding = below & (constraintCosts >= level)
        if not exceeding.any():
            break
        below &= ~exceeding

    shares = np.where(below, level - constraintCosts, 0.0) / elementCost
    # Raise the shares below the minimum to it, and scale the others down to make room, until none
    # falls below it.
    minimum = MINIMUM_SHARE_OF_AN_EQUAL_SHARE / nProcesses
    raised = np.zeros(nProcesses, dtype=bool)
    while True:
        raised |= shares < minimum
        shares = np.where(raised, minimum, shares * (1.0 - minimum * raised.sum()) / shares[~raised].sum())
        if not np.any(shares[~raised] < minimum):
            return shares


def _measuredWeights(numbers: list, typeInfos: list, measuredCosts: dict) -> np.ndarray:
    """Integer METIS weights from measured element costs, 1000 per median measured element.

    Parameters
    ----------
    numbers
        The element numbers, in partition order.
    typeInfos
        The type information of each of those elements, in the same order.
    measuredCosts
        The measured cost of elements, by number.

    Returns
    -------
    np.ndarray
        One positive integer weight per element.
    """

    measured = np.array([cost for cost in measuredCosts.values() if cost > 0.0])
    if not measured.size:
        return np.array([_elementWeight(typeInfo) for typeInfo in typeInfos], dtype=np.int64)

    nDofOf = {number: typeInfo.nDof for number, typeInfo in zip(numbers, typeInfos)}
    costPerDof = np.median(
        [measuredCosts[n] / max(1, nDofOf[n]) for n in measuredCosts if measuredCosts[n] > 0.0 and n in nDofOf]
    )
    unit = np.median(measured) / 1000.0

    weights = np.empty(len(numbers), dtype=np.int64)
    for position, (number, typeInfo) in enumerate(zip(numbers, typeInfos)):
        cost = measuredCosts.get(number, 0.0)
        if cost <= 0.0:
            cost = costPerDof * typeInfo.nDof if typeInfo.hasKernels else unit
        weights[position] = max(1, int(round(cost / unit)))
    return weights


def assignConstraints(constraints: dict, nParts: int) -> dict:
    """Assign every constraint to one process, round-robin in name order.

    Parameters
    ----------
    constraints
        The constraints of the model, by name.
    nParts
        The number of processes.

    Returns
    -------
    dict
        The rank of every constraint, by name.
    """

    # Dealt from the last rank downwards: rank 0 already carries the degrees of freedom no element
    # touches, and writes the output.
    return {name: (nParts - 1 - position) % nParts for position, name in enumerate(sorted(constraints))}
