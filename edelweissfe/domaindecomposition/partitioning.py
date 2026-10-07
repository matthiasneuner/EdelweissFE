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
state.
"""

import numpy as np

from edelweissfe.domaindecomposition.metis import partitionMeshDual
from edelweissfe.models.mesh import ElementTypeInfo, Mesh, MeshElement


def _elementWeight(typeInfo: ElementTypeInfo) -> int:
    """The expected cost of one element's increment, as a positive integer.

    Its number of degrees of freedom: a proxy for both its quadrature-point count and its kernel's
    arithmetic, and zero-cost facets (no kernels) weigh 1 so that METIS still places them.
    """

    return max(1, typeInfo.nDof) if typeInfo.hasKernels else 1


def partitionElementsOfMesh(mesh: Mesh, nParts: int, domainSize: int, communicator, measuredCosts: dict = None) -> dict:
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
                offsets, np.array(connectivity, dtype=np.int64), len(nodeIndex), nParts, weights, max(1, domainSize)
            )

        communicator.Bcast(parts, root=0)

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


def keepElementsWhereTheyWere(owners: dict, previousOwners: dict, nParts: int) -> dict:
    """Renumber the parts of a new partition so that as many elements as possible keep their process.

    METIS numbers the parts of a partition arbitrarily: a repartition close to the previous one may
    still give every part another number, and so move every element to another process. Each new part
    is given the number of the previous part it shares the most elements with, largest overlaps first;
    the parts left over keep their order. The parts themselves -- which elements are computed
    together -- do not change, only which process computes them.

    Parameters
    ----------
    owners
        The rank of every element, by number, in the new partition.
    previousOwners
        The rank of every element, by number, in the previous partition; the same elements.
    nParts
        The number of processes.

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

    renumbered = np.full(nParts, -1, dtype=np.int64)
    taken = np.zeros(nParts, dtype=bool)
    # Largest overlap first; ties in the order of the new, then the previous part, so that every
    # process arrives at the same renumbering.
    newParts, previousParts = np.unravel_index(np.argsort(-overlap, axis=None, kind="stable"), overlap.shape)
    for newPart, previousPart in zip(newParts, previousParts):
        if renumbered[newPart] < 0 and not taken[previousPart] and overlap[newPart, previousPart] > 0:
            renumbered[newPart] = previousPart
            taken[previousPart] = True
    leftOver = iter(np.flatnonzero(~taken).tolist())
    for newPart in range(nParts):
        if renumbered[newPart] < 0:
            renumbered[newPart] = next(leftOver)

    return dict(zip(numbers, renumbered[new].tolist()))


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
