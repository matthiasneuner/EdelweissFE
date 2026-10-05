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

The elements are partitioned by METIS on the root process, which broadcasts the result, so every
process holds the same partition regardless of whether METIS itself would reproduce it. The
partition covers every element of the model -- the kernel-less contact facets included, whose
lumped operators are zero but whose owner must still be unique.

Constraints are not partitioned by geometry: each is one object, evaluated by one process, and is
dealt out round-robin in name order. The assignment depends on the constraint names only, so it
survives a topology change -- a stateful constraint keeps the process that holds its authoritative
state.
"""

import numpy as np

from edelweissfe.domaindecomposition.metis import partitionMeshDual


def _elementWeight(element) -> int:
    """The expected cost of one element's increment, as a positive integer.

    Its number of degrees of freedom: a proxy for both its quadrature-point count and its kernel's
    arithmetic, and zero-cost facets (no kernels) weigh 1 so that METIS still places them.
    """

    return max(1, element.nDof) if element.hasKernels else 1


def partitionElements(elements: dict, nParts: int, domainSize: int, communicator, measuredCosts: dict = None) -> dict:
    """Assign every element to one of ``nParts`` processes.

    Parameters
    ----------
    elements
        The elements of the model, by number.
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

    numbers = list(elements.keys())
    parts = np.zeros(len(numbers), dtype=np.int64)

    if nParts > 1:
        if communicator.Get_rank() == 0:
            nodeIndex = {}
            offsets = np.zeros(len(numbers) + 1, dtype=np.int64)
            connectivity = []
            for position, number in enumerate(numbers):
                element = elements[number]
                for node in element.nodes:
                    connectivity.append(nodeIndex.setdefault(node.label, len(nodeIndex)))
                offsets[position + 1] = len(connectivity)

            if measuredCosts:
                weights = _measuredWeights(elements, numbers, measuredCosts)
            else:
                weights = np.array([_elementWeight(elements[number]) for number in numbers], dtype=np.int64)
            parts[:] = partitionMeshDual(
                offsets, np.array(connectivity, dtype=np.int64), len(nodeIndex), nParts, weights, max(1, domainSize)
            )

        communicator.Bcast(parts, root=0)

    return dict(zip(numbers, parts.tolist()))


def _measuredWeights(elements: dict, numbers: list, measuredCosts: dict) -> np.ndarray:
    """Integer METIS weights from measured element costs, 1000 per median measured element.

    Parameters
    ----------
    elements
        The elements, by number.
    numbers
        The element numbers, in partition order.
    measuredCosts
        The measured cost of elements, by number.

    Returns
    -------
    np.ndarray
        One positive integer weight per element.
    """

    measured = np.array([cost for cost in measuredCosts.values() if cost > 0.0])
    if not measured.size:
        return np.array([_elementWeight(elements[number]) for number in numbers], dtype=np.int64)

    costPerDof = np.median(
        [measuredCosts[n] / max(1, elements[n].nDof) for n in measuredCosts if measuredCosts[n] > 0.0]
    )
    unit = np.median(measured) / 1000.0

    weights = np.empty(len(numbers), dtype=np.int64)
    for position, number in enumerate(numbers):
        cost = measuredCosts.get(number, 0.0)
        if cost <= 0.0:
            element = elements[number]
            cost = costPerDof * element.nDof if element.hasKernels else unit
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
