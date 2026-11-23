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
"""The bound of the stiffness of penalty springs (:mod:`edelweissfe.constraints.base.penaltysprings`), and that every
penalty contact bounds the springs of its force: node to surface, and contact points against a rigid body (the
surface-to-surface contact: ``test_surfacetodeformablesurfacepenalty.py``)."""

import numpy as np
import pytest
from _hexa20cube import twoBlockModel  # noqa: E402  (tests/ is on sys.path)
from scipy.sparse import csr_matrix

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.penaltysprings import (
    SpringGradientOnOwnNodes,
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToSurfaceContact,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.journal.journal import Journal


def test_the_bound_is_the_sum_over_the_springs_of_the_absolute_gradients():
    """:math:`b = \\sum_p |k_p|\\, |w_p|\\, (|w_p| \\cdot \\boldsymbol{w})`, with the gradients given in parts on shared
    nodes (a sparse and a dense interpolation) and on nodes of each spring's own (some named by several springs),
    against the dense sum."""

    rng = np.random.default_rng(0)
    nDof, nSprings, nDim = 30, 7, 3
    stiffness = rng.normal(size=nSprings)
    weights = rng.uniform(0.5, 2.0, nDof)

    shared = rng.normal(size=(nSprings, 4)) * (rng.uniform(size=(nSprings, 4)) < 0.6)
    sharedDirections = rng.normal(size=(nSprings, nDim))
    dofsOfSharedNodes = np.arange(12).reshape((4, nDim))
    own = rng.normal(size=(nSprings, 2))
    ownDirections = rng.normal(size=(nSprings, nDim))
    # two nodes of each spring's own, distinct within a spring, some shared between springs
    ownNodes = np.array([rng.choice(6, size=2, replace=False) for _ in range(nSprings)])
    ownDofs = 12 + ownNodes[:, :, None] * nDim + np.arange(nDim)

    W = np.zeros((nSprings, nDof))
    for p in range(nSprings):
        W[p, dofsOfSharedNodes] += np.outer(shared[p], sharedDirections[p])
        np.add.at(W[p], ownDofs[p], np.outer(own[p], ownDirections[p]))
    expected = (np.abs(stiffness)[:, None] * np.abs(W)).T @ (np.abs(W) @ weights)

    for interpolation in (shared, csr_matrix(shared)):
        bound = np.zeros(nDof)
        gradient = [
            SpringGradientOnSharedNodes(interpolation, sharedDirections, dofsOfSharedNodes),
            SpringGradientOnOwnNodes(own, ownDirections, ownDofs),
        ]
        addBoundOfAbsoluteStiffnessOfSprings(bound, weights, stiffness, gradient)
        np.testing.assert_allclose(bound, expected, rtol=1e-13)


def _nodeToSurfaceContact(penetration=0.05, **options) -> NodeToSurfaceContact:
    """The slave nodes of the lower hexa20 cube's top face against the upper one's bottom face, searched once."""

    journal = Journal()
    model, slaveSurface, masterSurface = twoBlockModel(journal, penetration)
    configuration = NodeToDeformableSurfacePenaltySchema(penalty=1.0e4, searchDistance=1.0, **options)
    constraint = NodeToSurfaceContact(
        "theContact", model, slaveSurface, masterSurface, journal, configuration=configuration
    )
    constraint.updateConnectivity(model)
    return constraint


def _bounds(constraint, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The bound of the constraint, and the row sums of the absolute value of its tangent (the default of
    :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`), weighted by random columns."""

    weights = np.random.default_rng(3).uniform(0.5, 2.0, constraint.nDof)
    bound, ofTheTangent = np.zeros(constraint.nDof), np.zeros(constraint.nDof)
    constraint.addBoundOfAbsoluteStiffnessTimes(U, np.zeros_like(U), weights, bound, None)
    ConstraintBase.addBoundOfAbsoluteStiffnessTimes(constraint, U, np.zeros_like(U), weights, ofTheTangent, None)
    assert ofTheTangent.max() > 0.0
    return bound, ofTheTangent


@pytest.mark.parametrize("contactType", ["linear", "quadratic"])
def test_node_to_surface_bounds_the_normal_springs_of_its_force(contactType):
    """Without friction and with small sliding, the tangent of a slave node is its one normal spring,
    :math:`k\\, w \\otimes w`: the bound from the springs is the row sum of the absolute tangent itself."""

    constraint = _nodeToSurfaceContact(contactType=contactType, sliding="small")
    U = 1e-3 * np.random.default_rng(0).standard_normal(constraint.nDof)
    bound, ofTheTangent = _bounds(constraint, U)
    np.testing.assert_allclose(bound, ofTheTangent, rtol=1e-12, atol=1e-12 * ofTheTangent.max())


def test_node_to_surface_with_friction_bounds_the_sticking_springs():
    """With friction, a sticking slave node adds a spring of the tangential penalty along every direction of the
    tangent plane: the bound holds the row sums of the absolute tangent, normal and tangential."""

    constraint = _nodeToSurfaceContact(sliding="small", mu=0.3, tangentPenalty=2.0e4)
    bound, ofTheTangent = _bounds(constraint, np.zeros(constraint.nDof))
    assert np.all(bound >= ofTheTangent * (1 - 1e-12))
    withoutFriction, _ = _bounds(_nodeToSurfaceContact(sliding="small"), np.zeros(constraint.nDof))
    assert np.all(bound >= withoutFriction) and np.any(bound > 1.5 * withoutFriction)


def test_node_to_surface_with_finite_sliding_bounds_the_same_springs():
    """With finite sliding the gap is measured anew, with the same springs: just closed, where the normal forces are
    negligible, the bound is the row sum of the absolute tangent. The geometric stiffness (the normal force times
    the second derivative of the gap) is not bounded."""

    constraint = _nodeToSurfaceContact(penetration=1e-9, sliding="finite")
    bound, ofTheTangent = _bounds(constraint, np.zeros(constraint.nDof))
    np.testing.assert_allclose(bound, ofTheTangent, rtol=0, atol=1e-8 * ofTheTangent.max())
