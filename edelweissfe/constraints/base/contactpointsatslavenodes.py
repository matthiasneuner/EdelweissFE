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
"""
The contact points of a node-based contact formulation: the nodes of a slave surface, each standing for its tributary
area.
"""

import numpy as np
from scipy.sparse import csr_matrix, identity

from edelweissfe.constraints.base.contactpoints import (
    ContactPoints,
    areasOfContactPoints,
    thicknessesOfFacets,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.meshtools import currentNodeCoordinates


def tributaryAreasOfSlaveNodes(slaveFacets: list, model: FEModel) -> tuple[list, np.ndarray]:
    """The unique nodes of slave contact facets and the area each stands for, in the reference configuration, as the
    model integrates (:func:`~edelweissfe.constraints.base.contactpoints.areasOfContactPoints`): the sum of its shares of
    the measures of its incident facets (``nodalAreaShares``, set by the surface element generator), each facet with
    the thickness of its section in a plane model -- a node between facets of different thicknesses stands for its
    share of the area of each --, over the ring at the node's radius in an axisymmetric model.

    Parameters
    ----------
    slaveFacets
        The slave contact facet elements.
    model
        The model, with the sections of the elements the facets bound.

    Returns
    -------
    tuple[list, numpy.ndarray]
        The nodes, in first-encounter order (the order of the generator's '<prefix>_nodes' node set), and their
        tributary areas. A node with a zero share (``nodalWeights=serendipityOptimal``) is kept, inert, so that the
        order is unaffected.
    """

    measureOfNode = {}
    thicknessTimesMeasureOfNode = {}
    for slaveFacet, thickness in zip(slaveFacets, thicknessesOfFacets(slaveFacets, model)):
        for node, share in zip(slaveFacet.nodes, slaveFacet.nodalAreaShares):
            measureOfNode[node] = measureOfNode.get(node, 0.0) + share
            thicknessTimesMeasureOfNode[node] = thicknessTimesMeasureOfNode.get(node, 0.0) + thickness * share

    nodes = list(measureOfNode)
    measures = np.array(list(measureOfNode.values()), dtype=float)
    # the share-weighted thickness of a node, so that its area is the sum of its shares of the areas of its facets
    thicknesses = np.divide(
        np.array(list(thicknessTimesMeasureOfNode.values()), dtype=float),
        measures,
        out=np.ones_like(measures),
        where=measures > 0,
    )
    coordinates = np.array([node.coordinates for node in nodes]).reshape(len(nodes), model.domainSize)
    return nodes, areasOfContactPoints(measures, coordinates, thicknesses, model.axisymmetric)


class ContactPointsAtSlaveNodes(ContactPoints):
    """The nodes of a slave surface, used as contact points.

    A node is its own contact point: its interpolation vector is :math:`c = [1]`, it sits at
    :math:`\\boldsymbol{X} + \\boldsymbol{U}` (its origin :math:`\\boldsymbol{X}`, its total displacement
    :math:`\\boldsymbol{U}`), and it stands for its tributary area (:func:`tributaryAreasOfSlaveNodes`): times the
    thickness in a plane model, in an axisymmetric model for the ring it sweeps, :math:`2 \\pi r` times its tributary
    length at its radius (:func:`~edelweissfe.constraints.base.contactpoints.areasOfContactPoints`), measured when a
    constraint takes the points (:meth:`measureTheAreasAsIn`). A located node takes itself into the block of the
    constraint, once.

    Parameters
    ----------
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    constraintName
        The name of the owning constraint, for error messages.
    """

    def __init__(self, slaveSurface: ElementSet, constraintName: str):
        self.setName = slaveSurface.name
        self._constraintName = constraintName
        self._facets = list(slaveSurface)
        self.slaveNodes = list(dict.fromkeys(node for facet in self._facets for node in facet.nodes))
        self.nPoints = len(self.slaveNodes)
        self._referenceCoordinates = np.array([node.coordinates for node in self.slaveNodes])
        self.locatePoints(np.zeros(0, dtype=int))

    def measureTheAreasAsIn(self, model: FEModel, constraintName: str):
        """Measure the tributary areas of the slave nodes as the model integrates (:func:`tributaryAreasOfSlaveNodes`).

        Parameters
        ----------
        model
            The model of the constraint.
        constraintName
            The name of the constraint.
        """

        _, self.areas = tributaryAreasOfSlaveNodes(self._facets, model)
        self.axisymmetric = model.axisymmetric

    @property
    def surfaceNodes(self) -> list:
        """The slave nodes, in the order of the surface element generator's '<prefix>_nodes' node set.

        Returns
        -------
        list
            The nodes.
        """

        return self.slaveNodes

    def describe(self) -> str:
        """The number of slave nodes, for the journal.

        Returns
        -------
        str
            The description.
        """

        return f"{self.nPoints} slave nodes"

    def finiteElementNodes(self) -> set:
        """The slave nodes.

        Returns
        -------
        set
            The nodes.
        """

        return set(self.slaveNodes)

    def currentPointCoordinates(self, model: FEModel) -> np.ndarray:
        """The current coordinates of the slave nodes.

        Parameters
        ----------
        model
            The model tree, holding the displacement field.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        return currentNodeCoordinates(self.slaveNodes, model, self._referenceCoordinates)

    def locatePoints(self, points: np.ndarray) -> list:
        """Locate the given nodes: each takes itself into the block.

        Parameters
        ----------
        points
            The indices of the slave nodes to locate, ascending.

        Returns
        -------
        list
            The nodes.
        """

        self._referenceCoordinatesOfLocated = self._referenceCoordinates[points]
        return [self.slaveNodes[p] for p in points]

    def originsOfLocatedPoints(self) -> np.ndarray:
        """:math:`\\boldsymbol{X}`, the reference coordinates of the located nodes.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nLocated, nDim).
        """

        return self._referenceCoordinatesOfLocated

    def changesFromTheOriginsOfLocatedPoints(self, U: np.ndarray, dU: np.ndarray, nComponents: int) -> np.ndarray:
        """The total of the field at the located nodes; see
        :meth:`ContactPoints.changesFromTheOriginsOfLocatedPoints`.

        Parameters
        ----------
        U
            The field's solution of the slave part of the block.
        dU
            The increment; unused.
        nComponents
            The number of components of the field per node.

        Returns
        -------
        numpy.ndarray
            The changes, of shape (nLocated, number of components of the field).
        """

        return U.reshape((len(self._referenceCoordinatesOfLocated), nComponents))

    def addForceResidual(self, P: np.ndarray, points: np.ndarray, forces: np.ndarray, directions: np.ndarray):
        """Add :math:`f\\, n` at each given node; see :meth:`ContactPoints.addForceResidual`.

        Parameters
        ----------
        P
            The slave part of the residual contribution; augmented.
        points
            The located nodes that carry a force.
        forces
            The forces of their springs.
        directions
            The unit directions of their springs.
        """

        P.reshape((-1, directions.shape[1]))[points] += forces[:, None] * directions

    def interpolationOfLocatedPoint(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The position of a located node in the slave part of the block, and :math:`c = [1]`.

        Parameters
        ----------
        k
            The index of the located node.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The position of the node and the interpolation vector.
        """

        return np.array([k]), np.ones(1)

    def interpolationMatrixOfLocatedPoints(self) -> csr_matrix:
        """The identity: every located node is its own point; see
        :meth:`ContactPoints.interpolationMatrixOfLocatedPoints`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The identity over the located nodes.
        """

        return identity(len(self._referenceCoordinatesOfLocated), format="csr")

    def searchLayout(self) -> dict[str, np.ndarray]:
        """The labels of the slave nodes identify the contact points in a checkpoint.

        Returns
        -------
        dict[str, numpy.ndarray]
            The node labels.
        """

        return {"slaveNodes": np.array([node.label for node in self.slaveNodes], dtype=np.int64)}

    def nodalNormalForces(self, normalForces: np.ndarray) -> np.ndarray:
        """The normal force at each slave node, positive when repelled.

        Parameters
        ----------
        normalForces
            The normal force of each node (negative in compression).

        Returns
        -------
        numpy.ndarray
            The normal force at each slave node.
        """

        return -normalForces

    def isTouchedBy(self, model: FEModel, change: ModelChange) -> bool:
        """Whether a change of the mesh touched the source elements of the slave surface.

        Parameters
        ----------
        model
            The model tree, holding the recipes of the generated contact facet sets.
        change
            The change of the mesh.

        Returns
        -------
        bool
            True if the slave nodes must be regenerated.
        """

        recipe = model.contactFacetRecipes.get(self.setName)
        return recipe is not None and change.touchesSurface(recipe[0])

    def regenerated(self, model: FEModel) -> "ContactPointsAtSlaveNodes":
        """The slave nodes of the regenerated slave facets.

        Parameters
        ----------
        model
            The model tree, holding the regenerated facet set.

        Returns
        -------
        ContactPointsAtSlaveNodes
            The new contact points.
        """

        return ContactPointsAtSlaveNodes(model.elementSets[self.setName], self._constraintName)
