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

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.contactpoints import ContactPoints
from edelweissfe.constraints.base.contactpointsatslavenodes import (
    ContactPointsAtSlaveNodes,
)
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
    checkFacetsCarryParentFaces,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.frozencontactsearch import FrozenContactSearch
from edelweissfe.constraints.base.penaltylaw import penaltyLaw
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    SurfaceContactPenaltySchema,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.numerics.systemmatrixcontributor import localEntriesOfSubBlocks
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    closestFacets,
    cosOfSmoothNormalChangeLimit,
    facetNormalAndMeasure,
    rowDot,
)
from edelweissfe.utils.meshtools import currentNodeCoordinates
from edelweissfe.utils.parentfacegeometry import parentFaceShapeFunctions
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between the contact points of a slave body and a deformable master
surface, whose force is distributed with the *parent element face* shape functions of the master -- for every kind of
contact point: the quadrature points of the faces of finite elements (a Gauss-point-to-segment, GPTS, formulation),
the nodes of finite elements, and, from EdelweissMeshfree, the points of the faces of particles. What a kind of point
contributes -- how it moves, which nodes its force goes to, the area it stands for -- is its
:class:`~edelweissfe.constraints.base.contactpoints.ContactPoints`; everything else is written once, here.

Why the quadrature points exist alongside :mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty`: a
node-based penalty scheme cannot transmit a correct pressure across a *serendipity* (quad8/hexa20)
face, and no choice of nodal weights can repair it. The consistent nodal loads of a uniform
pressure :math:`p` on a quad8 face of area :math:`A` are :math:`-pA/12` at the corners and
:math:`+pA/3` at the midsides -- the corner load is *tensile*, while a unilateral spring can only
push. The discrete solution resolves that contradiction by opening the corner gaps: on a matched
hexa20 patch test all corner nodes lift off, carrying no load, and the transmitted force falls ~1.7%
short with a visibly undulating interface. See issue #112.

Integrating instead removes the contradiction at its root. The pressure is evaluated *pointwise*
at quadrature points, where it is non-negative as physics requires, and is then distributed with
the parent face's own shape functions, so the nodal force at node :math:`a`,

.. math::
    f_a = \\sum_q w_q \\, J_q \\, p_q \\, N_a(\\xi_q) ,

is negative wherever :math:`N_a` is -- a tensile *nodal* force arising from a non-negative
*pressure field*, with no unilateral condition ever imposed at a corner node. Note that quadrature
alone would achieve nothing: the flat facets' own linear shape functions are non-negative, so
distributing with those would leave the corners lifting off exactly as before. The parent-face
basis (:mod:`~edelweissfe.utils.parentfacegeometry`) is the load-bearing half of the formulation.

Scope of this implementation: frictionless normal penalty contact under ``sliding=small``, for both the implicit
and the explicit solver path. The normal force comes from an exchangeable law
(:mod:`~edelweissfe.constraints.base.penaltylaw`); Coulomb friction and augmented Lagrange are *not* implemented (the
node-based constraint has both), and they enter as a law with a per-point history, without touching the contact points,
the search or the gap. ``sliding=finite`` is rejected rather than approximated, see :class:`Constraint`.

What this formulation does *not* fix: pointwise enforcement at quadrature points still
over-constrains a non-matching interface, and the integrand is discontinuous inside a slave facet
wherever the assigned master facet changes, so that quadrature error does not vanish under
refinement of the rule. Exactness on non-matching meshes requires segment clipping and weak
enforcement through a multiplier field -- mortar. This is the strictly weaker, strictly simpler
member of that family, and it is what the corner-liftoff defect actually calls for.
"""

#: The fraction of the search distance beyond which the depth at which a point enters contact is reported: a point
#: that enters that deep in one increment has almost outrun the search.
_entryDepthWarningFraction = 0.5


class ContactState(IntEnum):
    """Where a contact point stands with respect to the master surface, as of the last update of the connectivity."""

    #: No facet within the search distance.
    FREE = 0
    #: A facet within the search distance, the point outside of it: the point is seen before it touches.
    APPROACHING = 1
    #: The point has entered the master body through its paired facet, and is pushed back out through it.
    IN_CONTACT = 2
    #: The point lies inside the master body without having been seen outside of it before: it has tunnelled through
    #: the search distance in one increment. It gets no force (the closest facet is no indication of the way out),
    #: and the constraint reports it.
    FOUND_INSIDE = 3


class ContactStiffnessView:
    """Provides structured 2-D sub-views for the sparse stiffness matrix slice of :class:`Constraint`.

    Each closed contact point couples the nodes its interpolation vector acts on to the nodes of its assigned master
    facet's parent face, and to nothing else -- so the contribution is one dense square block per located point, over
    that point's own ``[slave nodes, master parent face nodes]``.

    Attributes
    ----------
    blocks : list[numpy.ndarray]
        One view per located contact point, of shape ``(m, m)`` with
        ``m = (nSlaveNodesOfThePoint + nMasterParentNodes) * nDim``.
    """

    def __init__(self, flat_array: np.ndarray, blockSizes: list[int]):
        """Carve the flat value slice into one dense square block per located contact point.

        Parameters
        ----------
        flat_array
            This constraint's slice of the global VIJ value array.
        blockSizes
            The DOF count ``m`` of each located contact point's block, in assembly order.
        """

        self.blocks = []
        offset = 0
        for m in blockSizes:
            self.blocks.append(flat_array[offset : offset + m * m].reshape((m, m)))
            offset += m * m


#: The kinds of contact points an input file can choose, by the value of the ``contactPoints`` option.
_contactPointsFromInput = {
    "quadraturepoints": lambda configuration, slaveSurface, name: ContactPointsOnSlaveSurface(
        slaveSurface, configuration.nQuadraturePoints, name
    ),
    "nodes": lambda configuration, slaveSurface, name: ContactPointsAtSlaveNodes(slaveSurface, name),
}


@dataclass(frozen=True)
class SurfaceToDeformableSurfacePenaltySchema(SurfaceContactPenaltySchema):
    """The options this constraint accepts: the shared slave-side and penalty options of
    :class:`~edelweissfe.constraints.base.surfacecontactpenaltyschema.SurfaceContactPenaltySchema`,
    plus the master surface and the kind of contact points. ``penalty`` here is an interface stiffness modulus per
    unit area exactly as in the node-based constraint, multiplied by the area each contact point stands for.
    """

    masterSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the master surface.",
        dtype=str,
        default=None,
        required=True,
    )
    contactPoints: str = schemaField(
        description="The contact points of the slave surface: 'quadraturePoints' (nQuadraturePoints quadrature "
        "points on each slave facet, distributed with the slave parent face's shape functions) or 'nodes' (the "
        "slave nodes, each standing for its tributary area).",
        dtype=str,
        default="quadraturePoints",
    )


class Constraint(FrozenContactSearch, ForcesOnlyExplicitEvaluation, ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between the contact points of a slave body and a deformable master surface,
    distributed with the master's parent element face shape functions.

    Theoretical background
    -----------------------
    A contact point :math:`p` moves with an interpolation vector :math:`c` over the nodes it acts on: the parent face
    shape functions :math:`N^s(\\xi_q)` of a quadrature point of a slave facet, :math:`[1]` for a node, the vector
    :math:`G(\\boldsymbol{x}_q)` with which a particle moves a point on one of its faces (see
    :mod:`~edelweissfe.constraints.base.contactpoints`); it stands for an area :math:`A_p`. A quadrature point
    follows the *curved* parent surface, :math:`x_q = N^s(\\xi_q) \\cdot x^s`, so the slave-side faceting error of
    the node-based formulation disappears as well.

    Under ``sliding=small`` the projection is frozen at every update of the connectivity: the assigned master facet,
    the parametric location of the closest point *in that facet's parent face*, and the facet's unit normal
    :math:`\\bar{n}`. The gap is then

    .. math::
        g = \\bar{n} \\cdot \\left( \\boldsymbol{x}_p - N^m(\\xi_m) \\cdot x^m \\right) ,

    which is *linear* in the displacement DOFs, so the gradient
    :math:`w = \\bar{n} \\otimes [c, -N^m]` is constant, the geometric Hessian vanishes
    identically, and the tangent :math:`\\mathrm{d}f_n/\\mathrm{d}g \\, w \\otimes w` is
    symmetric. The force is :math:`-f_n\\, w` on the nodes, with the normal force :math:`f_n` of the penalty law at
    the stiffness ``penalty`` times :math:`A_p`.
    Structurally this is the node-based constraint's frozen form, with that formulation's slave coefficient ``1``
    replaced by the interpolation vector of the contact point -- which is precisely the generalization that lets a
    corner node of a serendipity face carry a tensile force, and a particle take part at all.

    Note that the gap is measured to the *parent surface* point :math:`N^m(\\xi_m) \\cdot x^m`, not
    to the flat facet point :math:`b \\cdot x^{\\text{facet}}`. This matters: distributing the force
    with :math:`N^m` while measuring the gap to the flat point would make the force distribution
    differ from the transpose of the gap gradient, i.e. a non-symmetric, variationally inconsistent
    operator. The flat facets serve only as the search and parametrization scaffold; the contact
    geometry itself is the curved parent surface. For straight-edged faces the two points coincide
    identically, so nothing is lost in the common case.

    The pairing
    -----------
    In every update of the connectivity, the contact points within ``searchDistance`` of the master surface (all of
    them, if no search distance is given) are paired with their closest facet, but only while they are outside the
    master body (:attr:`ContactState.APPROACHING`). Once a point is in contact (:attr:`ContactState.IN_CONTACT`), it
    is paired anew in every update with its closest facet, however deep it is, as it slides along the surface and
    from one face of the master onto the next where the surface is smooth -- but never across a crease, with a facet
    whose normal differs from the one it was pushed along until then by more than
    :data:`~edelweissfe.utils.facetcontactgeometry.cosOfSmoothNormalChangeLimit` allows: the closest facet of a point
    near a corner of the master may be on its side, of a point deep inside a thin body on its far side, and its normal
    would push the point out of the wrong side. A point that was out of reach at the previous update and is found inside the master body has
    tunnelled through the search distance in one increment (:attr:`ContactState.FOUND_INSIDE`): it gets no force and
    is reported, and the search distance must exceed the distance a point travels towards the surface in one
    increment -- the constraint warns when a point enters deeper than half of it. At the first update, the bodies are
    as the model gives them: a point found inside then is in contact (an initial overlap is pushed apart).

    The pairing is checked for a master surface that is convex where it is touched: at a re-entrant corner, the sign
    of the gap of the closest facet does not tell inside from outside.

    ``sliding=finite`` is rejected rather than approximated. It would require the exact gap gradient
    and Hessian of a point against a *curved* quadratic surface, whose closest-point projection is
    itself an iterative sub-problem -- a substantially different formulation from the flat-facet
    algebra of :mod:`~edelweissfe.utils.facetcontactgeometry`, not a variation on it.

    Like the node-based constraint, this is a
    :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either surface's source solid
    elements are refined mid-run, it rebuilds that side from the regenerated facets at its own next
    :meth:`updateConnectivity` tick. Since no frictional or multiplier history exists yet, the
    rebuild is unconditional and carries nothing across -- the point at which friction is added is
    the point at which that becomes a real problem, because quadrature points, unlike nodes, have no
    identity that survives a retiling.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    contactPoints
        The contact points of the slave side.
    masterSurface
        The element set of contact facet elements forming the master surface.
    journal
        The journal instance.
    configuration
        The options this constraint accepts, see :class:`SurfaceToDeformableSurfacePenaltySchema`. Its
        ``slaveSurface``, ``nQuadraturePoints`` and ``contactPoints`` say which contact points an input file builds
        (:meth:`fromConstraintDefinition`); here they are given.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = SurfaceToDeformableSurfacePenaltySchema

    #: The contact searches for the surface it touches.
    connectivityChangesWithMotion = True

    #: Journal sender tag: short and class-level, so that the fixed-width sender column never truncates it.
    identification = "SurfaceContact"

    def __init__(
        self,
        name: str,
        model: FEModel,
        contactPoints: ContactPoints,
        masterSurface: ElementSet,
        journal: Journal,
        *,
        configuration: SurfaceToDeformableSurfacePenaltySchema = SurfaceToDeformableSurfacePenaltySchema(),
    ):
        super().__init__(name, model)

        self.name = name
        self.journal = journal
        self._lastSeenTopologyVersion = model.topology.version
        model.topology.registerMeshDependent(self)

        self.nDim = model.domainSize

        self._masterSurfaceSetName = masterSurface.name

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")

        self.law = penaltyLaw(configuration.contactType)

        self.sliding = configuration.sliding.lower()
        if self.sliding != "small":
            raise ValueError(
                f"Constraint sliding '{self.sliding}' is not supported by "
                "surfaceToDeformableSurfacePenalty: only 'small' is implemented. Finite sliding "
                "would require the exact gap gradient and Hessian against a curved quadratic "
                "surface, whose closest-point projection is itself an iterative sub-problem."
            )

        self.searchDistance = configuration.searchDistance

        self.slave = contactPoints
        self._buildMaster(masterSurface)
        self._resetContactPoints()

        self.totalNormalForce = 0.0

        self.journal.message(
            f"contact '{self.name}': {self.slave.describe()}, {len(self.facetElements)} master facets, "
            f"sliding={self.sliding}, type={self.law.name}",
            self.identification,
        )

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition, with the contact points the option
        ``contactPoints`` chooses on the slave surface. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        contactPointsOfSurface = _contactPointsFromInput.get(configuration.contactPoints.lower())
        if contactPointsOfSurface is None:
            raise ValueError(
                f"Constraint '{name}': contactPoints '{configuration.contactPoints}' is not supported. Use "
                "'quadraturePoints' or 'nodes'."
            )
        return cls(
            name,
            model,
            contactPointsOfSurface(configuration, model.elementSets[configuration.slaveSurface], name),
            model.elementSets[configuration.masterSurface],
            journal,
            configuration=configuration,
        )

    def _buildMaster(self, masterSurface: ElementSet):
        """Build (or rebuild) everything cached about the master surface: its facets and their parent faces, the
        master nodes, each once, and the facets as rows of indices into them.

        Parameters
        ----------
        masterSurface
            The element set of master contact facets.
        """

        self.facetElements = list(masterSurface)

        checkFacetsCarryParentFaces(self.facetElements, "master", self._masterSurfaceSetName, self.name)

        parentFaceTypes = {facet.parentFaceType for facet in self.facetElements}
        if len(parentFaceTypes) > 1:
            raise ValueError(
                f"Constraint '{self.name}': the master surface '{self._masterSurfaceSetName}' mixes parent faces "
                f"{sorted(parentFaceTypes)}; a single element type is required."
            )
        self._masterParentFaceType = parentFaceTypes.pop() if parentFaceTypes else None

        masterParentNodes = {node for facet in self.facetElements for node in facet.parentFaceNodes}
        if not masterParentNodes.isdisjoint(self.slave.finiteElementNodes()):
            raise ValueError(
                f"Constraint '{self.name}': the contact points and the master surface "
                f"'{self._masterSurfaceSetName}' share nodes -- self-contact is not supported."
            )

        indexOfMasterNode = {}
        for facet in self.facetElements:
            for node in facet.nodes:
                indexOfMasterNode.setdefault(node, len(indexOfMasterNode))
        self._masterNodes = list(indexOfMasterNode)
        self._referenceCoordinatesOfMasterNodes = np.array([node.coordinates for node in self._masterNodes])
        self._masterNodesOfFacet = np.array(
            [[indexOfMasterNode[node] for node in facet.nodes] for facet in self.facetElements], dtype=np.intp
        )

        self._vertexParametricCoordsOfFacet = np.array([facet.vertexParametricCoords for facet in self.facetElements])
        self._masterParentRefCoords = np.array(
            [[node.coordinates for node in facet.parentFaceNodes] for facet in self.facetElements]
        )
        self.nMasterParentNodes = self._masterParentRefCoords.shape[1] if self.facetElements else 0

    def _resetContactPoints(self):
        """Size the per-point state for the current contact points, which start without a pairing."""

        nPoints = self.slave.nPoints

        self._stateOfPoint = np.full(nPoints, ContactState.FREE, dtype=np.int8)
        self._searchedBefore = False
        self._deepestEntryReported = 0.0

        # The frozen projection, refreshed in every update of the connectivity: the assigned master facet (-1 if
        # none), the parent face shape functions of the closest point within that facet's parent face, and the
        # facet's unit normal.
        self._assignedFacetIdx = np.full(nPoints, -1, dtype=np.intp)
        self._frozenMasterShapeFunctions = np.zeros((nPoints, self.nMasterParentNodes))
        self._frozenNormals = np.zeros((nPoints, self.nDim))

        # Per-point results of the current evaluation, for output.
        self._normalForceCurrent = np.zeros(nPoints)
        self._gapCurrent = np.zeros(nPoints)

        self._locatePairedPoints()
        self._lastReportedActivity = None

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return [["displacement"]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return self.nDim * len(self._nodes)

    @property
    def nPoints(self) -> int:
        """The number of contact points."""
        return self.slave.nPoints

    @property
    def slaveSurfaceNodes(self) -> list:
        """The nodes of an FE slave surface, see
        :attr:`~edelweissfe.constraints.base.contactpoints.ContactPoints.surfaceNodes`."""
        return self.slave.surfaceNodes

    @property
    def stateOfPoints(self) -> np.ndarray:
        """The :class:`ContactState` of every contact point, as of the last update of the connectivity.

        Returns
        -------
        numpy.ndarray
            The states.
        """

        return self._stateOfPoint

    @property
    def numberOfPointsFoundInside(self) -> int:
        """The number of contact points found inside the master body without having been seen outside of it, which
        get no force (:attr:`ContactState.FOUND_INSIDE`).

        Returns
        -------
        int
            The number.
        """

        return int(np.count_nonzero(self._stateOfPoint == ContactState.FOUND_INSIDE))

    def updateConnectivity(self, model: FEModel) -> bool:
        """Pair the contact points with the master surface at the configuration of the model, freeze their
        projections, and locate the paired points in the block (see the class documentation, *The pairing*).

        Called once per increment by the solver, before the discrete system is (re)built. The
        candidate search runs on the *flat* master facets, which keeps it robust; what is frozen from it is the
        parametric location of the closest point within the facet's *parent* face, from which the parent-face
        shape functions -- the ones that may go negative at a corner -- are evaluated.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        coordinatesOfPoint = self.slave.currentPointCoordinates(model)
        coordinatesOfFacet = currentNodeCoordinates(self._masterNodes, model, self._referenceCoordinatesOfMasterNodes)[
            self._masterNodesOfFacet
        ]
        facetCoords = list(coordinatesOfFacet)
        normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)

        previousState = self._stateOfPoint
        wasInContact = previousState == ContactState.IN_CONTACT
        wasSeen = wasInContact | (previousState == ContactState.APPROACHING) | (not self._searchedBefore)

        # A point in contact looks for its closest facet however deep it is, every other point within the search
        # distance.
        candidatesOfPoint = [None] * self.nPoints
        for points, searchDistance in (
            (np.flatnonzero(wasInContact), None),
            (np.flatnonzero(~wasInContact), self.searchDistance),
        ):
            if len(points):
                for point, candidates in zip(
                    points, closestFacetCandidates(coordinatesOfPoint[points], facetCoords, searchDistance)
                ):
                    candidatesOfPoint[point] = candidates
        closestFacet, weights, distance = closestFacets(coordinatesOfPoint, facetCoords, candidatesOfPoint)

        # A point in contact is pushed back out the way it came: it slides on over the master surface where that is
        # smooth, but does not move across a crease, onto a facet whose normal differs from the one it was pushed
        # along by more than the limit -- the side of a corner it is near, or the far side of a thin master body.
        acrossACrease = np.flatnonzero(
            wasInContact
            & (closestFacet >= 0)
            & (rowDot(normalOfFacet[closestFacet], self._frozenNormals) < cosOfSmoothNormalChangeLimit)
        )
        if len(acrossACrease):
            sameWay = [
                [
                    f
                    for f in candidatesOfPoint[point]
                    if normalOfFacet[f] @ self._frozenNormals[point] >= cosOfSmoothNormalChangeLimit
                ]
                for point in acrossACrease
            ]
            closestFacet[acrossACrease], weights[acrossACrease], distance[acrossACrease] = closestFacets(
                coordinatesOfPoint[acrossACrease], facetCoords, sameWay
            )

        reached = closestFacet >= 0
        if self.searchDistance is not None:
            reached &= wasInContact | (distance <= self.searchDistance)

        facetOfReached = closestFacet[reached]
        closestPoint = np.einsum("pa,pad->pd", weights[reached], coordinatesOfFacet[facetOfReached])
        gap = np.zeros(self.nPoints)
        gap[reached] = rowDot(normalOfFacet[facetOfReached], coordinatesOfPoint[reached] - closestPoint)

        penetrating = reached & (gap < 0.0)
        state = np.full(self.nPoints, ContactState.FREE, dtype=np.int8)
        state[reached & ~penetrating] = ContactState.APPROACHING
        state[penetrating & wasSeen] = ContactState.IN_CONTACT
        state[penetrating & ~wasSeen] = ContactState.FOUND_INSIDE

        self._reportEntriesAndPointsFoundInside(previousState, state, gap)
        self._stateOfPoint = state
        self._searchedBefore = True

        paired = (state == ContactState.APPROACHING) | (state == ContactState.IN_CONTACT)
        newAssignment = np.where(paired, closestFacet, -1)

        # The clamped barycentric weights locate the closest point in the flat facet; map that location into the
        # parent face's parametric space and evaluate the parent face's shape functions there.
        pairedFacet = newAssignment[paired]
        parametric = np.einsum("pa,pad->pd", weights[paired], self._vertexParametricCoordsOfFacet[pairedFacet])
        self._frozenMasterShapeFunctions[:] = 0.0
        self._frozenNormals[:] = 0.0
        if len(pairedFacet):
            self._frozenMasterShapeFunctions[paired] = parentFaceShapeFunctions(self._masterParentFaceType, parametric)
            self._frozenNormals[paired] = normalOfFacet[pairedFacet]

        return self._assignMasterFacets(newAssignment)

    def _reportEntriesAndPointsFoundInside(self, previousState: np.ndarray, state: np.ndarray, gap: np.ndarray):
        """Report in the journal the points newly found inside the master body, and a point that enters contact
        deeper than any reported before, beyond half of the search distance.

        Parameters
        ----------
        previousState
            The states before this update.
        state
            The states after it.
        gap
            The gaps after it.
        """

        newlyFoundInside = (state == ContactState.FOUND_INSIDE) & (previousState != ContactState.FOUND_INSIDE)
        if newlyFoundInside.any():
            self.journal.message(
                f"contact '{self.name}': {np.count_nonzero(newlyFoundInside)} contact point(s) found inside the "
                f"master body without having been seen outside of it within the search distance "
                f"({self.searchDistance:.3e}); they get no force. Increase the search distance, or reduce the time "
                "increment.",
                self.identification,
                level=1,
            )

        entering = (state == ContactState.IN_CONTACT) & (previousState == ContactState.APPROACHING)
        if self.searchDistance is not None and entering.any():
            entryDepth = float(-gap[entering].min())
            if (
                entryDepth > _entryDepthWarningFraction * self.searchDistance
                and entryDepth > self._deepestEntryReported
            ):
                self._deepestEntryReported = entryDepth
                self.journal.message(
                    f"contact '{self.name}': a contact point entered {entryDepth:.3e} deep in one increment, more "
                    f"than {_entryDepthWarningFraction:.0%} of the search distance ({self.searchDistance:.3e}): "
                    "points may soon tunnel through it unseen.",
                    self.identification,
                    level=1,
                )

    def _assignMasterFacets(self, newAssignment: np.ndarray) -> bool:
        """Assign each contact point its master facet (searched or restored), and locate the paired points."""

        hasChanged = not np.array_equal(newAssignment, self._assignedFacetIdx)
        self._assignedFacetIdx = newAssignment

        previousNodes = self._nodes
        self._locatePairedPoints()
        if self._nodes != previousNodes:
            hasChanged = True

        # A point that lost its facet has no gap: reset it to zero, so that getGaps() and the closed count below
        # do not keep its last value.
        self._gapCurrent[newAssignment < 0] = 0.0

        # Assembly is paid per assigned point, load is carried by the closed ones (counted at the last force
        # evaluation). Reported only when it changes.
        activity = (len(self._locatedPoints), int((self._gapCurrent < 0.0).sum()))
        if activity != self._lastReportedActivity:
            self._lastReportedActivity = activity
            self.journal.message(
                f"contact '{self.name}': {activity[0]}/{self.nPoints} active, {activity[1]} closed",
                self.identification,
                level=2,
            )

        return hasChanged

    def _locatePairedPoints(self):
        """Locate the paired points in the block: the nodes their contact points act on (the slave part), then, point
        after point, the parent face nodes of their master facets (the master part). Their frozen projections are
        gathered into arrays, once per update of the connectivity, so that an evaluation is a fixed handful of array
        operations rather than a loop over the points.
        """

        located = np.flatnonzero(self._assignedFacetIdx >= 0)
        self._locatedPoints = located
        self.withinReach = bool(len(located))

        slaveNodes = self.slave.locatePoints(located)
        facets = self._assignedFacetIdx[located]
        masterNodes = [node for f in facets for node in self.facetElements[f].parentFaceNodes]
        self._nodes = slaveNodes + masterNodes
        self._nSlaveDof = self.nDim * len(slaveNodes)

        self._masterShapeFunctionsOfLocated = self._frozenMasterShapeFunctions[located]
        self._normalsOfLocated = self._frozenNormals[located]
        self._masterRefOfLocated = self._masterParentRefCoords[facets].reshape(
            (len(located), self.nMasterParentNodes, self.nDim)
        )
        self._areasOfLocated = self.slave.areas[located]
        self._blockSizes = None
        self._closed = np.zeros(0, dtype=np.intp)

    def _searchLayout(self) -> dict[str, np.ndarray]:
        return self.slave.searchLayout() | {
            "masterFacets": np.array([facet.elNumber for facet in self.facetElements], dtype=np.int64),
        }

    def _frozenProjection(self) -> dict[str, np.ndarray]:
        return {
            "assignedFacetIdx": self._assignedFacetIdx,
            "stateOfPoint": self._stateOfPoint,
            "masterShapeFunctions": self._frozenMasterShapeFunctions,
            "normals": self._frozenNormals,
        }

    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        self._stateOfPoint = np.array(projection["stateOfPoint"], dtype=np.int8)
        self._searchedBefore = True
        self._frozenMasterShapeFunctions = np.array(projection["masterShapeFunctions"]).reshape(
            self._frozenMasterShapeFunctions.shape
        )
        self._frozenNormals = np.array(projection["normals"]).reshape(self._frozenNormals.shape)
        return self._assignMasterFacets(np.array(projection["assignedFacetIdx"], dtype=np.intp))

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild both sides if ``change`` touched the source elements of either: the contact points are
        regenerated from their regenerated facets if touched, and the master surface is rebuilt from its facet set.

        The facets themselves were already regenerated in the topology-update phase by the implicit
        ``surfaceFacets`` modifier; this constraint is a pure reader of the resulting element sets.
        Everything cached here is derived from the reference configuration, so the rebuild is
        wholesale -- see the class docstring on why that is acceptable only while there is no
        frictional or multiplier history to carry.
        """

        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = self.slave.isTouchedBy(model, change)
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        if touchedSlave:
            self.slave = self.slave.regenerated(model)
        self._buildMaster(model.elementSets[self._masterSurfaceSetName])
        self._resetContactPoints()
        return True

    def _activeBlockSizes(self) -> list[int]:
        """The DOF count of each located contact point's dense block, in assembly order; cached per update of the
        connectivity, since the implicit path asks for it twice per evaluation."""

        if self._blockSizes is None:
            self._blockSizes = [len(dofs) for dofs in self._dofsOfLocatedPoints()]
        return self._blockSizes

    def _dofsOfLocatedPoints(self) -> list[np.ndarray]:
        """Per located point, its degrees of freedom in the block: those of its slave nodes, then those of the parent
        face nodes of its master facet."""

        nMasterDof = self.nMasterParentNodes * self.nDim
        return [
            np.concatenate(
                (
                    self.slave.interpolationOfLocatedPoint(k)[0],
                    self._nSlaveDof + k * nMasterDof + np.arange(nMasterDof),
                )
            )
            for k in range(len(self._locatedPoints))
        ]

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """One dense block per located contact point, row by row, over its slave and master DOFs."""
        return localEntriesOfSubBlocks([(dofs, dofs) for dofs in self._dofsOfLocatedPoints()])

    def shapeBlock(self, values: np.ndarray) -> ContactStiffnessView:
        return ContactStiffnessView(values, self._activeBlockSizes())

    def computeKernels(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        K: ContactStiffnessView | None,
        timeStep: TimeStep,
    ):
        """Add the residual contribution of the contact forces, and the tangent unless ``K`` is None.

        The gaps of all located points are evaluated at once: the contact points say where they are, the master
        points ride the *curved* parent surfaces (see the class docstring on why the master point must not be taken
        on the flat facet instead).
        """

        self.totalNormalForce = 0.0
        self._normalForceCurrent[:] = 0.0
        self._closed = np.zeros(0, dtype=np.intp)

        located = self._locatedPoints
        if not len(located):
            return

        nSlaveDof = self._nSlaveDof
        Nm = self._masterShapeFunctionsOfLocated
        nBar = self._normalsOfLocated

        gaps = self._gapsOfLocatedPoints(U_np, dU)
        self._gapCurrent[located] = gaps

        closed = np.flatnonzero(self.law.isClosed(gaps))
        self._closed = closed
        if not len(closed):
            return

        penaltyTimesArea = self.penalty * self._areasOfLocated[closed]
        f_n, stiffness = self.law.normalForce(penaltyTimesArea, gaps[closed])

        # The residual contribution is the negative of the contact force on the nodes, f_n * w with
        # w = kron([c, -Nm], nBar): the contact points distribute their part with their interpolation vectors, the
        # master facets theirs with their parent face shape functions.
        self.slave.addNormalForceResidual(P[:nSlaveDof], closed, f_n, nBar[closed])
        wMaster = ((-Nm[closed])[:, :, None] * nBar[closed][:, None, :]).reshape((len(closed), -1))
        P[nSlaveDof:].reshape((len(located), -1))[closed] += f_n[:, None] * wMaster

        self._normalForceCurrent[located[closed]] = f_n
        self.totalNormalForce = float(f_n.sum())

        if K is not None:
            # The tangent is only requested by the implicit path: one dense block per located point.
            for i, k in enumerate(closed):
                c = self.slave.interpolationOfLocatedPoint(k)[1]
                w = np.concatenate(((c[:, None] * nBar[k][None, :]).ravel(), wMaster[i]))
                K.blocks[k] += stiffness[i] * np.outer(w, w)

    def _gapsOfLocatedPoints(self, U_np: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """The gaps of all located points at once: the contact points say where they are, the master points ride the
        *curved* parent surfaces (see the class docstring on why the master point must not be taken on the flat facet
        instead).

        Parameters
        ----------
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.

        Returns
        -------
        numpy.ndarray
            The gap of each located point, negative in contact.
        """

        nSlaveDof = self._nSlaveDof
        xS = self.slave.coordinatesOfLocatedPoints(U_np[:nSlaveDof], dU[:nSlaveDof])
        xM = self._masterRefOfLocated + U_np[nSlaveDof:].reshape(self._masterRefOfLocated.shape)
        relative = xS - np.einsum("ai,aij->aj", self._masterShapeFunctionsOfLocated, xM)
        return np.einsum("aj,aj->a", self._normalsOfLocated, relative)

    def computeBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add the row sums of the absolute value of the stiffness of the closed points, weighted by column; see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeBoundOfAbsoluteStiffnessTimes`.

        A closed point :math:`p` has the stiffness :math:`k_p\\, w_p \\otimes w_p`, with :math:`w_p = [c, -N^m] \\otimes
        \\bar{n}` and the stiffness :math:`k_p = \\mathrm{d}f_n/\\mathrm{d}g` its law gives it at its gap (the one of the
        tangent). The absolute value of a sum is at most the sum of the absolute values, so

        .. math::

            b_a \\mathrel{+}= \\sum_p k_p\\, |w_{p,a}| \\, \\big(|w_p| \\cdot \\boldsymbol{w}\\big) :

        one product of the absolute interpolation vectors with the weights per point, and one distribution back, as the
        force is distributed.
        """

        located = self._locatedPoints
        if not len(located):
            return

        gaps = self._gapsOfLocatedPoints(U_np, dU)
        closed = np.flatnonzero(self.law.isClosed(gaps))
        if not len(closed):
            return

        _, stiffness = self.law.normalForce(self.penalty * self._areasOfLocated[closed], gaps[closed])

        nSlaveDof = self._nSlaveDof
        absoluteNormals = np.abs(self._normalsOfLocated[closed])
        absoluteSlaveInterpolation = abs(self.slave.interpolationMatrixOfLocatedPoints())[closed]
        absoluteMasterShapeFunctions = np.abs(self._masterShapeFunctionsOfLocated[closed])
        slaveWeights = weights[:nSlaveDof].reshape((-1, self.nDim))
        masterWeights = weights[nSlaveDof:].reshape((len(located), self.nMasterParentNodes, self.nDim))[closed]

        # |w_p| . weights, then k_p times it
        weightedByPoint = rowDot(absoluteNormals, absoluteSlaveInterpolation @ slaveWeights) + np.einsum(
            "ai,aij,aj->a", absoluteMasterShapeFunctions, masterWeights, absoluteNormals
        )
        stiffnessTimesWeighted = stiffness * weightedByPoint

        bound[:nSlaveDof].reshape((-1, self.nDim))[:] += absoluteSlaveInterpolation.T @ (
            stiffnessTimesWeighted[:, None] * absoluteNormals
        )
        bound[nSlaveDof:].reshape((len(located), self.nMasterParentNodes, self.nDim))[closed] += (
            stiffnessTimesWeighted[:, None, None]
            * absoluteMasterShapeFunctions[:, :, None]
            * absoluteNormals[:, None, :]
        )

    @property
    def contactForceOnSlave(self) -> np.ndarray:
        """The resultant contact force on the contact points at the last evaluation, :math:`-\\sum f_n \\bar{n}`.

        Returns
        -------
        numpy.ndarray
            The force.
        """

        closed = self._closed
        return -(self._normalForceCurrent[self._locatedPoints[closed]][:, None] * self._normalsOfLocated[closed]).sum(
            axis=0
        )

    @property
    def contactForceOnMaster(self) -> np.ndarray:
        """The resultant contact force on the master nodes at the last evaluation, the sum of the nodal forces
        :math:`f_n N^m_a \\bar{n}`; equal and opposite to :attr:`contactForceOnSlave` to round-off, since the parent
        face shape functions sum to one.

        Returns
        -------
        numpy.ndarray
            The force.
        """

        closed = self._closed
        f_n = self._normalForceCurrent[self._locatedPoints[closed]]
        sumOfNm = self._masterShapeFunctionsOfLocated[closed].sum(axis=1)
        return ((f_n * sumOfNm)[:, None] * self._normalsOfLocated[closed]).sum(axis=0)

    @property
    def penaltyEnergy(self) -> float:
        """The energy stored in the penalty springs at the last evaluation, for an energy balance.

        Returns
        -------
        float
            The energy.
        """

        points = self._locatedPoints[self._closed]
        return float(self.law.storedEnergy(self.penalty * self.slave.areas[points], self._gapCurrent[points]).sum())

    @property
    def deepestPenetration(self) -> float:
        """The largest penetration of a contact point at the last evaluation, a positive length; zero without
        contact.

        Returns
        -------
        float
            The penetration.
        """

        points = self._locatedPoints[self._closed]
        return float(-self._gapCurrent[points].min()) if len(points) else 0.0

    @property
    def numberOfPointsInContact(self) -> int:
        """The number of contact points that carried a force at the last evaluation.

        Returns
        -------
        int
            The number.
        """

        return len(self._closed)

    def getNormalPressures(self) -> np.ndarray:
        """The current normal contact pressures (positive in compression), one per contact point.

        Ordered like the contact points: for quadrature points, by slave facet and, within a facet, by quadrature
        point -- *not* by node, unlike the node-based constraint's accessor of the same name. A ``fromExpression``
        field output reading this therefore cannot be tied to a node set; use a reduction (``np.max``, ``np.mean``,
        ...) or :meth:`getSlaveNodalNormalForces` instead.
        """

        return self.slave.normalPressures(self._normalForceCurrent)

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating), ordered like
        :meth:`getNormalPressures`.

        A point with no assigned master facet reports 0, not a stale value from when it last had
        one -- see :meth:`updateConnectivity`. Returned as a copy, so a caller cannot corrupt the
        state a later augmented-Lagrange update would read.
        """

        return self._gapCurrent.copy()

    def getSlaveNodalNormalForces(self) -> np.ndarray:
        """The normal component of the contact force delivered to each slave surface node, for contact points on a
        slave surface of finite elements.

        Positive means pushed along the master's outward normal, i.e. repelled; **negative means
        pulled**, which is exactly the tensile corner load a serendipity face requires and a
        node-based scheme cannot produce. This is the accessor that demonstrates the formulation
        works, and the one a patch test should assert on.

        Ordered like the surface element generator's ``<prefix>_nodes`` node set of the slave
        surface (see :attr:`slaveSurfaceNodes`), so a ``fromExpression`` output over that set lines
        up. A parent-face node that is not itself a facet node -- possible only with
        ``triangulation=corner``, where the facets keep no midside nodes -- receives contact force
        but has no slot in that set, and is omitted here; use ``triangulation=midside`` for a
        complete picture.
        """

        return self.slave.nodalNormalForces(self._normalForceCurrent)
