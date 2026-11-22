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
from scipy.spatial import cKDTree

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.contactpoints import ContactPoints
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    checkFacetsCarryParentFaces,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.frozencontactsearch import FrozenContactSearch
from edelweissfe.constraints.base.penaltylaw import penaltyLaw, penaltyLawsAndTheTie
from edelweissfe.constraints.base.penaltysprings import (
    PenaltySprings,
    SpringGradientOnOwnNodes,
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
)
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    SurfaceContactPenaltySchema,
    contactPointsOfTheSlaveSurface,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.numerics.systemmatrixcontributor import localEntriesOfSubBlocks
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    FacetedSurface,
    closestFacetCandidates,
    closestFacets,
    facetNormalAndMeasure,
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
the search or the gap. With the law ``type=tied`` the same constraint *ties* its points to the master surface instead,
in every direction, with the gap or overlap at the pairing as the state at rest -- e.g. particles to finite elements,
see :class:`Constraint`, *The tie*. ``sliding=finite`` is rejected rather than approximated, see :class:`Constraint`.

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
    #: The point is tied to the point of the master surface it was paired with (the law ``tied``), for good.
    TIED = 4


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
    initialOverlap: str = schemaField(
        description="What an overlap of a contact point found at the first contact search stands for. 'pushApart': a "
        "real interference fit (a groove cut smaller than the thread), pushed apart -- at once, or over "
        "initialOverlapRampTime. 'adjust': an error of the geometry (flat faces standing for a curved surface): the "
        "overlap is kept as a stress-free offset of the point's gap for good, so that the point touches the master at "
        "the start without a force, and a point that separates and comes back meets the surface where it touched it. "
        "Not with type=tied, which keeps any gap or overlap at the pairing as the state at rest.",
        dtype=str,
        default="pushApart",
    )
    initialOverlapRampTime: float = schemaField(
        description="With initialOverlap=pushApart: the time over which the overlap found at the first contact "
        "search is pushed apart: it is taken out of the gap at first and fed back in smoothly (starting and ending "
        "at rest) over this time. 0: pushed apart at once.",
        dtype=float,
        default=0.0,
    )
    contactType: str = schemaField(
        description="The penalty law: 'linear' (linear force, constant stiffness with jump) or 'quadratic' "
        "(quadratic force, linear stiffness), or 'tied': the contact points are tied to the master surface, in "
        "tension and compression and in every direction, by linear springs of the penalty stiffness; each point is "
        "paired once, at the first contact search, and the gap or overlap it has then is the state at rest.",
        dtype=str,
        default="linear",
        optionName="type",
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
    the parametric location of the closest point *in that facet's parent face*, and the direction of the gap
    :math:`\\bar{n}`, the facet's unit normal except at a crease of the master (see *The gap at the corners of the
    master*). The gap is then

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

    The gap at the corners of the master
    ------------------------------------
    The gap is frozen as the first-order expansion of the *signed distance* of the point to the master surface at the
    update (:meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.signedDistances`): :math:`\\bar{n}` is the
    direction in which that distance grows. Where the closest point lies inside a facet, or on a seam of a smooth
    surface, that is the facet's normal. Where it lies on a *crease* -- an edge or a vertex where the facets meet at
    more than the angle of
    :data:`~edelweissfe.utils.facetcontactgeometry.cosOfSmoothNormalChangeLimit`, such as the crest and the root of a
    thread -- it is the line from the closest point to the contact point, and whether the point is inside or outside
    is told by the angle-weighted pseudonormal of the crease, which is right at convex and re-entrant corners alike.
    So a point beside a crest is outside, with the distance to the crest as its gap, and a point inside a root is
    pushed out the shortest way. On the boundary of a master surface that ends, the plane of its last facet goes on.

    The pairing
    -----------
    In every update of the connectivity, the contact points within ``searchDistance`` of the master surface (all of
    them, if no search distance is given) are paired with their closest facet, but only while they are outside the
    master body (:attr:`ContactState.APPROACHING`). Once a point is in contact (:attr:`ContactState.IN_CONTACT`), it
    is paired anew in every update with its closest facet, however deep it is, as it slides along the surface, from
    one face of the master onto the next and around a re-entrant corner -- but never across a convex crease (see
    :meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.acrossAConvexCrease`): the closest facet of a point
    near a convex corner of the master may be on its side, of a point deep inside a thin body on its far side, and
    would push the point out of the wrong side. A point that leaves the master body is released at once, also
    beside a crest. A point that was out of reach at the previous update and is found inside the master body has
    tunnelled through the search distance in one increment (:attr:`ContactState.FOUND_INSIDE`): it gets no force and
    is reported, and the search distance must exceed the distance a point travels towards the surface in one
    increment -- the constraint warns when a point enters deeper than half of it.

    The first update
    ----------------
    At the first update, the bodies are as the model gives them: a point found inside then is in contact. What its
    overlap :math:`g_0` stands for, ``initialOverlap`` says:

    - ``pushApart`` (the default): a real **interference fit**, such as a groove cut smaller than the thread. The
      overlap is pushed apart: at once, or with ``initialOverlapRampTime`` gradually -- :math:`g_0` is subtracted
      from the gap at first, and fed back in over that time, smoothly, starting and ending at rest.
    - ``adjust``: an **error of the geometry**, such as flat faces of particles standing for the curved hole a screw
      fits, whose chords lie inside the screw's curved faces. :math:`g_0` is subtracted from the gap for good: the
      master surface is taken as adjusted to touch the point, the point rests there without a force, and its gap is
      measured from there -- a point that separates and comes back meets the surface where it touched it.

    Either way the offset is a constant of the gap, so the tangent does not change, and the law (with it which
    points are closed, and the bound of the stiffness for the critical time step) sees the offset gap. The offsets
    belong to the contact points and are checkpointed; a new mesh of the master keeps them, contact points
    regenerated by a refinement have none.

    The tie
    -------
    With the law ``type=tied`` (:class:`~edelweissfe.constraints.base.penaltylaw.TiedPenaltyLaw`) the constraint
    does not let its points touch the master surface, it *ties* them to it -- e.g. the faces of particles near a
    screw to the finite elements of the far field. It is the same constraint, the same contact points, the same frozen
    projection, the same parent face shape functions; what differs is the pairing and the law:

    - **paired once.** At the first update, every point within ``searchDistance`` of the master surface (every point,
      without a search distance) is paired with the closest point of its closest facet, and stays paired with that
      *material* point of the master's parent face, :math:`N^m(\\xi_m) \\cdot x^m`, for good: tension does not
      release it, sliding does not move it on. It is paired anew only when the mesh of either side changes (see
      below).
    - **at rest where it is.** The vector :math:`\\boldsymbol{r}_0` from the master point to the point at the pairing
      is where the point rests: a gap or an overlap of the two surfaces -- of a coarse far field, whose faceted surface
      lies inside the curved surface of fine particles -- is part of the stress-free state, not closed or pushed apart.
    - **held in every direction.** A spring of the penalty stiffness along every axis holds the point there, in
      tension as in compression; together, with the displacement from rest
      :math:`\\boldsymbol{d} = \\boldsymbol{x}_p - N^m \\cdot x^m - \\boldsymbol{r}_0`, the force on the point is

      .. math::
          \\boldsymbol{f} = -k A_p \\, \\boldsymbol{d} , \\qquad
          \\boldsymbol{K} = k A_p \\, [c, -N^m] \\otimes [c, -N^m] \\otimes \\boldsymbol{I} ,

      each spring along its axis :math:`\\boldsymbol{e}_i` exactly the spring of a contact with :math:`\\bar{n}`
      replaced by :math:`\\boldsymbol{e}_i`. The force is linear in the displacements, its tangent constant and
      symmetric, its energy :math:`\\tfrac{1}{2} k A_p |\\boldsymbol{d}|^2`.

    The interface is as compliant as the springs: a traction :math:`t` opens it by :math:`t / k`. Choose ``penalty``
    well above the stiffness of the adjoining material over the size of its elements or particles, :math:`E / h`,
    and within what the time step of an explicit analysis can carry. The rest offset is a fixed vector, so a rotation
    of the interface by :math:`\\theta` is resisted by :math:`k |\\boldsymbol{r}_0| \\theta`: small for a small gap at
    rest, and zero without one.

    On a **change of the mesh** of either side (a refinement), the points are tied anew at the next update, and each
    new point takes the displacement from rest of the tied point of before closest to it, at the last evaluation: if
    only the master side was retiled, every point keeps its own, and the forces of the tie stay as they were.

    ``sliding=finite`` is rejected rather than approximated. It would require the exact gap gradient
    and Hessian of a point against a *curved* quadratic surface, whose closest-point projection is
    itself an iterative sub-problem -- a substantially different formulation from the flat-facet
    algebra of :mod:`~edelweissfe.utils.facetcontactgeometry`, not a variation on it.

    Like the node-based constraint, this is a
    :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either surface's source solid
    elements are refined mid-run, it rebuilds that side from the regenerated facets at its own next
    :meth:`updateConnectivity` tick. Since no frictional or multiplier history exists yet, the
    rebuild of a contact is unconditional and carries nothing across -- the point at which friction is added is
    the point at which that becomes a real problem, because quadrature points, unlike nodes, have no
    identity that survives a retiling. A tie hands its displacements from rest on, from point to closest point.

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

        self.law = penaltyLaw(configuration.contactType, penaltyLawsAndTheTie)

        self.sliding = configuration.sliding.lower()
        if self.sliding != "small":
            raise ValueError(
                f"Constraint sliding '{self.sliding}' is not supported by "
                "surfaceToDeformableSurfacePenalty: only 'small' is implemented. Finite sliding "
                "would require the exact gap gradient and Hessian against a curved quadratic "
                "surface, whose closest-point projection is itself an iterative sub-problem."
            )

        self.searchDistance = configuration.searchDistance

        initialOverlap = configuration.initialOverlap.lower()
        if initialOverlap not in ("pushapart", "adjust"):
            raise ValueError(
                f"Constraint '{name}': initialOverlap '{configuration.initialOverlap}' is not supported. Use "
                "'pushApart' or 'adjust'."
            )
        #: True if an overlap at the first contact search is kept as a stress-free offset of the gap for good
        #: (``initialOverlap=adjust``), False if it is pushed apart (``pushApart``).
        self.adjustsTheInitialOverlap = initialOverlap == "adjust"
        self.initialOverlapRampTime = configuration.initialOverlapRampTime
        if self.initialOverlapRampTime < 0.0:
            raise ValueError("The initial overlap ramp time must not be negative.")
        if self.adjustsTheInitialOverlap and self.initialOverlapRampTime > 0.0:
            raise ValueError(
                f"Constraint '{name}': an adjusted initial overlap is never pushed apart; it has no ramp time "
                "(initialOverlapRampTime is for initialOverlap=pushApart)."
            )
        if self.law.tiesThePoints and (self.adjustsTheInitialOverlap or self.initialOverlapRampTime > 0.0):
            raise ValueError(
                f"Constraint '{name}': a tie keeps the gap or overlap of its points at the pairing as the state at "
                "rest; initialOverlap and initialOverlapRampTime are for contact."
            )
        # The time of the first contact search, from which the initial overlap is offset; None before it, and
        # without an offset (an overlap pushed apart at once).
        self._initialOverlapTime = None

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
        return cls(
            name,
            model,
            contactPointsOfTheSlaveSurface(configuration, model, name),
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
        ).reshape((len(self.facetElements), self.nDim))
        self._masterFacets = FacetedSurface(self._masterNodesOfFacet)

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

        # The overlap of each point at the first contact search (zero or negative), recorded if it is not pushed apart
        # at once: the gap does not count it, for good (adjusted) or until the ramp has fed it in (pushed apart).
        self._initialOverlap = np.zeros(nPoints)

        # A tie: where each tied point is at rest relative to the master point it is tied to (the vector from that
        # point to it at the pairing), and its displacement from there and its position at the last evaluation. A
        # change of the mesh hands the displacements on to the points that replace them (refresh).
        self._restingOffset = np.zeros((nPoints, self.nDim))
        self._displacementFromRest = np.zeros((nPoints, self.nDim))
        self._positionCurrent = np.zeros((nPoints, self.nDim))
        self._displacementsFromRestHandedOn = None

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

        if self.law.tiesThePoints:
            if self._searchedBefore:
                # A tied point stays tied to its master point; only where its own nodes are may change (particles).
                return self._assignMasterFacets(self._assignedFacetIdx)
            return self._tieThePoints(model)

        coordinatesOfPoint = self.slave.currentPointCoordinates(model)
        coordinatesOfFacet, normalOfFacet, features = self._masterFacetsInTheConfigurationOf(model)
        facetCoords = list(coordinatesOfFacet)

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

        reached = closestFacet >= 0
        if self.searchDistance is not None:
            reached &= wasInContact | (distance <= self.searchDistance)

        # The signed distance to the closest facet tells inside from outside, also at the edges and the vertices of a
        # master that is not convex.
        gap = np.zeros(self.nPoints)
        direction = np.zeros((self.nPoints, self.nDim))
        gap[reached], direction[reached] = self._masterFacets.signedDistances(
            coordinatesOfPoint[reached],
            closestFacet[reached],
            weights[reached],
            coordinatesOfFacet,
            normalOfFacet,
            features,
        )
        penetrating = reached & (gap < 0.0)

        # A point in contact is pushed back out the way it came: it slides on over the master surface where that is
        # smooth and around a re-entrant crease, but does not move across a convex crease, onto the side of a corner
        # it is near or the far side of a thin master body.
        previousFacet = self._assignedFacetIdx
        mustStay = np.flatnonzero(penetrating & wasInContact)
        mustStay = mustStay[
            self._masterFacets.acrossAConvexCrease(
                previousFacet[mustStay], closestFacet[mustStay], coordinatesOfFacet, normalOfFacet
            )
        ]
        if len(mustStay):
            sameWay = []
            for point in mustStay:
                candidates = np.union1d(candidatesOfPoint[point], [previousFacet[point]]).astype(np.intp)
                across = self._masterFacets.acrossAConvexCrease(
                    np.full(len(candidates), previousFacet[point]), candidates, coordinatesOfFacet, normalOfFacet
                )
                sameWay.append(candidates[~across].tolist())
            closestFacet[mustStay], weights[mustStay], _ = closestFacets(
                coordinatesOfPoint[mustStay], facetCoords, sameWay
            )
            gap[mustStay], direction[mustStay] = self._masterFacets.signedDistances(
                coordinatesOfPoint[mustStay],
                closestFacet[mustStay],
                weights[mustStay],
                coordinatesOfFacet,
                normalOfFacet,
                features,
            )

        state = np.full(self.nPoints, ContactState.FREE, dtype=np.int8)
        state[reached & ~penetrating] = ContactState.APPROACHING
        state[penetrating & wasSeen] = ContactState.IN_CONTACT
        state[penetrating & ~wasSeen] = ContactState.FOUND_INSIDE

        self._reportEntriesAndPointsFoundInside(previousState, state, gap)
        self._stateOfPoint = state
        firstSearch = not self._searchedBefore
        self._searchedBefore = True

        paired = (state == ContactState.APPROACHING) | (state == ContactState.IN_CONTACT)
        newAssignment = np.where(paired, closestFacet, -1)
        pairedFacet = newAssignment[paired]
        self._freezeProjections(paired, pairedFacet, weights[paired], direction[paired])

        offsetsTheInitialOverlap = self.adjustsTheInitialOverlap or self.initialOverlapRampTime > 0.0
        if firstSearch and self._initialOverlapTime is None and offsetsTheInitialOverlap:
            self._recordTheInitialOverlap(model, coordinatesOfPoint[paired], pairedFacet, paired)

        return self._assignMasterFacets(newAssignment)

    def _recordTheInitialOverlap(
        self, model: FEModel, coordinatesOfPoints: np.ndarray, facets: np.ndarray, paired: np.ndarray
    ):
        """Record the overlap of the paired points at the first contact search as the offsets of their gaps (see the
        class documentation, *The first update*), and report the deepest.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.
        coordinatesOfPoints
            The current coordinates of the paired points.
        facets
            The master facet of each paired point.
        paired
            Which points are paired, a mask over all points.
        """

        self._initialOverlapTime = model.time
        self._initialOverlap[paired] = np.minimum(self._parentFaceGaps(model, coordinatesOfPoints, facets, paired), 0.0)

        overlapping = np.count_nonzero(self._initialOverlap)
        if overlapping:
            treatment = (
                "adjusted: kept as the state at rest"
                if self.adjustsTheInitialOverlap
                else f"pushed apart over {self.initialOverlapRampTime:.3e}"
            )
            self.journal.message(
                f"contact '{self.name}': {overlapping} point(s) overlap the master at the first search, the deepest by "
                f"{-self._initialOverlap.min():.3e}; {treatment}",
                self.identification,
            )

    def _masterFacetsInTheConfigurationOf(self, model: FEModel) -> tuple[np.ndarray, np.ndarray, dict]:
        """The master facets in the configuration of the model: their coordinates, unit normals, and features (see
        :meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.features`).

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.

        Returns
        -------
        tuple
            The coordinates of the facets (nFacets, nVerticesOfAFacet, nDim), their normals, and their features.
        """

        coordinatesOfFacet = currentNodeCoordinates(self._masterNodes, model, self._referenceCoordinatesOfMasterNodes)[
            self._masterNodesOfFacet
        ]
        normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
        return coordinatesOfFacet, normalOfFacet, self._masterFacets.features(coordinatesOfFacet, normalOfFacet)

    def _freezeProjections(self, paired: np.ndarray, facets: np.ndarray, weights: np.ndarray, directions: np.ndarray):
        """Freeze the projections of the paired points, and clear those of the others.

        The clamped barycentric weights locate the closest point in the flat facet; that location is mapped into the
        parent face's parametric space, and the parent face's shape functions are evaluated there. Frozen with them is
        the direction in which the gap grows.

        Parameters
        ----------
        paired
            Which points are paired, a mask over all points.
        facets
            The facet of each paired point.
        weights
            The barycentric weights of the closest point of each paired point in its facet.
        directions
            The direction in which the gap of each paired point grows.
        """

        parametric = np.einsum("pa,pad->pd", weights, self._vertexParametricCoordsOfFacet[facets])
        self._frozenMasterShapeFunctions[:] = 0.0
        self._frozenNormals[:] = 0.0
        if len(facets):
            self._frozenMasterShapeFunctions[paired] = parentFaceShapeFunctions(self._masterParentFaceType, parametric)
            self._frozenNormals[paired] = directions

    def _tieThePoints(self, model: FEModel) -> bool:
        """Tie every contact point within ``searchDistance`` of the master surface (every point, without a search
        distance) to the point of the master's parent face closest to it, at the configuration of the model, for good
        (see the class documentation, *The tie*). Where it is then, relative to that point, is where it rests; after
        a change of the mesh, displaced from there as the point it replaces was.

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
        coordinatesOfFacet, normalOfFacet, features = self._masterFacetsInTheConfigurationOf(model)
        facetCoords = list(coordinatesOfFacet)

        closestFacet, weights, distance = closestFacets(
            coordinatesOfPoint,
            facetCoords,
            closestFacetCandidates(coordinatesOfPoint, facetCoords, self.searchDistance),
        )
        tied = closestFacet >= 0
        if self.searchDistance is not None:
            tied &= distance <= self.searchDistance

        _, direction = self._masterFacets.signedDistances(
            coordinatesOfPoint[tied], closestFacet[tied], weights[tied], coordinatesOfFacet, normalOfFacet, features
        )
        self._freezeProjections(tied, closestFacet[tied], weights[tied], direction)
        self._stateOfPoint = np.where(tied, ContactState.TIED, ContactState.FREE).astype(np.int8)
        self._searchedBefore = True

        offset = self._relativePositionsToParentFaces(model, coordinatesOfPoint[tied], closestFacet[tied], tied)
        if self._displacementsFromRestHandedOn is not None:
            # Each new point is displaced from rest as the point of before closest to it was: the master side was
            # retiled under the same points, or the slave side made new points next to the old ones.
            positions, displacements = self._displacementsFromRestHandedOn
            _, closestBefore = cKDTree(positions).query(coordinatesOfPoint[tied])
            offset -= displacements[closestBefore]
            self._displacementsFromRestHandedOn = None
        self._restingOffset[tied] = offset

        nTied = int(np.count_nonzero(tied))
        largestGap = float(np.linalg.norm(offset, axis=1).max()) if nTied else 0.0
        self.journal.message(
            f"tie '{self.name}': {nTied}/{self.nPoints} points tied, largest gap or overlap at rest {largestGap:.3e}",
            self.identification,
        )
        if nTied < self.nPoints and self.searchDistance is not None:
            self.journal.message(
                f"tie '{self.name}': {self.nPoints - nTied} point(s) farther than the search distance "
                f"({self.searchDistance:.3e}) from the master surface are not tied.",
                self.identification,
                level=1,
            )

        return self._assignMasterFacets(np.where(tied, closestFacet, -1))

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

    def _parentFaceGaps(
        self, model: FEModel, coordinatesOfPoints: np.ndarray, facets: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """The gaps of paired points as an evaluation measures them, to the point of the master's parent face, in the
        configuration of the model: their relative positions (:meth:`_relativePositionsToParentFaces`) along their
        frozen directions.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.
        coordinatesOfPoints
            The current coordinates of the points.
        facets
            The master facet of each point.
        points
            The indices of the points, whose frozen projection is used.

        Returns
        -------
        numpy.ndarray
            The gaps.
        """

        return np.einsum(
            "pd,pd->p",
            self._frozenNormals[points],
            self._relativePositionsToParentFaces(model, coordinatesOfPoints, facets, points),
        )

    def _relativePositionsToParentFaces(
        self, model: FEModel, coordinatesOfPoints: np.ndarray, facets: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """Where paired points are relative to their points of the master's parent face, in the configuration of the
        model.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.
        coordinatesOfPoints
            The current coordinates of the points.
        facets
            The master facet of each point.
        points
            The indices of the points, whose frozen projection is used.

        Returns
        -------
        numpy.ndarray
            The vectors from the master points to the points, (nPoints, nDim).
        """

        parentNodes = [node for f in facets for node in self.facetElements[f].parentFaceNodes]
        reference = self._masterParentRefCoords[facets]
        parentCoordinates = currentNodeCoordinates(parentNodes, model, reference.reshape((-1, self.nDim))).reshape(
            reference.shape
        )
        masterPoints = np.einsum("pa,pad->pd", self._frozenMasterShapeFunctions[points], parentCoordinates)
        return coordinatesOfPoints - masterPoints

    def _uncountedFractionOfInitialOverlapAt(self, timeStep: TimeStep | None) -> float:
        """The fraction of the initial overlap the gap does not count at the time of ``timeStep``. Adjusted: one, for
        good. Pushed apart: zero at once; over a ramp time one at the first contact search, zero after the ramp time,
        in between :math:`1 - s + \\sin(2 \\pi s) / (2 \\pi)` of the elapsed fraction :math:`s` of the ramp
        time, which starts and ends at rest.

        Parameters
        ----------
        timeStep
            The time step of the evaluation; without an initial overlap to ramp, it may be None.

        Returns
        -------
        float
            The fraction.
        """

        if self._initialOverlapTime is None:
            return 0.0
        if self.adjustsTheInitialOverlap:
            return 1.0
        if timeStep is None:
            return 0.0
        s = min(max((timeStep.totalTime - self._initialOverlapTime) / self.initialOverlapRampTime, 0.0), 1.0)
        return 1.0 - s + np.sin(2.0 * np.pi * s) / (2.0 * np.pi)

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
        # evaluation). Reported only when it changes; a tie reported its points when it tied them.
        activity = (len(self._locatedPoints), int((self._gapCurrent < 0.0).sum()))
        if activity != self._lastReportedActivity and not self.law.tiesThePoints:
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
        self._initialOverlapOfLocated = self._initialOverlap[located]
        self._restingOffsetOfLocated = self._restingOffset[located]
        self._forcesOfTheTie = np.zeros((len(located), self.nDim))
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
            "initialOverlap": self._initialOverlap,
            "restingOffset": self._restingOffset,
            # Empty before the first search.
            "initialOverlapTime": np.array([] if self._initialOverlapTime is None else [self._initialOverlapTime]),
        }

    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        self._stateOfPoint = np.array(projection["stateOfPoint"], dtype=np.int8)
        self._searchedBefore = True
        self._frozenMasterShapeFunctions = np.array(projection["masterShapeFunctions"]).reshape(
            self._frozenMasterShapeFunctions.shape
        )
        self._frozenNormals = np.array(projection["normals"]).reshape(self._frozenNormals.shape)
        self._initialOverlap = np.array(projection["initialOverlap"]).reshape(self._initialOverlap.shape)
        self._restingOffset = np.array(projection["restingOffset"]).reshape(self._restingOffset.shape)
        initialOverlapTime = np.asarray(projection["initialOverlapTime"]).ravel()
        self._initialOverlapTime = float(initialOverlapTime[0]) if len(initialOverlapTime) else None
        assignment = np.array(projection["assignedFacetIdx"], dtype=np.intp)
        if self.slave.locatedOnlyWhenConnected:
            # Located at the next update of the connectivity, once the entities of the points are connected.
            self._assignedFacetIdx = assignment
            return False
        return self._assignMasterFacets(assignment)

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild both sides if ``change`` touched the source elements of either: the contact points are
        regenerated from their regenerated facets if touched, and the master surface is rebuilt from its facet set.

        The facets themselves were already regenerated in the topology-update phase by the implicit
        ``surfaceFacets`` modifier; this constraint is a pure reader of the resulting element sets.
        Everything cached here is derived from the reference configuration, so the rebuild is
        wholesale -- see the class docstring on why that is acceptable only while there is no
        frictional or multiplier history to carry. A tie carries its state: the points are tied anew at the next
        update, each displaced from rest as the tied point of before closest to it was at the last evaluation. The
        offsets of an initial overlap belong to the contact points: they stay with them if only the master was
        retiled, and are lost with them if the contact points are regenerated (reported).
        """

        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = self.slave.isTouchedBy(model, change)
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        tied = self._stateOfPoint == ContactState.TIED
        handedOn = (self._positionCurrent[tied], self._displacementFromRest[tied]) if tied.any() else None

        initialOverlap = self._initialOverlap
        if touchedSlave:
            self.slave = self.slave.regenerated(model)
            if initialOverlap.any():
                self.journal.message(
                    f"contact '{self.name}': the contact points were regenerated; the offsets of their initial "
                    f"overlap (deepest {-initialOverlap.min():.3e}) are lost, and an overlap they still have is pushed "
                    "apart at once.",
                    self.identification,
                    level=1,
                )
        self._buildMaster(model.elementSets[self._masterSurfaceSetName])
        self._resetContactPoints()
        if not touchedSlave:
            self._initialOverlap = initialOverlap
        self._displacementsFromRestHandedOn = handedOn
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
        """Add the residual contribution of the penalty springs at the located points, and the tangent unless ``K``
        is None.

        All located points are evaluated at once: where they are relative to their master points
        (:meth:`_relativePositionsOfLocatedPoints`), the springs this gives them (:meth:`_springsOfLocatedPoints`: the
        contact's one along :math:`\\bar{n}`, the tie's one along every axis), and the forces of the springs on the
        nodes (:meth:`_addForcesOfSprings`).
        """

        self.totalNormalForce = 0.0
        self._normalForceCurrent[:] = 0.0
        self._closed = np.zeros(0, dtype=np.intp)

        located = self._locatedPoints
        if not len(located):
            return

        xS, relative = self._relativePositionsOfLocatedPoints(U_np, dU)
        springs = self._springsOfLocatedPoints(relative, timeStep)
        for spring in springs:
            self._addForcesOfSprings(P, K, spring)

        # For the output: the gap the law sees, along the normal frozen at the pairing, and the force along it.
        nBar = self._normalsOfLocated
        self._gapCurrent[located] = self._gapsOfLocatedPoints(relative, timeStep)
        if self.law.tiesThePoints:
            forces = np.column_stack([spring.forces for spring in springs])
            self._closed = np.arange(len(located))
            self._positionCurrent[located] = xS
            self._displacementFromRest[located] = relative - self._restingOffsetOfLocated
            self._forcesOfTheTie = forces
            self._normalForceCurrent[located] = np.einsum("aj,aj->a", nBar, forces)
            self.totalNormalForce = float(self._normalForceCurrent[located].sum())
        elif springs:
            (spring,) = springs
            self._closed = spring.points
            self._normalForceCurrent[located[spring.points]] = spring.forces
            self.totalNormalForce = float(spring.forces.sum())

    def _relativePositionsOfLocatedPoints(self, U_np: np.ndarray, dU: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where the located points are, and where relative to their master points: the contact points say where they
        are, the master points ride the *curved* parent surfaces (see the class docstring on why the master point must
        not be taken on the flat facet instead). The gap of a contact is the component of the relative position along
        :math:`\\bar{n}`, the stretch of a tie's springs its departure from the resting offset.

        Parameters
        ----------
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The current coordinates of the located points, and the vectors from their master points to them, each
            (nLocated, nDim).
        """

        nSlaveDof = self._nSlaveDof
        xS = self.slave.coordinatesOfLocatedPoints(U_np[:nSlaveDof], dU[:nSlaveDof])
        xM = self._masterRefOfLocated + U_np[nSlaveDof:].reshape(self._masterRefOfLocated.shape)
        return xS, xS - np.einsum("ai,aij->aj", self._masterShapeFunctionsOfLocated, xM)

    def _springsOfLocatedPoints(self, relative: np.ndarray, timeStep: TimeStep | None) -> list[PenaltySprings]:
        """The penalty springs at the located points, from where they are relative to their master points.

        - A contact has one spring along :math:`\\bar{n}` at every closed point. Its gap is
          :math:`g = \\bar{n} \\cdot (x - x^m)`, less the part of the initial overlap it does not count (all of an
          adjusted one, the part of one pushed apart that the ramp has not fed in yet; :meth:`_gapsOfLocatedPoints`):
          a constant offset of the gap, which leaves its gradient, and so the tangent, as it is. The law says which
          points are closed, and their force and stiffness.
        - A tie has one spring along every axis :math:`e_i` at every point. Its "gap" is the displacement of the point
          from rest along the axis, :math:`(x - x^m - r_0) \\cdot e_i`, with the law's force and stiffness.

        The force (:meth:`_addForcesOfSprings`) and the bound of the stiffness
        (:meth:`computeBoundOfAbsoluteStiffnessTimes`) go through the same springs.

        Parameters
        ----------
        relative
            Where the located points are relative to their master points, see
            :meth:`_relativePositionsOfLocatedPoints`.
        timeStep
            The time step of the evaluation, for the ramp of an initial overlap.

        Returns
        -------
        list[PenaltySprings]
            The springs; empty for a contact with no closed point.
        """

        penaltyTimesArea = self.penalty * self._areasOfLocated

        if self.law.tiesThePoints:
            everyPoint = np.arange(len(relative))
            displacementFromRest = relative - self._restingOffsetOfLocated
            springs = []
            for axis, direction in enumerate(np.eye(self.nDim)):
                forces, stiffness = self.law.normalForce(penaltyTimesArea, displacementFromRest[:, axis])
                springs.append(
                    PenaltySprings(everyPoint, np.broadcast_to(direction, relative.shape), forces, stiffness)
                )
            return springs

        gaps = self._gapsOfLocatedPoints(relative, timeStep)
        closed = np.flatnonzero(self.law.isClosed(gaps))
        if not len(closed):
            return []
        forces, stiffness = self.law.normalForce(penaltyTimesArea[closed], gaps[closed])
        return [PenaltySprings(closed, self._normalsOfLocated[closed], forces, stiffness)]

    def _gapsOfLocatedPoints(self, relative: np.ndarray, timeStep: TimeStep | None) -> np.ndarray:
        """The gaps of the located points as the law sees them: :math:`g = \\bar{n} \\cdot (x - x^m)`, less the part
        of the initial overlap not counted at the time of ``timeStep`` -- all of an adjusted one, the part of one
        pushed apart over a ramp time that the ramp has not fed in yet (see the class documentation, *The first
        update*).

        Parameters
        ----------
        relative
            Where the located points are relative to their master points, see
            :meth:`_relativePositionsOfLocatedPoints`.
        timeStep
            The time step of the evaluation.

        Returns
        -------
        numpy.ndarray
            The gaps.
        """

        gaps = np.einsum("aj,aj->a", self._normalsOfLocated, relative)
        uncountedFraction = self._uncountedFractionOfInitialOverlapAt(timeStep)
        if uncountedFraction:
            gaps = gaps - uncountedFraction * self._initialOverlapOfLocated
        return gaps

    def _addForcesOfSprings(self, P: np.ndarray, K: ContactStiffnessView | None, springs: PenaltySprings):
        """Add the residual contribution of penalty springs at located points, each along a unit direction
        :math:`\\bar{n}` (the normal of a contact, an axis of a tie), and their tangent unless ``K`` is None.

        The residual contribution is the negative of the force on the nodes, :math:`f\\, w` with
        :math:`w = [c, -N^m] \\otimes \\bar{n}`: the contact points distribute their part with their interpolation
        vectors, the master facets theirs with their parent face shape functions. The tangent is
        :math:`\\mathrm{d}f/\\mathrm{d}g \\; w \\otimes w`.

        Parameters
        ----------
        P
            The residual contribution of the constraint; augmented.
        K
            The tangent of the constraint, one dense block per located point; augmented. None on the explicit path.
        springs
            The springs, see :meth:`_springsOfLocatedPoints`.
        """

        points, directions, forces, stiffness = springs.points, springs.directions, springs.forces, springs.stiffness
        nSlaveDof = self._nSlaveDof
        self.slave.addForceResidual(P[:nSlaveDof], points, forces, directions)
        wMaster = ((-self._masterShapeFunctionsOfLocated[points])[:, :, None] * directions[:, None, :]).reshape(
            (len(points), -1)
        )
        P[nSlaveDof:].reshape((len(self._locatedPoints), -1))[points] += forces[:, None] * wMaster

        if K is not None:
            # The tangent is only requested by the implicit path: one dense block per located point.
            for i, k in enumerate(points):
                c = self.slave.interpolationOfLocatedPoint(k)[1]
                w = np.concatenate(((c[:, None] * directions[i][None, :]).ravel(), wMaster[i]))
                K.blocks[k] += stiffness[i] * np.outer(w, w)

    def computeBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add the row sums of the absolute value of the stiffness of the springs, weighted by column; see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeBoundOfAbsoluteStiffnessTimes`.

        The springs are those of the force (:meth:`_springsOfLocatedPoints`): the contact's one along :math:`\\bar{n}`
        at every closed point, with its gap (less the initial overlap not counted) and its frozen direction (the
        facet normal or, at a crease of the master, the direction to the closest point); the tie's one along every axis
        at every point. A spring :math:`p` has the gradient :math:`w_p = [c, -N^m] \\otimes \\bar{n}_p` and the
        stiffness :math:`k_p = \\mathrm{d}f/\\mathrm{d}g` its law gives it (the one of the tangent); the bound is the one
        of every penalty spring
        (:func:`~edelweissfe.constraints.base.penaltysprings.addBoundOfAbsoluteStiffnessOfSprings`), with the
        contact points' interpolation over the slave nodes they share and the parent face nodes of each point's master
        facet. The springs of a tie along different axes act on different components, so for a tie this is the row
        sum itself.
        """

        located = self._locatedPoints
        if not len(located):
            return

        _, relative = self._relativePositionsOfLocatedPoints(U_np, dU)
        springs = self._springsOfLocatedPoints(relative, timeStep)
        if not springs:
            return

        nSlaveDof = self._nSlaveDof
        slaveInterpolationOfLocated = self.slave.interpolationMatrixOfLocatedPoints()
        dofsOfSlaveNodes = np.arange(nSlaveDof).reshape((-1, self.nDim))
        dofsOfMasterNodesOfLocated = nSlaveDof + np.arange(len(located) * self.nMasterParentNodes * self.nDim).reshape(
            (len(located), self.nMasterParentNodes, self.nDim)
        )
        for spring in springs:
            points = spring.points
            gradient = [
                SpringGradientOnSharedNodes(slaveInterpolationOfLocated[points], spring.directions, dofsOfSlaveNodes),
                SpringGradientOnOwnNodes(
                    -self._masterShapeFunctionsOfLocated[points], spring.directions, dofsOfMasterNodesOfLocated[points]
                ),
            ]
            addBoundOfAbsoluteStiffnessOfSprings(bound, weights, spring.stiffness, gradient)

    @property
    def contactForceOnSlave(self) -> np.ndarray:
        """The resultant contact force on the contact points at the last evaluation, :math:`-\\sum f_n \\bar{n}`.

        Returns
        -------
        numpy.ndarray
            The force.
        """

        closed = self._closed
        if self.law.tiesThePoints:
            return -self._forcesOfTheTie.sum(axis=0)
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
        sumOfNm = self._masterShapeFunctionsOfLocated[closed].sum(axis=1)
        if self.law.tiesThePoints:
            return (sumOfNm[:, None] * self._forcesOfTheTie).sum(axis=0)
        f_n = self._normalForceCurrent[self._locatedPoints[closed]]
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
        if self.law.tiesThePoints:
            return float(
                self.law.storedEnergy(
                    self.penalty * self.slave.areas[points][:, None], self._displacementFromRest[points]
                ).sum()
            )
        return float(self.law.storedEnergy(self.penalty * self.slave.areas[points], self._gapCurrent[points]).sum())

    @property
    def deepestPenetration(self) -> float:
        """The largest penetration of a contact point at the last evaluation, a positive length, measured as the law
        sees it (beyond an adjusted initial overlap); zero without contact.

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

    def getTractions(self) -> np.ndarray:
        """The traction on each contact point at the last evaluation: the force on it per unit of the area it stands
        for, :math:`-f_n \\bar{n} / A_p` in contact, :math:`-\\boldsymbol{f} / A_p` in a tie, with its normal and its
        tangential part; zero where a point carries no force. Ordered like :meth:`getNormalPressures`.

        Returns
        -------
        numpy.ndarray
            The tractions, (nPoints, nDim).
        """

        located = self._locatedPoints
        if self.law.tiesThePoints:
            forces = -self._forcesOfTheTie
        else:
            forces = -self._normalForceCurrent[located][:, None] * self._normalsOfLocated
        tractions = np.zeros((self.nPoints, self.nDim))
        tractions[located] = forces / self.slave.areas[located][:, None]
        return tractions

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating) as the law sees it, less an initial
        overlap it does not count, ordered like :meth:`getNormalPressures`.

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
