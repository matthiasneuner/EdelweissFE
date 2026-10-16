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
"""Octree refinement of HEX20 elements: subdivision, 2:1 balancing, hanging-node classification and
the conformity check -- all decided topologically, in exact integer arithmetic.

Vocabulary (see also :mod:`edelweissfe.adaptivity.rootentities`)
-----------------------------------------------------------------
Cell
    An element of the octree: a root element, or a child of a refined cell. A cell that has not been
    refined is *active* -- a leaf of the octree, and an element of the model. A refined cell stays in
    the octree, inactive, as the parent of its children: the octree is the history of the refinement.
Reference box
    Every cell lies inside one root element and covers an axis-aligned box of that root's reference
    lattice: an integer origin and an integer edge length (``2M`` for the root itself, divided by the
    split factor at every level). A cell's own nodes therefore sit at exact lattice points.
Own node / hanging node
    A node is one of a cell's *own* nodes if it is in the cell's connectivity. A node that lies
    on the boundary of an active cell without being one of its own nodes *hangs* on it: its value is
    tied to the cell's trace on that edge or face by a hanging-node constraint.
Conformity
    The active mesh is conforming up to the hanging-node constraints iff every node on the boundary of
    an active cell is an own node or a hanging node (:meth:`AdaptiveMesh.check_conformity`).

Physical coordinates are not decided here, and not stored here either: a cell is subdivided with the
isoparametric map of its nodes' reference coordinates as the model's mesh
(:class:`~edelweissfe.models.mesh.Mesh`) describes them at that moment -- so curved parents stay curved,
and a node another modifier has moved is subdivided from where it is now -- and the nodes a
subdivision creates are described into that mesh. No decision about which nodes coincide, touch or
hang is ever taken from coordinates.

Storage
-------
The octree is held as arrays, one row per cell (:class:`OctreeCells`), rather than as one Python
object per cell; the cells are read through named accessors (:meth:`AdaptiveMesh.level`,
:meth:`AdaptiveMesh.children`, ...). This keeps the mirror of a mesh of a hundred thousand elements at
some ten megabytes, held identically by every process of a domain-decomposed run.
"""

from collections import defaultdict
from fractions import Fraction
from itertools import product
from typing import NamedTuple

import numpy as np

from edelweissfe.adaptivity.rootentities import (
    LABEL_DTYPE,
    RootEntityTable,
    grownToCapacity,
    requireLabelsFitTheirArrays,
)
from edelweissfe.numerics.mpctransformation import _flattenChainedRecords
from edelweissfe.utils.exceptions import TopologyError

#: The deepest refinement level the exact reference lattice can represent. The lattice resolution is
#: ``splitFactor ** MAXIMUM_REFINEMENT_DEPTH`` units per half root edge, a 64-bit integer for a split
#: factor up to 8; refining beyond this depth raises.
MAXIMUM_REFINEMENT_DEPTH = 20


class NodeRegistry:
    """Hands out node labels for the nodes that refinement creates.

    A new node is named by its *identity* -- which corners of its parent span it, and in what exact
    integer proportion (see :meth:`AdaptiveMesh._childConnectivity`) -- so every parent that touches
    the point derives the same identity and gets the same label. Identities are kept per body
    (``componentId``): two bodies touching at a flush interface never share a node. An identity is
    stored packed into bytes, the labels and weights as 64-bit integers.

    The coordinates of a new node are held here only until it is described in the mesh
    (:meth:`take_undescribed`), at the end of a refinement pass.

    Parameters
    ----------
    reserve_labels
        The ``count -> range`` label allocator of the model this registry mirrors, i.e.
        ``TopologyPipeline.reserveNodeNumbers``, so model and octree draw from one counter and cannot collide.
        Without one (a standalone octree, e.g. in tests) the registry mints ``max + 1`` itself.
    """

    def __init__(self, reserve_labels=None):
        self._labelOfIdentity = {}  # packed (componentId, identity) -> label
        self._undescribed = {}  # label -> coordinates of a new node not yet described in the mesh
        self._maxLabel = 0
        self._reserveLabels = reserve_labels

    def reserve_labels_up_to(self, label: int):
        """Raise the label high-water mark, so new labels cannot collide with labels already used --
        those of the root elements, and those of nodes outside the refineable mesh, e.g. of contact
        facets."""
        self._maxLabel = max(self._maxLabel, int(label))

    def label_of_new_node(self, identity, coord, componentId: int = 0) -> int:
        """The label of the node with the given identity, minting a fresh one the first time.

        Only minting draws a label; looking up a known identity -- the common case, since most new
        nodes are shared by several children -- consumes none.

        Parameters
        ----------
        identity
            The spanning corner labels and their integer weights, as ``((label, weight), ...)``,
            ascending.
        coord
            The node's coordinates, kept until the node is described in the mesh.
        componentId
            The body of the node.
        """
        key = np.array([componentId, *(v for pair in identity for v in pair)], dtype=np.int64).tobytes()
        label = self._labelOfIdentity.get(key)
        if label is None:
            label = self._mint()
            self._labelOfIdentity[key] = label
            self._undescribed[label] = np.array(coord, dtype=float)
        return label

    def is_undescribed(self, label: int) -> bool:
        """Whether ``label`` is a new node not yet described in the mesh."""
        return label in self._undescribed

    def coordinates_of_undescribed(self, label: int) -> np.ndarray:
        """The coordinates of a new node not yet described in the mesh."""
        return self._undescribed[label]

    def take_undescribed(self) -> tuple:
        """The new nodes not yet described in the mesh, in the order they were minted, as labels and
        coordinates; they are forgotten here."""
        labels = list(self._undescribed)
        coordinates = [self._undescribed[label] for label in labels]
        self._undescribed = {}
        return labels, coordinates

    def _mint(self) -> int:
        """One fresh label, from the model's allocator if this registry was given one."""
        if self._reserveLabels is None:
            label = self._maxLabel + 1
        else:
            (label,) = self._reserveLabels(1)
            if label <= self._maxLabel:
                # the allocator is monotonic and was told about every label the registry knows, so
                # this means registry and model were wired to different models
                raise ValueError(
                    "the node allocator handed out label {:d}, which the refinement registry "
                    "already uses; registry and model are out of sync".format(label)
                )
        requireLabelsFitTheirArrays([label])
        self._maxLabel = label
        return label


class OctreeCells:
    """The cells of the octree as a struct of arrays: one row per cell, the row being the cell's id.

    Every cell ever created keeps its row (a refined cell is the parent of its children), and ids
    start at 1, so row 0 is unused. The children of a cell are created together and have consecutive
    ids.

    ==================  ===========  =========================================================
    array               dtype        per cell
    ==================  ===========  =========================================================
    ``connectivity``    int32        the labels of its own nodes, in the topology's local order
    ``level``           int8         0 for a root, the parent's level + 1 for a child
    ``parent``          int32        the cell it was split from; -1 for a root
    ``firstChild``      int32        its first child; -1 while it is not refined
    ``root``            int32        the root element it lies in
    ``component``       int32        the body it belongs to
    ``isActive``        bool         not refined: a leaf of the octree, an element of the model
    ``boxOrigin``       int64 (3)    the origin of its reference box in its root's lattice
    ``boxEdgeLength``   int64        the edge length of its reference box
    ``nodeKeyIds``      int32        per own node, its key's number (see
                                     :meth:`AdaptiveMesh.node_keys`); -1 until computed
    ==================  ===========  =========================================================

    Parameters
    ----------
    nodesPerCell
        The number of nodes of a cell (20 for HEX20).
    """

    def __init__(self, nodesPerCell: int):
        self.count = 1  # the next id; ids start at 1
        self.connectivity = np.zeros((0, nodesPerCell), dtype=LABEL_DTYPE)
        self.level = np.zeros(0, dtype=np.int8)
        self.parent = np.zeros(0, dtype=np.int32)
        self.firstChild = np.zeros(0, dtype=np.int32)
        self.root = np.zeros(0, dtype=np.int32)
        self.component = np.zeros(0, dtype=np.int32)
        self.isActive = np.zeros(0, dtype=bool)
        self.boxOrigin = np.zeros((0, 3), dtype=np.int64)
        self.boxEdgeLength = np.zeros(0, dtype=np.int64)
        self.nodeKeyIds = np.zeros((0, nodesPerCell), dtype=np.int32)

    def append(self, number: int) -> range:
        """Make rows for ``number`` new cells (all values zero; the caller fills them in) and return
        their ids."""
        first, end = self.count, self.count + number
        if end > self.level.shape[0]:
            # grown by a quarter, not doubled: a mirror of a hundred thousand roots refined in a few
            # places would otherwise carry a hundred thousand empty rows
            capacity = max(end, self.level.shape[0] + self.level.shape[0] // 4)
            self.connectivity = grownToCapacity(self.connectivity, capacity)
            self.level = grownToCapacity(self.level, capacity)
            self.parent = grownToCapacity(self.parent, capacity)
            self.firstChild = grownToCapacity(self.firstChild, capacity)
            self.root = grownToCapacity(self.root, capacity)
            self.component = grownToCapacity(self.component, capacity)
            self.isActive = grownToCapacity(self.isActive, capacity)
            self.boxOrigin = grownToCapacity(self.boxOrigin, capacity)
            self.boxEdgeLength = grownToCapacity(self.boxEdgeLength, capacity)
            self.nodeKeyIds = grownToCapacity(self.nodeKeyIds, capacity)
        self.count = end
        return range(first, end)


class HangingNode(NamedTuple):
    """A hanging node: its value is ``sum(weights[i] * value(masters[i]))``, the trace of the master
    element on the edge or face (``kind``) the node lies on."""

    slave: int
    kind: str
    masters: list
    weights: np.ndarray


class InternalPlane(NamedTuple):
    """A lattice plane inside a root element: reference coordinate ``axis`` equals ``value``."""

    rootEid: int
    axis: int
    value: int


class Rectangle(NamedTuple):
    """An axis-aligned rectangle ``[uMin, uMax] x [vMin, vMax]`` of lattice coordinates on a face."""

    uMin: int
    uMax: int
    vMin: int
    vMax: int

    def overlaps(self, other: "Rectangle") -> bool:
        """True if the two rectangles overlap with positive area (touching edges do not count)."""
        return min(self.uMax, other.uMax) > max(self.uMin, other.uMin) and min(self.vMax, other.vMax) > max(
            self.vMin, other.vMin
        )


class FaceOnEntity(NamedTuple):
    """An element face: the entity it lies on (a :class:`~edelweissfe.adaptivity.rootentities.RootFace`
    or an :class:`InternalPlane`) and the rectangle it covers there."""

    entity: object
    rectangle: Rectangle


class AdaptiveMesh:
    """Octree hierarchy of HEX20 elements: refinement, 2:1 balancing and hanging-node classification.

    Every decision about how elements and nodes relate -- which nodes coincide, which node lies on
    which face or edge, which elements share a face, which nodes hang and with which weights, which new
    nodes join a node set -- is taken topologically and in exact integer arithmetic, never from physical
    coordinates: each cell knows its root and its exact reference box, and the root mesh's
    vertices, edges and faces are named by their corner labels
    (:class:`~edelweissfe.adaptivity.rootentities.RootEntityTable`). This holds on curved, warped and
    arbitrarily oriented unstructured meshes alike. The only requirement is a conforming root mesh
    without collapsed elements. :meth:`check_conformity` verifies the result after every adaptation.

    The node coordinates are the mesh's (``modelMesh``): a cell is subdivided from the reference
    coordinates its nodes have there when it is refined, and :meth:`describe_new_nodes` describes the
    nodes a refinement pass created there.

    Parameters
    ----------
    modelMesh
        The :class:`~edelweissfe.models.mesh.Mesh` describing the nodes -- of the root elements, and
        those refinement creates.
    splitFactor
        ``n``: a refined element is split into ``n`` parts per axis, i.e. ``n**3`` children.
    topology
        The element :class:`~edelweissfe.adaptivity.topologybase.TopologyBase` (default: HEX20).
    reserve_labels
        The model's node-label allocator, see :class:`NodeRegistry`.
    """

    def __init__(self, modelMesh, splitFactor: int = 2, topology=None, reserve_labels=None):
        if topology is None:
            from edelweissfe.adaptivity.hex20topology import Hex20Topology

            topology = Hex20Topology()
        self.topology = topology
        self.modelMesh = modelMesh
        self.registry = NodeRegistry(reserve_labels=reserve_labels)
        self.splitFactor = splitFactor
        #: M: reference coordinate xi of a root is the lattice integer (xi + 1) * M
        self.latticeUnits = splitFactor**MAXIMUM_REFINEMENT_DEPTH
        if 2 * self.latticeUnits >= 2**62:
            raise ValueError(
                f"a split factor of {splitFactor} is too large for the exact 64-bit reference lattice "
                f"({MAXIMUM_REFINEMENT_DEPTH} levels); it can be at most 8"
            )
        self._cells = OctreeCells(len(self.topology.reference_node_lattice))
        self.elementSets = {}  # name -> set(eid)      (children inherit membership on refine)
        self.nodeSets = {}  # name -> set(node label)
        self.surfaces = {}  # name -> set((eid, faceID))  (element-based, Marmot faceID convention)
        #: the node keys of the cells' own nodes, numbered: the number of a key packed into an integer
        #: (RootEntityTable.pack), and the packed key of each number
        self._numberOfPackedKey = {}
        self._packedKeys = []
        n = splitFactor
        lattice = self.topology.reference_node_lattice  # own nodes, 0..2 per axis
        #: per child (in subdivision order ix * n * n + iy * n + iz): its nodes on the parent's lattice
        #: with 2n units per edge -- the child at position i along an axis covers [2i, 2i + 2] there
        self._childLattice = [
            [tuple(2 * i + int(p[k]) for k, i in enumerate(position)) for p in lattice]
            for position in product(range(n), repeat=3)
        ]
        #: the parent's own nodes on the same lattice, so a child node landing on one reuses it
        self._ownNodeAt = {tuple(int(v) * n for v in p): slot for slot, p in enumerate(lattice)}
        self._cornerSlots = self.topology.corner_slots
        self.rootEntities = RootEntityTable(self.topology, self.latticeUnits)
        #: result of _nodesOnElementBoundaries for the current active mesh; reset by every change
        self._boundaryNodesCache = None

    # ---- topological containers ----
    def define_element_set(self, name, eids):
        self.elementSets[name] = set(eids)

    def define_node_set(self, name, labels):
        self.nodeSets[name] = set(labels)

    def define_surface(self, name, pairs):
        """pairs: iterable of (eid, faceID) with Marmot faceID (1-6)."""
        self.surfaces[name] = set(pairs)

    # ---- the cells ----
    def cell_ids(self) -> range:
        """The ids of all cells, active and refined, ascending."""
        return range(1, self._cells.count)

    def is_active(self, eid: int) -> bool:
        """Whether cell ``eid`` is active: not refined, a leaf of the octree and an element of the model."""
        return bool(self._cells.isActive[eid])

    def level(self, eid: int) -> int:
        """The refinement level of cell ``eid``: 0 for a root element."""
        return int(self._cells.level[eid])

    def parent(self, eid: int) -> int | None:
        """The cell ``eid`` was split from; None for a root element."""
        parent = int(self._cells.parent[eid])
        return None if parent < 0 else parent

    def children(self, eid: int) -> list:
        """The children of cell ``eid`` in subdivision order (see :meth:`refine`); empty while it is not
        refined."""
        first = int(self._cells.firstChild[eid])
        return [] if first < 0 else list(range(first, first + self.splitFactor**3))

    def connectivity(self, eid: int) -> list:
        """The labels of the own nodes of cell ``eid``, in the topology's local node order."""
        return self._cells.connectivity[eid].tolist()

    def component(self, eid: int) -> int:
        """The body cell ``eid`` belongs to."""
        return int(self._cells.component[eid])

    def root_of(self, eid: int) -> int:
        """The root element cell ``eid`` lies in."""
        return int(self._cells.root[eid])

    def reference_box(self, eid: int) -> tuple:
        """The reference box of cell ``eid`` in its root's lattice, as its origin (three integers) and
        its edge length."""
        return tuple(self._cells.boxOrigin[eid].tolist()), int(self._cells.boxEdgeLength[eid])

    def node_coordinates(self, labels) -> np.ndarray:
        """The reference coordinates of nodes, one row per label: as the mesh describes them, or, for a
        node of the current refinement pass not yet described there, as its subdivision placed it."""
        labels = list(labels)
        registry = self.registry
        undescribed = [registry.is_undescribed(label) for label in labels]
        described = [label for label, isNew in zip(labels, undescribed) if not isNew]
        rows = iter(self.modelMesh.nodeCoordinates[self.modelMesh.rowsOfNodes(described)])
        return np.array(
            [
                registry.coordinates_of_undescribed(label) if isNew else next(rows)
                for label, isNew in zip(labels, undescribed)
            ]
        )

    def element_coordinates(self, eid: int) -> np.ndarray:
        """The reference coordinates of the own nodes of cell ``eid``, one row per node."""
        return self.node_coordinates(self.connectivity(eid))

    def describe_new_nodes(self) -> list:
        """Describe the nodes created since the last call in the mesh, in the order they were created.

        Called once per refinement pass rather than per refined cell, so that the mesh indexes its
        nodes once per pass.

        Returns
        -------
        list
            Their labels.
        """
        labels, coordinates = self.registry.take_undescribed()
        if labels:
            self.modelMesh.addNodes(labels, coordinates)
        return labels

    # ---- building the hierarchy ----
    def add_root(self, labels, componentId: int = 0) -> int:
        """Add a root (level-0) element by its 20 node labels, in C3D20 order; see :meth:`add_roots`."""
        (eid,) = self.add_roots([labels], [componentId])
        return eid

    def add_roots(self, connectivities, componentIds) -> range:
        """Add root (level-0) elements by their node labels, in C3D20 order.

        The labels are the model's own, and each must be described in the mesh. ``componentIds`` are
        the connected bodies the elements belong to: node identities are kept per body, so two bodies
        sharing a flush interface are never welded together.

        Parameters
        ----------
        connectivities
            Per root, its node labels.
        componentIds
            Per root, its body.

        Returns
        -------
        range
            The ids of the new roots, in the order given.

        Raises
        ------
        ValueError
            For a label not described in the mesh, or a collapsed root element. A non-manifold root
            mesh raises in :meth:`check_root_mesh`, or at the first query that needs the root mesh.
        """
        connectivities = np.asarray(connectivities, dtype=np.int64).reshape(len(componentIds), -1)
        requireLabelsFitTheirArrays(connectivities)
        known = self.modelMesh.hasNodes(connectivities.reshape(-1))
        if not known.all():
            unknown = np.unique(connectivities.reshape(-1)[~known])
            raise ValueError(f"root element node(s) {unknown[:10].tolist()} are not described in the mesh")
        cells = self._cells
        eids = cells.append(connectivities.shape[0])
        rows = slice(eids.start, eids.stop)
        cells.connectivity[rows] = connectivities
        cells.level[rows] = 0
        cells.parent[rows] = -1
        cells.firstChild[rows] = -1
        cells.root[rows] = np.arange(eids.start, eids.stop)
        cells.component[rows] = np.asarray(componentIds, dtype=np.int64)
        cells.isActive[rows] = True
        cells.boxOrigin[rows] = 0
        cells.boxEdgeLength[rows] = 2 * self.latticeUnits
        cells.nodeKeyIds[rows] = -1
        self.registry.reserve_labels_up_to(int(connectivities.max(initial=0)))
        self.rootEntities.add_roots(np.arange(eids.start, eids.stop), connectivities, componentIds)
        self._boundaryNodesCache = None
        return eids

    def check_root_mesh(self):
        """Raise unless the root mesh is a manifold; see
        :meth:`~edelweissfe.adaptivity.rootentities.RootEntityTable.check_manifold`."""
        self.rootEntities.check_manifold()

    def _addChildren(self, parent: int, connectivities: list, positions: list) -> list:
        """Add the children of cell ``parent``, one per connectivity, in subdivision order; a child's
        reference box is the parent's, split ``splitFactor`` ways per axis, at its position (three
        integers in ``range(splitFactor)``)."""
        cells = self._cells
        eids = cells.append(len(connectivities))
        edgeLength = int(cells.boxEdgeLength[parent]) // self.splitFactor
        parentOrigin = cells.boxOrigin[parent].tolist()
        for eid, connectivity, position in zip(eids, connectivities, positions):
            cells.connectivity[eid] = connectivity
            cells.level[eid] = cells.level[parent] + 1
            cells.parent[eid] = parent
            cells.firstChild[eid] = -1
            cells.root[eid] = cells.root[parent]
            cells.component[eid] = cells.component[parent]
            cells.isActive[eid] = True
            cells.boxOrigin[eid] = [parentOrigin[k] + position[k] * edgeLength for k in range(3)]
            cells.boxEdgeLength[eid] = edgeLength
            cells.nodeKeyIds[eid] = -1
        self._boundaryNodesCache = None
        return list(eids)

    def _childConnectivity(self, parentConn, childIndex, coords, componentId) -> list:
        """The node labels of one child, decided topologically.

        Each child node is either one of the parent's own nodes -- an exact lattice match, reused -- or
        a new node spanned by some of the parent's corners. Its identity is then the set of spanning
        corner *labels* with exact integer (multilinear) weights; a neighbour sharing that face or edge
        names the same corners with the same weights, so both arrive at one label for one point.
        """
        extent = 2 * self.splitFactor
        lattice = self.topology.reference_node_lattice
        conn = []
        for slot, point in enumerate(self._childLattice[childIndex]):
            own = self._ownNodeAt.get(point)
            if own is not None:
                conn.append(parentConn[own])
                continue
            spanning = []
            for cornerSlot in self._cornerSlots:
                weight = 1
                for k in range(3):
                    weight *= point[k] if lattice[cornerSlot][k] == 2 else extent - point[k]
                if weight:
                    spanning.append((parentConn[cornerSlot], weight))
            spanning.sort()
            conn.append(self.registry.label_of_new_node(tuple(spanning), coords[slot], componentId))
        return conn

    def refine(self, eid) -> list:
        """Split an active element into ``splitFactor**3`` children; deactivate it and keep element sets,
        surfaces and node sets consistent.

        Children are returned in subdivision order, ``ix * n * n + iy * n + iz``
        (see :meth:`~edelweissfe.adaptivity.topologybase.TopologyBase.subdivision_children_param`).
        They are placed with the isoparametric map of the element's node coordinates in the mesh, as
        they are now.

        Raises
        ------
        TopologyError
            Beyond :data:`MAXIMUM_REFINEMENT_DEPTH`.
        """
        if not self.is_active(eid):
            return self.children(eid)
        if self.level(eid) >= MAXIMUM_REFINEMENT_DEPTH:
            raise TopologyError(f"AMR: element {eid} is at the maximum refinement depth {MAXIMUM_REFINEMENT_DEPTH}")
        n = self.splitFactor
        parentConn = self.connectivity(eid)
        componentId = self.component(eid)
        connectivities, positions = [], []
        for childIndex, (childCoords, position) in enumerate(
            zip(self.topology.subdivide(self.node_coordinates(parentConn), n), product(range(n), repeat=3))
        ):
            # children stay in the parent's body, and take their node identities from it
            connectivities.append(self._childConnectivity(parentConn, childIndex, childCoords, componentId))
            positions.append(position)
        kids = self._addChildren(eid, connectivities, positions)
        self._cells.isActive[eid] = False
        self._cells.firstChild[eid] = kids[0]
        self._boundaryNodesCache = None

        # element sets + section assignment: children inherit every membership of the parent
        for members in self.elementSets.values():
            if eid in members:
                members.update(kids)

        # surfaces: (parent, faceID) -> (child, faceID) for the children tiling that face
        for pairs in self.surfaces.values():
            faceids_here = [fid for (peid, fid) in pairs if peid == eid]
            for fid in faceids_here:
                pairs.discard((eid, fid))
                for j in self.topology.face_child_indices(self.topology.faceid_to_face[fid], self.splitFactor):
                    pairs.add((kids[j], fid))

        self._inheritNodeSetMembership(parentConn, kids)
        return kids

    def _inheritNodeSetMembership(self, parentConn, kids):
        """A new node joins a node set iff it lies on a parent face or edge whose nodes are all in the set
        -- decided from the node's exact lattice position in the parent.

        A node set is thus inherited *through faces and edges*: a set holding all nodes of a face gains
        the new nodes on that face (even if it was meant as the face's perimeter only), and a set never
        gains new nodes inside the parent. Where that is not the intended membership, define the set
        through an element set or a surface instead.
        """
        extent = 2 * self.splitFactor
        onEntities = {}  # new node label -> parent faces and edges it lies on
        for childIndex, kid in enumerate(kids):
            for label, point in zip(self.connectivity(kid), self._childLattice[childIndex]):
                if label not in onEntities and point not in self._ownNodeAt:
                    onEntities[label] = set(self.topology.entities_containing(point, extent))
        entities = [tuple(f) for f in self.topology.faces] + [tuple(ed) for ed in self.topology.edges]
        for S in self.nodeSets.values():
            inSet = [entity for entity in entities if all(parentConn[i] in S for i in entity)]
            if not inSet:
                continue
            for label, containing in onEntities.items():
                if any(entity in containing for entity in inSet):
                    S.add(label)

    # ---- queries ----
    def active(self) -> list:
        """The active (leaf) elements, in ascending eid order."""
        return np.flatnonzero(self._cells.isActive[: self._cells.count]).tolist()

    def node_lattice_points(self, eid) -> list:
        """The lattice points, in the element's root, of the element's own nodes (slot order)."""
        origin, edgeLength = self.reference_box(eid)
        half = edgeLength // 2
        return [tuple(origin[k] + int(p[k]) * half for k in range(3)) for p in self.topology.reference_node_lattice]

    def node_keys(self, eid) -> list:
        """The :class:`~edelweissfe.adaptivity.rootentities.NodeKey` of each of the element's own nodes,
        in slot order."""
        return [self._nodeKey(number) for number in self._nodeKeyNumbers(eid)]

    def _nodeKeyNumbers(self, eid) -> list:
        """The numbers of the keys of the element's own nodes, in slot order: two nodes are at one point
        iff their numbers are equal. Computed once, the first time they are asked for: an element's
        reference box never changes. A key is numbered in the order keys are first met, and stored
        packed (:meth:`~edelweissfe.adaptivity.rootentities.RootEntityTable.pack`)."""
        numbers = self._cells.nodeKeyIds[eid]
        if numbers[0] < 0:
            numberOf = self._numberOfPackedKey
            for slot, key in enumerate(self.rootEntities.keys(self.root_of(eid), self.node_lattice_points(eid))):
                packed = self.rootEntities.pack(key)
                number = numberOf.get(packed)
                if number is None:
                    number = numberOf[packed] = len(self._packedKeys)
                    self._packedKeys.append(packed)
                numbers[slot] = number
        return numbers.tolist()

    def _nodeKey(self, number: int):
        """The :class:`~edelweissfe.adaptivity.rootentities.NodeKey` of a key number (see
        :meth:`_nodeKeyNumbers`)."""
        return self.rootEntities.unpack(self._packedKeys[number])

    def _nodesOnElementBoundaries(self) -> list:
        """Every node on the boundary of an active element that is not one of its own nodes, as
        ``(eid, label, pointInElement)``: the node's lattice point relative to that element's box
        origin, the element spanning ``[0, edge length]`` per axis.

        Exact and exhaustive by construction: every node sits at an exact point of its root, and the root
        mesh's entities are named by their corner labels, so a node on an element's boundary is found
        whichever element -- or neighbouring root -- it belongs to. Also verifies that no point carries
        two nodes and no node sits at two points. Only a refined root, or a root sharing a vertex with
        one, can hold such a node (elsewhere the mesh is the conforming root mesh itself), so only their
        elements are examined. Cached until the mesh next changes.
        """
        if self._boundaryNodesCache is not None:
            return self._boundaryNodesCache
        cells = self._cells
        act = np.flatnonzero(cells.isActive[: cells.count])
        rootOfActive = cells.root[act]
        refinedRoots = set(rootOfActive[cells.level[act] > 0].tolist())
        relevantRoots = set()
        for root in refinedRoots:
            relevantRoots |= self.rootEntities.roots_touching(root)
        labelOfKey, keyOfLabel = {}, {}
        for eid in act[np.isin(rootOfActive, list(relevantRoots))].tolist():
            for label, key in zip(self.connectivity(eid), self._nodeKeyNumbers(eid)):
                if labelOfKey.setdefault(key, label) != label or keyOfLabel.setdefault(label, key) != key:
                    raise TopologyError(
                        f"AMR conformity: node {label} and node {labelOfKey[key]} occupy one point "
                        f"{self._nodeKey(key)} (or node {label} sits at two points); refinement split or merged "
                        "a node"
                    )

        nodesOfRoot = {root: [] for root in relevantRoots}  # root -> [(label, lattice point in that root)]
        for keyNumber, label in labelOfKey.items():
            key = self._nodeKey(keyNumber)
            for root in self.rootEntities.roots_sharing(key):
                if root in nodesOfRoot:
                    nodesOfRoot[root].append((label, self.rootEntities.local_point(root, key)))

        # per node, descend its root's octree to the active elements whose closed box contains it:
        # O(nodes x depth), not O(elements x nodes) per root
        found = []
        for root, nodes in nodesOfRoot.items():
            for label, point in nodes:
                for eid in self._activeElementsContaining(root, point):
                    if label in self.connectivity(eid):
                        continue
                    origin, edgeLength = self.reference_box(eid)
                    pointInElement = tuple(point[k] - origin[k] for k in range(3))
                    if any(v == 0 or v == edgeLength for v in pointInElement):
                        found.append((eid, label, pointInElement))
        self._boundaryNodesCache = found
        return found

    def _activeElementsContaining(self, rootEid, point) -> list:
        """The active elements of root ``rootEid`` whose closed box contains a lattice point, found by
        descending the octree: the child along each axis follows from the point's position, and a point
        on a split plane belongs to the children on both sides."""
        n = self.splitFactor
        found, stack = [], [rootEid]
        while stack:
            eid = stack.pop()
            if self.is_active(eid):
                found.append(eid)
                continue
            origin, edgeLength = self.reference_box(eid)
            childEdgeLength = edgeLength // n
            candidates = []
            for k in range(3):
                i, remainder = divmod(point[k] - origin[k], childEdgeLength)
                candidates.append([c for c in (i - 1, i) if 0 <= c < n] if remainder == 0 else [i])
            firstChild = int(self._cells.firstChild[eid])
            for ix, iy, iz in product(*candidates):
                stack.append(firstChild + ix * n * n + iy * n + iz)
        return found

    # ---- conformity ----
    def check_conformity(self, slaveLabels):
        """Raise unless the active mesh is conforming up to the given hanging-node slaves.

        Exact, and independent of how the hanging nodes were found. Two conditions:

        * One node per point: a :class:`~edelweissfe.adaptivity.rootentities.NodeKey` maps to one label,
          and a label to one key. A split point would be two unconnected nodes; a merged one would weld
          distinct points.
        * Every node on the boundary of an active element is one of its own nodes or a hanging-node
          slave. Anything else is a free node on the element's boundary: the interface is not conforming.

        Parameters
        ----------
        slaveLabels
            The labels of all hanging-node slaves (e.g. the keys of :meth:`hanging_mpc_records`).

        Raises
        ------
        TopologyError
            Naming the offending nodes and elements.
        """
        slaveLabels = set(slaveLabels)
        unconstrained = [(label, eid) for eid, label, _ in self._nodesOnElementBoundaries() if label not in slaveLabels]
        if unconstrained:
            nodes = sorted({label for label, _ in unconstrained})
            raise TopologyError(
                f"AMR conformity: {len(nodes)} node(s) lie on the boundary of an active element without "
                f"being one of its nodes or a hanging-node slave -- the coarse-fine interface is not "
                f"conforming there. First (node, element) pairs: {sorted(unconstrained)[:10]}"
            )

    # ---- 2:1 balancing ----
    def _facesOnEntities(self, eid) -> list:
        """The six faces of an element as :class:`FaceOnEntity`, exact and topological.

        A face on the boundary of the element's root lies on a root face, named by its corner labels,
        and its rectangle is measured in that face's canonical frame, so a neighbouring root derives the
        identical entity and rectangle. A face inside the root lies on an :class:`InternalPlane`, with
        the rectangle in the other two lattice coordinates. Two elements share a face iff they have faces
        on one entity whose rectangles overlap with positive area.
        """
        rootEid = self.root_of(eid)
        origin, length = self.reference_box(eid)
        rootExtent = 2 * self.latticeUnits
        lattice = self.topology.reference_node_lattice
        faces = []
        for faceIndex, face in enumerate(self.topology.faces):
            rows = lattice[list(face)]
            axis = next(k for k in range(3) if np.all(rows[:, k] == rows[0, k]))
            value = origin[axis] + int(rows[0, axis]) * length // 2
            u, v = (k for k in range(3) if k != axis)
            if value in (0, rootExtent):
                corners = []
                for du, dv in ((0, 0), (length, length)):
                    point = [0, 0, 0]
                    point[axis], point[u], point[v] = value, origin[u] + du, origin[v] + dv
                    entity, position = self.rootEntities.root_face_position(rootEid, faceIndex, tuple(point))
                    corners.append(position)
                (u0, v0), (u1, v1) = corners
                faces.append(FaceOnEntity(entity, Rectangle(min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1))))
            else:
                rectangle = Rectangle(origin[u], origin[u] + length, origin[v], origin[v] + length)
                faces.append(FaceOnEntity(InternalPlane(rootEid, axis, value), rectangle))
        return faces

    def balance_2to1(self) -> int:
        """Refine coarser elements until no face-adjacent active pair differs by more than one level.

        Face adjacency is exact and topological (see :meth:`_facesOnEntities`), so balancing holds on
        curved and warped meshes as on flat ones. Returns the number of extra elements refined.

        Only *face* neighbours are balanced. Elements touching only along an edge or at a vertex may
        differ by two or more levels; the interface stays conforming (the finer edge nodes hang on the
        coarsest element's edge, through exactly resolved constraint chains), so this is a grading, not a
        correctness, limitation. Edge/vertex balancing is deliberately not implemented: it would refine
        additional elements and change the results of multi-level runs.
        """
        nExtra = 0
        while True:
            act = self.active()
            level = dict(zip(act, self._cells.level[act].tolist()))
            # only a pair whose levels differ by two or more can violate the rule: a mesh whose active
            # levels span less -- every mesh refined by a single level -- is balanced
            minLevel, maxLevel = min(level.values()), max(level.values())
            if maxLevel - minLevel < 2:
                break
            finerFacesOn = defaultdict(list)  # entity -> [(level, rectangle)] of the finer candidates
            for eid in act:
                if level[eid] >= minLevel + 2:
                    for face in self._facesOnEntities(eid):
                        finerFacesOn[face.entity].append((level[eid], face.rectangle))
            toRefine = set()
            for a in act:
                if level[a] + 2 > maxLevel:
                    continue
                for face in self._facesOnEntities(a):
                    if any(
                        finerLevel >= level[a] + 2 and face.rectangle.overlaps(rectangle)
                        for finerLevel, rectangle in finerFacesOn.get(face.entity, ())
                    ):
                        toRefine.add(a)
                        break
            if not toRefine:
                break
            for eid in toRefine:
                self.refine(eid)
                nExtra += 1
        return nExtra

    # ---- hanging nodes ----
    def classify_hanging(self) -> list:
        """All hanging nodes of the active mesh, as :class:`HangingNode`, sorted by slave label.

        A hanging node is a node on the boundary of an active element that is not one of its own nodes
        (:meth:`_nodesOnElementBoundaries`: exact, exhaustive, no geometry). Its master entity is the
        lowest-dimensional one of that element containing it -- an edge or a face -- and, among several
        such elements, the coarsest, which guarantees continuity with the coarsest trace. Its weights are
        the master element's own shape functions at the node's exact reference point, which on that edge
        or face are non-zero only for the entity's nodes: the exact coarse trace, on curved and warped
        elements alike. They are verified in exact arithmetic before use.
        """
        best = {}  # slave -> (dimension, level, eid, pointInElement, kind, entity)
        for eid, label, pointInElement in self._nodesOnElementBoundaries():
            _, edgeLength = self.reference_box(eid)
            kind, entity = self.topology.lowest_entity_containing(pointInElement, edgeLength)
            if kind not in ("edge", "face"):
                raise TopologyError(f"AMR: node {label} lies on a corner of element {eid} without being its node")
            candidate = (1 if kind == "edge" else 2, self.level(eid), eid, pointInElement, kind, entity)
            current = best.get(label)
            if current is None or candidate[:2] < current[:2]:
                best[label] = candidate

        hanging = []
        for label, (_, _, eid, pointInElement, kind, entity) in sorted(best.items()):
            _, edgeLength = self.reference_box(eid)
            xi = tuple(Fraction(2 * v, edgeLength) - 1 for v in pointInElement)
            exactWeights = self.topology.shape_functions_exact(*xi)
            self._verifyTraceWeights(label, eid, xi, entity, exactWeights)
            weights = np.array([float(exactWeights[i]) for i in entity])
            conn = self.connectivity(eid)
            hanging.append(HangingNode(label, kind, [conn[i] for i in entity], weights))
        return hanging

    def _verifyTraceWeights(self, label, eid, xi, entity, exactWeights):
        """Raise unless ``exactWeights`` are the exact trace weights of a node at ``xi`` on ``entity``.

        Checked in exact rational arithmetic, so without any tolerance: the weight of every node off the
        entity vanishes, and the entity's weights reproduce the constant, the three reference coordinates
        and the six quadratic monomials :math:`\\xi_a \\xi_b` at the node -- which catches a wrong entity
        or orientation, not just a wrong formula.
        """
        reference = [tuple(int(v) - 1 for v in p) for p in self.topology.reference_node_lattice]
        exponents = [(0, 0, 0)] + [tuple(int(k == a) for k in range(3)) for a in range(3)]
        exponents += [tuple(int(k == a) + int(k == b) for k in range(3)) for a in range(3) for b in range(a, 3)]

        def monomial(point, exponent):
            return point[0] ** exponent[0] * point[1] ** exponent[1] * point[2] ** exponent[2]

        onEntity = set(entity)
        offEntityWeightVanishes = all(w == 0 for i, w in enumerate(exactWeights) if i not in onEntity)
        reproduces = all(
            sum(exactWeights[i] * monomial(reference[i], exponent) for i in entity) == monomial(xi, exponent)
            for exponent in exponents
        )
        if not (offEntityWeightVanishes and reproduces):
            raise TopologyError(
                f"AMR: the hanging-node weights of node {label} on element {eid} (reference point {xi}) "
                "are not the exact trace of that element's shape functions"
            )

    def hanging_mpc_records(self) -> dict:
        """The hanging-node constraints as ``{slave: [(master, weight), ...]}``, every master an
        independent node: a chain -- a master that is itself hanging -- is substituted through, with the
        same flattening the multi-point-constraint transformation applies across all constraints."""
        raw = [(h.slave, list(zip(h.masters, h.weights))) for h in self.classify_hanging()]
        return {slave: sorted(masters) for slave, masters in _flattenChainedRecords(raw)}
