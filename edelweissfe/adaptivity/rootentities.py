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
"""The vertices, edges and faces of the *root* mesh, and where on them a refinement node lies --
decided without looking at a single physical coordinate.

Vocabulary
----------
Root element
    An element of the mesh before any refinement. Every refined element lies inside exactly one.
Reference lattice
    The reference cube of a root element, :math:`[-1, 1]^3`, carries an integer lattice: reference
    coordinate :math:`\\xi` corresponds to the integer :math:`(\\xi + 1)\\,M`, so the cube spans
    :math:`[0, 2M]` per axis. :math:`M` is a power of the split factor, fine enough that every node a
    subdivision can create sits on a lattice point -- so all positions are *exact integers*.
Root entity
    A vertex, edge or face of the root mesh, named by the labels of its corner nodes. Two root elements
    that share a face therefore derive the identical name for it, whatever the orientation of their
    local axes.
Node key
    Where a node sits: the lowest-dimensional root entity containing it, and its exact integer
    position on that entity, measured in a frame that the entity's corner labels fix (see
    :class:`NodeKey`). Two nodes are at one point iff they have one key.

This is the mechanism established AMR libraries use (deal.II, p4est, MFEM): whether two points
coincide and whether a point lies on a face are questions of topology and exact arithmetic, so no
tolerance can make them wrong on a curved or warped mesh.

Storage
-------
The table is held as arrays, not as one Python object per root, vertex or face: the corner labels and
the body of every root element, one row per root, and an index from each vertex of the root mesh to
the roots sharing it, built once all roots are known (:meth:`RootEntityTable.check_manifold`). The
edges and faces need no table of their own: the roots sharing an edge or a face are the roots sharing
all of its corners that have it as an edge or a face. A :class:`NodeKey` can be packed into one
integer (:meth:`RootEntityTable.pack`), which is how the octree stores the keys of its nodes.
"""

from typing import NamedTuple

import numpy as np

#: The integer type of node labels in the arrays of the refinement: a node label must stay below
#: ``2**31``. The exact positions on the reference lattice are 64-bit integers.
LABEL_DTYPE = np.int32


def requireLabelsFitTheirArrays(labels):
    """Raise unless every node label fits :data:`LABEL_DTYPE`.

    Parameters
    ----------
    labels
        The node labels.

    Raises
    ------
    ValueError
        For a label outside the range of :data:`LABEL_DTYPE`.
    """

    labels = np.asarray(labels)
    if labels.size and (labels.min() < 0 or labels.max() > np.iinfo(LABEL_DTYPE).max):
        raise ValueError(
            "node labels must lie in [0, {:d}] for adaptive refinement, found {:d}..{:d}".format(
                np.iinfo(LABEL_DTYPE).max, int(labels.min()), int(labels.max())
            )
        )


class NodeKey(NamedTuple):
    """Where a node sits: a root entity and the node's exact integer position on it.

    ============  ====================================  ===========================================
    ``kind``      ``entity``                            ``position``
    ============  ====================================  ===========================================
    ``vertex``    the vertex's node label               ``()``
    ``edge``      its two corner labels, ascending      ``(t,)`` from the smaller label, in [0, 2M]
    ``face``      its four corner labels, ascending     ``(u, v)`` in the canonical face frame
    ``interior``  the root element's id                 ``(x, y, z)`` in the root's lattice
    ============  ====================================  ===========================================

    ``component`` is the body the node belongs to: two bodies that touch at a flush interface
    (a tie, a contact pair, a crack plane) share no entity, even where their coordinates coincide.
    """

    kind: str
    component: int
    entity: object
    position: tuple


class RootFace(NamedTuple):
    """A face of the root mesh, named by the ascending labels of its four corner nodes."""

    component: int
    cornerLabels: tuple


#: The kinds of node keys, in the order of their code in a packed key (:meth:`RootEntityTable.pack`).
_KINDS = ("vertex", "edge", "face", "interior")
#: Per kind: how many labels (or ids) name its entity, and how many integers its position has.
_ENTITY_LENGTH = {"vertex": 1, "edge": 2, "face": 4, "interior": 1}
_POSITION_LENGTH = {"vertex": 0, "edge": 1, "face": 2, "interior": 3}
#: The radix of a label, an element id or a body in a packed key.
_LABEL_RADIX = 2**32


class _VertexIndex(NamedTuple):
    """The roots sharing each vertex of the root mesh, in compressed rows.

    A vertex is named by ``component * 2**32 + label``; ``vertices`` holds the names in ascending order,
    and the roots of ``vertices[i]`` are ``roots[first[i]:first[i + 1]]``, ascending.
    """

    vertices: np.ndarray
    first: np.ndarray
    roots: np.ndarray


class RootEntityTable:
    """The vertices, edges and faces of the root mesh, keyed by root corner labels.

    Parameters
    ----------
    topology
        The :class:`~edelweissfe.adaptivity.topologybase.TopologyBase` of the root elements.
    latticeUnits
        :math:`M`: reference coordinate :math:`\\xi` of a root element is the lattice integer
        :math:`(\\xi + 1)\\,M`.
    """

    def __init__(self, topology, latticeUnits: int):
        self._topology = topology
        self._latticeUnits = latticeUnits
        self._cornerSlots = topology.corner_slots
        #: a corner's position in the root lattice (every coordinate 0 or 2M)
        self._cornerPoint = {
            slot: tuple(int(v) * latticeUnits for v in topology.reference_node_lattice[slot])
            for slot in self._cornerSlots
        }
        cornerSet = set(self._cornerSlots)
        #: per face: its corner slots in loop order; per edge: its two corner slots
        self._faceCorners = [[slot for slot in face if slot in cornerSet] for face in topology.faces]
        self._edgeCorners = [[slot for slot in edge if slot in cornerSet] for edge in topology.edges]
        #: per root element, in the row of its id: the labels of its corner nodes, in the order of
        #: ``corner_slots``, and its body; rows of ids that are not roots stay unused
        self._cornerLabels = np.zeros((0, len(self._cornerSlots)), dtype=LABEL_DTYPE)
        self._component = np.zeros(0, dtype=np.int64)
        self._isRoot = np.zeros(0, dtype=bool)
        #: built on first use, once all roots are known; reset by adding a root
        self._vertexIndex = None
        #: positions on the reference lattice range over [0, 2M]
        self._positionRadix = 2 * latticeUnits + 1

    def add_root(self, rootEid: int, connectivity, componentId: int = 0):
        """Register a root element by its node labels (in the topology's local node order).

        Raises
        ------
        ValueError
            For a collapsed root element (a corner label repeated), whose faces and edges cannot be
            named by their corners. A non-manifold root mesh is detected once all roots are known, by
            :meth:`check_manifold`.
        """

        self.add_roots([rootEid], [connectivity], [componentId])

    def add_roots(self, rootEids, connectivities, componentIds):
        """Register root elements by their node labels; see :meth:`add_root`.

        Parameters
        ----------
        rootEids
            The ids of the root elements.
        connectivities
            Per root, its node labels in the topology's local node order.
        componentIds
            Per root, its body.
        """

        rootEids = np.asarray(rootEids, dtype=np.int64).reshape(-1)
        if rootEids.shape[0] == 0:
            return
        connectivities = np.asarray(connectivities, dtype=np.int64).reshape(rootEids.shape[0], -1)
        requireLabelsFitTheirArrays(connectivities)
        corners = connectivities[:, self._cornerSlots]
        sortedCorners = np.sort(corners, axis=1)
        collapsed = np.flatnonzero((sortedCorners[:, 1:] == sortedCorners[:, :-1]).any(axis=1))
        if collapsed.shape[0]:
            raise ValueError(
                f"root element {int(rootEids[collapsed[0]])} repeats a corner node label (a collapsed element); "
                "its faces and edges cannot be named by their corner labels"
            )
        end = int(rootEids.max()) + 1
        if end > self._isRoot.shape[0]:
            capacity = max(end, 2 * self._isRoot.shape[0])
            self._cornerLabels = grownToCapacity(self._cornerLabels, capacity)
            self._component = grownToCapacity(self._component, capacity)
            self._isRoot = grownToCapacity(self._isRoot, capacity)
        self._cornerLabels[rootEids] = corners
        self._component[rootEids] = np.asarray(componentIds, dtype=np.int64).reshape(-1)
        self._isRoot[rootEids] = True
        self._vertexIndex = None

    def check_manifold(self):
        """Raise unless the root mesh is a manifold: no face shared by more than two roots.

        Builds the index from the vertices to the roots sharing them, which every query needs; a query
        runs this check itself, so calling it once all roots are added only makes a defective root mesh
        fail at setup.

        Raises
        ------
        ValueError
            For a face owned by more than two roots, naming it.
        """

        self._vertices()

    def key(self, rootEid: int, point) -> NodeKey:
        """The :class:`NodeKey` of a lattice point (three integers in [0, 2M]) of root ``rootEid``."""
        return self.keys(rootEid, [point])[0]

    def keys(self, rootEid: int, points) -> list:
        """The :class:`NodeKey` of each of several lattice points of root ``rootEid``; see :meth:`key`."""
        corners = self._corners(rootEid)
        component = int(self._component[rootEid])
        return [self._computeKey(rootEid, corners, component, tuple(point)) for point in points]

    def pack(self, key: NodeKey) -> int:
        """One integer naming a :class:`NodeKey`, so that two keys are equal iff their integers are.

        The digits of the integer are, from the most significant one: the body, the labels (or the root
        id) naming the entity, the position on the entity, each digit in a radix it cannot reach, and
        the kind. :meth:`unpack` is its inverse.
        """

        value = key.component
        for label in (key.entity,) if key.kind in ("vertex", "interior") else key.entity:
            value = value * _LABEL_RADIX + label
        for coordinate in key.position:
            value = value * self._positionRadix + coordinate
        return value * len(_KINDS) + _KINDS.index(key.kind)

    def unpack(self, value: int) -> NodeKey:
        """The :class:`NodeKey` an integer from :meth:`pack` names."""
        value, kindCode = divmod(value, len(_KINDS))
        kind = _KINDS[kindCode]
        position = []
        for _ in range(_POSITION_LENGTH[kind]):
            value, coordinate = divmod(value, self._positionRadix)
            position.append(coordinate)
        entity = []
        for _ in range(_ENTITY_LENGTH[kind]):
            value, label = divmod(value, _LABEL_RADIX)
            entity.append(label)
        entity.reverse()
        position.reverse()
        return NodeKey(kind, value, entity[0] if kind in ("vertex", "interior") else tuple(entity), tuple(position))

    def roots_sharing(self, key: NodeKey) -> list:
        """The root elements whose closure contains the entity of ``key``."""
        if key.kind == "interior":
            return [key.entity]
        if key.kind == "vertex":
            return self._rootsOfVertex(key.component, key.entity).tolist()
        corners = key.entity
        withAll = set(self._rootsOfVertex(key.component, corners[0]).tolist())
        for label in corners[1:]:
            withAll &= set(self._rootsOfVertex(key.component, label).tolist())
        isEntityOf = self._isEdgeOf if key.kind == "edge" else self._isFaceOf
        return sorted(root for root in withAll if isEntityOf(root, corners))

    def roots_touching(self, rootEid: int) -> set:
        """The root elements sharing at least one vertex (hence possibly an edge or a face) with
        ``rootEid``, including itself."""
        component = int(self._component[rootEid])
        return {
            root
            for label in self._cornerLabels[rootEid].tolist()
            for root in self._rootsOfVertex(component, label).tolist()
        }

    def root_face_position(self, rootEid: int, faceIndex: int, point) -> tuple:
        """The root face ``faceIndex`` (into ``topology.faces``) of ``rootEid`` as a :class:`RootFace`,
        and the position of ``point`` (a lattice point on it, possibly on its boundary) in the face's
        canonical frame -- identical from either root sharing the face."""
        corners = self._corners(rootEid)
        faceCorners = self._faceCorners[faceIndex]
        origin, first, second = self._canonicalFaceFrame(faceCorners, corners)
        face = RootFace(int(self._component[rootEid]), tuple(sorted(corners[s] for s in faceCorners)))
        return face, self._faceCoordinates(point, origin, first, second)

    def local_point(self, rootEid: int, key: NodeKey) -> tuple:
        """The lattice point, in root element ``rootEid``, of a key whose entity lies in that root's
        closure (see :meth:`roots_sharing`)."""
        corners = self._corners(rootEid)
        slotOf = {label: slot for slot, label in corners.items()}
        if key.kind == "interior":
            return key.position
        if key.kind == "vertex":
            return self._cornerPoint[slotOf[key.entity]]
        if key.kind == "edge":
            a, b = (self._cornerPoint[slotOf[label]] for label in key.entity)
            (t,) = key.position
            return tuple(a[k] + t * (b[k] - a[k]) // (2 * self._latticeUnits) for k in range(3))
        faceCorners = next(fc for fc in self._faceCorners if tuple(sorted(corners[s] for s in fc)) == key.entity)
        origin, first, second = (self._cornerPoint[s] for s in self._canonicalFaceFrame(faceCorners, corners))
        u, v = key.position
        twoM = 2 * self._latticeUnits
        return tuple(origin[k] + (u * (first[k] - origin[k]) + v * (second[k] - origin[k])) // twoM for k in range(3))

    # ---- internals ----
    def _corners(self, rootEid: int) -> dict:
        """The corner labels of a root element, by corner slot."""
        return dict(zip(self._cornerSlots, self._cornerLabels[rootEid].tolist()))

    def _vertices(self) -> _VertexIndex:
        """The index from the vertices of the root mesh to the roots sharing them, built once all roots
        are known; checks that the root mesh is a manifold when it is built (see :meth:`check_manifold`)."""
        if self._vertexIndex is None:
            rootEids = np.flatnonzero(self._isRoot)
            names = (self._component[rootEids, None] * _LABEL_RADIX + self._cornerLabels[rootEids]).reshape(-1)
            roots = np.repeat(rootEids, len(self._cornerSlots))
            order = np.lexsort((roots, names))
            names, roots = names[order], roots[order]
            vertices, first = np.unique(names, return_index=True)
            self._vertexIndex = _VertexIndex(vertices, np.append(first, names.shape[0]), roots)
            self._requireManifold(rootEids)
        return self._vertexIndex

    def _requireManifold(self, rootEids: np.ndarray):
        """Raise if a face of the root mesh, named by its body and its four corner labels, belongs to more
        than two roots."""
        cornerColumn = {slot: column for column, slot in enumerate(self._cornerSlots)}
        faces = []
        for faceCorners in self._faceCorners:
            labels = np.sort(self._cornerLabels[np.ix_(rootEids, [cornerColumn[s] for s in faceCorners])], axis=1)
            faces.append(np.column_stack([self._component[rootEids], labels]))
        faces = np.concatenate(faces)
        unique, counts = np.unique(faces, axis=0, return_counts=True)
        if (counts > 2).any():
            face = unique[np.flatnonzero(counts > 2)[0]]
            owners = np.tile(rootEids, len(self._faceCorners))[np.flatnonzero((faces == face).all(axis=1))]
            raise ValueError(
                f"the face with corner labels {face[1:].tolist()} is shared by more than two root "
                f"elements ({sorted(owners.tolist())}); the root mesh is not a manifold"
            )

    def _rootsOfVertex(self, component: int, label: int) -> np.ndarray:
        """The roots sharing a vertex, ascending."""
        index = self._vertices()
        name = component * _LABEL_RADIX + label
        i = int(np.searchsorted(index.vertices, name))
        if i == index.vertices.shape[0] or index.vertices[i] != name:
            return index.roots[:0]
        return index.roots[index.first[i] : index.first[i + 1]]

    def _computeKey(self, rootEid: int, corners: dict, component: int, point) -> NodeKey:
        kind, entity = self._topology.lowest_entity_containing(point, 2 * self._latticeUnits)
        if kind == "vertex":
            return NodeKey("vertex", component, corners[entity], ())
        if kind == "edge":
            a, b = (slot for slot in entity if slot in corners)
            if corners[a] > corners[b]:
                a, b = b, a
            return NodeKey("edge", component, (corners[a], corners[b]), (self._edgeParameter(point, a, b),))
        if kind == "face":
            faceCorners = [slot for slot in entity if slot in corners]
            origin, first, second = self._canonicalFaceFrame(faceCorners, corners)
            labels = tuple(sorted(corners[s] for s in faceCorners))
            return NodeKey("face", component, labels, self._faceCoordinates(point, origin, first, second))
        return NodeKey("interior", component, rootEid, tuple(point))

    def _isEdgeOf(self, rootEid, labels) -> bool:
        corners = self._corners(rootEid)
        return any({corners[a], corners[b]} == set(labels) for a, b in self._edgeCorners)

    def _isFaceOf(self, rootEid, labels) -> bool:
        corners = self._corners(rootEid)
        return any({corners[s] for s in faceCorners} == set(labels) for faceCorners in self._faceCorners)

    def _edgeParameter(self, point, a, b) -> int:
        """Position along the root edge from corner ``a`` (0) to corner ``b`` (2M)."""
        pa, pb = self._cornerPoint[a], self._cornerPoint[b]
        axis = next(k for k in range(3) if pa[k] != pb[k])
        return abs(point[axis] - pa[axis])

    def _canonicalFaceFrame(self, faceCorners, corners):
        """The face frame fixed by the corner labels::

                second ------ .
                  |           |       origin: the corner with the smallest label
                  |           |       first:  its loop neighbour with the smaller label
                origin ---- first     second: its other loop neighbour

        so both roots sharing the face choose the same frame, whatever their local numbering."""
        k = min(range(4), key=lambda i: corners[faceCorners[i]])
        previous, following = faceCorners[(k - 1) % 4], faceCorners[(k + 1) % 4]
        first, second = (previous, following) if corners[previous] < corners[following] else (following, previous)
        return faceCorners[k], first, second

    def _faceCoordinates(self, point, origin, first, second) -> tuple:
        """Position of ``point`` in the face frame: the distances from the origin along the two frame
        edges, which are lattice-axis-aligned, each spanning [0, 2M]."""
        o, e1, e2 = (self._cornerPoint[s] for s in (origin, first, second))
        axisU = next(k for k in range(3) if e1[k] != o[k])
        axisV = next(k for k in range(3) if e2[k] != o[k])
        return (abs(point[axisU] - o[axisU]), abs(point[axisV] - o[axisV]))


def grownToCapacity(array: np.ndarray, capacity: int) -> np.ndarray:
    """A copy of ``array`` with ``capacity`` rows: its rows, then zeros."""
    grown = np.zeros((capacity,) + array.shape[1:], dtype=array.dtype)
    grown[: array.shape[0]] = array
    return grown
