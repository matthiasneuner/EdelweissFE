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
#  Alexander Dummer alexander.dummer@uibk.ac.at
#  Paul Hofer Paul.Hofer@uibk.ac.at
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

from edelweissfe.points.node import Node
from edelweissfe.sets.orderedset import LocalOrderedSet
from edelweissfe.utils.exceptions import TopologyError


class NodeSet(LocalOrderedSet):
    """A basic node set.
    It has a label, and a list containing the unique nodes.

    **Local and complete** (see :class:`~edelweissfe.sets.orderedset.LocalOrderedSet`), as an
    element set: a node set of a model resolved from its mesh is a :class:`NodeSetOfMesh`, which
    holds the nodes of its set held in this process, and is complete if that is all of them. A set of
    node objects like this one holds the nodes it was given, and is always complete. Reading a set as
    a whole -- iterating it, indexing it, ``len()``, :attr:`nodes` -- requires it to be complete; the
    local part is :meth:`localNodes`.

    Parameters
    ----------
    label
        The unique label for this element set.
    nodes
        A list of nodes.
    """

    memberName = "node"

    def __init__(
        self,
        label: str,
        nodes,
    ):
        self.allowedObjectTypes = [Node]

        super().__init__(label, nodes)

    @property
    def nodes(self):
        """The nodes of the set, in set order, as a read-only view.

        Raises
        ------
        TopologyError
            If the set is not complete in this process; see :meth:`localNodes`.
        """

        if not self.isComplete:
            self._raiseNotComplete("reading .nodes of")
        return self.items

    def localNodes(self):
        """The nodes of the set held in this process, in set order, as a read-only view: every node
        of the set if it is complete, e.g. in a serial run.

        Returns
        -------
        KeysView
            The local nodes.
        """

        return self.localMembers()

    def nodeLabelsOfWholeSet(self) -> list:
        """The labels of every node of the set, in set order: here, of its nodes.

        Returns
        -------
        list
            The node labels.
        """

        return [node.label for node in self]

    def requireDescribedBy(self, mesh):
        """State that this set is the node set of its name in the given mesh, to be updated from it.

        Parameters
        ----------
        mesh
            The mesh.

        Raises
        ------
        TopologyError
            Always: a set of node objects is described by no mesh, and a node set of the mesh cannot
            take its name.
        """

        raise TopologyError(
            "node set {:} is a set of node objects a model entity made, and the mesh cannot define a node set "
            "of the same name".format(self.name)
        )


class NodeSetOfMesh(NodeSet):
    """A node set of a model, resolved from its mesh: it holds the nodes of the set that are held in
    this process (see :class:`NodeSet`).

    The mesh describes the nodes and the node sets as data (labels and coordinates; a node set as
    an ordered list of labels), and the model creates the node objects from it
    (:meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh`), which resolves every node set of
    the mesh to such a set (:meth:`~edelweissfe.models.femodel.FEModel.resolveNodeSetOfMesh`, the only
    place one is made). In a serial run that is every node of the set. A process that holds only
    some of the nodes holds only that part of a set, says so through :attr:`isComplete`, and raises
    if read as a whole. The labels of the whole set are known in every process
    (:meth:`nodeLabelsOfWholeSet`).

    Parameters
    ----------
    name
        The name of the set, in the mesh as here.
    nodes
        The nodes of the set held in this process, in set order.
    mesh
        The :class:`~edelweissfe.models.mesh.Mesh` describing this set under the same name.
    """

    def __init__(self, name: str, nodes, mesh):
        #: The mesh describing this set.
        self.mesh = mesh
        super().__init__(name, nodes)

    @property
    def isComplete(self) -> bool:
        """True if every node of the set in the mesh is held in this process. Derived, not stored,
        so that no change of the members can leave it stale. Counting suffices, since the set holds
        only nodes of its set in the mesh."""

        return len(self.data) == len(self.mesh.nodeSets[self.name])

    def sizeOfWholeSet(self) -> int:
        """The number of nodes of the whole set, from the mesh.

        Returns
        -------
        int
            The size of the set.
        """

        return len(self.mesh.nodeSets[self.name])

    def nodeLabelsOfWholeSet(self) -> list:
        """The labels of every node of the set, in set order -- also of those not held here, read
        from the mesh.

        Returns
        -------
        list
            The node labels.
        """

        return self.mesh.nodeSets[self.name].tolist()

    def requireDescribedBy(self, mesh):
        """State that this set is the node set of its name in the given mesh, to be updated from it.

        Parameters
        ----------
        mesh
            The mesh.

        Raises
        ------
        TopologyError
            If this set was resolved from another mesh.
        """

        if mesh is not self.mesh:
            raise TopologyError(
                "node set {:} was resolved from another mesh than the model's -- a model that replaces its mesh "
                "drops the node sets of the old one".format(self.name)
            )

    def _whyIncomplete(self) -> str:
        """Why nodes of the set are not held here.

        Returns
        -------
        str
            The explanation.
        """

        return "this process holds only some of the nodes of the mesh"
