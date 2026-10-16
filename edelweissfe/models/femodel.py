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
# Created on Fri Jan 27 19:53:45 2017

# @author: Matthias Neuner

import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from operator import attrgetter

import h5py
import numpy as np

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.elementdistribution import ElementDistribution
from edelweissfe.models.mesh import Mesh
from edelweissfe.models.modelchange import TopologyRecord
from edelweissfe.models.topologypipeline import TopologyPipeline
from edelweissfe.numerics.globaldofnumbering import GlobalDofNumbering
from edelweissfe.numerics.parallelizationutilities import (
    chunked_iterable,
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import (
    ElementSet,
    ElementSetOfMesh,
    ElementSetOfSurfaceFace,
)
from edelweissfe.sets.nodeset import NodeSet, NodeSetOfMesh
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.utils.checkpoint import readRestartDataInto, writeRestartDataOf
from edelweissfe.utils.exceptions import RestartError, TopologyError
from edelweissfe.variables.fieldvariable import FieldVariable
from edelweissfe.variables.scalarvariable import ScalarVariable


class FEModel:
    """This is is a standard finite element model tree.
    It takes care of the correct number of variables,
    for nodes and scalar degrees of freedem, and it manages the fields.

    What the model holds:

    * :attr:`mesh` (:class:`~edelweissfe.models.mesh.Mesh`) holds the nodes (labels and reference
      coordinates), the node sets, the element connectivity, the element sets and the surfaces, as
      data.
    * :attr:`nodes` holds the node objects created from the mesh (:meth:`createNodesOfMesh`),
      :attr:`elements` the element objects. Neither is assigned directly: :meth:`prepareYourself` refuses a node or an element
      the mesh does not describe. (Auxiliary elements -- contact facets, point masses -- are added
      by :meth:`createAuxiliaryElement`, which describes them in the mesh as well.)

    The model is therefore built in two stages. First the mesh is described in :attr:`mesh`. Then
    :meth:`createNodesOfMesh` and :meth:`createElementsOfMesh` create the node and element objects
    from it and resolve the node sets, element sets and surfaces to them. The fields at the nodes,
    and with them the layout of the degrees of freedom, follow from the mesh alone (see
    :meth:`~edelweissfe.models.mesh.Mesh.typeOf`).


    Parameters
    ----------
    dimension
        The dimension of the model.
    """

    identification = "FEModel"

    def __init__(self, dimension: int):
        self.time = 0.0  #: Current time of the model.
        #: The node objects created from :attr:`mesh`, by label, in mesh order. Never assigned directly:
        #: describe a node in :attr:`mesh` and call :meth:`createNodesOfMesh`.
        self.nodes = {}
        #: The mesh, as data: the nodes, the node sets, the element connectivity, the element sets and
        #: the surfaces.
        self.mesh = Mesh()
        #: The element objects created from :attr:`mesh`, by number, in mesh order. Never assigned
        #: directly: describe an element in :attr:`mesh` and call :meth:`createElementsOfMesh`.
        self.elements = {}
        #: Which elements of the mesh are created (:meth:`createElementsOfMesh` asks it): every one.
        #: In a domain-decomposed run, the elements this process creates, and how results of the whole
        #: model are read from the processes that computed them (see
        #: :mod:`~edelweissfe.models.elementdistribution`).
        self.elementDistribution = ElementDistribution()
        #: The node sets of the model, by name: those of the mesh resolved to the local nodes
        #: (:class:`~edelweissfe.sets.nodeset.NodeSetOfMesh`), and sets of node objects a model entity
        #: made itself (e.g. the slave nodes a tie tied).
        self.nodeSets = {}
        #: The mesh, and how many of its leading rows all have node objects here
        #: (:meth:`createNodesOfMesh`).
        self._rowsWithNodeObjects = (None, 0)
        #: The mesh, its node sets' version and the number of nodes when the node sets were last
        #: resolved (:meth:`createNodesOfMesh`).
        self._nodeSetsResolvedFor = None
        #: Where the process holds the nodes of its subdomain only: the labels of the nodes made here during
        #: the run whose node field values were not written here yet (:meth:`nodesAwaitingValues`).
        self._nodesAwaitingValues = set()
        #: Whether the node fields were made (:meth:`_prepareVariablesAndFields`) -- in a process holding no
        #: node of the mesh then, none.
        self._nodeFieldsPrepared = False
        self.nodeFields = {}  #: NodeFields in the model.
        self.elementSets = {}  #: ElementSets in the model.
        self.sections = {}  #: Sections in the model.
        self.surfaces = {}  #: Surface definitions in the model.
        self.constraints = {}  #: Constraints in the model.
        self.constraintSets = {}  #: ConstraintsSets in the model.
        self.multiPointConstraints = {}  #: Multi-point (DOF-elimination) constraints in the model.
        self.modelModifiers = {}  #: Model modifiers (dynamic topology / mesh mutation entities) in the model.
        self.contactFacetRecipes = {}  #: facet elSet name -> (surfaceName, prefix, triangulation, nodalWeights).
        self.materials = {}  #: Materials in the model.
        self.analyticalFields = {}  #: AnalyticalFields in the model.
        self.scalarVariables = {}  #: ScalarVariables in the model.
        self.additionalParameters = {}  #: Additional information.
        self.rigidBodies = {}  #: RigidBodies in the model.
        self.elementProperties = []  #: Element properties.
        self.domainSize = dimension  #: Spatial dimension of the model
        self.fieldOutputController = None  #: Set once by the driver; lets in-model entities (e.g. AMR markers) look up a named *fieldOutput by value, not just by declaration.
        #: How the mesh may change during a run; see :class:`~edelweissfe.models.topologypipeline.TopologyPipeline`.
        self.topology = TopologyPipeline(self)
        #: The state of every element of the mesh, by number, for the restart checkpoint being written,
        #: where it does not come from the local element objects; see
        #: :meth:`elementStatesFromElsewhere`. None otherwise.
        self._elementStatesFromElsewhere = None
        #: The node fields of the whole model, per field and entry, for the restart checkpoint being
        #: written, where the node fields here hold only some of the nodes; see
        #: :meth:`nodeFieldsFromElsewhere`. None otherwise.
        self._nodeFieldsFromElsewhere = None

    def __copy__(self):
        """A shallow copy of the model, with a topology pipeline that acts on the copy.

        A shallow copy shares every attribute with the original, which would include the pipeline --
        and a pipeline changes the model it was created for. The copy therefore gets its own pipeline
        bound to it (see :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.boundTo`), which
        is what a reduced model needs, e.g. EdelweissMeshfree's active sub-model.
        """

        clone = object.__new__(type(self))
        # shared with the original too: the mesh, the nodes, and the caches of createNodesOfMesh, which
        # name the mesh they hold for, so a copy given another mesh starts afresh
        clone.__dict__.update(self.__dict__)
        clone.topology = self.topology.boundTo(clone)
        return clone

    def moveNodes(self, labels, coordinates):
        """Change the reference coordinates of nodes: in the mesh and, in place, at the local node
        objects -- what a model entity or modifier changing the reference geometry calls (a tie
        adjusting its slave nodes onto the master surface, a snap onto a cylinder).

        Parameters
        ----------
        labels
            The node labels.
        coordinates
            The new reference coordinates, one row per node.
        """

        mesh = self.mesh
        mesh.moveNodes(labels, coordinates)
        rows = mesh.rowsOfNodes(labels)
        moved = mesh.nodeCoordinates
        nodes = self.nodes
        for label, row in zip(np.asarray(labels).tolist(), rows.tolist()):
            if label in nodes:
                nodes[label].coordinates[:] = moved[row]

    def createNodesOfMesh(self, isHeldHere: Callable[[int], bool] | None = None):
        """Create the node objects of the mesh, and resolve its node sets to them.

        Nodes are created in mesh order, after those created before, so :attr:`nodes` lists them in
        the order they were described -- the order of the degrees of freedom of every field. Each
        node's coordinates are a copy of its reference coordinates in the mesh.

        Calling it again creates the nodes described since: a late generator, a model entity that needs
        the objects of nodes it just described (a rigid body's surface nodes and reference point), a
        refinement's new nodes. Node sets that already exist are updated in place (references held to
        them stay valid).

        Which nodes are held here, the model's :attr:`elementDistribution` decides
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.isNodeHeldHere`): every node in a
        serial run; in a process of a domain-decomposed run holding the nodes of its subdomain only, those
        of its elements and those a model entity reads whole (:meth:`holdNodes`).

        Parameters
        ----------
        isHeldHere
            Which nodes of the mesh get an object; None for those the element distribution holds here.
        """

        if isHeldHere is None:
            isHeldHere = self.elementDistribution.isNodeHeldHere
        mesh = self.mesh
        mesh.requireEachNodeDescribedOnce()
        nodes = self.nodes
        nodesBefore = len(nodes)
        # The leading rows of the mesh whose nodes all have objects here need no second look: a model
        # describing its mesh in many steps calls this after each of them.
        firstRow = self._rowsWithNodeObjects[1] if self._rowsWithNodeObjects[0] is mesh else 0
        labels = mesh.nodeLabels[firstRow:].tolist()
        coordinates = mesh.nodeCoordinates[firstRow:]
        firstNotHeld = len(labels)
        # a node made during a run, where only some nodes are held, has no values until they are written
        awaitsValues = self._nodeFieldsPrepared and self.elementDistribution.holdsNodesOfSubdomainOnly
        for offset, label in enumerate(labels):
            if label in nodes:
                continue
            if isHeldHere(label):
                if not self.topology.isOpen:
                    raise TopologyError(
                        "node {:} was created outside a topology change: only model modifiers may create "
                        "or delete nodes, inside TopologyPipeline.changes()".format(label)
                    )
                nodes[label] = Node(label, coordinates[offset].copy())
                if awaitsValues:
                    self._nodesAwaitingValues.add(label)
            else:
                firstNotHeld = min(firstNotHeld, offset)
        self._rowsWithNodeObjects = (mesh, firstRow + firstNotHeld)
        if self.elementDistribution.holdsNodesOfSubdomainOnly and 0 < nodesBefore < len(nodes):
            # a node skipped before may be held now, behind nodes of later rows (where every node is held,
            # nodes are created in mesh order)
            self.putNodesInMeshOrder()

        # resolved again only if the nodes or the node sets of the mesh changed since
        resolvedFor = (mesh, mesh.nodeSetsVersion, len(nodes))
        if resolvedFor != self._nodeSetsResolvedFor:
            self.resolveNodeSetsOfMesh()
            self._nodeSetsResolvedFor = resolvedFor

    def putNodesInMeshOrder(self):
        """Order :attr:`nodes` as the mesh lists them, in place, after nodes were created later than
        nodes of later rows of the mesh (a node held here from then on: a reader holding it, an element
        moving here). The order of :attr:`nodes` is the order of the degrees of freedom of every field."""

        rows = self.mesh.rowsOfNodes(list(self.nodes))
        if np.all(rows[1:] > rows[:-1]):
            return
        items = list(self.nodes.items())
        self.nodes.clear()
        self.nodes.update(items[i] for i in np.argsort(rows, kind="stable").tolist())

    def holdNodes(self, labels):
        """Hold nodes of the mesh in this process from now on, whichever elements it holds, and make their
        node objects: how a model entity that reads nodes whole in every process -- a constraint's node set,
        the nodes of contact facets -- gets them where the process holds the nodes of its subdomain only
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.holdNodes`). In a serial run,
        and wherever every node is held, it changes nothing.

        Parameters
        ----------
        labels
            The node labels.
        """

        labels = [int(label) for label in labels]
        self.elementDistribution.holdNodes(labels)
        nodes = self.nodes
        if any(label not in nodes for label in labels):
            # making node objects is a change of the topology, also where a reader holds nodes after the
            # model was set up
            with self.topology.changes():
                self.holdNodesOfDistribution()

    def holdNodesOfDistribution(self):
        """Make the node objects of the nodes the element distribution holds here, and drop the others
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.isNodeHeldHere`), keeping
        :attr:`nodes` in mesh order; after the nodes held here changed -- elements moved between the
        processes, or a reader holds nodes --, inside a topology change. Once the node fields exist, they
        are resized for the nodes now held (their values kept where a node stays), and the field variables
        linked to them again.
        """

        isHeldHere = self.elementDistribution.isNodeHeldHere
        dropped = [label for label in self.nodes if not isHeldHere(label)]
        if dropped and not self.topology.isOpen:
            raise TopologyError(
                "nodes were dropped outside a topology change: node objects are created and dropped only inside "
                "TopologyPipeline.changes()"
            )
        for label in dropped:
            del self.nodes[label]
            self._nodesAwaitingValues.discard(label)
        nodesBefore = set(self.nodes)
        # created afresh from the first row: a node skipped before may be held now
        self._rowsWithNodeObjects = (None, 0)
        self.createNodesOfMesh()
        if dropped:
            self._nodeSetsResolvedFor = None
            self.resolveNodeSetsOfMesh()
        if self._nodeFieldsPrepared and (dropped or len(self.nodes) != len(nodesBefore)):
            newNodes = [node for label, node in self.nodes.items() if label not in nodesBefore]
            self._resizeNodeFieldsForNodes(Journal(verbose=False), newNodes)
            # every node: the resize made new arrays, of which the field variables are views
            self._linkFieldVariableObjects(list(self.nodes.values()))

    def nodesAwaitingValues(self) -> set:
        """The labels of the nodes made here during the run whose node field values nothing here wrote:
        where the process holds the nodes of its subdomain only, a node made during a run starts with
        zeros until the values are written -- a refinement interpolating them, the process an element moved
        from sending them, or the processes holding them sending them after a topology update
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.completeNodesAwaitingValues`).
        Empty where every node is held.

        Returns
        -------
        set
            The node labels.
        """

        return self._nodesAwaitingValues

    def nodeValuesWritten(self, labels):
        """Note that the node field values of nodes were written here (:meth:`nodesAwaitingValues`).

        Parameters
        ----------
        labels
            The node labels.
        """

        self._nodesAwaitingValues.difference_update(labels)

    def coordinatesOfNodes(self, labels) -> list:
        """The coordinates of nodes of the mesh: those of the node object where this process holds one,
        else the reference coordinates the mesh describes -- the same, unless something moves the node
        objects (a rigid body, which every process holds every node for).

        Parameters
        ----------
        labels
            The node labels.

        Returns
        -------
        list
            The coordinates, one array per node.
        """

        nodes, mesh = self.nodes, self.mesh
        return [
            nodes[label].coordinates if label in nodes else mesh.nodeCoordinates[mesh.rowsOfNodes([label])[0]]
            for label in labels
        ]

    def wholeNodeSet(self, name: str, reader: str) -> NodeSet:
        """A node set of the mesh, held whole in this process: how a model entity reading every node of a
        set in every process -- a constraint whose copy every process holds -- reads it. Where the process
        holds the nodes of its subdomain only, the nodes of the set are held here from now on
        (:meth:`holdNodes`). The process evaluating a constraint reads them all; every other process holds
        them only so that its copy of the constraint is the same object as the evaluating one's, and can
        be synchronized from it (and evaluated, if the constraint moves there).

        Parameters
        ----------
        name
            The name of the node set in the mesh.
        reader
            What reads it, for the message of a set that is still not complete.

        Returns
        -------
        NodeSet
            The node set, complete.
        """

        if name not in self.mesh.nodeSets:
            # a set of node objects a model entity made (e.g. the slave nodes a tie tied): held whole
            nodeSet = self.nodeSets[name]
        else:
            self.holdNodes(self.mesh.nodeSets[name].tolist())
            nodeSet = self.resolveNodeSetOfMesh(name)
        nodeSet.requireComplete(reader)
        return nodeSet

    def resolveNodeSetsOfMesh(self):
        """Make every node set of the mesh hold the local nodes; sets that already exist are
        updated in place, so references held to them stay valid."""

        for name in self.mesh.nodeSets:
            self.resolveNodeSetOfMesh(name)

    def resolveNodeSetOfMesh(self, name: str) -> NodeSetOfMesh:
        """Make ``nodeSets[name]`` hold the local nodes of the mesh's node set ``name``.

        Parameters
        ----------
        name
            The name of the node set in :attr:`mesh`.

        Returns
        -------
        NodeSetOfMesh
            The local nodes of the set, in set order; complete if that is all of them.
        """

        nodes = self.nodes
        held = [nodes[label] for label in self.mesh.nodeSets[name].tolist() if label in nodes]

        nodeSet = self.nodeSets.get(name)
        if nodeSet is None:
            nodeSet = self.nodeSets[name] = NodeSetOfMesh(name, held, self.mesh)
        else:
            nodeSet.requireDescribedBy(self.mesh)
            if list(nodeSet.localNodes()) != held:
                nodeSet.replaceMembers(held)
        return nodeSet

    def defineNodeSetOfMesh(self, name: str, labels) -> NodeSetOfMesh:
        """Describe (or redefine) a node set in the mesh, and resolve it to the local nodes: how a
        model entity or a model modifier defines a node set of nodes of the mesh during a run.

        Parameters
        ----------
        name
            The name of the set.
        labels
            The node labels, in order.

        Returns
        -------
        NodeSetOfMesh
            The node set.
        """

        self.mesh.setNodeSet(name, labels)
        return self.resolveNodeSetOfMesh(name)

    def redefineNodeSet(self, name: str, labels: list):
        """Give a node set new members, in place (references held to it stay valid): a node set of the
        mesh is redefined in the mesh and resolved again (:meth:`defineNodeSetOfMesh`); a set of node
        objects a model entity made (e.g. the slave nodes a tie tied) is given the node objects.

        Parameters
        ----------
        name
            The name of the node set.
        labels
            The labels of its new members, in order.
        """

        if name in self.mesh.nodeSets:
            self.defineNodeSetOfMesh(name, labels)
        else:
            self.nodeSets[name].replaceMembers([self.nodes[label] for label in labels])

    def checkNodesAgreeWithMesh(self):
        """Raise unless :attr:`nodes` holds nodes of the mesh, in mesh order, and every node of the
        mesh the element distribution holds here -- every node, unless a process of a domain-decomposed run
        holds the nodes of its subdomain only
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.isNodeHeldHere`).

        The order matters: the nodes of every field, and so the degrees of freedom, are numbered in
        the order of :attr:`nodes`, and the numbering of the mesh
        (:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`) in mesh order.

        Raises
        ------
        TopologyError
            If a node of the model is not described in the mesh, a node of the mesh is not local,
            or the nodes are not in mesh order.
        """

        heldLabels = np.fromiter(self.nodes.keys(), dtype=np.int64, count=len(self.nodes))
        isHeldHere = self.elementDistribution.isNodeHeldHere
        meshLabels = self.mesh.nodeLabels
        if self.elementDistribution.holdsNodesOfSubdomainOnly:
            meshLabels = meshLabels[np.fromiter((isHeldHere(label) for label in meshLabels.tolist()), dtype=bool)]
        if not np.array_equal(heldLabels, meshLabels):
            notDescribed = sorted(set(heldLabels.tolist()) - set(meshLabels.tolist()))
            notHeld = sorted(set(meshLabels.tolist()) - set(heldLabels.tolist()))
            raise TopologyError(
                "the nodes of the model do not agree with the mesh: {:} node(s) (e.g. {:}) are not described in "
                "the mesh, {:} node(s) of the mesh (e.g. {:}) have no node object{:} -- nodes are described in "
                "model.mesh and created by FEModel.createNodesOfMesh, in mesh order".format(
                    len(notDescribed),
                    notDescribed[:5],
                    len(notHeld),
                    notHeld[:5],
                    "" if notDescribed or notHeld else ", and they are not in mesh order",
                )
            )

    def createAuxiliaryNodes(self, nodes: list):
        """Add nodes a model entity (a discrete rigid body) makes itself, instead of the input
        describing them in the mesh; they are described in the mesh as well, from these objects.

        This special path is kept only so that the float32 arithmetic of the rigid bodies stays bitwise:
        a discrete rigid body makes its surface nodes with the coordinates of its surface file in that
        file's precision (single precision for an STL file), which the mesh (double precision) would
        not reproduce. Every other node is described in :attr:`mesh` first and created from there, by
        :meth:`createNodesOfMesh` -- which is called first here, so that the nodes stay in mesh order.

        Parameters
        ----------
        nodes
            The nodes, already carrying labels obtained from
            :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveNodeNumbers`.
        """

        self.createNodesOfMesh()
        self.mesh.addNodes([node.label for node in nodes], [node.coordinates for node in nodes])
        self.mesh.requireEachNodeDescribedOnce()
        for node in nodes:
            self.nodes[node.label] = node

    def createAuxiliaryElement(self, element, hostElement: int | None = None):
        """Add an auxiliary element to the model: one a model entity (a contact surface, a rigid body)
        makes itself, instead of the input describing it in the mesh; it is added to the mesh as well.

        This is the special path. Every other element is described in :attr:`mesh` first and created
        from there, by :meth:`createElementsOfMesh` (or :meth:`createElementOfMesh`).

        In a domain-decomposed run, an auxiliary element is made in every process -- it is
        surface-sized -- and :attr:`elementDistribution` decides which process computes it
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.placeAuxiliaryElement`).

        Parameters
        ----------
        element
            The element, already carrying a number obtained from :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveElementNumbers`.
        hostElement
            The number of its host element: the element of the mesh it lies on (the solid element
            whose face a facet tiles), or None if it has none.
        """

        self._checkElementCanBeAdded(element.elNumber)
        record = self.mesh.addAuxiliaryElement(element, hostElement)
        self.elements[element.elNumber] = element
        self.elementDistribution.placeAuxiliaryElement(record)

    def createElementOfMesh(self, number: int):
        """Create the element object of one element of the mesh, and add it to the model.

        This is how every element described in :attr:`mesh` comes into existence -- at setup through
        :meth:`createElementsOfMesh`, during a run through a model modifier (e.g. the children of a
        refinement). It needs nothing but the mesh and the nodes, so an element can be created in any
        process at any time, e.g. when it migrates between processes. Its section and properties are
        assigned separately (:meth:`~edelweissfe.sections.base.sectionbase.Section.assignSectionToElement`).

        Parameters
        ----------
        number
            The number of the element in the mesh.

        Returns
        -------
        element
            The new element.
        """

        self._checkElementCanBeAdded(number)
        record = self.mesh.elements[number]
        element = self.mesh.elementClassOf(record)(record.elType, number)
        nodes = self.nodes
        element.setNodes([nodes[label] for label in record.nodeLabels])
        self.elements[number] = element
        return element

    def createChildElementOfMesh(self, number: int, parentNumber: int):
        """Create the element object of an element a model modifier described in place of another --
        the child of a refined element.

        In a domain-decomposed run the child is created where its parent is computed
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.placeChildElement`), and
        the other processes create nothing.

        Parameters
        ----------
        number
            The number of the child in the mesh.
        parentNumber
            The number of the element it replaces (in part), still described in the mesh.

        Returns
        -------
        element | None
            The new element; None if it is created in another process.
        """

        self.elementDistribution.placeChildElement(number, parentNumber)
        if not self.elementDistribution.isLocal(number):
            return None
        return self.createElementOfMesh(number)

    def createElementsOfMesh(self):
        """Create the element objects of the mesh, and resolve its element sets and surfaces to them.

        Elements are created in mesh order, so :attr:`elements` lists them in the order they were
        described. Each element set of the mesh becomes an
        :class:`~edelweissfe.sets.elementset.ElementSetOfMesh` of the local elements, and each surface
        an :class:`~edelweissfe.surfaces.entitybasedsurface.EntityBasedSurface` of those sets.

        Which elements are created, :attr:`elementDistribution` decides
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.isLocal`): every element
        in a serial run; in a domain-decomposed run with a distributed model, only the local ones --
        those a process computes.

        Calling it again is harmless: elements already created are kept (the distribution is asked
        again for the others), and sets and surfaces that already exist are updated in place
        (references held to them stay valid), so a mesh described in several steps can be made in
        several steps. :meth:`prepareYourself` checks that every element of the mesh was either
        local or declared not local, and that every element of the model is described in the mesh.
        """

        isLocal = self.elementDistribution.isLocal
        for number in self.mesh.elements:
            if number not in self.elements and isLocal(number):
                self.createElementOfMesh(number)

        self.resolveSetsAndSurfacesOfMesh()

    def dropElementOfMesh(self, number: int):
        """Drop the element object of an element of the mesh from this process; the counterpart of
        :meth:`createElementOfMesh`. The mesh keeps the element.

        Used when an element moves to another process (load balancing in a domain-decomposed run):
        the element still exists, only no longer here. Its sets and surfaces are updated by
        :meth:`resolveSetsAndSurfacesOfMesh`.

        Parameters
        ----------
        number
            The number of the element in the mesh.
        """

        if not self.topology.isOpen:
            raise TopologyError(
                "element {:} was dropped outside a topology change: element objects are created and dropped "
                "only inside TopologyPipeline.changes()".format(number)
            )
        del self.elements[number]

    def putElementsInMeshOrder(self):
        """Order :attr:`elements` as the mesh lists them, in place, after elements were created later
        than the others (e.g. when they moved here from another process).

        The order of :attr:`elements` is the order their contributions are summed in, and a serial
        run sums them in mesh order.
        """

        position = {number: index for index, number in enumerate(self.mesh.elements)}
        ordered = sorted(self.elements.items(), key=lambda item: position[item[0]])
        self.elements.clear()
        self.elements.update(ordered)

    def resolveSetsAndSurfacesOfMesh(self):
        """Make every element set and surface of the mesh hold the local elements; sets and
        surfaces that already exist are updated in place, so references held to them stay valid."""

        for name in self.mesh.elementSets:
            self.resolveElementSetOfMesh(name)

        for name in self.mesh.surfaces:
            self.resolveSurfaceOfMesh(name)

    def resolveElementSetOfMesh(self, name: str) -> ElementSetOfMesh:
        """Make ``elementSets[name]`` hold the elements of the mesh's element set ``name`` created here.

        An existing :class:`~edelweissfe.sets.elementset.ElementSetOfMesh` of that name is updated in place,
        so that references to it stay valid.

        Parameters
        ----------
        name
            The name of the element set in :attr:`mesh`.

        Returns
        -------
        ElementSetOfMesh
            The elements of the set created here, in set order; complete if that is all of them.
        """

        numbers = self.mesh.elementSets[name]
        elements = self.elements
        created = [elements[number] for number in numbers if number in elements]

        elementSet = self.elementSets.get(name)
        if elementSet is None:
            elementSet = self.elementSets[name] = ElementSetOfMesh(
                name, created, self.mesh, self.nodes, self.elementDistribution
            )
        else:
            if list(elementSet.localElements()) != created:
                elementSet.replaceMembers(created)
            # the set may have changed in the mesh even where the local part did not
            elementSet.describedBy(self.mesh, self.nodes, self.elementDistribution)
        return elementSet

    def resolveSurfaceOfMesh(self, name: str):
        """Make ``surfaces[name]`` the surface of the mesh with the given name, on the elements created
        here; an existing surface is updated in place if its faces changed.

        A face given by an element set refers to the resolved set itself, as a surface defined in the
        input file always did; a face given by element numbers becomes an
        :class:`~edelweissfe.sets.elementset.ElementSetOfSurfaceFace`. Either holds the local elements of
        the face, and knows whether that is all of them.

        Parameters
        ----------
        name
            The name of the surface in :attr:`mesh`.
        """

        faces = {}
        for face, surfaceFace in self.mesh.surfaces[name].items():
            if surfaceFace.elementSetName is not None:
                faces[face] = self.elementSets[surfaceFace.elementSetName]
            else:
                local = [self.elements[number] for number in surfaceFace.elementNumbers if number in self.elements]
                faces[face] = ElementSetOfSurfaceFace(
                    name, face, local, self.mesh, self.nodes, self.elementDistribution
                )

        surface = self.surfaces.get(name)
        if surface is None:
            self.surfaces[name] = EntityBasedSurface(name, faces)
        elif surface.keys() != faces.keys() or any(
            list(surface[face].localElements()) != list(faces[face].localElements()) for face in faces
        ):
            surface.replaceData(faces)

    def requireCompleteMesh(self, reader: str):
        """State that ``reader`` needs every element of the mesh, not only the local ones; the
        whole-model counterpart of :meth:`~edelweissfe.sets.elementset.ElementSet.requireComplete`.

        Parameters
        ----------
        reader
            Who reads the whole model, for the error message.

        Raises
        ------
        TopologyError
            If some element of the mesh was not created in this process.
        """

        notCreatedHere = self.mesh.elements.keys() - self.elements.keys()
        if notCreatedHere:
            raise TopologyError(
                "{:} needs every element of the model, but {:} of them have no element object here: {:}".format(
                    reader, len(notCreatedHere), self.elementDistribution.whyElementsAreMissingHere()
                )
            )

    def wholeElementSet(self, name: str, reader: str) -> ElementSet:
        """The element set of the given name, which ``reader`` needs whole -- every element of it in
        this process, e.g. the contact facets of a surface, which every process makes
        (:meth:`createAuxiliaryElement`).

        Parameters
        ----------
        name
            The name of the element set.
        reader
            Who reads the set, for the error message.

        Returns
        -------
        ElementSet
            The element set.

        Raises
        ------
        TopologyError
            If only part of the set was created in this process
            (:meth:`~edelweissfe.sets.elementset.ElementSet.requireComplete`).
        """

        elementSet = self.elementSets[name]
        elementSet.requireComplete(reader)
        return elementSet

    def checkElementsAgreeWithMesh(self):
        """Raise unless :attr:`elements` and :attr:`mesh` describe the same elements, in both directions.

        * Every element of the mesh was either local or declared not local by :attr:`elementDistribution`
          -- an element described after the elements were made, and never made, would otherwise
          silently be missing from the model.
        * Every element of :attr:`elements` is described in the mesh -- an element assigned to
          :attr:`elements` directly would otherwise silently be missing from everything the mesh
          determines, e.g. the fields at its nodes and the layout of the degrees of freedom.

        Raises
        ------
        TopologyError
            If some element of the mesh was neither local nor declared not local, or some element of the
            model is not described in the mesh.
        """

        neverCreated = sorted(
            number
            for number in self.mesh.elements.keys() - self.elements.keys()
            if self.elementDistribution.isLocal(number)
        )
        if neverCreated:
            raise TopologyError(
                "{:} element(s) of the mesh (e.g. {:}) were described but never created -- elements described "
                "after FEModel.createElementsOfMesh must be made by calling it again".format(
                    len(neverCreated), neverCreated[:5]
                )
            )

        notDescribed = sorted(self.elements.keys() - self.mesh.elements.keys())
        if notDescribed:
            raise TopologyError(
                "{:} element(s) of the model (e.g. {:}) are not described in the mesh -- elements are created "
                "from the mesh: describe them in model.mesh and call FEModel.createElementsOfMesh".format(
                    len(notDescribed), notDescribed[:5]
                )
            )

    def _checkElementCanBeAdded(self, elNumber: int):
        """Raise unless an element with this number may be added to the model now."""

        if not self.topology.isOpen:
            raise TopologyError(
                "element {:} was created outside a topology change: only model modifiers may create "
                "or delete elements, inside TopologyPipeline.changes()".format(elNumber)
            )
        if elNumber in self.elements:
            raise TopologyError(
                "element number {:} is already taken -- element numbers are reserved via "
                "TopologyPipeline.reserveElementNumbers() and never recycled".format(elNumber)
            )

    def removeElement(self, elNumber: int):
        """Remove an element from the mesh and, if it is local, from the model; see
        :meth:`removeElements`.

        Parameters
        ----------
        elNumber
            The number of the element to remove.
        """

        self.removeElements((elNumber,))

    def removeElements(self, numbers):
        """Remove elements from the mesh -- and so from its element sets and surfaces
        (:meth:`~edelweissfe.models.mesh.Mesh.removeElements`) -- and, those local, from the model.
        Their numbers are retired, never reissued. The element sets and surfaces of the model follow
        when they are next resolved (:meth:`resolveSetsAndSurfacesOfMesh`).

        Parameters
        ----------
        numbers
            The numbers of the elements to remove.
        """

        numbers = list(numbers)
        if not self.topology.isOpen:
            raise TopologyError(
                "element(s) {:} were deleted outside a topology change: only model modifiers may create "
                "or delete elements, inside TopologyPipeline.changes()".format(numbers[:5])
            )

        self.mesh.removeElements(numbers)
        for number in numbers:
            self.elements.pop(number, None)

    def _activateNodeFieldsFromMesh(
        self,
    ):
        """Create the FieldVariables at the nodes, for the fields every element of the mesh has at
        its nodes.

        Which fields an element has at which node depends on its type only, so this needs the mesh
        and no element object: the fields -- and hence the layout of the degrees of freedom -- are
        the same in every process, whichever elements it created.
        """

        nodes = self.nodes
        typeOf = self.mesh.typeOf
        for record in self.mesh.elements.values():
            for label, elementNodeFields in zip(record.nodeLabels, typeOf(record).fields):
                node = nodes.get(label)
                if node is None:
                    # held by another process only (see createNodesOfMesh)
                    continue
                nodeFields = node.fields
                for field in elementNodeFields:
                    if field not in nodeFields:
                        nodeFields[field] = FieldVariable(node, field)

    def _activateNodeFieldsOfNodes(self, nodes: list):
        """Create the FieldVariables of the given nodes, for the fields the elements of the mesh have at
        them (:meth:`_activateNodeFieldsFromMesh`), from the global numbering of the mesh -- arrays, so that
        the cost of a few new nodes (elements moving here) is not a walk over the whole mesh.

        Parameters
        ----------
        nodes
            The nodes.
        """

        if not nodes:
            return
        carrying = GlobalDofNumbering.ofModel(self).nodesCarryingField
        rows = self.mesh.rowsOfNodes([node.label for node in nodes])
        for field, carriedAt in carrying.items():
            for node in (node for node, carries in zip(nodes, carriedAt[rows].tolist()) if carries):
                if field not in node.fields:
                    node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromConstraints(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        constraints.
        """

        for constraint in self.constraints.values():
            for node, nodeFields in zip(constraint.nodes, constraint.fieldsOnNodes):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _createNodeFieldsFromNodes(self, nodes: list, nodeSets: list) -> dict[str, NodeField]:
        """Bundle nodal FieldVariables together in contiguous NodeFields.

        Parameters
        ----------
        nodes
            The list of Nodes from which the NodeFields should be created.
        nodeSets
            The list of NodeSets, which should be considered in the index map of the NodeFields.

        Returns
        -------
        dict[str,NodeField]
            The dictionary containing the NodeField instances for every active field."""

        domainSize = self.domainSize

        theNodeFields = dict()
        for field in phenomena.keys():
            theNodeField = NodeField(field, getFieldSize(field, domainSize), nodes)

            if theNodeField.nodes:
                theNodeFields[field] = theNodeField

        return theNodeFields

    def _linkFieldVariableObjects(self, nodes):
        """Link NodeFields to individual FieldVariable objects.

        Parameters
        ----------
        nodes
            Nodes to be linked

        Inside a deferring :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.changes` window this only notes the request; the link is
        made once, for these nodes as they are then, when the window closes.
        """

        if self.topology.isDeferringFieldBookkeeping:
            self.topology.deferFieldVariableLink(nodes)
            return

        for node in nodes:
            for field, fieldVariable in node.fields.items():
                nodeField = self.nodeFields[field]
                idx = nodeField._indicesOfNodesInArray[node]
                fieldVariable.values = self.nodeFields[field]["U"][idx, :]

        return

    def _requestAdditionalScalarVariable(self, name: str):
        """Create a new scalar variables

        Parameters
        ----------
        name
            The name of the variable.

        Returns
        -------
        ScalarVariable
            The instance of the variable.
        """
        if name in self.scalarVariables:
            raise Exception("ScalarVariable with name {:} already exists!".format(name))

        self.scalarVariables[name] = ScalarVariable()
        return self.scalarVariables[name]

    def _createAndAssignScalarVariableForConstraints(self, journal: Journal):
        """Create ScalarVariables for constraints.

        Parameters
        ----------
        journal
            The journal intance.
        """
        # we may have additional scalar degrees of freedom, not associated with any node (e.g, lagrangian multipliers of constraints)

        for constraintName, constraint in self.constraints.items():
            nAdditionalScalarVariables = constraint.getNumberOfAdditionalNeededScalarVariables()
            if nAdditionalScalarVariables > 0:
                journal.message(
                    "Constraint {:} requests {:} additional ScalarVariables".format(
                        constraintName, nAdditionalScalarVariables
                    ),
                    self.identification,
                    2,
                )

                scalarVariables = [
                    self._requestAdditionalScalarVariable("{:}_{:}".format(constraintName, i))
                    for i in range(nAdditionalScalarVariables)
                ]

                constraint.assignAdditionalScalarVariables(scalarVariables)

    def _prepareVariablesAndFields(self, journal):
        """Prepare all variables and fields for a simulation.

        Parameters
        ----------
        journal
            The journal instance.
        """
        journal.message(
            "Activating fields on nodes from the mesh and the constraints",
            self.identification,
        )
        self._activateNodeFieldsFromMesh()
        self._populateNodeFieldVariablesFromConstraints()

        journal.message("Bundling fields on nodes to NodeFields", self.identification)
        # of the nodes held here (NodeField reads the local part of a node set)
        self.nodeFields = self._createNodeFieldsFromNodes(self.nodeSets["all"], self.nodeFields.values())
        self._nodeFieldsPrepared = True

        journal.message("Assembling ScalarVariables", self.identification)
        self.scalarVariables = dict()
        self._createAndAssignScalarVariableForConstraints(journal)

    def _resizeNodeFieldsForNodes(self, journal: Journal, newNodes: list | None = None):
        """Resize the existing NodeFields in place for the current ``self.nodeSets["all"]``,
        instead of rebuilding :attr:`nodeFields` from scratch as :meth:`_prepareVariablesAndFields`
        does. Used by mesh mutators (e.g. AMR's ``hadaptivity._materialize``) so that NodeField
        identity -- and hence any :class:`~edelweissfe.fields.nodefield.NodeFieldSubset` or
        reference a consumer cached -- survives a topology change. ScalarVariables (e.g. Lagrange
        multipliers of constraints) are rebuilt, but values are preserved by name for constraints
        that still exist, so their converged state survives the refinement too.

        Everything here is recomputed from the current elements, constraints and ``nodeSets["all"]``
        -- the node order of the resized fields is the model's node creation order, which is why one
        call after several mutations yields the same layout as one call per mutation. Inside a
        deferring :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.changes` window the call is therefore only noted, and made once when
        the window closes.

        Parameters
        ----------
        journal
            The journal instance.
        newNodes
            The nodes created since the node fields were last sized, if only these need their fields
            activated (elements moved here: see :meth:`holdNodesOfDistribution`); None to activate the fields
            of every node from the mesh.
        """
        if self.topology.isDeferringFieldBookkeeping:
            self.topology.deferNodeFieldResize(journal)
            return

        journal.message(
            "Activating fields on nodes from the mesh and the constraints",
            self.identification,
        )
        if newNodes is None:
            self._activateNodeFieldsFromMesh()
        else:
            self._activateNodeFieldsOfNodes(newNodes)
        self._populateNodeFieldVariablesFromConstraints()

        journal.message("Resizing NodeFields", self.identification)
        nodes = self.nodeSets["all"]
        for nodeField in self.nodeFields.values():
            nodeField.resize(nodes)

        # a phenomenon activated for the first time (e.g. by a newly materialized constraint) has
        # no NodeField yet -- create it exactly as _prepareVariablesAndFields would, and keep the node
        # fields in the order of the phenomena, as there: the order the fields are numbered in
        # (GlobalDofNumbering). The dict and the NodeFields keep their identity.
        for field in phenomena.keys():
            if field not in self.nodeFields:
                newNodeField = NodeField(field, getFieldSize(field, self.domainSize), nodes)
                if newNodeField.nodes:
                    # with the entries every node field has from the start (see the driver)
                    newNodeField.createFieldValueEntry("U")
                    newNodeField.createFieldValueEntry("P")
                    self.nodeFields[field] = newNodeField
        inPhenomenaOrder = [(field, self.nodeFields[field]) for field in phenomena if field in self.nodeFields]
        if list(self.nodeFields) != [field for field, _ in inPhenomenaOrder]:
            self.nodeFields.clear()
            self.nodeFields.update(inPhenomenaOrder)

        journal.message("Assembling ScalarVariables", self.identification)
        # rebuilding scalarVariables cold-starts every value (ScalarVariable() defaults to 0.0), which
        # would silently discard converged Lagrange-multiplier values on every AMR refinement, even
        # though node fields are warm-started. Snapshot by name and restore the overlapping subset so
        # unchanged constraints keep their converged multiplier and only genuinely new ones cold-start.
        previousScalarVariableValues = {name: v.value for name, v in self.scalarVariables.items()}
        self.scalarVariables = dict()
        self._createAndAssignScalarVariableForConstraints(journal)
        for name, v in self.scalarVariables.items():
            if name in previousScalarVariableValues:
                v.value = previousScalarVariableValues[name]

    def _prepareElements(self, journal: Journal):
        """Prepare elements for a simulation.
        In detail, sections are assigned.


        Parameters
        ----------
        journal
            The journal instance.
        """
        self.assignSectionsAndPropertiesToElements(self.elements)

    def materialPropertiesToExport(self) -> list[tuple]:
        """Collective in a distributed run. The material properties of every element of every section
        that writes them to a file (``>>writeMaterialPropertiesToFile``): the whole model's, in mesh
        order. Who writes the files decides the writer -- the driver.

        In a domain-decomposed run, each element's properties come from the process owning it: in a
        distributed run they are gathered
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.resultsOfWholeSet`), so
        every process must call this, and only one of them writes the files.

        Returns
        -------
        list[tuple]
            Per such section, the section and, per element set of it, the element numbers in set
            order and their material properties, one row per element (None for a set without
            elements), as :meth:`~edelweissfe.sections.base.sectionbase.Section.writeMaterialPropertiesFile`
            takes them.
        """

        distribution = self.elementDistribution
        exports = []
        for section in self.sections.values():
            if not section.writeMaterialPropertiesToFile:
                continue
            rowsOfSets = []
            for elementSet in section.elSets:
                owned = distribution.ownedElements(elementSet.localElements())
                properties = np.array([element._materialProperties for element in owned]) if owned else None
                rowsOfSets.append(
                    (
                        elementSet.elementNumbersOfWholeSet(),
                        distribution.resultsOfWholeSet(elementSet, [element.elNumber for element in owned], properties),
                    )
                )
            exports.append((section, rowsOfSets))
        return exports

    def assignSectionsAndPropertiesToElements(self, elements: dict):
        """Assign sections and element properties to elements: to every element when the model is
        prepared, and to elements created later, e.g. elements that moved here from another process.

        Each element receives the same in either case, in the same order: every section whose
        element sets contain it, then every element property of such a set.

        Parameters
        ----------
        elements
            The elements, by number; members of the element sets already (see
            :meth:`resolveSetsAndSurfacesOfMesh`).
        """

        for section in self.sections.values():
            for elementSet in section.elSets:
                for element in elementSet.localElements():
                    if element.elNumber in elements:
                        section.assignSectionToElement(element, self)

        for elementProperty in self.elementProperties:
            for element in self.elementSets[elementProperty.elSetName].localElements():
                if element.elNumber in elements:
                    elementProperty.assignToElement(element)

        self._requireMaterialAssigned(elements)

    @staticmethod
    def _requireMaterialAssigned(elements: dict):
        """Raise unless every given element was assigned a material.

        Parameters
        ----------
        elements
            The elements, by number.
        """

        materialAssigned = np.fromiter(map(attrgetter("hasMaterial"), elements.values()), dtype=bool)
        if not materialAssigned.all():
            elementIds = np.array([str(elId) for elId in elements.keys()])[np.logical_not(materialAssigned)]
            raise Exception(f"No material was assigned to element(s) with id(s) {', '.join(elementIds)}.")

    def prepareYourself(self, journal: Journal):
        """Prepare the model for a simulation.
        Creates the variables, bundles the fields,
        and initializes elements.


        Parameters
        ----------
        journal
            The journal instance.
        """

        self.checkNodesAgreeWithMesh()
        self.checkElementsAgreeWithMesh()
        self.topology.adoptSetupElementNumbers()
        self.topology.ensureSurfaceFacetModifier(journal)
        self.topology.checkModelModifierDomains()
        self._prepareVariablesAndFields(journal)
        self._prepareElements(journal)
        self.topology.recordSetupFingerprint()

    def advanceToTime(self, time: float):
        """Accept the current state of the model and sub instances, and
        set the new time.

        Parameters
        ----------
        time
            The new time.
        """

        self.time = time
        self.acceptStatesOf(self.elements, self.constraints)

    def acceptStatesOf(self, elements: dict, constraints: dict):
        """Let the given elements and constraints, and every multi-point constraint, accept their
        computed state: :meth:`advanceToTime` without the time, for part of the model. A solver that
        computes only part of the model accepts that part -- the state of an element it did not
        compute is not its to accept.

        Parameters
        ----------
        elements
            The elements whose computed state is accepted, by number.
        constraints
            The constraints whose computed state is accepted, by name.
        """

        self._acceptElementStates(elements)

        # Left serial deliberately. Measured on the 337 471-dof explicit anchor pry-out model, where
        # the element loop below costs 19.09 ms per call: these two together cost 0.015 ms, i.e. less
        # than a tenth of a percent of it. There is nothing here to parallelize.
        for constraint in constraints.values():
            constraint.acceptLastState()

        for mpc in self.multiPointConstraints.values():
            mpc.acceptLastState()

    def _acceptElementStates(self, elements: dict):
        """Let every given element accept its computed state, across the available threads.

        An element's ``acceptLastState`` touches only that element's own state buffers -- for a
        Marmot element it is the single ``self._stateVars[:] = self._stateVarsTemp`` copy -- so the
        elements are independent of one another and this loop parallelizes without any coordination.

        It is worth parallelizing because of how the explicit solver uses it, not because of how the
        implicit ones do. An implicit analysis calls this once per *converged* increment, amortised
        over a Newton loop and a linear solve, where 19 ms disappears into the noise. Explicit
        dynamics calls it on every one of millions of increments: on the anchor pry-out model it was
        15.6 % of the whole step, the single largest cost outside the element kernels themselves, and
        every millisecond of it was serial Python holding 31 of 32 threads idle.

        Parameters
        ----------
        elements
            The elements whose state is accepted, by number.
        """

        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        if numThreads == 1 or len(elements) < numThreads:
            for element in elements.values():
                element.acceptLastState()
            return

        def acceptChunk(chunk):
            for element in chunk:
                element.acceptLastState()

        # The same chunk granularity the parallel element computation uses: four chunks per thread,
        # enough to even out the spread between cheap facet elements and expensive solid ones
        # without paying dispatch overhead per element.
        chunkSize = max(1, len(elements) // (numThreads * 4))
        chunks = chunked_iterable(elements.values(), chunkSize)

        # list(), not a bare call: Executor.map is lazy, and leaving the iterator unconsumed would
        # return here with workers still writing element state.
        list(getThreadPool(numThreads).map(acceptChunk, chunks))

    def elementStatesForCheckpoint(self) -> Iterator[tuple[int, np.ndarray]]:
        """The converged state of every element of the mesh, by number, for a restart checkpoint:
        read from the element objects -- or from the states given for this checkpoint
        (:meth:`elementStatesFromElsewhere`).

        Returns
        -------
        Iterator[tuple[int, np.ndarray]]
            ``(number, state)`` pairs, one per element of the mesh.

        Raises
        ------
        TopologyError
            If some element of the mesh has no element object here, and no states were given.
        """

        if self._elementStatesFromElsewhere is not None:
            return iter(self._elementStatesFromElsewhere.items())
        self.requireCompleteMesh("A restart checkpoint")
        return ((number, element.getStateVars()) for number, element in self.elements.items())

    @contextmanager
    def elementStatesFromElsewhere(self, states: dict | None):
        """A context in which a restart checkpoint reads the element states from ``states`` instead of
        the local element objects (:meth:`elementStatesForCheckpoint`): a process that holds only
        some of the elements -- in a domain-decomposed run -- writes a checkpoint of the whole model
        from the states gathered from the processes holding the others.

        Parameters
        ----------
        states
            The converged state of every element of the mesh, by number; None to read them from the
            element objects, as without the context.

        Yields
        ------
        None
        """

        self._elementStatesFromElsewhere = states
        try:
            yield
        finally:
            self._elementStatesFromElsewhere = None

    @contextmanager
    def nodeFieldsFromElsewhere(self, nodeFields: dict | None):
        """A context in which a restart checkpoint reads the node fields from ``nodeFields`` instead of
        :attr:`nodeFields`: a process that holds only some of the nodes -- in a domain-decomposed run --
        writes a checkpoint of the whole model from the node fields gathered from the processes computing
        the others
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.nodeFieldsOfWholeModel`).

        Parameters
        ----------
        nodeFields
            Per field, every entry at every node of the mesh carrying the field, in mesh order; None to
            read them from :attr:`nodeFields`, as without the context.

        Yields
        ------
        None
        """

        self._nodeFieldsFromElsewhere = nodeFields
        try:
            yield
        finally:
            self._nodeFieldsFromElsewhere = None

    def nodeFieldsForCheckpoint(self) -> dict:
        """Every entry of every node field at every node of the mesh carrying it, per field and entry, for
        a restart checkpoint: :attr:`nodeFields` -- or the node fields given for this checkpoint
        (:meth:`nodeFieldsFromElsewhere`).

        Returns
        -------
        dict
            Per field, a dict of the entries' values.

        Raises
        ------
        TopologyError
            If this process holds only some of the nodes, and no node fields were given.
        """

        if self._nodeFieldsFromElsewhere is not None:
            return self._nodeFieldsFromElsewhere
        if not self.elementDistribution.holdsEveryNodeHere:
            raise TopologyError(
                "A restart checkpoint reads the node fields of the whole model, but this process holds only some "
                "of the nodes: the node fields must be gathered for it (FEModel.nodeFieldsFromElsewhere)"
            )
        return {nf.name: nf._values for nf in self.nodeFields.values()}

    def writeRestart(self, restartFile: h5py.File):
        """Write the current (converged) state of the model to a restart checkpoint.

        Does not serialize model topology (nodes/elements/sets/sections/materials) -- only state
        that a rebuild from the original ``.inp`` file cannot reproduce: node field values, element
        quadrature-point history, scalar variables (e.g. Lagrange multipliers), and stateful
        constraints' internal history (e.g. frictional contact).

        Parameters
        ----------
        restartFile
            An open, writable :class:`h5py.File` (or group) to write the checkpoint into.
        """

        f = restartFile

        f.attrs["time"] = self.time

        nodeFieldsGroup = f.create_group("nodeFields")
        # In a domain-decomposed run, also the nodes held in other processes only (see
        # nodeFieldsForCheckpoint).
        for name, entries in self.nodeFieldsForCheckpoint().items():
            nodeFieldGroup = nodeFieldsGroup.create_group(name)
            for entryName, entryValues in entries.items():
                nodeFieldGroup.create_dataset(entryName, data=entryValues)

        scalarVariablesGroup = f.create_group("scalarVariables")
        for name, scalarVariable in self.scalarVariables.items():
            scalarVariablesGroup.attrs[name] = scalarVariable.value

        writeRestartDataOf(f.create_group("modelModifiers"), self.modelModifiers)

        # Every element of the model: in a domain-decomposed run, also those created and computed in
        # other processes (see elementStatesForCheckpoint).
        elementsGroup = f.create_group("elements")
        for elNumber, stateVars in self.elementStatesForCheckpoint():
            elementsGroup.create_dataset(str(elNumber), data=stateVars)

        writeRestartDataOf(f.create_group("constraints"), self.constraints)

        # The ordered record of every applied model-modifier decision (e.g. AMR refinements). A
        # resumed run replays these through the modifiers' own apply() -- see readRestart and
        # TopologyPipeline.replayHistory -- rather than serializing the resulting topology directly, so there
        # is exactly one code path that mutates topology, live or replayed.
        historyGroup = f.create_group("topologyHistory")
        historyGroup.attrs["setupFingerprint"] = self.topology.setupFingerprint()
        historyGroup.attrs["count"] = len(self.topology.history)
        for index, record in enumerate(self.topology.history):
            recordGroup = historyGroup.create_group("{:06d}".format(index))
            recordGroup.attrs["modifier"] = record.modifier
            recordGroup.attrs["roundNumber"] = record.roundNumber
            recordGroup.attrs["time"] = record.time
            recordGroup.attrs["fingerprint"] = record.fingerprint
            for entryName, entryValues in record.plan.items():
                recordGroup.create_dataset(entryName, data=entryValues)

    def rowsOfNodeFieldInWholeField(self, nodeField, numbering: GlobalDofNumbering) -> np.ndarray | None:
        """The row of each node of a node field of the model in the node field of the whole model -- among
        the nodes of the mesh carrying the field, in mesh order, as a checkpoint and a vector of the whole
        model (the global numbering) lay it out: None -- every row, in order -- where this process
        holds every node (:attr:`~edelweissfe.models.elementdistribution.ElementDistribution.holdsEveryNodeHere`);
        else the rows of the nodes it holds.

        Parameters
        ----------
        nodeField
            A node field of the model.
        numbering
            The global numbering of the degrees of freedom of the model.

        Returns
        -------
        np.ndarray | None
            The rows; None for every row, in order.
        """

        if self.elementDistribution.holdsEveryNodeHere:
            return None
        meshRows = self.mesh.rowsOfNodes([node.label for node in nodeField.nodes])
        return numbering.rankInField[nodeField.name][meshRows].astype(np.int64)

    def readRestart(self, restartFile: h5py.File, journal: Journal = None):
        """Read the state of the model from a restart checkpoint written by :meth:`writeRestart`.

        The model must already have been rebuilt from the original ``.inp`` file (same topology)
        and :meth:`prepareYourself` called, before this is called.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        journal
            Optional Journal for the progress messages of the topology replay.
        """

        f = restartFile

        self.time = f.attrs["time"]

        # Must run before every restore below: replaying the topology history can materialize
        # elements/nodes a plain rebuild from the .inp file cannot reproduce (e.g. AMR-refined
        # children) -- the node-field and element-statevar restores that follow address elements by
        # label and would silently miss anything not already in self.elements/self.nodeFields yet.
        historyGroup = f["topologyHistory"]
        if self.topology.setupFingerprint() != historyGroup.attrs["setupFingerprint"]:
            raise RestartError(
                "the model rebuilt from the input file is not the one the checkpoint was written from: "
                "its mesh (nodes, elements, coordinates, numbering) differs"
            )
        records = []
        for index in range(int(historyGroup.attrs["count"])):
            recordGroup = historyGroup["{:06d}".format(index)]
            records.append(
                TopologyRecord(
                    modifier=str(recordGroup.attrs["modifier"]),
                    roundNumber=int(recordGroup.attrs["roundNumber"]),
                    time=float(recordGroup.attrs["time"]),
                    plan={entryName: values[:] for entryName, values in recordGroup.items()},
                    fingerprint=str(recordGroup.attrs["fingerprint"]),
                )
            )
        self.topology.replayHistory(records, journal)
        readRestartDataInto(f["modelModifiers"], self.modelModifiers)
        # Bring the mesh-dependent consumers (ties, contact surfaces) up to date with the replayed
        # mesh here, as part of restoring the model: the uninterrupted run's consumers had caught up
        # with every change before the checkpoint was written, so a resumed run must start the same.
        self.topology.refreshMeshDependents()

        # A checkpoint holds every node carrying a field, in mesh order; a process holding only some of
        # the nodes reads their rows.
        numbering = None if self.elementDistribution.holdsEveryNodeHere else GlobalDofNumbering.ofModel(self)
        for nf in self.nodeFields.values():
            storedField = f["nodeFields"].get(nf.name)
            if storedField is None:
                raise RestartError("the checkpoint holds no node field {:}".format(nf.name))

            # Iterate the checkpoint's entries, not the model's -- entries created only later by a
            # solver (e.g. the explicit solver's 'V') would otherwise never be restored.
            rows = self.rowsOfNodeFieldInWholeField(nf, numbering)
            for entryName, storedValues in storedField.items():
                if entryName not in nf:
                    nf.createFieldValueEntry(entryName)
                if rows is None:
                    storedValues.read_direct(nf[entryName])
                else:
                    nf[entryName][:] = storedValues[()][rows]
        # every node held here holds the values of the checkpoint now
        self._nodesAwaitingValues.clear()

        for name, scalarVariable in self.scalarVariables.items():
            scalarVariable.value = f["scalarVariables"].attrs[name]

        # One uniform loop, by element number, with nothing swallowed -- sound only because the replay
        # above reproduces the original numbering exactly, verified against the recorded fingerprint.
        # The one element skipped is an element of the mesh that is not local to this process (a
        # domain-decomposed run creates only its own); its process restores it. Any other missing
        # element means the replayed model does not match the one checkpointed, which must be
        # reported, not silently skipped.
        for elementKey, stateVars in f["elements"].items():
            elNumber = int(elementKey)
            element = self.elements.get(elNumber)
            if element is None and elNumber in self.mesh.elements and not self.elementDistribution.isLocal(elNumber):
                continue
            if element is None:
                raise RestartError(
                    "the checkpoint holds state for element {:}, which does not exist after the "
                    "topology replay -- the replayed model does not match the one checkpointed".format(elNumber)
                )
            element.setStateVars(stateVars[:])

        # A rigid body moves its surface nodes into the current configuration, a pure function of
        # its reference point's restored DOFs: put them where the uninterrupted run had them.
        for rigidBody in self.rigidBodies.values():
            rigidBody.updateKinematics()

        readRestartDataInto(f["constraints"], self.constraints)


def printPrettyModelSummary(model: FEModel, journal: Journal):
    identification = "PrettyModelSummary"

    def wrapList(theList):
        for line in textwrap.wrap("[" + ", ".join(theList) + "]"):
            journal.message(
                "  {:<20} ".format(line),
                identification,
                0,
            )

    journal.message(
        "Finite element model with spatial dimension {:} has".format(model.domainSize),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15} ".format("nodes:", len(model.nodes)),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15} ".format("node sets:", len(model.nodeSets)),
        identification,
        0,
    )
    wrapList(model.nodeSets.keys())
    journal.message(
        " {:<20}{:<15} ".format("node fields:", len(model.nodeFields)),
        identification,
        0,
    )
    wrapList(["{:} ({:})".format(k, len(v.nodes)) for k, v in model.nodeFields.items()])
    journal.message(
        " {:<20}{:<15}".format("elements: ", len(model.elements)),
        identification,
        0,
    )
    # A domain-decomposed process may create only some of the elements of the mesh.
    if len(model.mesh.elements) > len(model.elements):
        journal.message(
            " {:<20}{:<15}".format("in the mesh: ", len(model.mesh.elements)),
            identification,
            0,
        )
    journal.message(
        " {:<20}{:<15}".format("element sets: ", len(model.elementSets)),
        identification,
        0,
    )
    wrapList(model.elementSets.keys())
    if model.constraints:
        journal.message(
            " {:<20}{:<15}".format("constraints: ", len(model.constraints)),
            identification,
            0,
        )
    if model.multiPointConstraints:
        journal.message(
            " {:<20}{:<15}".format("multi-point constraints: ", len(model.multiPointConstraints)),
            identification,
            0,
        )
    if model.scalarVariables:
        journal.message(
            " {:<20}{:<15}".format("scalar variables: ", len(model.scalarVariables)),
            identification,
            0,
        )
    journal.message(
        " {:<20}{:<15}".format("materials: ", len(model.materials)),
        identification,
        0,
    )
    wrapList(model.materials.keys())

    if model.analyticalFields:
        journal.message(
            " {:<20}{:<15}".format("analytical fields: ", len(model.analyticalFields)),
            identification,
            0,
        )
        wrapList(model.analyticalFields.keys())
