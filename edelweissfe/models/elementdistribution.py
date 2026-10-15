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
"""Which elements of the mesh a process creates, and which it owns.

A model is built in two stages: the mesh is described as data, then the element objects are made
from it (:mod:`~edelweissfe.models.mesh`). Between the two stages an :class:`ElementDistribution`
decides which elements are *local* to this process -- have an element object here -- and which it
*owns*: computes, and reports the results and the state of. A serial run creates and owns every
element, and that is what this class does; every method below is trivial here.

A domain-decomposed run may give each process only the elements of its subdomain (*distributed*
elements), or every element (*replicated* elements); its distribution
(:class:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements`) then also says
how a result of a whole element set is gathered from the processes owning the elements. Code that
reads results of a whole element set goes through the methods below, and so reads the same thing in
both cases. Those are all the places the model meets the decomposition:

=================================  ==================================================================  ===================
method                             called by                                                           serial behaviour
=================================  ==================================================================  ===================
:meth:`decideLocalElements`        the input file, once the mesh is described                          nothing to decide
:meth:`isLocal`                    :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`    always True
:meth:`placeAuxiliaryElement`      :meth:`~edelweissfe.models.femodel.FEModel.createAuxiliaryElement`  nothing to decide
:meth:`placeChildElement`          ``FEModel.createChildElementOfMesh``, for refined elements          nothing to decide
:meth:`updateLocalElements`        the topology pipeline, after a mesh change                          nothing to do
:meth:`ownedElements`              element field outputs                                               all of them
:meth:`resultsOfWholeSet`          element field outputs                                               the results given
:meth:`isNodeHeldHere`             :meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh`       always True
:meth:`holdNodes`                  :meth:`~edelweissfe.models.femodel.FEModel.holdNodes`               nothing to do
=================================  ==================================================================  ===================

**Which nodes a process holds.** The nodes are described in the mesh, and a node object is made
for each node of the mesh *held* here (:meth:`isNodeHeldHere`): every node, here and wherever a
process must read nodes outside its subdomain. A distributed job whose processes compute their
subdomains only -- no contact search, no refinement, no rigid body
(:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsToHoldEveryNode`) -- holds in each
process the nodes of its elements and those a model entity reads whole (:meth:`holdNodes`); what reads
the nodes of the whole model -- a per-node field output, a checkpoint -- gathers their values from the
processes computing them (:meth:`nodeValuesOfWholeSet`, :meth:`nodeFieldsOfWholeModel`).

**Replicated or distributed: what a model entity declares.** Whether a domain-decomposed job may
distribute its elements depends on what its constraints, generators and model modifiers read. Each
of these classes declares it once, in the class attribute ``replicatedElementsReason``
(:class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`,
:class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`,
:class:`~edelweissfe.generators.base.generatorbase.GeneratorBase`,
:class:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase`): ``None`` if it reads
only what every process holds whole -- the mesh, the nodes and their fields, contact facets and rigid
bodies -- and no element object; otherwise why it needs every element object in every process. The
base classes default to one of the reasons below, so a new class needs to declare nothing: a job
using it simply runs with replicated elements -- more memory per process, the same result. Setting
``None`` is a claim, to be backed by a test case in ``testfiles/mpi`` that runs distributed
(:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsToReplicateElements` collects the
reasons of a job).
"""

import numpy as np

from edelweissfe.models.mesh import Mesh, MeshElement
from edelweissfe.numerics.globaldofnumbering import GlobalDofNumbering
from edelweissfe.utils.exceptions import TopologyError

#: The default reason of a constraint: not yet verified, by a distributed test case, to read only what
#: every process holds.
NOT_VERIFIED_WITH_DISTRIBUTED_ELEMENTS = (
    "is not yet known to read only what every process holds (the mesh, the nodes, contact facets, rigid bodies)"
)

#: The reason of a constraint with scalar variables of its own (Lagrange multipliers): only the
#: implicit solvers solve for them, and the domain-decomposed solver is explicit.
IMPLICIT_ONLY = "introduces scalar variables (Lagrange multipliers), which only the implicit solvers solve for"

#: The default reason of a generator, which is free to read and change the model.
NOT_KNOWN_TO_ONLY_DESCRIBE_THE_MESH = "is not known to only describe the mesh"

#: The default reason of a model modifier, which is free to read and change the element objects.
CHANGES_THE_MESH_READING_ELEMENT_OBJECTS = (
    "changes the mesh during the run, reading the element objects of the whole model"
)


class ElementDistribution:
    """Every element of the mesh is local, and owned, here; see the module
    documentation."""

    #: Whether every process creates every element of the mesh.
    replicatesElements = True

    #: Whether a process holds only the nodes of its subdomain -- those of its elements, and those a
    #: model entity reads whole (:meth:`holdNodes`) --, decided once per job, the same in every process.
    #: Here: every node is held.
    holdsNodesOfSubdomainOnly = False

    #: Changed whenever elements move between processes, so that whatever was derived from the
    #: elements this process holds or reports (e.g. the result views of a field output) is derived
    #: again. Here: never.
    ownershipVersion = 0

    def decideLocalElements(self, mesh: Mesh, domainSize: int):
        """Decide, once the mesh is described and before any element exists, which of its elements
        this process creates. Here: all of them, so there is nothing to decide.

        Parameters
        ----------
        mesh
            The mesh, as described by the input file and the generators.
        domainSize
            The spatial dimension.
        """

    def placeChildElement(self, childNumber: int, parentNumber: int):
        """Decide where an element a model modifier describes in place of another is computed: the
        child of a refined element is computed by the process that computed its parent, which holds
        the parent's state to transfer. Here: every element is local, so there is nothing to
        decide.

        Parameters
        ----------
        childNumber
            The number of the new element, already described in the mesh.
        parentNumber
            The number of the element it replaces (in part), still described in the mesh.
        """

    def placeAuxiliaryElement(self, record: MeshElement):
        """Decide where an auxiliary element (a contact facet, the point mass of a rigid
        body; see :meth:`~edelweissfe.models.femodel.FEModel.createAuxiliaryElement`) is computed. Such an
        element is surface-sized and made in every process, by the same code, with the same number.
        Here: every element is local, so there is nothing to decide.

        Parameters
        ----------
        record
            The element as described in the mesh, with its host element
            (:attr:`~edelweissfe.models.mesh.MeshElement.hostElement`).
        """

    def updateLocalElements(self, model):
        """After a model modifier changed the mesh, create and drop element objects so that this process
        holds exactly those it needs for the changed mesh. Here: the modifier created every new element itself, so there is
        nothing left to do.

        Parameters
        ----------
        model
            The model tree, its mesh changed.
        """

    def whyElementsAreMissingHere(self) -> str:
        """Why elements of the mesh have no element object here, for an error message. Here: every
        element of the mesh is created, so such elements were described after the elements were made.

        Returns
        -------
        str
            The explanation.
        """

        return (
            "the others were described in the mesh but never created -- elements described after "
            "FEModel.createElementsOfMesh must be made by calling it again"
        )

    def isLocal(self, number: int) -> bool:
        """Whether this process creates the element with the given number; asked by
        :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`.

        Parameters
        ----------
        number
            The element number.

        Returns
        -------
        bool
            Always True.
        """

        return True

    @property
    def holdsEveryNodeHere(self) -> bool:
        """Whether this process holds every node of the mesh: here, always; not in a process holding the
        nodes of its subdomain only (:attr:`holdsNodesOfSubdomainOnly`).

        Returns
        -------
        bool
            Whether every node is held here.
        """

        return True

    def isNodeHeldHere(self, label: int) -> bool:
        """Whether this process holds the node with the given label -- has a node object of it --;
        asked by :meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh`.

        Parameters
        ----------
        label
            The node label.

        Returns
        -------
        bool
            Always True.
        """

        return True

    def holdNodes(self, labels):
        """Hold the given nodes of the mesh here from now on, whichever elements this process holds: the
        nodes a model entity reads whole in every process (a node set of a constraint, the nodes of the
        contact facets). Here: every node is held already.

        Parameters
        ----------
        labels
            The node labels.
        """

    def ownedElements(self, elements) -> list:
        """Those of the given elements whose results and state this process reports: the elements it
        computes. Here: all of them.

        Parameters
        ----------
        elements
            Local elements, e.g. an :class:`~edelweissfe.sets.elementset.ElementSet`.

        Returns
        -------
        list
            The elements this process owns, in the order given.
        """

        return list(elements)

    def resultsOfWholeSet(
        self, elementSet, numbersOwnedHere: list, results: np.ndarray | None, toEveryProcess: bool = True
    ) -> np.ndarray | None:
        """The results of every element of a set, in set order, from the results of the elements
        this process owns. Here: the results given, since it owns every element.

        Parameters
        ----------
        elementSet
            The element set.
        numbersOwnedHere
            The numbers of the elements of the set this process owns (:meth:`ownedElements`), in set
            order.
        results
            Their results, one row per element; None if there are none.
        toEveryProcess
            Whether every process receives the results, or the process writing the output only;
            here, the one process.

        Returns
        -------
        np.ndarray | None
            The results of every element of the set, one row per element, in set order.
        """

        return results

    def nodesProvidingValuesHere(self, model) -> set:
        """The labels of the nodes whose values this process contributes when the values of nodes are
        gathered (:meth:`nodeValuesOfWholeSet`): nodes whose degrees of freedom it integrates, so that
        its node fields hold them current. Here: every node.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        set
            The node labels.
        """

        return set(model.nodes)

    def nodeValuesOfWholeSet(
        self, labels: np.ndarray, labelsHere: np.ndarray, valuesHere: np.ndarray, toEveryProcess: bool = True
    ) -> np.ndarray | None:
        """Collective in a domain-decomposed run. The values of nodes -- one row per node, e.g. an entry of
        a node field at the nodes of a set --, in the order given, from the rows each process contributes
        (:meth:`nodesProvidingValuesHere`). Here: the rows given, every node being held here.

        Parameters
        ----------
        labels
            The labels of the nodes whose values are gathered, in the order of the result.
        labelsHere
            The labels of the nodes whose values this process contributes.
        valuesHere
            Their values, one row per node.
        toEveryProcess
            Whether every process receives the values, or the process writing the output only.

        Returns
        -------
        np.ndarray | None
            The values, one row per node of ``labels``; None in a process not receiving them.
        """

        return assembleNodeValues(labels, [(labelsHere, valuesHere)])

    def nodeFieldsOfWholeModel(self, model, piece: dict, toEveryProcess: bool = False) -> dict | None:
        """Collective in a domain-decomposed run. The node fields of the whole model, as a checkpoint holds
        them (:func:`assembleNodeFields`), gathered to rank 0 -- or to every process -- from the piece each
        process read (:func:`nodeFieldValuesHere`). Here: assembled from the one piece.

        Parameters
        ----------
        model
            The model tree.
        piece
            The piece of this process.
        toEveryProcess
            Whether every process receives them, or rank 0 only.

        Returns
        -------
        dict | None
            Per field, a dict of the entries' values; None in a process not receiving them.
        """

        return assembleNodeFields(model, [piece])


def nodeFieldValuesHere(model) -> dict:
    """The values of the node fields of a model at the nodes whose values this process contributes
    (:meth:`ElementDistribution.nodesProvidingValuesHere`), per field: their labels, and every entry at
    them -- this process' piece of :meth:`ElementDistribution.nodeFieldsOfWholeModel`.

    Parameters
    ----------
    model
        The model tree.

    Returns
    -------
    dict
        Per field, the labels and a dict of the entries' values, one row per label.
    """

    provided = model.elementDistribution.nodesProvidingValuesHere(model)
    piece = {}
    for name, nodeField in model.nodeFields.items():
        rows = [row for row, node in enumerate(nodeField.nodes) if node.label in provided]
        labels = np.asarray([nodeField.nodes[row].label for row in rows], dtype=np.int64)
        piece[name] = (labels, {entry: nodeField[entry][rows] for entry in nodeField.entries()})
    return piece


def assembleNodeFields(model, pieces: list) -> dict:
    """The node fields of the whole model -- per field, every entry at every node of the mesh carrying
    the field, in mesh order, as a checkpoint holds them --, from the pieces of the processes
    (:func:`nodeFieldValuesHere`).

    Parameters
    ----------
    model
        The model tree.
    pieces
        The pieces of the processes.

    Returns
    -------
    dict
        Per field, in the order of the fields, a dict of the entries' values.
    """

    numbering = GlobalDofNumbering.ofModel(model)
    wholeFields = {}
    for field in numbering.fields:
        labels = numbering.nodeLabels[numbering.nodesCarryingField[field]]
        entries = list(dict.fromkeys(entry for piece in pieces if field in piece for entry in piece[field][1]))
        wholeFields[field] = {
            entry: assembleNodeValues(
                labels,
                [
                    (piece[field][0], piece[field][1][entry])
                    for piece in pieces
                    if field in piece and entry in piece[field][1]
                ],
            )
            for entry in entries
        }
    return wholeFields


def assembleNodeValues(labels: np.ndarray, pieces: list) -> np.ndarray:
    """The values of nodes in a given order, from pieces of labels and values -- rows a node appears in
    in several pieces hold the same bits, and the last one is taken.

    Parameters
    ----------
    labels
        The labels of the nodes, in the order of the result.
    pieces
        Pairs of labels and values, one row per label.

    Returns
    -------
    np.ndarray
        The values, one row per node of ``labels``.

    Raises
    ------
    TopologyError
        If a node of ``labels`` is in no piece.
    """

    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    whole, filled = None, np.zeros(labels.shape[0], dtype=bool)
    for pieceLabels, pieceValues in pieces:
        pieceLabels = np.asarray(pieceLabels, dtype=np.int64)
        if not pieceLabels.shape[0]:
            continue
        rows = order[np.minimum(np.searchsorted(labels, pieceLabels, sorter=order), labels.shape[0] - 1)]
        if not np.array_equal(labels[rows], pieceLabels):
            raise TopologyError("values of nodes were contributed that were not asked for")
        if whole is None:
            whole = np.empty((labels.shape[0],) + np.shape(pieceValues)[1:], dtype=np.asarray(pieceValues).dtype)
        whole[rows] = pieceValues
        filled[rows] = True
    if not filled.all():
        raise TopologyError(
            "the values of {:} node(s) (e.g. {:}) were gathered from the processes, but none contributed them".format(
                int((~filled).sum()), labels[~filled][:5].tolist()
            )
        )
    if whole is None:
        whole = np.empty((0,))
    return whole
