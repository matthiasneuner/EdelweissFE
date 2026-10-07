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
=================================  ==================================================================  ===================

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
        the parent's state to transfer. Here: every element is computed here, so there is nothing to
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
        Here: every element is computed here, so there is nothing to decide.

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
            The elements owned here, in the order given.
        """

        return list(elements)

    def resultsOfWholeSet(self, elementSet, numbersOwnedHere: list, results: np.ndarray | None) -> np.ndarray:
        """The results of every element of a set, in set order, from the results of the elements
        owned here. Here: the results given, since every element is owned here.

        Parameters
        ----------
        elementSet
            The element set.
        numbersOwnedHere
            The numbers of the elements of the set owned here (:meth:`ownedElements`), in set
            order.
        results
            Their results, one row per element; None if there are none.

        Returns
        -------
        np.ndarray
            The results of every element of the set, one row per element, in set order.
        """

        return results
