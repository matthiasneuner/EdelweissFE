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
"""Which process creates which element: the whole model in every process, or each its own part.

A domain-decomposed job decides, once for the whole job and in one place
(:func:`elementDistributionOfThisJob`), how its processes hold the model:

* **Distributed** (:class:`DistributedElements`): the mesh is partitioned before any element exists,
  and each process creates only the elements it computes -- plus those few others it must evaluate
  itself (see :meth:`DistributedElements.decideWhichElementsAreCreatedHere`). The memory of the
  elements -- most of a model -- is divided among the processes.
* **The whole model in every process** (the plain
  :class:`~edelweissfe.models.elementdistribution.ElementDistribution`): every process creates every
  element, as a serial run does, and the processes keep the states of the elements they do not
  compute current by synchronizing them. That is the fallback for whatever still reads the whole
  model during a run; :func:`reasonsForTheWholeModel` is the rule that names it, and the job reports
  the reason once.

Nothing here imports ``mpi4py`` itself: a distribution under an MPI launcher is given the communicator
of :mod:`.mpienvironment`, and a serial run gets the serial distribution.
"""

import numpy as np

from edelweissfe.config.constraints import getConstraintClass
from edelweissfe.config.generators import getGeneratorClass
from edelweissfe.config.modelmodifiers import getModelModifierClass
from edelweissfe.domaindecomposition.mpienvironment import (
    numberOfProcesses,
    worldCommunicator,
)
from edelweissfe.domaindecomposition.partitioning import (
    partitionElementsOfMesh,
    processOfElementMadeByOwner,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.elementdistribution import ElementDistribution
from edelweissfe.models.mesh import Mesh, MeshElement
from edelweissfe.utils.exceptions import TopologyError


def reasonsForTheWholeModel(inputfile: dict) -> list[str]:
    """Why a job must hold the whole model in every process: the rule of the fallback.

    A job needs every element in every process if something reads, or changes, the whole model
    while it runs, and has not yet been taught to gather what it reads from the processes computing
    it:

    * **model modifiers that read element objects of the whole model** (see
      :attr:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.wholeModelReason`),
      e.g. the surface snap; adaptive refinement (``hAdaptivity``) is not among them: it reads the
      mesh, which every process holds whole, and creates the children of a refined element where
      the parent is computed;
    * **constraints not known to read only what every process holds** (see
      :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.wholeModelReason`). A
      constraint is evaluated whole by one process, but it reads no element object of the solid
      mesh: contact and ties read the contact facets of their surfaces, the nodes and the rigid
      bodies, which every process holds whole (a facet and the point mass of a rigid body are made
      by every process itself, see :meth:`DistributedElements.placeElementMadeByOwner`). Ties,
      penalty contact and the other forces-only penalty constraints are verified so; the
      constraints of the implicit solvers (Lagrange multipliers, indirect load control) still name
      a reason (:mod:`~edelweissfe.constraints.base.wholemodel`);
    * **generators that do more than describe the mesh** (see
      :attr:`~edelweissfe.generators.base.generatorbase.GeneratorBase.wholeModelReason`): code
      running on element objects while the mesh is described (``executePythonCode``, ``cubit``);
      contact facets and rigid bodies are not among them;
    * **generators run after the keywords** (``executeAfterManualGeneration=True``) **that describe
      elements of the mesh** (see
      :attr:`~edelweissfe.generators.base.generatorbase.GeneratorBase.describesElementsOfMesh`):
      they would describe elements after the mesh was partitioned, which no process would compute;
      one that makes only elements of its own (contact facets, a rigid body) may run late;
    * **expression field outputs over an element set** (``>>fromExpression, elSet=``): the expression
      reads the element objects of the whole set itself, which cannot be gathered.

    Parameters
    ----------
    inputfile
        The parsed input file.

    Returns
    -------
    list[str]
        The reasons; empty if the elements can be distributed over the processes.
    """

    reasons = []
    for definition in inputfile["modelModifier"]:
        reason = getModelModifierClass(definition["type"]).wholeModelReason
        if reason is not None:
            reasons.append("model modifier {:} ({:}) {:}".format(definition["name"], definition["type"], reason))
    for definition in inputfile["constraint"]:
        reason = getConstraintClass(definition["type"]).wholeModelReason
        if reason is not None:
            reasons.append("constraint {:} ({:}) {:}".format(definition["name"], definition["type"], reason))
    for definition in inputfile["modelGenerator"]:
        generatorClass = getGeneratorClass(definition["generator"])
        if generatorClass.wholeModelReason is not None:
            reasons.append(
                "generator {:} ({:}) {:}".format(
                    definition["name"], definition["generator"], generatorClass.wholeModelReason
                )
            )
        if definition.get("executeAfterManualGeneration", False) and generatorClass.describesElementsOfMesh:
            reasons.append(
                "generator {:} ({:}) runs after the mesh is partitioned (executeAfterManualGeneration)".format(
                    definition["name"], definition["generator"]
                )
            )
    for definition in inputfile["fieldOutput"]:
        for fieldOutput in definition["moduleoptions"].get("fromExpression", []):
            if fieldOutput["elSet"]:
                reasons.append(
                    "expression field output {:} reads the elements of element set {:}".format(
                        fieldOutput["name"], fieldOutput["elSet"]
                    )
                )
    return reasons


def elementDistributionOfThisJob(inputfile: dict, journal: Journal) -> ElementDistribution:
    """How the processes of this job hold the model; decided once per job, and reported once.

    A serial job creates every element. A job on several MPI processes distributes the elements over
    them, unless :func:`reasonsForTheWholeModel` names a reason to hold the whole model in every
    process.

    Parameters
    ----------
    inputfile
        The parsed input file.
    journal
        The journal, for the report.

    Returns
    -------
    ElementDistribution
        The distribution, to be set as the model's
        :attr:`~edelweissfe.models.femodel.FEModel.elementDistribution` before the model is built.
    """

    if numberOfProcesses() == 1:
        return ElementDistribution()

    reasons = reasonsForTheWholeModel(inputfile)
    if reasons:
        journal.message(
            "Whole model on every process because: {:}".format("; ".join(reasons)), "DomainDecomposition", 0
        )
        return ElementDistribution()

    journal.message(
        "Distributed model: each of the {:} processes creates only the elements it computes".format(
            numberOfProcesses()
        ),
        "DomainDecomposition",
        0,
    )
    return DistributedElements(worldCommunicator())


class DistributedElements(ElementDistribution):
    """Each process creates only the elements of its subdomain; see the module documentation.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    """

    createsEveryElement = False

    def __init__(self, communicator):
        self.communicator = communicator
        self.rank = communicator.Get_rank()

        #: The rank of every element of the mesh, by number: the process computing it.
        self.owners = None
        #: The numbers of the elements created in this process.
        self._createdHere = set()
        #: Changed by every :meth:`moveElementsTo`.
        self.ownershipVersion = 0
        #: How many elements this process received from another one over the run, by
        #: :meth:`moveElementsTo`; a diagnostic.
        self.nElementsReceived = 0
        #: The state of every element of the model, by number, on rank 0, between
        #: :meth:`gatherStatesForCheckpoint` and :meth:`forgetGatheredStates`.
        self._gatheredStates = None

    def decideWhichElementsAreCreatedHere(self, mesh: Mesh, domainSize: int):
        """Partition the mesh, and decide which elements this process creates. Collective.

        A process creates the elements it computes -- its part of the partition -- and the elements
        made by their owners (see :meth:`placeElementMadeByOwner`). The loads acting on an element
        are evaluated where the element is computed, and exchanged like its forces (see
        :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.loadsOnSubdomain`), so no
        process needs an element of another one for its loads.

        Parameters
        ----------
        mesh
            The mesh, as described by the input file and the generators.
        domainSize
            The spatial dimension.
        """

        self.owners = partitionElementsOfMesh(mesh, self.communicator.Get_size(), domainSize, self.communicator)

        self._createdHere = self._elementsCreatedFor(mesh, self.owners)

    def _elementsCreatedFor(self, mesh: Mesh, owners: dict) -> set:
        """The numbers of the elements this process creates under a partition: its own, and every
        element made by its owner (see :meth:`placeElementMadeByOwner`).

        Parameters
        ----------
        mesh
            The mesh.
        owners
            The rank of every element of the mesh, by number.

        Returns
        -------
        set
            The element numbers.
        """

        own = {number for number, owner in owners.items() if owner == self.rank}
        madeByOwners = {number for number, record in mesh.elements.items() if record.isMadeByOwner}
        return own | madeByOwners

    def moveElementsTo(self, model, owners: dict) -> tuple[int, int, int]:
        """Adopt a new partition: move every element whose process changes to its new process
        (migration). Collective; only where every process holds the converged state of the elements
        it computes, i.e. after an increment was accepted.

        An element is completely described by the mesh, the replicated definitions -- its sections,
        material and element properties -- and its state. So, in this order:

        1. each process sends the state of every element it computed and no longer computes to the
           element's new process;
        2. each process drops the objects of the elements it no longer needs, and creates those it
           now needs -- its new elements (:meth:`decideWhichElementsAreCreatedHere`) -- from the
           mesh, in mesh order;
        3. the element sets and surfaces are resolved to the elements now created here (and, before
           the new elements are created, to those kept), the new
           elements receive their sections and element properties, as at setup, and every element
           this process now computes, but did not compute before, receives the state its previous
           process sent.

        A migrated element is therefore the element its previous process held, bit for bit, as a
        resumed restart's element is. What else is derived from the elements held here -- the
        degree-of-freedom indices of the elements, the subdomain, the increment plan, the result
        views of field outputs -- must be derived again; :attr:`ownershipVersion` says that it
        changed.

        Parameters
        ----------
        model
            The model tree.
        owners
            The new rank of every element of the mesh, by number.

        Returns
        -------
        tuple[int, int, int]
            How many element objects this process created, dropped, and received the state of.
        """

        rank = self.rank
        previousOwners = self.owners

        outgoing = [{} for _ in range(self.communicator.Get_size())]
        for number, owner in owners.items():
            if previousOwners[number] == rank and owner != rank:
                outgoing[owner][number] = model.elements[number].getStateVars()
        incoming = self.communicator.alltoall(outgoing)
        del outgoing

        # Element objects change process; the mesh does not change.
        self.owners = owners
        created, dropped = self._createAndDropElementsCreatedFor(model)

        nReceived = 0
        for states in incoming:
            for number, state in states.items():
                model.elements[number].setStateVars(state)
                nReceived += 1

        self.ownershipVersion += 1
        self.nElementsReceived += nReceived
        return len(created), len(dropped), nReceived

    def _createAndDropElementsCreatedFor(self, model) -> tuple[dict, list]:
        """Create and drop element objects so that this process holds exactly those it creates under
        the current partition (:attr:`owners`): drop those it no longer needs, and create those it
        lacks from the mesh, in mesh order, with their sections and element properties, as at setup.

        The dropped elements leave their sets and surfaces before the new ones are created, so that
        they are released first.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        tuple[dict, list]
            The elements created, by number, and the numbers of the elements dropped.
        """

        mesh = model.mesh
        createdHere = self._elementsCreatedFor(mesh, self.owners)
        toDrop = [number for number in model.elements if number not in createdHere]
        toCreate = [number for number in mesh.elements if number in createdHere and number not in model.elements]

        with model.topology.changes():
            for number in toDrop:
                model.dropElementOfMesh(number)
            model.resolveSetsAndSurfacesOfMesh()
            created = {number: model.createElementOfMesh(number) for number in toCreate}
        model.putElementsInMeshOrder()
        model.resolveSetsAndSurfacesOfMesh()
        model.assignSectionsAndPropertiesToElements(created)

        self._createdHere = createdHere
        return created, toDrop

    def placeChildElement(self, childNumber: int, parentNumber: int):
        """The child of a refined element is computed by the process that computed its parent: that
        process creates it, and transfers the state of its own parent to it. See
        :meth:`~edelweissfe.models.elementdistribution.ElementDistribution.placeChildElement`.

        Parameters
        ----------
        childNumber
            The number of the new element, already described in the mesh.
        parentNumber
            The number of the element it replaces (in part), still described in the mesh.
        """

        owner = self.owners[parentNumber]
        self.owners[childNumber] = owner
        if owner == self.rank:
            self._createdHere.add(childNumber)

    def placeElementMadeByOwner(self, record: MeshElement):
        """An element its owner made itself -- a contact facet, the point mass of a rigid body -- is
        made in every process: it is surface-sized, and a constraint evaluated in any process may
        read it (a contact search reads the facets of a whole surface). It is computed, and reported,
        by one process, the one :func:`~.partitioning.processOfElementMadeByOwner` names. Made before
        the mesh is partitioned, the partition places it. See
        :meth:`~edelweissfe.models.elementdistribution.ElementDistribution.placeElementMadeByOwner`.

        Parameters
        ----------
        record
            The element as described in the mesh, with its host element.
        """

        self._createdHere.add(record.number)
        if self.owners is not None:
            self.owners[record.number] = processOfElementMadeByOwner(record, self.owners)

    def createAndDropElementsOfChangedMesh(self, model):
        """After a model modifier changed the mesh -- every process changes it identically -- forget
        the elements no longer in it, and create and drop elements so that this process holds exactly
        those it now needs (see :meth:`decideWhichElementsAreCreatedHere`): its own, which the
        modifier created already (:meth:`placeChildElement`). Local: the decision reads only the
        mesh, which every process holds whole.

        Parameters
        ----------
        model
            The model tree, its mesh changed.

        Raises
        ------
        TopologyError
            If the modifier described an element without placing it on a process.
        """

        mesh = model.mesh
        unplaced = [number for number in mesh.elements if number not in self.owners]
        if unplaced:
            raise TopologyError(
                "{:} element(s) (e.g. {:}) were described during the run without a process to compute them: a "
                "model modifier that describes elements must place them (ElementDistribution.placeChildElement)".format(
                    len(unplaced), unplaced[:5]
                )
            )
        self.owners = {number: self.owners[number] for number in mesh.elements}

        self._createAndDropElementsCreatedFor(model)

    def isCreatedHere(self, number: int) -> bool:
        """Whether this process creates the element with the given number.

        Parameters
        ----------
        number
            The element number.

        Returns
        -------
        bool
            Whether it is created here.

        Raises
        ------
        TopologyError
            Before the mesh was partitioned, and for an element described after it: no process owns
            it.
        """

        if self.owners is None:
            raise TopologyError(
                "element {:} is to be created before the mesh was partitioned over the processes".format(number)
            )
        if number not in self.owners:
            raise TopologyError(
                "element {:} was described after the mesh was partitioned over the processes, so no process "
                "computes it".format(number)
            )
        return number in self._createdHere

    def elementsReportedHere(self, elements) -> list:
        """Those of the given elements this process computes, and so reports.

        Parameters
        ----------
        elements
            Elements created here.

        Returns
        -------
        list
            The elements computed here, in the order given.
        """

        return [element for element in elements if self.owners[element.elNumber] == self.rank]

    def resultsOfWholeSet(self, elementSet, numbersReportedHere: list, results: np.ndarray | None) -> np.ndarray:
        """The results of every element of a set, in the order of the set in the mesh, gathered from
        the processes computing them -- to every process, so that a field output is the same in
        every process, as without decomposition. Collective.

        Parameters
        ----------
        elementSet
            The element set; resolved from the mesh.
        numbersReportedHere
            The numbers of the elements of the set computed here, in set order.
        results
            Their results, one row per element; None if there are none.

        Returns
        -------
        np.ndarray
            The results of every element of the set, one row per element, in set order.

        Raises
        ------
        TopologyError
            If an element of the set is computed by no process.
        """

        numbers = elementSet.elementNumbersOfWholeSet()
        pieces = self.communicator.allgather((list(numbersReportedHere), results))

        rowOf = {number: row for row, number in enumerate(numbers)}
        whole = None
        filled = np.zeros(len(numbers), dtype=bool)
        for pieceNumbers, pieceResults in pieces:
            if pieceResults is None:
                continue
            if whole is None:
                whole = np.empty((len(numbers),) + pieceResults.shape[1:], dtype=pieceResults.dtype)
            rows = [rowOf[number] for number in pieceNumbers]
            whole[rows] = pieceResults
            filled[rows] = True

        if not filled.all():
            raise TopologyError(
                "the results of element set {:} were gathered from the processes, but {:} of its elements "
                "were computed by none".format(elementSet.name, int((~filled).sum()))
            )
        return whole

    def gatherStatesForCheckpoint(self, elements: dict):
        """Gather the state of every element of the model to rank 0, each from the process computing
        it, so that rank 0 can write a restart checkpoint of the whole model. Collective.

        Parameters
        ----------
        elements
            The elements created here, by number.
        """

        computedHere = {
            number: element.getStateVars() for number, element in elements.items() if self.owners[number] == self.rank
        }
        gathered = self.communicator.gather(computedHere, root=0)
        if gathered is not None:
            self._gatheredStates = {}
            for states in gathered:
                self._gatheredStates.update(states)

    def forgetGatheredStates(self):
        """Release the states :meth:`gatherStatesForCheckpoint` gathered."""

        self._gatheredStates = None

    def statesOfElements(self, elements: dict):
        """The state of every element of the model, by number, as gathered to rank 0 by
        :meth:`gatherStatesForCheckpoint`.

        Parameters
        ----------
        elements
            The elements created here, by number; their states are among those gathered.

        Returns
        -------
        Iterator
            ``(number, state)`` pairs, one per element of the model.

        Raises
        ------
        RuntimeError
            If the states were not gathered for this checkpoint.
        """

        if self._gatheredStates is None:
            raise RuntimeError(
                "a restart checkpoint of a distributed model needs the element states gathered from every "
                "process first (DistributedElements.gatherStatesForCheckpoint)"
            )
        return iter(self._gatheredStates.items())
