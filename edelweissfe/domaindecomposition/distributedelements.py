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

from edelweissfe.config.generators import getGeneratorClass
from edelweissfe.config.stepactions import stepActionFactory
from edelweissfe.domaindecomposition.mpienvironment import (
    numberOfProcesses,
    worldCommunicator,
)
from edelweissfe.domaindecomposition.partitioning import partitionElementsOfMesh
from edelweissfe.journal.journal import Journal
from edelweissfe.models.elementdistribution import ElementDistribution
from edelweissfe.models.mesh import Mesh
from edelweissfe.utils.exceptions import TopologyError


def reasonsForTheWholeModel(inputfile: dict) -> list[str]:
    """Why a job must hold the whole model in every process: the rule of the fallback.

    A job needs every element in every process if something reads, or changes, the whole model
    while it runs, and has not yet been taught to gather what it reads from the processes computing
    it:

    * **model modifiers** -- adaptive refinement above all -- change the mesh during the run, and
      their topology logic and state transfer read the whole mesh;
    * **constraints** -- contact, ties and the like -- are evaluated whole by one process, search the
      whole surface, and may couple nodes of any subdomain;
    * **generators that do more than describe the mesh** (see
      :attr:`~edelweissfe.generators.base.generatorbase.GeneratorBase.wholeModelReason`): code
      running on element objects while the mesh is described (``executePythonCode``, ``cubit``),
      or contact facets and rigid bodies made by every process itself;
    * **generators run after the keywords** (``executeAfterManualGeneration=True``): they may describe
      elements after the mesh was partitioned, which no process would compute;
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
        reasons.append(
            "model modifier {:} ({:}) changes the mesh during the run".format(definition["name"], definition["type"])
        )
    for definition in inputfile["constraint"]:
        reasons.append(
            "constraint {:} ({:}) is evaluated whole by one process".format(definition["name"], definition["type"])
        )
    for definition in inputfile["modelGenerator"]:
        reason = getGeneratorClass(definition["generator"]).wholeModelReason
        if reason is not None:
            reasons.append("generator {:} ({:}) {:}".format(definition["name"], definition["generator"], reason))
        if definition.get("executeAfterManualGeneration", False):
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


def _stepActionDefinitions(inputfile: dict) -> list[tuple[type, dict]]:
    """The step action class and parsed definition of every step action of every step of the input
    file.

    Parameters
    ----------
    inputfile
        The parsed input file.

    Returns
    -------
    list[tuple[type, dict]]
        ``(stepActionClass, definition)`` pairs, in deck order.
    """

    return [
        (stepActionFactory(actionType), definition)
        for step in inputfile["step"]
        for actionType, definitions in step["moduleoptions"].items()
        for definition in definitions
    ]


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
    return DistributedElements(worldCommunicator(), _stepActionDefinitions(inputfile))


class DistributedElements(ElementDistribution):
    """Each process creates only the elements of its subdomain; see the module documentation.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    stepActionDefinitions
        The class and parsed definition of every step action of the job; those loading elements
        name them (:meth:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase.elementsLoadedByDefinition`).
    """

    createsEveryElement = False

    def __init__(self, communicator, stepActionDefinitions: list[tuple[type, dict]]):
        self.communicator = communicator
        self.rank = communicator.Get_rank()
        self._stepActionDefinitions = stepActionDefinitions

        #: The rank of every element of the mesh, by number: the process computing it.
        self.owners = None
        #: The numbers of the elements created in this process.
        self._createdHere = set()
        #: The state of every element of the model, by number, on rank 0, between
        #: :meth:`gatherStatesForCheckpoint` and :meth:`forgetGatheredStates`.
        self._gatheredStates = None

    def decideWhichElementsAreCreatedHere(self, mesh: Mesh, domainSize: int):
        """Partition the mesh, and decide which elements this process creates. Collective.

        A process creates the elements it computes -- its part of the partition -- and, in addition,
        every element carrying a load (a distributed load on its surface, a body load) that shares a
        node with one of its own elements. A load is not exchanged between processes: each process
        adds the loads at the degrees of freedom it integrates itself, in the order of the load's
        elements, which is what keeps the result bit-identical to a serial run (see
        :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.distributedLoadsOnSubdomain`).
        Such an element is created for its loads only: it is neither computed nor reported here.

        Parameters
        ----------
        mesh
            The mesh, as described by the input file and the generators.
        domainSize
            The spatial dimension.
        """

        self.owners = partitionElementsOfMesh(mesh, self.communicator.Get_size(), domainSize, self.communicator)

        own = {number for number, owner in self.owners.items() if owner == self.rank}
        nodesOfOwnElements = {label for number in own for label in mesh.elements[number].nodeLabels}

        loaded = set()
        for stepActionClass, definition in self._stepActionDefinitions:
            loaded.update(stepActionClass.elementsLoadedByDefinition(definition, mesh))

        loadedNeighbours = {
            number for number in loaded - own if not nodesOfOwnElements.isdisjoint(mesh.elements[number].nodeLabels)
        }
        self._createdHere = own | loadedNeighbours

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

    def resultsOfWholeSet(self, elementSet, reportedHere: list, results: np.ndarray | None) -> np.ndarray:
        """The results of every element of a set, in the order of the set in the mesh, gathered from
        the processes computing them -- to every process, so that a field output is the same in
        every process, as without decomposition. Collective.

        Parameters
        ----------
        elementSet
            The element set; resolved from the mesh.
        reportedHere
            The elements of the set computed here, in set order.
        results
            Their results, one row per element; None if there are none.

        Returns
        -------
        np.ndarray
            The results of every element of the set, one row per element, in set order.

        Raises
        ------
        TopologyError
            If the set is not described in the mesh, or an element of the set is computed by no
            process.
        """

        if elementSet.mesh is None or elementSet.name not in elementSet.mesh.elementSets:
            raise TopologyError(
                "element set {:} is not described in the mesh, so its results cannot be gathered from the "
                "processes".format(elementSet.name)
            )
        numbers = elementSet.mesh.elementSets[elementSet.name]
        pieces = self.communicator.allgather(([element.elNumber for element in reportedHere], results))

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
