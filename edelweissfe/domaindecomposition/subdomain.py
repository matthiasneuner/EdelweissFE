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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
"""The subdomain one MPI process computes: the model partition of a domain-decomposed solver.

A :class:`Subdomain` is the :class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` of one
process of several. It computes the elements METIS assigns to it and the constraints dealt to it,
and integrates the degrees of freedom those touch (:mod:`.partitioning`). Its element forces are
completed at the interface with the neighbouring subdomains (:mod:`.subdomaininterface`), the
constraint forces are shared with every process, sums are formed over all processes in rank order,
and the vectors and states of the whole model are made current from the processes computing them
(:mod:`.statesynchronization`). Every sum that decides the solution is formed in the order it is
formed without decomposition, so a run is bit-identical to one in a single process.
"""

import hashlib
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.partitioning import (
    assignConstraints,
    partitionElements,
)
from edelweissfe.domaindecomposition.statesynchronization import (
    ModelStateSynchronization,
)
from edelweissfe.domaindecomposition.subdomaininterface import (
    ConstraintForceExchange,
    InterfaceForceAssembly,
    SubdomainInterface,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.solvers.base.modelpartition import ModelPartition
from edelweissfe.solvers.base.parallelelementcomputation import ElementPlan
from edelweissfe.utils.exceptions import ConditionalStop, CutbackRequest, StepFailed

#: How :meth:`Subdomain.agreedOnByAllParts` ranks what went wrong: the most severe outcome of any
#: process is the one every process raises.
_NO_FAILURE, _CUTBACK, _CONDITIONAL_STOP, _FAILURE = 0, 1, 2, 3


class Subdomain(ModelPartition):
    """The subdomain this MPI process computes; see the module documentation.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model.
    journal
        The journal, for the subdomain report.
    identification
        The solver's identification, for the journal.
    loadBalanceTolerance
        How far the slowest process may fall behind the mean, as a fraction, before
        :meth:`rebalance` repartitions; 0 disables it.
    """

    def __init__(self, communicator, journal, identification: str, loadBalanceTolerance: float):
        self.communicator = communicator
        self.rank = communicator.Get_rank()
        self.nProcesses = communicator.Get_size()
        self.journal = journal
        self.identification = identification
        self.loadBalanceTolerance = loadBalanceTolerance

        #: The rank of every element and of every constraint, and the element keys the partition was
        #: made for.
        self._elementOwners = None
        self._partitionedElementKeys = None
        self._constraintOwners = None

        self._ownedElements = {}
        self._ownedConstraints = {}

        self._interface = None
        self._inSubdomain = None
        self._stateSynchronization = None
        self._constraintForceExchange = None
        #: The position of every element in the model, by number: the order interface forces are
        #: summed in.
        self._elementPositions = {}
        #: The elements whose degrees of freedom reach into this subdomain; see loadedElements.
        self._elementsTouchingSubdomain = set()

        #: What the last definition was made for; a rebalance redefines from it.
        self._model = None
        self._dofManager = None
        self._mpcTransformation = None

    # --- ModelPartition: defining the subdomain --------------------------------------------------

    def define(
        self,
        model: FEModel,
        dofManager: DofManager,
        mpcTransformation: MultiPointConstraintTransformation | None,
        topologyChanged: bool,
    ):
        """Partition the model if its elements changed, and determine this process' subdomain
        degrees of freedom and interface; see :meth:`ModelPartition.define`. Collective.

        A rebuild with the same elements -- a new step, a changed material property -- keeps the
        partition, and must: the states of the elements another process computed are current only
        after a synchronization, and only a topology change is guaranteed to follow one.
        """

        self._model = model
        self._dofManager = dofManager
        self._mpcTransformation = mpcTransformation

        with performancetiming.timeit("subdomain definition"):
            if topologyChanged:
                self._checkReplicatedLayout(model, dofManager)

                if self._elementOwners is None or model.elements.keys() != self._partitionedElementKeys:
                    self._partition(model, measuredCosts=None)

                self._constraintOwners = self._constraintOwnersOf(model)

                self._adoptOwnership(model)

            self._defineInterface(model, ownershipChanged=topologyChanged)

        self._reportSubdomains(topologyChanged)

    def _partition(self, model: FEModel, measuredCosts: dict | None):
        """Partition the elements, by estimated or by measured cost. Collective.

        Parameters
        ----------
        model
            The model tree.
        measuredCosts
            The measured cost per increment of elements, by number, on rank 0; None to estimate.
        """

        with performancetiming.timeit("partition"):
            self._elementOwners = partitionElements(
                model.elements, self.nProcesses, model.domainSize, self.communicator, measuredCosts
            )
        self._partitionedElementKeys = set(model.elements.keys())

    def _constraintOwnersOf(self, model: FEModel) -> dict:
        """The rank of every constraint of the model: the current assignment while the model has the
        constraints it was made for, otherwise the assignment the next definition adopts.

        The second case is what lets a connectivity update right after a topology change, before the
        subdomain is defined afresh, search on the process that is going to evaluate a constraint.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        dict
            The rank of every constraint, by name.
        """

        if self._constraintOwners is None or model.constraints.keys() != self._constraintOwners.keys():
            return assignConstraints(model.constraints, self.nProcesses)
        return self._constraintOwners

    def _adoptOwnership(self, model: FEModel):
        """Collect the elements and constraints this process owns under the current partition.

        Parameters
        ----------
        model
            The model tree.
        """

        self._elementPositions = {number: position for position, number in enumerate(model.elements)}

        self._ownedElements = {
            number: element for number, element in model.elements.items() if self._elementOwners[number] == self.rank
        }
        self._ownedConstraints = {
            name: constraint
            for name, constraint in model.constraints.items()
            if self._constraintOwners[name] == self.rank
        }

    def _defineInterface(self, model: FEModel, ownershipChanged: bool):
        """Determine the subdomain degrees of freedom and the interface from the owned entities; with
        new ownership, also the layout of the state synchronization. Collective.

        Parameters
        ----------
        model
            The model tree.
        ownershipChanged
            Whether the elements this process owns may have changed since the last call.
        """

        dofManager = self._dofManager
        touched = [dofManager.idcsOfElementsInDofVector[element] for element in self._ownedElements.values()]
        touched += [
            dofManager.idcsOfConstraintsInDofVector[constraint] for constraint in self._ownedConstraints.values()
        ]
        touched = np.concatenate(touched) if touched else np.empty(0, dtype=np.int64)
        touched = self._closeOverMultiPointConstraints(touched, dofManager.nDof)

        self._interface = SubdomainInterface(self.communicator, touched, dofManager.nDof)

        self._inSubdomain = np.zeros(dofManager.nDof, dtype=bool)
        self._inSubdomain[self._interface.subdomainDofs] = True
        self._elementsTouchingSubdomain = {
            element
            for element in model.elements.values()
            if self._inSubdomain[dofManager.idcsOfElementsInDofVector[element]].any()
        }

        if ownershipChanged:
            self._stateSynchronization = ModelStateSynchronization(
                self.communicator, model.elements, self._elementOwners, model.constraints, self._constraintOwners
            )

        self._constraintForceExchange = ConstraintForceExchange(
            self.communicator, model.constraints, self._ownedConstraints, dofManager.idcsOfConstraintsInDofVector
        )

    def _closeOverMultiPointConstraints(self, touched: np.ndarray, nDof: int) -> np.ndarray:
        """Add to the touched degrees of freedom every one linked to them by a multi-point
        constraint, transitively.

        A slave's force is folded onto its masters and its velocity interpolated from theirs, so a
        process integrating any degree of freedom of such a group must integrate all of them, and
        hold the complete force at each; see
        :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.slaveMasterDofPairs`.

        Parameters
        ----------
        touched
            The degrees of freedom this process' elements and constraints touch.
        nDof
            The size of the equation system.

        Returns
        -------
        np.ndarray
            The touched degrees of freedom and their multi-point-constraint closure.
        """

        if self._mpcTransformation is None:
            return touched

        slaves, masters = self._mpcTransformation.slaveMasterDofPairs()
        if not slaves.size:
            return touched

        links = coo_matrix((np.ones(slaves.shape[0]), (slaves, masters)), shape=(nDof, nDof))
        _, group = connected_components(links, directed=False)

        linked = np.zeros(nDof, dtype=bool)
        linked[slaves] = True
        linked[masters] = True

        groupsTouched = np.unique(group[touched[linked[touched]]])
        return np.concatenate([touched, np.flatnonzero(linked & np.isin(group, groupsTouched))])

    def _checkReplicatedLayout(self, model: FEModel, dofManager: DofManager):
        """Refuse to continue unless every process built the same model and degree-of-freedom layout.

        The interface exchange addresses degrees of freedom by index, which is only meaningful if
        every process numbered them identically; after a refinement that rests on every process
        having refined identically. A fingerprint of the element numbers, the node order of every
        field and the size of the system is compared across all processes. Collective.

        Parameters
        ----------
        model
            The model tree.
        dofManager
            The degree-of-freedom layout.

        Raises
        ------
        RuntimeError
            On every process, if any two fingerprints differ.
        """

        digest = hashlib.sha1()
        digest.update(np.asarray(list(model.elements.keys()), dtype=np.int64).tobytes())
        for name, field in model.nodeFields.items():
            digest.update(name.encode())
            digest.update(np.asarray([node.label for node in field.nodes], dtype=np.int64).tobytes())
        digest.update(np.int64(dofManager.nDof).tobytes())

        fingerprints = self.communicator.allgather(digest.hexdigest())
        if len(set(fingerprints)) > 1:
            raise RuntimeError(
                "The processes hold different models: their layout fingerprints are {:}. Every process "
                "must build, and refine, the model identically for the subdomain interface to be "
                "meaningful.".format(", ".join(sorted(set(fingerprints))))
            )

    def _reportSubdomains(self, topologyChanged: bool):
        """Report the size of every subdomain and of its interface. Collective.

        Parameters
        ----------
        topologyChanged
            Whether the partition may have changed; reported more prominently then.
        """

        statistics = self.communicator.gather(
            (
                sum(element.hasKernels for element in self._ownedElements.values()),
                self._interface.subdomainDofs.shape[0],
                self._interface.nInterfaceDofs,
                len(self._interface.neighbours),
            ),
            root=0,
        )
        if statistics is None:
            return

        elementCounts = np.array([entry[0] for entry in statistics])
        imbalance = elementCounts.max() / max(elementCounts.mean(), 1e-300)
        self.journal.message(
            "Subdomains: {:} element(s) per process (max/mean {:.3f}); subdomain DOFs {:}; interface DOFs {:}; "
            "neighbours {:}".format(
                "/".join(str(count) for count in elementCounts),
                imbalance,
                "/".join(str(entry[1]) for entry in statistics),
                "/".join(str(entry[2]) for entry in statistics),
                "/".join(str(entry[3]) for entry in statistics),
            ),
            self.identification,
            0 if topologyChanged else 2,
        )

        if topologyChanged and self._stateSynchronization is not None:
            withoutState = len(self._stateSynchronization.elementsWithoutState)
            if withoutState:
                self.journal.message(
                    "{:} element(s) expose no state (getStateVars); their state is not synchronized "
                    "between processes, so output and refinement read it only where it was computed.".format(
                        withoutState
                    ),
                    self.identification,
                    0,
                )

    # --- ModelPartition: what is computed here ---------------------------------------------------

    @property
    def elements(self) -> dict:
        """The elements this process owns."""

        return self._ownedElements

    @property
    def constraints(self) -> dict:
        """The constraints this process owns."""

        return self._ownedConstraints

    @property
    def dofs(self) -> np.ndarray:
        """The subdomain degrees of freedom: those the owned elements and constraints touch, closed
        over the multi-point constraints. The others hold whatever the last synchronization gave
        them until the next one overwrites them."""

        return self._interface.subdomainDofs

    def integratedValuesOf(self, vector: np.ndarray) -> np.ndarray:
        """The entries at the subdomain degrees of freedom, gathered into a copy."""

        return vector[self._interface.subdomainDofs]

    def storeIntegratedValues(self, vector: np.ndarray, values: np.ndarray):
        """Scattered back to the subdomain degrees of freedom."""

        vector[self._interface.subdomainDofs] = values

    def integratedEntries(self, indices: np.ndarray) -> np.ndarray:
        """Those among the subdomain degrees of freedom."""

        return np.flatnonzero(self._inSubdomain[indices])

    def ownedEntries(self, indices: np.ndarray) -> np.ndarray:
        """Those this process owns: the lowest-ranked process integrating a degree of freedom owns it."""

        return self._interface.ownedDofMask[indices]

    def loadedElements(self, elements) -> list:
        """Those reaching into the subdomain, owned or not."""

        return [element for element in elements if element in self._elementsTouchingSubdomain]

    def constraintsSearchedHere(self, model: FEModel, constraints: dict) -> dict:
        """Those this process owns, or -- right after a topology change -- is going to own.

        A search -- a contact search above all -- is run by the constraint's owner alone, at the
        periodic contact update and at a topology check alike: only the owner's evaluation reads its
        outcome. Another process' copy keeps the footprint of the last state synchronization, with
        the mesh refreshes since applied to it as to the owner's; such a footprint names nodes of
        the model, so a degree-of-freedom layout built with it is complete, and the layout
        fingerprint compared after every build makes sure of that.
        """

        owners = self._constraintOwnersOf(model)
        return {name: constraint for name, constraint in constraints.items() if owners[name] == self.rank}

    # --- ModelPartition: completing the results of the whole model -------------------------------

    def interfaceAssemblyFor(self, plan: ElementPlan) -> InterfaceForceAssembly:
        """The exchange of the plan's contributions at the interface, summed in model order."""

        entryElementPositions = np.repeat(
            [self._elementPositions[number] for number in plan.elements],
            [element.nDof for element in plan.elements.values()],
        )
        return InterfaceForceAssembly(self._interface, plan.entryDofs, entryElementPositions)

    def shareFromOwners(self, vector: DofVector):
        """Every entry from its owner."""

        self._interface.gatherFromOwners(vector)

    def addConstraintForces(self, forces: dict, P: DofVector):
        """Share the forces of the owned constraints with every process, and add those of all.

        The degrees of freedom are the owner's: another process' copy of a contact constraint
        couples the nodes of its last synchronization.
        """

        with performancetiming.timeit("constraint force exchange"):
            self._constraintForceExchange.addAllConstraintForces(
                {name: constraintForce.forces for name, constraintForce in forces.items()}, P
            )

    @performancetiming.timeit("subdomain synchronization")
    def synchronizeStates(self, includeElements: bool):
        """From the processes that computed them."""

        if includeElements:
            self._stateSynchronization.synchronizeElementStates()
        self._stateSynchronization.synchronizeConstraintStates()

    def sumAcrossParts(self, values: list[float]) -> list[float]:
        """Summed in ascending rank order, so that they are the same bits on every process."""

        gathered = self.communicator.allgather(list(values))
        totals = list(gathered[0])
        for contributions in gathered[1:]:
            totals = [total + value for total, value in zip(totals, contributions)]
        return totals

    def minAcrossParts(self, value: float) -> float:
        """By an MPI reduction."""

        return self.communicator.allreduce(value, op=MPI.MIN)

    def anyPart(self, flag: bool) -> bool:
        """By an MPI reduction."""

        return bool(self.communicator.allreduce(bool(flag), op=MPI.LOR))

    def requireSameOnAllParts(self, value, description: str):
        """Compared over all processes."""

        values = self.communicator.allgather(value)
        if any(other != values[0] for other in values[1:]):
            raise RuntimeError(
                "The processes disagree on {:} ({:}); they must compute the replicated parts of the model "
                "identically.".format(description, values)
            )

    @contextmanager
    def agreedOnByAllParts(self, operation: str):
        """Every process reports what went wrong in it, and every process raises the most severe."""

        failure = None
        try:
            yield
        except Exception as exception:
            failure = exception

        if failure is None:
            outcome = _NO_FAILURE
        elif isinstance(failure, CutbackRequest):
            outcome = _CUTBACK
        elif isinstance(failure, ConditionalStop):
            outcome = _CONDITIONAL_STOP
        else:
            outcome = _FAILURE

        status = np.array([outcome], dtype=np.int32)
        # Where the processes wait for the slowest one: timed on its own, it is the load imbalance.
        with performancetiming.timeit("subdomain wait"):
            self.communicator.Allreduce(MPI.IN_PLACE, status, op=MPI.MAX)

        if status[0] == _NO_FAILURE:
            return
        if status[0] == _CONDITIONAL_STOP:
            raise ConditionalStop() from failure

        reports = self.communicator.allgather(
            None if failure is None else "{:}: {:}".format(type(failure).__name__, failure)
        )
        message = "; ".join(
            "process {:}: {:}".format(rank, report) for rank, report in enumerate(reports) if report is not None
        )
        if status[0] == _CUTBACK:
            raise CutbackRequest(message, 0.5) from failure
        raise StepFailed("{:} failed in {:}".format(operation, message)) from failure

    def shareOfModelTotal(self, value: float) -> float:
        """Rank 0 carries the total."""

        return value if self.rank == 0 else 0.0

    # --- ModelPartition: load balancing ----------------------------------------------------------

    def measuresElementCosts(self) -> bool:
        """Whenever load balancing is enabled."""

        return bool(self.loadBalanceTolerance)

    def rebalance(self, plan: ElementPlan, costs: np.ndarray | None, nIncrements: int) -> bool:
        """Repartition with the measured element costs if the slowest process has fallen more than
        ``loadBalanceTolerance`` behind the mean; see :meth:`ModelPartition.rebalance`."""

        tolerance = self.loadBalanceTolerance
        if not tolerance or self.nProcesses == 1 or costs is None or not nIncrements:
            return False

        busyTimes = np.array(self.communicator.allgather(float(costs.sum())))
        imbalance = busyTimes.max() / max(busyTimes.mean(), 1e-300)
        if imbalance <= 1.0 + tolerance:
            return False

        measured = dict(zip(plan.elements.keys(), (costs / nIncrements).tolist()))
        gathered = self.communicator.gather(measured, root=0)
        measuredCosts = None
        if gathered is not None:
            measuredCosts = {}
            for elementCosts in gathered:
                measuredCosts.update(elementCosts)

        self.journal.message(
            "Load imbalance {:.3f} (slowest process / mean kernel time) exceeds 1 + {:}: repartitioning "
            "with the measured element costs".format(imbalance, tolerance),
            self.identification,
            1,
        )

        model = self._model
        self._partition(model, measuredCosts)
        self._adoptOwnership(model)
        with performancetiming.timeit("subdomain definition"):
            self._defineInterface(model, ownershipChanged=True)
        self._reportSubdomains(topologyChanged=True)
        return True
