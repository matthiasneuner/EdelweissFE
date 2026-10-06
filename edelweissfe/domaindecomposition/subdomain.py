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
"""The subdomain one MPI process computes, and what it exchanges with the others.

A :class:`Subdomain` decides which part of the model its process computes -- the elements METIS
assigns to it, the constraints dealt to it, and the degrees of freedom those touch
(:mod:`.partitioning`) -- and describes it as a
:class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` (:attr:`Subdomain.partition`). It
also carries out every exchange between the processes the domain-decomposed solver needs: the
element forces are completed at the interface with the neighbouring subdomains
(:mod:`.subdomaininterface`), the constraint forces are shared with every process, sums are formed
over all processes in rank order, failures are agreed on, and the vectors and states of the whole
model are made current from the processes computing them (:mod:`.statesynchronization`). Every sum
that decides the solution is formed in the order it is formed without decomposition, so a run is
bit-identical to one in a single process.
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
    partitionElementsOfMesh,
)
from edelweissfe.domaindecomposition.statesynchronization import (
    ModelStateSynchronization,
    elementsWithoutState,
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
from edelweissfe.stepactions.base.bodyloadbase import BodyLoadBase
from edelweissfe.stepactions.base.distributedloadbase import DistributedLoadBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    CutbackRequest,
    StepFailed,
    TopologyError,
)

#: How :meth:`Subdomain.agreedOnByAllParts` ranks what went wrong: the most severe outcome of any
#: process is the one every process raises.
_NO_FAILURE, _CUTBACK, _CONDITIONAL_STOP, _FAILURE = 0, 1, 2, 3


class DistributedLoadOnSubdomain:
    """A distributed load, restricted to the faces of the elements reaching into a subdomain.

    What the shared load assembly
    (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.computeDistributedLoads`)
    reads of a :class:`~edelweissfe.stepactions.base.distributedloadbase.DistributedLoadBase`, with
    a smaller surface.

    Parameters
    ----------
    load
        The distributed load.
    surface
        The faces it acts on in the subdomain: element lists by face, in the load's order.
    """

    def __init__(self, load: DistributedLoadBase, surface: dict):
        self.load = load
        self.surface = surface

    @property
    def loadType(self) -> str:
        """The load's type."""

        return self.load.loadType

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The load's current magnitude.

        Parameters
        ----------
        timeStep
            The time step.

        Returns
        -------
        np.ndarray
            The magnitude.
        """

        return self.load.getCurrentLoad(timeStep)


class BodyLoadOnSubdomain:
    """A body load, restricted to the elements reaching into a subdomain.

    What the shared load assembly
    (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.computeBodyForces`)
    reads of a :class:`~edelweissfe.stepactions.base.bodyloadbase.BodyLoadBase`, with a smaller
    element set.

    Parameters
    ----------
    load
        The body load.
    elementSet
        The elements it acts on in the subdomain, in the load's order.
    """

    def __init__(self, load: BodyLoadBase, elementSet: list):
        self.load = load
        self.elementSet = elementSet

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The load's current magnitude.

        Parameters
        ----------
        timeStep
            The time step.

        Returns
        -------
        np.ndarray
            The magnitude.
        """

        return self.load.getCurrentLoad(timeStep)


class Subdomain:
    """The subdomain this MPI process computes, and its exchanges with the others; see the module
    documentation.

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
        #: The elements whose degrees of freedom reach into this subdomain, and the loads acting on
        #: them, by load; see distributedLoadsOnSubdomain and bodyLoadsOnSubdomain.
        self._elementsTouchingSubdomain = set()
        self._loadsOnSubdomain = {}

        #: The elements, constraints and degrees of freedom this process computes, as of the last
        #: definition.
        self.partition = None

        self._reportedFixedPartition = False

        #: What the last definition was made for; a rebalance redefines from it.
        self._model = None
        self._dofManager = None
        self._mpcTransformation = None

    # --- Defining the subdomain -------------------------------------------------------------------

    def define(
        self,
        model: FEModel,
        dofManager: DofManager,
        mpcTransformation: MultiPointConstraintTransformation | None,
        topologyChanged: bool,
    ):
        """Partition the model if its elements changed, and determine this process' subdomain
        degrees of freedom and interface: the :attr:`partition`. Collective.

        Called whenever the equation system has been built (``topologyChanged``) or its constraints
        re-located (a contact search moved them). A rebuild with the same elements -- a new step, a
        changed material property -- keeps the partition, and must: the states of the elements
        another process computed are current only after a synchronization, and only a topology
        change is guaranteed to follow one.

        Parameters
        ----------
        model
            The model tree.
        dofManager
            The degree-of-freedom layout of the current equation system.
        mpcTransformation
            The multi-point-constraint transformation of the current equation system, or None.
        topologyChanged
            True after the equation system was built afresh; False when only the constraints were
            re-located.
        """

        self._model = model
        self._dofManager = dofManager
        self._mpcTransformation = mpcTransformation

        with performancetiming.timeit("subdomain definition"):
            if topologyChanged:
                self._checkReplicatedLayout(model, dofManager)

                if self._elementOwners is None or model.mesh.elements.keys() != self._partitionedElementKeys:
                    self._partition(model, measuredCosts=None)

                self._constraintOwners = self._constraintOwnersOf(model)

                self._adoptOwnership(model)

            self._defineInterface(model, ownershipChanged=topologyChanged)

        self._reportSubdomains(topologyChanged)
        if self._elementOwners is not None and not self._reportedFixedPartition:
            self._reportFixedPartition(model)

    def _reportFixedPartition(self, model: FEModel):
        """Report once that a distributed model keeps its first partition although load balancing is
        enabled; see :meth:`measuresElementCosts`.

        Parameters
        ----------
        model
            The model tree.
        """

        if self.loadBalanceTolerance and not model.elementDistribution.createsEveryElement:
            self.journal.message(
                "Load balancing is off: the elements were created in their processes, and moving them between "
                "processes (migration) is not implemented yet",
                self.identification,
                0,
            )
        self._reportedFixedPartition = True

    def _partition(self, model: FEModel, measuredCosts: dict | None):
        """Partition the elements of the mesh, by estimated or by measured cost. Collective.

        A model whose processes each created only their own elements was partitioned before its
        elements were created (see
        :meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.decideWhichElementsAreCreatedHere`),
        on the same mesh and by the same estimate: that partition is adopted, since an element can
        only be computed where it exists.

        Parameters
        ----------
        model
            The model tree.
        measuredCosts
            The measured cost per increment of elements, by number, on rank 0; None to estimate.

        Raises
        ------
        TopologyError
            If the mesh of a distributed model changed since it was partitioned.
        """

        distribution = model.elementDistribution
        if distribution.createsEveryElement:
            with performancetiming.timeit("partition"):
                self._elementOwners = partitionElementsOfMesh(
                    model.mesh, self.nProcesses, model.domainSize, self.communicator, measuredCosts
                )
        else:
            if measuredCosts is not None or model.mesh.elements.keys() != distribution.owners.keys():
                raise TopologyError(
                    "the elements of this model were partitioned over the processes before they were created, "
                    "and cannot be repartitioned"
                )
            self._elementOwners = distribution.owners
        self._partitionedElementKeys = set(model.mesh.elements.keys())

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

        # The order of the mesh, which is the order of the elements in a model that created all of
        # them, and so the order a serial run sums their contributions in.
        self._elementPositions = {number: position for position, number in enumerate(model.mesh.elements)}

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
            # The states of the elements other processes compute are kept current only where every
            # element was created; a distributed model gathers what it reads instead.
            self._stateSynchronization = ModelStateSynchronization(
                self.communicator,
                model.elements if model.elementDistribution.createsEveryElement else {},
                self._elementOwners,
                model.constraints,
                self._constraintOwners,
            )
            self._refuseElementsWithoutState(model)

        self._constraintForceExchange = ConstraintForceExchange(
            self.communicator, model.constraints, self._ownedConstraints, dofManager.idcsOfConstraintsInDofVector
        )
        self._loadsOnSubdomain = {}

        self.partition = ModelPartition(
            self._ownedElements, self._ownedConstraints, self._interface.subdomainDofs, self._interface.ownedDofMask
        )

    def _refuseElementsWithoutState(self, model: FEModel):
        """Refuse a model with elements that expose no state, on more than one process.

        Such an element's state cannot be sent to another process: the output and a checkpoint
        written by rank 0 would read the state its copy was built with, and a repartition would
        continue it from there. Each process checks the elements it created.

        Parameters
        ----------
        model
            The model tree.

        Raises
        ------
        NotImplementedError
            If an element does not implement ``getStateVars``.
        """

        withoutState = elementsWithoutState(model.elements)
        if withoutState and self.nProcesses > 1:
            raise NotImplementedError(
                "{:} element(s) of the model, e.g. element {:} ({:}), expose no state (getStateVars), "
                "so their state cannot be exchanged between processes; this model cannot be "
                "domain-decomposed. Run it with a serial solver.".format(
                    len(withoutState), withoutState[0], type(model.elements[withoutState[0]]).__name__
                )
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
        """Refuse to continue unless every process built the same mesh and degree-of-freedom layout.

        The interface exchange addresses degrees of freedom by index, which is only meaningful if
        every process numbered them identically; after a refinement that rests on every process
        having refined identically. A fingerprint of the mesh -- the element numbers and the node
        labels of every element -- the degree of freedom of every node of every field, the node
        coordinates and the size of the system is compared across all processes. It is made from
        the mesh and the nodes, which every process holds whole, and not from the element objects,
        of which a process may hold only its own. Collective.

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
        meshElements = model.mesh.elements
        digest.update(np.asarray(list(meshElements.keys()), dtype=np.int64).tobytes())
        if meshElements:
            digest.update(
                np.concatenate([np.asarray(record.nodeLabels, dtype=np.int64) for record in meshElements.values()])
                .astype(np.int64)
                .tobytes()
            )
        dofsOfFieldVariables = dofManager.idcsOfFieldVariablesInDofVector
        for name, field in model.nodeFields.items():
            digest.update(name.encode())
            digest.update(np.asarray([node.label for node in field.nodes], dtype=np.int64).tobytes())
            digest.update(
                np.concatenate(
                    [np.atleast_1d(dofsOfFieldVariables[node.fields[name]]) for node in field.nodes]
                    or [np.empty(0, dtype=np.int64)]
                )
                .astype(np.int64)
                .tobytes()
            )
        if model.nodes:
            digest.update(
                np.concatenate([np.asarray(node.coordinates, dtype=float) for node in model.nodes.values()]).tobytes()
            )
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
                len(self._model.elements),
            ),
            root=0,
        )
        if statistics is None:
            return

        elementCounts = np.array([entry[0] for entry in statistics])
        imbalance = elementCounts.max() / max(elementCounts.mean(), 1e-300)
        self.journal.message(
            "Subdomains: {:} element(s) per process (max/mean {:.3f}); subdomain DOFs {:}; interface DOFs {:}; "
            "neighbours {:}; elements created {:} of {:}".format(
                "/".join(str(count) for count in elementCounts),
                imbalance,
                "/".join(str(entry[1]) for entry in statistics),
                "/".join(str(entry[2]) for entry in statistics),
                "/".join(str(entry[3]) for entry in statistics),
                "/".join(str(entry[4]) for entry in statistics),
                len(self._model.mesh.elements),
            ),
            self.identification,
            0 if topologyChanged else 2,
        )

    # --- What is computed here ----------------------------------------------------------------

    def distributedLoadsOnSubdomain(self, distributedLoads) -> list[DistributedLoadOnSubdomain]:
        """The given distributed loads, each restricted to the elements reaching into the subdomain,
        owned or not: each process adds the loads at its subdomain degrees of freedom itself, since
        loads are not exchanged.

        Parameters
        ----------
        distributedLoads
            The distributed loads of the step.

        Returns
        -------
        list[DistributedLoadOnSubdomain]
            The restricted loads, in the order given.
        """

        restricted = []
        for load in distributedLoads:
            onSubdomain = self._loadsOnSubdomain.get(load)
            if onSubdomain is None:
                self._requireLoadedElementsCreatedHere(
                    load.surface.name,
                    [
                        number
                        for numbers in self._model.mesh.elementNumbersOfSurface(load.surface.name).values()
                        for number in numbers
                    ],
                )
                onSubdomain = DistributedLoadOnSubdomain(
                    load,
                    {
                        faceID: [element for element in elementSet if element in self._elementsTouchingSubdomain]
                        for faceID, elementSet in load.surface.items()
                    },
                )
                self._loadsOnSubdomain[load] = onSubdomain
            restricted.append(onSubdomain)
        return restricted

    def bodyLoadsOnSubdomain(self, bodyLoads) -> list[BodyLoadOnSubdomain]:
        """The given body loads, each restricted to the elements reaching into the subdomain; see
        :meth:`distributedLoadsOnSubdomain`.

        Parameters
        ----------
        bodyLoads
            The body loads of the step.

        Returns
        -------
        list[BodyLoadOnSubdomain]
            The restricted loads, in the order given.
        """

        restricted = []
        for load in bodyLoads:
            onSubdomain = self._loadsOnSubdomain.get(load)
            if onSubdomain is None:
                self._requireLoadedElementsCreatedHere(
                    load.elementSet.name, self._model.mesh.elementSets[load.elementSet.name]
                )
                onSubdomain = BodyLoadOnSubdomain(
                    load, [element for element in load.elementSet if element in self._elementsTouchingSubdomain]
                )
                self._loadsOnSubdomain[load] = onSubdomain
            restricted.append(onSubdomain)
        return restricted

    def _requireLoadedElementsCreatedHere(self, loadedSetName: str, numbers: list):
        """Refuse a load whose elements reaching into the subdomain were not all created here.

        Each process adds the loads at its subdomain degrees of freedom itself, from the loaded
        elements reaching into the subdomain. A distributed model creates those that share a node
        with an element of the subdomain (see
        :meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.decideWhichElementsAreCreatedHere`);
        a subdomain reaching further -- through a constraint or a multi-point constraint -- could miss
        a load silently, and is refused instead.

        Parameters
        ----------
        loadedSetName
            The name of the loaded surface or element set, for the message.
        numbers
            The numbers of the loaded elements, as described in the mesh.

        Raises
        ------
        TopologyError
            If a loaded element reaches into the subdomain without having been created here.
        """

        model = self._model
        if model.elementDistribution.createsEveryElement:
            return

        dofsOf = self._dofManager.idcsOfFieldVariablesInDofVector
        nodes = model.nodes
        for number in numbers:
            if number in model.elements:
                continue
            for label in model.mesh.elements[number].nodeLabels:
                if any(
                    self._inSubdomain[dofsOf[fieldVariable]].any() for fieldVariable in nodes[label].fields.values()
                ):
                    raise TopologyError(
                        "element {:} of {:} carries a load and reaches into the subdomain of process {:}, but was "
                        "not created there".format(number, loadedSetName, self.rank)
                    )

    def constraintsSearchedHere(self, model: FEModel, constraints: dict) -> dict:
        """Those of the given constraints whose connectivity search is run here: those this process
        owns, or -- right after a topology change -- is going to own.

        A search -- a contact search above all -- is run by the constraint's owner alone, at the
        periodic contact update and at a topology check alike: only the owner's evaluation reads its
        outcome. Another process' copy keeps the footprint of the last state synchronization, with
        the mesh refreshes since applied to it as to the owner's. Such a stale footprint is read by
        one thing, the degree-of-freedom layout of the next equation system, and it is harmless
        there because every copy equals its owner whenever the mesh, and with it the activation of
        fields on nodes, changes; :meth:`requireConstraintCopiesCurrent` makes sure of that.

        Parameters
        ----------
        model
            The model tree.
        constraints
            Constraints, by name.

        Returns
        -------
        dict
            Those searched here, by name, in the order given.
        """

        owners = self._constraintOwnersOf(model)
        return {name: constraint for name, constraint in constraints.items() if owners[name] == self.rank}

    def requireConstraintCopiesCurrent(self, model: FEModel):
        """Refuse to change the mesh unless every process' copy of every constraint couples the
        nodes and fields its owner's couples. Collective.

        A topology change activates fields on nodes for every constraint of the model -- every copy,
        in every process -- and a copy differing from its owner would give its process a different
        degree-of-freedom layout. The solver calls this right before a topology update, where every
        copy has just been synchronized (:meth:`synchronizeStates`).

        Parameters
        ----------
        model
            The model tree.

        Raises
        ------
        RuntimeError
            In every process, if a copy differs from its owner.
        """

        digest = hashlib.sha1()
        for name, constraint in model.constraints.items():
            digest.update(name.encode())
            digest.update(np.asarray([node.label for node in constraint.nodes], dtype=np.int64).tobytes())
            digest.update(repr(constraint.fieldsOnNodes).encode())
        self.requireSameOnAllParts(
            digest.hexdigest(), "the nodes and fields the constraints couple before a topology update"
        )

    # --- Completing the results of the whole model ------------------------------------------------

    def interfaceAssemblyFor(self, plan: ElementPlan) -> InterfaceForceAssembly:
        """The exchange of a plan's element contributions at the interface, summed in model order.
        Collective among neighbours.

        Parameters
        ----------
        plan
            A plan of elements computed here.

        Returns
        -------
        InterfaceForceAssembly
            The exchange.
        """

        entryElementPositions = np.repeat(
            [self._elementPositions[number] for number in plan.elements],
            [element.nDof for element in plan.elements.values()],
        )
        return InterfaceForceAssembly(self._interface, plan.entryDofs, entryElementPositions)

    def shareFromOwners(self, vector: DofVector):
        """Make a vector, correct at the degrees of freedom owned here, the same complete vector in
        every process, in place: every entry from its owner. Collective.

        Parameters
        ----------
        vector
            A vector of the whole model.
        """

        self._interface.gatherFromOwners(vector)

    def addConstraintForces(self, forces: dict, P: DofVector):
        """Share the forces of the constraints evaluated here with every process, and add those of
        every constraint of the model into ``P``, in model order. Collective.

        The degrees of freedom are the owner's: another process' copy of a contact constraint
        couples the nodes of its last synchronization.

        Parameters
        ----------
        forces
            The :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.ConstraintForce` of every
            constraint evaluated here, by name.
        P
            The net nodal force vector.
        """

        if not self._constraintOwners:
            return
        with performancetiming.timeit("constraint force exchange"):
            self._constraintForceExchange.addAllConstraintForces(
                {name: constraintForce.forces for name, constraintForce in forces.items()}, P
            )

    @performancetiming.timeit("subdomain synchronization")
    def synchronizeStates(self, includeElements: bool):
        """Give every stateful constraint -- and, if asked, every element -- of this process' model
        the state the process computing it last left it in. Collective.

        Parameters
        ----------
        includeElements
            Whether the element states are synchronized as well.
        """

        if includeElements:
            self._stateSynchronization.synchronizeElementStates()
        self._stateSynchronization.synchronizeConstraintStates()

    def sumAcrossParts(self, values: list[float]) -> list[float]:
        """The sums of values over all processes, added in ascending rank order, so that they are the
        same bits in every process. Collective.

        Parameters
        ----------
        values
            This process' contributions.

        Returns
        -------
        list[float]
            The sum of every entry over all processes.
        """

        gathered = self.communicator.allgather(list(values))
        totals = list(gathered[0])
        for contributions in gathered[1:]:
            totals = [total + value for total, value in zip(totals, contributions)]
        return totals

    def minAcrossParts(self, value: float) -> float:
        """The minimum of a value over all processes. Collective.

        Parameters
        ----------
        value
            This process' value.

        Returns
        -------
        float
            The minimum.
        """

        return self.communicator.allreduce(value, op=MPI.MIN)

    def anyPart(self, flag: bool) -> bool:
        """Whether a flag is set in any process. Collective.

        Parameters
        ----------
        flag
            This process' flag.

        Returns
        -------
        bool
            Whether any process set it.
        """

        return bool(self.communicator.allreduce(bool(flag), op=MPI.LOR))

    def requireSameOnAllParts(self, value, description: str):
        """Refuse to continue unless every process holds the same value. Collective.

        Parameters
        ----------
        value
            This process' value; anything comparable for equality.
        description
            What the value says, for the message.

        Raises
        ------
        RuntimeError
            In every process, if two processes hold different values.
        """

        values = self.communicator.allgather(value)
        if any(other != values[0] for other in values[1:]):
            raise RuntimeError(
                "The processes disagree on {:} ({:}); they must compute the replicated parts of the model "
                "identically.".format(description, values)
            )

    @contextmanager
    def agreedOnByAllParts(self, operation: str):
        """A context in which an exception raised in one process is raised in every process.
        Collective.

        A process raising alone would leave the others waiting for it in the next exchange forever.
        Every process reports what went wrong in it, and every process raises the most severe: a
        :class:`~edelweissfe.utils.exceptions.ConditionalStop` or a
        :class:`~edelweissfe.utils.exceptions.CutbackRequest` anywhere is raised as such everywhere
        -- a cutback with the smallest size any process requested -- so that every process takes the
        same path out of the step. Nothing inside the context may itself communicate, unless every
        process reaches it whatever another process did before: a process that raised would skip
        it. (The field outputs of a distributed model gather their element results inside the
        output context, every process in the same order, before anything that runs in one process
        only -- the output managers of rank 0.)

        Parameters
        ----------
        operation
            What is done in the context, for the message.

        Yields
        ------
        None
        """

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
            None
            if failure is None
            else (
                "{:}: {:}".format(type(failure).__name__, failure),
                failure.cutbackSize if outcome == _CUTBACK else None,
            )
        )
        message = "; ".join(
            "process {:}: {:}".format(rank, report[0]) for rank, report in enumerate(reports) if report is not None
        )
        if status[0] == _CUTBACK:
            cutbackSize = min(report[1] for report in reports if report is not None and report[1] is not None)
            raise CutbackRequest(message, cutbackSize) from failure
        raise StepFailed("{:} failed in {:}".format(operation, message)) from failure

    # --- Load balancing -----------------------------------------------------------------------------

    def measuresElementCosts(self) -> bool:
        """Whether the element kernels are to be timed, for :meth:`rebalance`: whenever load
        balancing is enabled, and the model can be repartitioned.

        A model whose processes each created only their own elements is not repartitioned: an
        element changing process would have to be created by its new process and its state sent
        there (migration), which is not implemented yet. Its first partition is kept, and the
        subdomain reports that once (:meth:`define`).

        Returns
        -------
        bool
            Whether to time them.
        """

        return bool(self.loadBalanceTolerance) and self._model.elementDistribution.createsEveryElement

    def rebalance(self, plan: ElementPlan, costs: np.ndarray | None, nIncrements: int) -> bool:
        """Repartition with the measured element costs if the slowest process has fallen more than
        ``loadBalanceTolerance`` behind the mean. Collective; only when every element state has just
        been synchronized, since an element changing process must arrive with its current state.

        Parameters
        ----------
        plan
            The plan of the elements whose kernels are computed here.
        costs
            The measured kernel time of every element of the plan, in plan order, or None.
        nIncrements
            The increments the costs were measured over.

        Returns
        -------
        bool
            Whether the partition changed; then everything derived from :attr:`partition` must be
            derived again.
        """

        tolerance = self.loadBalanceTolerance
        if not tolerance or self.nProcesses == 1 or costs is None or not nIncrements:
            return False
        if not self._model.elementDistribution.createsEveryElement:
            # Repartitioning a distributed model would move elements between processes; see
            # measuresElementCosts.
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
