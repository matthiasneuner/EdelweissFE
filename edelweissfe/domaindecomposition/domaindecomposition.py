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
"""The decomposition of a model over MPI processes: the subdomain each computes, and what they exchange.

A :class:`DomainDecomposition` decides which part of the model its process computes -- the elements METIS
assigns to it, the constraints dealt to it, and the degrees of freedom those touch
(:mod:`.partitioning`) -- and describes it as a
:class:`~edelweissfe.solvers.base.subdomain.Subdomain` (:attr:`DomainDecomposition.subdomain`). It
also carries out every exchange between the processes the domain-decomposed solver needs: the
element forces are completed at the interface with the neighbouring subdomains
(:mod:`.subdomaininterface`), the constraint forces are sent to the processes integrating their
degrees of freedom (:class:`~.subdomaininterface.ConstraintForceExchange`), sums are formed
over all processes in rank order, and the vectors and states of the whole model are made current
from the processes computing them (:mod:`.statesynchronization`). The loads of the elements a
subdomain computes are described in :mod:`.loadsonsubdomain`, and how the processes agree -- on a
failure in any of them, among others -- in :mod:`.communicator`. Every sum
that decides the solution is formed in the order it is formed without decomposition, so a run is
bit-identical to one in a single process.
"""

import hashlib
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from mpi4py import MPI

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.communicator import Communicator
from edelweissfe.domaindecomposition.loadsonsubdomain import (
    BodyLoadOnSubdomain,
    DistributedLoadOnSubdomain,
    LoadsOnSubdomain,
)
from edelweissfe.domaindecomposition.partitioning import (
    assignConstraintsByMajority,
    elementSharesBesideConstraints,
    keepElementsWhereTheyWere,
    partitionElementsOfMesh,
)
from edelweissfe.domaindecomposition.statesynchronization import (
    ModelStateSynchronization,
    elementsWithoutState,
)
from edelweissfe.domaindecomposition.subdomaininterface import (
    ConstraintForceExchange,
    ElementProcessesAtDofs,
    ElementProcessesAtNodes,
    InterfaceForceAssembly,
    InterfaceLoadAssembly,
    SubdomainInterface,
    ValuesFromOwners,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.globaldofnumbering import GlobalDofNumbering
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.solvers.base.parallelelementcomputation import ElementPlan
from edelweissfe.solvers.base.subdomain import Subdomain
from edelweissfe.utils.exceptions import TopologyError


@dataclass(frozen=True)
class MeasuredLoad:
    """What one process measured of its load since its increment plan was made, for
    :meth:`DomainDecomposition.rebalance`.

    Parameters
    ----------
    elementCosts
        The cost of every element of the plan per increment, in plan order -- the weights of a
        repartition: the least kernel time measured in the ``nIncrements`` increments since the last
        check, which a descheduled process cannot inflate, or the element number --, or None if
        nothing was measured.
    nIncrements
        The increments it was measured over.
    incrementsUntilNextCheck
        The increments until the next check, for the expected gain of a repartition; None to
        repartition whenever the tolerance is exceeded, for costs that are not times (element
        numbers).
    elementTime
        What the process is busy with its elements per increment, in the unit of the costs: the wall
        time of its element loop for measured costs; the sum of the element numbers for element
        numbers.
    constraintTime
        The time per increment it is busy evaluating its constraints, in the same unit; 0 where the
        costs are no times.
    constraintTimes
        The time per increment of each constraint it evaluates, by name; empty where the costs are no
        times.
    """

    elementCosts: np.ndarray | None
    nIncrements: int
    incrementsUntilNextCheck: int | None
    elementTime: float
    constraintTime: float
    constraintTimes: dict


class DomainDecomposition:
    """The decomposition of the model over the MPI processes: the subdomain this process computes,
    and its exchanges with the others; see the module documentation.

    Parameters
    ----------
    communicator
        The communicator of the processes sharing the model
        (:class:`~edelweissfe.domaindecomposition.communicator.Communicator`).
    journal
        The journal, for the subdomain report.
    identification
        The solver's identification, for the journal.
    loadBalanceTolerance
        How far the slowest process may fall behind the mean, as a fraction, before
        :meth:`rebalance` repartitions; 0 disables it.
    """

    def __init__(self, communicator: Communicator, journal, identification: str, loadBalanceTolerance: float):
        self.communicator = communicator
        self.rank = communicator.Get_rank()
        self.nProcesses = communicator.Get_size()
        self.journal = journal
        self.identification = identification
        self.loadBalanceTolerance = loadBalanceTolerance
        #: What the last repartition cost, in seconds, in the slowest process; None before the first.
        self._lastRepartitionCost = None
        #: What moving the elements of the last migration cost alone, in seconds, in the slowest
        #: process -- without building the equation system again; None before the first.
        self._lastMigrationCost = None
        #: The mean time one element keeps its process busy per increment -- the element loops' wall
        #: times over the number of elements --, at the last rebalancing check that measured times;
        #: None before.
        self._meanElementCost = None
        #: The measured time per increment of every constraint, by name, and of all elements together, at
        #: the last rebalancing check that measured times; empty and None before. Kept by name, so that
        #: the shares of the processes follow the constraints wherever they are evaluated.
        self._constraintCostsByName = {}
        self._elementTime = None
        #: The share of the element work each process is given beside its constraints, by rank
        #: (:func:`~edelweissfe.domaindecomposition.partitioning.elementSharesBesideConstraints`), for the
        #: current constraint owners; None for equal shares.
        self._processShares = None

        #: The rank of every element, by number and as an array in mesh order, which processes' elements
        #: touch each node, and the mesh and its version the partition was made for (:meth:`_setPartition`).
        self._elementOwners = None
        self._processOfElements = None
        self._elementsAtNodes = None
        self._partitionedMesh = None
        #: The rank of every constraint, by name, and the mesh and its version it was assigned for.
        self._constraintOwners = None
        self._constraintOwnersMesh = None
        #: Whether the solver stated, since the constraints were last assigned, that every copy of every
        #: constraint is its owner's (:meth:`constraintCopiesAreSynchronized`).
        self._constraintCopiesSynchronized = False

        self._ownedConstraints = {}

        #: The global numbering of the degrees of freedom the layout was last checked against.
        self._numbering = None

        self._interface = None
        self._stateSynchronization = None
        self._constraintForceExchange = None
        #: The element numbers of the mesh, sorted, and the position in the mesh of each: the order
        #: interface forces are summed in.
        self._sortedElementNumbers = None
        self._meshPositionOfSorted = None

        #: What the last definition was made for; a rebalance redefines from it.
        self._model = None
        self._mpcTransformation = None

        #: The subdomain of this process -- the elements, constraints and degrees of freedom it
        #: computes --, as of the last definition (:meth:`define`); None before the first, and while
        #: elements move between the processes.
        self.subdomain: Subdomain | None = None

        self._forgetElements()

    def _forgetElements(self):
        """Hold no element and nothing indexed by the elements: the state before the first
        definition, and the one :meth:`moveElements` starts from, so that no element a process drops
        is kept alive here."""

        self._ownedElements = {}
        #: The loads of the elements of the subdomain, and their assembly, for the loads last asked
        #: for; see loadsOnSubdomain.
        self._loadsOnSubdomain = None
        # forgotten with the elements (see __init__)
        self.subdomain = None
        #: The degree-of-freedom layout of the last definition; its entity indices hold the elements.
        self._dofManager = None

    # --- Defining the subdomain -------------------------------------------------------------------

    def define(
        self,
        model: FEModel,
        dofManager: DofManager,
        mpcTransformation: MultiPointConstraintTransformation | None,
        topologyChanged: bool,
    ):
        """Collective. Partition the model if its elements changed, and determine this process' subdomain
        degrees of freedom and interface: the :attr:`subdomain`.

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

                self._constraintOwnersOf(model)

                self._adoptOwnership(model)

            self._defineInterface(model, ownershipChanged=topologyChanged)

        self._reportSubdomains(topologyChanged)

    def _partition(self, model: FEModel, measuredCosts: dict | None, byMeasuredCosts: bool):
        """Collective. Partition the elements of the mesh, by estimated or by measured cost.

        A model whose processes each created only their own elements was partitioned before its
        elements were created (see
        :meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.decideLocalElements`),
        on the same mesh and by the same estimate: that partition is adopted, since an element can
        only be computed where it exists -- after a refinement with the children of a refined
        element where their parent was computed
        (:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.placeChildElement`).
        Repartitioned with measured costs, its elements are then to be moved to their new processes
        (:meth:`moveElements`).

        Parameters
        ----------
        model
            The model tree.
        measuredCosts
            The measured cost per increment of elements, by number, on rank 0; None elsewhere.
        byMeasuredCosts
            Whether to partition by the measured costs (in every process), or by the estimate.

        Raises
        ------
        TopologyError
            If the mesh of a distributed model changed without its distribution following.
        """

        distribution = model.elementDistribution
        if not distribution.replicatesElements and model.mesh.elements.keys() != distribution.owners.keys():
            raise TopologyError(
                "the mesh of this distributed model changed, but the processes computing its elements were not "
                "decided for the changed mesh (ElementDistribution.updateLocalElements)"
            )

        if distribution.replicatesElements or byMeasuredCosts:
            with performancetiming.timeit("partition"):
                owners = partitionElementsOfMesh(
                    model.mesh, self.nProcesses, model.domainSize, self.communicator, measuredCosts, self._processShares
                )
                if byMeasuredCosts:
                    # A repartition of the same elements: those that can stay in their process do.
                    owners = keepElementsWhereTheyWere(
                        owners, self._elementOwners, self.nProcesses, self._processShares
                    )
            self._setPartition(model, owners)
        else:
            # A copy: the distribution changes its own as the mesh changes (see
            # DistributedElements.placeChildElement), and this partition must not change with it.
            self._setPartition(model, dict(distribution.owners))

    def _setPartition(self, model: FEModel, owners: dict):
        """Adopt a partition of the mesh, and derive from it, once, what is read per mesh node: the process
        of every element in mesh order, and which processes' elements touch each field at each node
        (:class:`~.subdomaininterface.ElementProcessesAtNodes`).

        Parameters
        ----------
        model
            The model tree.
        owners
            The rank of every element of the mesh, by number.
        """

        mesh = model.mesh
        connectivity = mesh.connectivity()
        self._elementOwners = owners
        self._processOfElements = np.fromiter(
            (owners[number] for number in connectivity.numbers.tolist()),
            dtype=np.int32,
            count=connectivity.numbers.shape[0],
        )
        self._elementsAtNodes = ElementProcessesAtNodes.ofMesh(
            connectivity, self._processOfElements, mesh.nNodes, self.nProcesses
        )
        self._partitionedMesh = (mesh, mesh.elementsVersion)

    @staticmethod
    def _isFor(meshAndVersion: tuple | None, model: FEModel) -> bool:
        """Whether something derived from a mesh at a version is for the current mesh of the model: in
        constant time, by the mesh's count of changes (:attr:`~edelweissfe.models.mesh.Mesh.elementsVersion`).

        Parameters
        ----------
        meshAndVersion
            The mesh and its version it was derived for, or None.
        model
            The model tree.

        Returns
        -------
        bool
            True if it is current.
        """

        return (
            meshAndVersion is not None
            and meshAndVersion[0] is model.mesh
            and meshAndVersion[1] == model.mesh.elementsVersion
        )

    def _constraintOwnersOf(self, model: FEModel) -> dict:
        """Collective. The rank of every constraint of the model: the current assignment while it was made
        for the current mesh and the current set of constraints, otherwise a new assignment
        (:meth:`_assignConstraints`), made now -- the one the next definition adopts. A mesh not yet
        partitioned is partitioned first.

        This is the one place that decides when the constraints are assigned afresh: where the mesh or
        the set of constraints changed. Assigning them afresh after a migration, too, would be decided
        here.

        A connectivity update right after a topology change, before the subdomain is defined afresh,
        searches on the process that is going to evaluate a constraint because of this.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        dict
            The rank of every constraint, by name.
        """

        if not self._isFor(self._partitionedMesh, model):
            self._partition(model, measuredCosts=None, byMeasuredCosts=False)
        if not (
            self._isFor(self._constraintOwnersMesh, model)
            and self._constraintOwners is not None
            and model.constraints.keys() == self._constraintOwners.keys()
        ):
            self._assignConstraints(model)
        return self._constraintOwners

    def constraintCopiesAreSynchronized(self):
        """State that every process' copy of every constraint is its owner's -- the model was synchronized
        to every process, with the constraint states, since the last increment was accepted. The solver
        states it before a topology update (:meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.updateTopology`);
        assigning the constraints afresh requires it (:meth:`_assignConstraints`), and consumes it."""

        self._constraintCopiesSynchronized = True

    def _assignConstraints(self, model: FEModel):
        """Collective. Assign every constraint to the process holding most of its nodes under the current
        partition (:func:`~edelweissfe.domaindecomposition.partitioning.assignConstraintsByMajority`): the
        nodes its search may couple, or, for a constraint without a search, those it couples.

        Made from what every process holds alike -- the mesh, the partition, the nodes of the copies of
        the constraints --, and checked to be the same in every process. A constraint whose process
        changes is evaluated from then on from the new process' copy, which must therefore be its
        owner's: except for the first assignment, the solver must have stated that every copy is
        current (:meth:`constraintCopiesAreSynchronized`) -- it does before a topology update, which
        follows a synchronization of the whole model to every process.

        The shares of the element work the processes are given beside their constraints follow the
        constraints: computed from the measured cost of each constraint, by name, for its new process.
        Where every process holds every element, a partition made for other shares is made again for
        these, keeping the elements where they were as far as the shares allow; a distributed model
        partitions with them when it moves its elements
        (:meth:`rebalanceAfterTopologyChange`, which assigns the constraints first).

        Parameters
        ----------
        model
            The model tree.

        Raises
        ------
        RuntimeError
            If the constraints were assigned before, and no synchronization of their copies was stated
            since; or, in every process, if the processes assigned them differently.
        """

        if self._constraintOwners is not None and not self._constraintCopiesSynchronized:
            raise RuntimeError(
                "The constraints are to be assigned to processes afresh, which evaluate them from their copies "
                "from then on; but the copies were not stated to be synchronized from their owners since they "
                "were last assigned (DomainDecomposition.constraintCopiesAreSynchronized)."
            )
        self._constraintCopiesSynchronized = False

        mesh = model.mesh
        nodeRows = {}
        for name, constraint in model.constraints.items():
            # the nodes a search may couple; those coupled now for a constraint without a search
            nodes = constraint.nodesReadByConnectivityUpdate() or constraint.nodes
            nodeRows[name] = mesh.rowsOfNodes([node.label for node in nodes])
        owners = assignConstraintsByMajority(nodeRows, self._elementsAtNodes.lowestProcessAtNodes(), self.nProcesses)
        self.communicator.requireSameOnAllRanks(tuple(owners.items()), "the processes the constraints are assigned to")
        self._constraintOwners = owners
        self._constraintOwnersMesh = (mesh, mesh.elementsVersion)

        shares = self._sharesFor(owners, model)
        sharesChanged = (shares is None) != (self._processShares is None) or (
            shares is not None and not np.array_equal(shares, self._processShares)
        )
        self._processShares = shares
        if sharesChanged and model.elementDistribution.replicatesElements:
            with performancetiming.timeit("partition"):
                partitioned = partitionElementsOfMesh(
                    mesh, self.nProcesses, model.domainSize, self.communicator, processShares=shares
                )
                partitioned = keepElementsWhereTheyWere(partitioned, self._elementOwners, self.nProcesses, shares)
            self._setPartition(model, partitioned)

    def _constraintCostsOfProcesses(self, owners: dict) -> np.ndarray:
        """The measured cost per increment of the constraints each process evaluates under an assignment,
        from the cost of every constraint by name; zeros before anything was measured.

        Parameters
        ----------
        owners
            The rank of every constraint, by name.

        Returns
        -------
        np.ndarray
            The cost, by rank.
        """

        costs = np.zeros(self.nProcesses)
        for name, cost in self._constraintCostsByName.items():
            if name in owners:
                costs[owners[name]] += cost
        return costs

    def _sharesFor(self, owners: dict, model: FEModel) -> np.ndarray | None:
        """The shares of the element work for an assignment of the constraints
        (:func:`~edelweissfe.domaindecomposition.partitioning.elementSharesBesideConstraints`), from the
        last measurement; None for equal shares, and before anything was measured.

        Parameters
        ----------
        owners
            The rank of every constraint, by name.
        model
            The model tree.

        Returns
        -------
        np.ndarray | None
            The shares, by rank.
        """

        if self._elementTime is None:
            return None
        connectivity = model.mesh.connectivity()
        return elementSharesBesideConstraints(
            self._elementTime,
            self._constraintCostsOfProcesses(owners),
            connectivity.numbers.shape[0] - connectivity.nAuxiliary,
        )

    def _adoptOwnership(self, model: FEModel):
        """Collect the elements and constraints this process owns under the current partition.

        Parameters
        ----------
        model
            The model tree.
        """

        # The order of the mesh, which is the order of the elements in a model that created all of
        # them, and so the order a serial run sums their contributions in.
        numbers = model.mesh.connectivity().numbers
        self._meshPositionOfSorted = np.argsort(numbers, kind="stable")
        self._sortedElementNumbers = numbers[self._meshPositionOfSorted]

        self._ownedElements = {
            number: element for number, element in model.elements.items() if self._elementOwners[number] == self.rank
        }
        self._ownedConstraints = {
            name: constraint
            for name, constraint in model.constraints.items()
            if self._constraintOwners[name] == self.rank
        }

    def _defineInterface(self, model: FEModel, ownershipChanged: bool):
        """Collective. Determine the subdomain degrees of freedom and the interface from the owned entities;
        with new ownership, also the layout of the state synchronization.

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

        self._interface = SubdomainInterface(
            self.communicator,
            # expanded to the degrees of freedom for this interface only; held per node, a fraction of it
            ElementProcessesAtDofs.ofNodes(self._elementsAtNodes, self._numbering),
            touched,
            None if self._mpcTransformation is None else self._mpcTransformation.slaveMasterDofPairs(),
        )

        if ownershipChanged:
            # The states of the elements other processes compute are kept current only where every
            # element was created; a distributed model gathers what it reads instead.
            self._stateSynchronization = ModelStateSynchronization(
                self.communicator,
                model.elements if model.elementDistribution.replicatesElements else {},
                self._elementOwners,
                model.constraints,
                self._constraintOwners,
            )
            self._refuseElementsWithoutState(model)

        self._constraintForceExchange = ConstraintForceExchange(
            self._interface, model.constraints, self._ownedConstraints, dofManager.idcsOfConstraintsInDofVector
        )
        self._loadsOnSubdomain = None

        self.subdomain = Subdomain(self._ownedElements, self._ownedConstraints, self._interface.subdomainDofs)

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

    def _checkReplicatedLayout(self, model: FEModel, dofManager: DofManager):
        """Collective. Refuse to continue unless the DofManager numbers the degrees of freedom as the global
        numbering of the mesh does
        (:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`), and every process built
        the same mesh and degree-of-freedom layout (:func:`layoutFingerprint`).

        The interface exchange addresses degrees of freedom by index, and the load assembly tags a
        load contribution by its place in the element sets and surfaces of the mesh; both are only
        meaningful if every process numbered and ordered them identically, which after a refinement
        rests on every process having refined identically. The global numbering of the mesh is what a
        process holding only some of the nodes will number its degrees of freedom by.

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

        numbering = GlobalDofNumbering.ofModel(model)
        numbering.requireAgreementWith(dofManager, model.nodeFields, model.scalarVariables)
        self._numbering = numbering
        fingerprints = self.communicator.allgather(layoutFingerprint(model, numbering, self._mpcTransformation))
        if len(set(fingerprints)) > 1:
            raise RuntimeError(
                "The processes hold different models: their layout fingerprints are {:}. Every process "
                "must build, and refine, the model identically -- the same elements, element sets and "
                "surfaces in the same order, the same degrees of freedom and multi-point constraints -- for "
                "the subdomain interface and the order of the loads to be meaningful.".format(
                    ", ".join(sorted(set(fingerprints)))
                )
            )

    def _reportSubdomains(self, topologyChanged: bool):
        """Collective. Report the size of every subdomain and of its interface, and the constraints each
        process evaluates with the force entries they send per increment, one line each.

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
                len(self._ownedConstraints),
                self._constraintForceExchange.nEntriesSent,
            ),
            root=0,
        )
        if statistics is None:
            return

        def perProcess(column: int) -> str:
            return "/".join(str(entry[column]) for entry in statistics)

        elementCounts = np.array([entry[0] for entry in statistics])
        imbalance = elementCounts.max() / max(elementCounts.mean(), 1e-300)
        level = 0 if topologyChanged else 2
        self.journal.message(
            "Subdomains: {:} element(s) per process (max/mean {:.3f}); subdomain DOFs {:}; interface DOFs {:}; "
            "neighbours {:}; elements created {:} of {:}".format(
                perProcess(0),
                imbalance,
                perProcess(1),
                perProcess(2),
                perProcess(3),
                perProcess(4),
                len(self._model.mesh.elements),
            ),
            self.identification,
            level,
        )
        self.journal.message(
            "Constraints per process: {:}; their force entries sent per increment: {:}; constraint DOFs beyond "
            "the elements', gathered: {:}".format(
                perProcess(5), perProcess(6), self._interface.nConstraintDofsBeyondElements
            ),
            self.identification,
            level,
        )

    # --- The subdomain -------------------------------------------------------------------------

    def loadsOnSubdomain(self, distributedLoads, bodyLoads) -> LoadsOnSubdomain:
        """The given loads, restricted to the elements of the subdomain, with their nodal forces tagged
        for the assembly at the interface (:meth:`loadAssemblyFor`); made again when the loads or the
        subdomain changed. Local: it reads the mesh, which every process holds whole, and so may be
        called where a process can fail alone
        (:meth:`~edelweissfe.domaindecomposition.communicator.Communicator.allRanksFailTogether`).

        A load acting on an element is a contribution of that element: it is evaluated by the
        process computing the element, with that process' current solution -- the element's degrees
        of freedom are all integrated there -- and its nodal forces are sent to the neighbours
        integrating the same degrees of freedom (:class:`~.subdomaininterface.InterfaceLoadAssembly`).
        Every contribution is tagged with its place in the order a single process adds the loads in
        (:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleLoads`):
        the distributed loads in the order given, the faces of each surface, the elements of each
        face as the mesh describes it; then the body loads, the elements of each set.

        Parameters
        ----------
        distributedLoads
            The distributed loads of the step, in deck order.
        bodyLoads
            The body loads of the step, in deck order.

        Returns
        -------
        LoadsOnSubdomain
            The restricted loads.

        Raises
        ------
        TopologyError
            If a loaded local element is not where the mesh describes it in its surface or set.
        """

        distributedLoads, bodyLoads = list(distributedLoads), list(bodyLoads)
        if self._loadsOnSubdomain is not None and self._loadsOnSubdomain.isFor(distributedLoads, bodyLoads):
            return self._loadsOnSubdomain

        mesh = self._model.mesh
        dofsOf = self._dofManager.idcsOfElementsInDofVector
        entryDofs, entryOrder = [], []
        loadsBefore = 0

        def computedHere(loadedSetName: str, elements, numbersInMesh: list) -> list:
            """The elements of a loaded set in the subdomain, in set order, with their entries tagged by
            their place among all load contributions; advances the count of contributions."""

            nonlocal loadsBefore
            positionInMesh = {number: position for position, number in enumerate(numbersInMesh)}
            computed = []
            previous = -1
            for element in elements:
                position = positionInMesh.get(element.elNumber)
                if position is None or position < previous:
                    raise TopologyError(
                        "element {:} of {:} carries a load, but not in the place the mesh describes".format(
                            element.elNumber, loadedSetName
                        )
                    )
                previous = position
                if element.elNumber in self._ownedElements:
                    computed.append(element)
                    entryDofs.append(dofsOf[element])
                    entryOrder.append(np.full(element.nDof, loadsBefore + position, dtype=np.int64))
            loadsBefore += len(numbersInMesh)
            return computed

        restrictedDistributedLoads = []
        for load in distributedLoads:
            numbersOfFaces = mesh.elementNumbersOfSurface(load.surface.name)
            restrictedDistributedLoads.append(
                DistributedLoadOnSubdomain(
                    load,
                    {
                        faceID: computedHere(load.surface.name, elementSet.localElements(), numbersOfFaces[faceID])
                        for faceID, elementSet in load.surface.items()
                    },
                )
            )
        restrictedBodyLoads = [
            BodyLoadOnSubdomain(
                load,
                computedHere(
                    load.elementSet.name, load.elementSet.localElements(), load.elementSet.elementNumbersOfWholeSet()
                ),
            )
            for load in bodyLoads
        ]

        self._loadsOnSubdomain = LoadsOnSubdomain(
            distributedLoads,
            bodyLoads,
            restrictedDistributedLoads,
            restrictedBodyLoads,
            np.concatenate(entryDofs) if entryDofs else np.empty(0, dtype=np.int64),
            np.concatenate(entryOrder) if entryOrder else np.empty(0, dtype=np.int64),
        )
        return self._loadsOnSubdomain

    def loadAssemblyFor(self, loads: LoadsOnSubdomain) -> InterfaceLoadAssembly:
        """Collective among neighbours when built. The assembly of the nodal forces of loads restricted
        to this subdomain at the interface; built the first time it is asked for: every process asks for
        it after the same :meth:`loadsOnSubdomain`, which is made again in every process at the same
        time.

        Parameters
        ----------
        loads
            The loads, as :meth:`loadsOnSubdomain` restricted them.

        Returns
        -------
        InterfaceLoadAssembly
            The assembly.
        """

        if loads.assembly is None:
            loads.assembly = InterfaceLoadAssembly(self._interface, loads.entryDofs, loads.entryLoadOrder)
        return loads.assembly

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
        """Collective. Refuse to change the mesh unless every process' copy of every constraint couples the
        nodes and fields its owner's couples.

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
        self.communicator.requireSameOnAllRanks(
            digest.hexdigest(), "the nodes and fields the constraints couple before a topology update"
        )

    @property
    def ownedDofMask(self) -> np.ndarray:
        """Whether each degree of freedom of the model is owned by this process: True at most at its
        subdomain degrees of freedom, and at each degree of freedom of the model in exactly one process. A
        degree of freedom at the interface is integrated by several processes, but a quantity to be
        counted once for the model -- the work done at a prescribed degree of freedom, a kinetic
        energy -- is counted by its owner only.

        Returns
        -------
        np.ndarray
            The mask, one entry per degree of freedom of the model.
        """

        return self._interface.ownedDofMask

    # --- Completing the results of the whole model ------------------------------------------------

    def interfaceAssemblyFor(self, plan: ElementPlan) -> InterfaceForceAssembly:
        """Collective among neighbours. The exchange of a plan's element contributions at the interface,
        summed in model order.

        Parameters
        ----------
        plan
            A plan of elements of the subdomain.

        Returns
        -------
        InterfaceForceAssembly
            The exchange.
        """

        planNumbers = np.fromiter(plan.elements.keys(), dtype=np.int64, count=len(plan.elements))
        positions = self._meshPositionOfSorted[np.searchsorted(self._sortedElementNumbers, planNumbers)]
        entryElementPositions = np.repeat(positions, [element.nDof for element in plan.elements.values()])
        return InterfaceForceAssembly(self._interface, plan.entryDofs, entryElementPositions)

    def allgatherOwnedValues(self, vector: DofVector):
        """Collective. Make a vector, correct at the degrees of freedom this process owns, the same complete
        vector in every process, in place: every entry from its owner.

        Parameters
        ----------
        vector
            A vector of the whole model.
        """

        self._interface.allgatherOwnedValues(vector)

    def gatherOwnedValuesToRoot(self, vector: DofVector):
        """Collective. Make a vector, correct at the degrees of freedom this process owns, the complete
        vector on rank 0, in place: every entry from its owner. The other processes keep theirs, complete
        at the degrees of freedom they integrate only.

        Parameters
        ----------
        vector
            A vector of the whole model.
        """

        self._interface.gatherOwnedValuesToRoot(vector)

    @property
    def interface(self) -> SubdomainInterface:
        """The interface of the current definition: the degrees of freedom this process integrates,
        and the owner of every degree of freedom of the model.

        Returns
        -------
        SubdomainInterface
            The interface.
        """

        return self._interface

    def valuesFromOwners(self, neededDofs: np.ndarray) -> ValuesFromOwners:
        """Collective. How this process receives the entries of vectors at the given degrees of freedom from
        their owners, point to point (:class:`~.subdomaininterface.ValuesFromOwners`).

        Parameters
        ----------
        neededDofs
            The degrees of freedom whose entries this process needs.

        Returns
        -------
        ValuesFromOwners
            The exchange.
        """

        return ValuesFromOwners(self.communicator, neededDofs, self._interface.dofOwners)

    def receiveNewlyIntegratedValues(self, before: SubdomainInterface, vectors: list[DofVector]):
        """Collective. After the subdomain was defined afresh for the same degree-of-freedom layout -- a
        contact search moved a constraint, or elements moved between the processes -- complete the
        vectors at the degrees of freedom this process integrates now but did not before, from the
        processes that owned them before.

        A process holds a vector current at the degrees of freedom it integrates only (between two
        synchronizations of the whole model, see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.acceptIncrement`); a degree of
        freedom it starts to integrate holds whatever the last synchronization left there. Its owner
        under the previous definition integrated it, and so holds it current: the entries are its
        bits.

        Parameters
        ----------
        before
            The interface of the previous definition.
        vectors
            The vectors carried over to the new definition.
        """

        newlyIntegrated = np.setdiff1d(self._interface.subdomainDofs, before.subdomainDofs, assume_unique=True)
        ValuesFromOwners(self.communicator, newlyIntegrated, before.dofOwners).receive(vectors)

    def addConstraintForces(self, forces: dict, P: DofVector):
        """Collective among neighbours. Send the forces of the constraints of the subdomain to the processes
        integrating their degrees of freedom, and add those of every constraint acting on a subdomain
        degree of freedom into ``P``, in model order
        (:class:`~.subdomaininterface.ConstraintForceExchange`).

        The degrees of freedom are the owner's: another process' copy of a contact constraint
        couples the nodes of its last synchronization. ``P`` holds the constraint forces at the
        subdomain degrees of freedom only.

        Parameters
        ----------
        forces
            The :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.ConstraintForce` of every
            constraint of the subdomain, by name.
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
    def synchronizeStates(self, includeElements: bool, toEveryProcess: bool = True):
        """Collective. Give every stateful constraint -- and, if asked, every element -- of this process'
        model the state the process computing it last left it in.

        The constraint states are received by every process, or by rank 0 only, where the output is
        written; the copies elsewhere then keep the state of their last synchronization. The element
        states -- synchronized only where every process holds every element -- are always received
        by every process: a repartition may give any element to any process.

        Parameters
        ----------
        includeElements
            Whether the element states are synchronized as well.
        toEveryProcess
            Whether every process receives the constraint states, or rank 0 only.
        """

        if includeElements:
            self._stateSynchronization.synchronizeElementStates()
        self._stateSynchronization.synchronizeConstraintStates(toEveryProcess)

    def elementStatesOwnedHere(self, model: FEModel) -> dict:
        """The converged state of every element this process owns, by number, in model order: its
        part of a restart checkpoint of the whole model. Local.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        dict
            The states, by element number.
        """

        owned = model.elementDistribution.ownedElements(model.elements.values())
        return {element.elNumber: element.getStateVars() for element in owned}

    def gatherElementStatesToRoot(self, statesHere: dict) -> dict | None:
        """Collective. Gather the element states of every process (:meth:`elementStatesOwnedHere`) to rank
        0, for the restart checkpoint rank 0 writes.

        Parameters
        ----------
        statesHere
            The states of the elements this process owns, by number.

        Returns
        -------
        dict | None
            On rank 0, the state of every element of the model, by number -- the elements of rank 0
            first, then those of rank 1, and so on; None elsewhere.
        """

        gathered = self.communicator.gather(statesHere, root=0)
        if gathered is None:
            return None
        states = {}
        for statesOfProcess in gathered:
            states.update(statesOfProcess)
        return states

    # --- Load balancing -----------------------------------------------------------------------------

    def measuresElementCosts(self) -> bool:
        """Whether the element kernels are to be timed, for :meth:`rebalance`: whenever load
        balancing is enabled.

        Returns
        -------
        bool
            Whether to time them.
        """

        return bool(self.loadBalanceTolerance)

    def rebalance(self, plan: ElementPlan, load: MeasuredLoad) -> bool:
        """Collective. Repartition with the measured costs if the busiest process has fallen more than
        ``loadBalanceTolerance`` behind what a partition can attain, and if that is worth it. Only right
        after an increment was accepted and, where every process holds the whole model, every element
        state synchronized, since an element changing process must arrive with its current state.

        A process is busy with its elements and with the constraints it evaluates. The constraints
        stay where they are (see :mod:`~edelweissfe.domaindecomposition.partitioning`), so a
        partition can attain at best that every process is as busy as the mean -- or, if one
        process' constraints alone cost more, as busy as that process. The new partition gives each
        process the share of the elements that comes closest to it
        (:func:`~edelweissfe.domaindecomposition.partitioning.elementSharesBesideConstraints`): the
        process evaluating a large contact computes correspondingly fewer elements.

        Worth it means: the time the repartition is expected to save until the next check -- what the
        busiest process takes beyond the attainable time and the tolerance, per increment, times the
        increments until the next check -- exceeds what the last repartition cost
        (:meth:`recordRepartitionCost`). Both are wall times. The first repartition is always made.
        This keeps a model that cannot be balanced better -- fewer elements than processes, or a
        single constraint costing more than all elements, say -- from repartitioning, and a
        distributed one from migrating, on every check.

        Every process takes the same decision: it is made from the busy times of all processes,
        gathered to every process.

        Where every process holds the whole model, the subdomain is defined afresh at once. A
        distributed model has to move its elements to their new processes first
        (:meth:`elementsMustMove`, :meth:`moveElements`).

        Parameters
        ----------
        plan
            The plan of the elements of the subdomain that have kernels.
        load
            What this process measured.

        Returns
        -------
        bool
            Whether the partition changed; then everything derived from :attr:`subdomain` must be
            derived again.
        """

        costs, nIncrements = load.elementCosts, load.nIncrements
        tolerance = self.loadBalanceTolerance
        if not tolerance or self.nProcesses == 1 or costs is None or not nIncrements:
            return False

        incrementsUntilNextCheck = load.incrementsUntilNextCheck
        measuredTimes = np.array(self.communicator.allgather((load.elementTime, load.constraintTime)))
        elementTimes, constraintTimes = measuredTimes[:, 0], measuredTimes[:, 1]
        busyTimes = elementTimes + constraintTimes
        imbalance, timeBeyondTolerance = self._imbalance(busyTimes, constraintTimes)
        if incrementsUntilNextCheck is not None:
            nElements = self.communicator.allreduce(costs.shape[0])
            self._meanElementCost = float(elementTimes.sum()) / max(nElements, 1)
        self._elementTime = float(elementTimes.sum())
        self._constraintCostsByName = {}
        for costsOfProcess in self.communicator.allgather(load.constraintTimes):
            self._constraintCostsByName.update(costsOfProcess)
        self._processShares = self._sharesFor(self._constraintOwners, self._model)
        if timeBeyondTolerance <= 0.0:
            self.journal.message(
                "Load imbalance {:.3f} (the busiest process / the attainable) within 1 + {:}".format(
                    imbalance, tolerance
                ),
                self.identification,
                2,
            )
            return False

        if incrementsUntilNextCheck is not None and self._lastRepartitionCost is not None:
            expectedGain = timeBeyondTolerance * incrementsUntilNextCheck
            if expectedGain <= self._lastRepartitionCost:
                self.journal.message(
                    "Load imbalance {:.3f} exceeds 1 + {:}, but repartitioning would save an expected {:.3g} s until "
                    "the next check, less than the {:.3g} s the last repartition cost: not repartitioning".format(
                        imbalance, tolerance, expectedGain, self._lastRepartitionCost
                    ),
                    self.identification,
                    2,
                )
                return False

        measured = dict(zip(plan.elements.keys(), costs.tolist()))
        gathered = self.communicator.gather(measured, root=0)
        measuredCosts = None
        if gathered is not None:
            measuredCosts = {}
            for elementCosts in gathered:
                measuredCosts.update(elementCosts)

        self.journal.message(
            "Load imbalance {:.3f} (the busiest process / the attainable) exceeds 1 + {:}: repartitioning with the "
            "element costs{:}".format(
                imbalance,
                tolerance,
                (
                    ""
                    if self._processShares is None
                    else ", beside constraints costing {:} ms per increment".format(
                        "/".join("{:.2f}".format(1e3 * cost) for cost in constraintTimes)
                    )
                ),
            ),
            self.identification,
            1,
        )

        model = self._model
        self._partition(model, measuredCosts, byMeasuredCosts=True)
        if not self.elementsMustMove():
            self._adoptOwnership(model)
            with performancetiming.timeit("subdomain definition"):
                self._defineInterface(model, ownershipChanged=True)
            self._reportSubdomains(topologyChanged=True)
        return True

    def _imbalance(self, busyTimes: np.ndarray, constraintTimes: np.ndarray) -> tuple[float, float]:
        """How far the busiest process is behind what a partition of the elements can attain: the mean
        of the processes' busy times, or the time of the costliest constraints of one process if that
        is more (a constraint is not divided among processes).

        Parameters
        ----------
        busyTimes
            The time each process is busy, with its elements and constraints, by rank.
        constraintTimes
            The time of the constraints each process evaluates, by rank, in the same unit.

        Returns
        -------
        tuple[float, float]
            The busiest time over the attainable one, and the time the busiest process takes beyond
            the attainable time times ``1 + loadBalanceTolerance`` (positive if out of balance).
        """

        attainable = max(float(busyTimes.mean()), float(constraintTimes.max()), 1e-300)
        busiest = float(busyTimes.max())
        return busiest / attainable, busiest - (1.0 + self.loadBalanceTolerance) * attainable

    def recordRepartitionCost(self, seconds: float):
        """Collective. Record what the last repartition cost -- deciding it, partitioning, moving the
        elements and deriving everything again -- for the next :meth:`rebalance` to weigh against its
        gain: the longest of all processes, so that every process decides the same.

        Parameters
        ----------
        seconds
            The wall time of the repartition in this process.
        """

        self._lastRepartitionCost = self.communicator.allreduce(seconds, op=MPI.MAX)

    def rebalanceAfterTopologyChange(self, model: FEModel, horizon: int):
        """Collective. After a model modifier changed the mesh of a distributed model, move elements if the
        partition it inherited -- the children of a refined element are computed where their parent was
        -- is out of balance, and if that pays. Right after the topology update, before the equation
        system is built again, which it is anyway: a migration here costs only the moving of the
        elements.

        No new measurement is needed for the imbalance: it is estimated from the number of elements
        with kernels each process computes, at the mean measured cost of an element, beside the
        measured cost of the constraints it evaluates (as in :meth:`rebalance`; before any
        measurement, from the number of elements alone). It pays if the time it is expected to save
        -- what the busiest process takes beyond the attainable time and ``loadBalanceTolerance``,
        times ``horizon`` increments -- exceeds what moving the elements cost last time (only the
        moving, see :meth:`moveElements`). Before the first migration, or before any element cost was
        measured, the tolerance alone decides. The new partition is the estimate-weighted partition of
        the changed mesh, for the shares of the last measurement, with every element that can stay
        where it is kept there (:func:`keepElementsWhereTheyWere`). The decision is logged at level
        2. A model held whole on every process is partitioned afresh anyway.

        Parameters
        ----------
        model
            The model tree, its mesh changed.
        horizon
            The increments over which the gain is expected: until the end of the step, or as many as
            passed since the previous topology change, whichever is fewer.
        """

        distribution = model.elementDistribution
        tolerance = self.loadBalanceTolerance
        if distribution.replicatesElements or not tolerance or self.nProcesses == 1:
            return

        # The constraints are assigned for the changed mesh on the partition it inherited, and the shares
        # of the element work follow them; a new partition is then made for those shares.
        self._model = model
        self._constraintOwnersOf(model)
        mesh = model.mesh
        inherited = self._elementOwners
        counts = np.zeros(self.nProcesses)
        for group in mesh.connectivity().groups:
            if group.typeInfo.hasKernels:
                counts += np.bincount(self._processOfElements[group.positions], minlength=self.nProcesses)
        if self._meanElementCost is None:
            imbalance, timeBeyondTolerance = self._imbalance(counts, np.zeros(self.nProcesses))
        else:
            constraintCosts = self._constraintCostsOfProcesses(self._constraintOwners)
            busyTimes = counts * self._meanElementCost + constraintCosts
            imbalance, timeBeyondTolerance = self._imbalance(busyTimes, constraintCosts)
        if timeBeyondTolerance <= 0.0:
            self.journal.message(
                "After the topology change: element imbalance {:.3f} within 1 + {:}".format(imbalance, tolerance),
                self.identification,
                2,
            )
            return

        if self._meanElementCost is not None and self._lastMigrationCost is not None:
            expectedGain = timeBeyondTolerance * horizon
            if expectedGain <= self._lastMigrationCost:
                self.journal.message(
                    "After the topology change: element imbalance {:.3f} exceeds 1 + {:}, but moving elements would "
                    "save an expected {:.3g} s over {:} increments, less than the {:.3g} s moving them cost last time: "
                    "not moving".format(imbalance, tolerance, expectedGain, horizon, self._lastMigrationCost),
                    self.identification,
                    2,
                )
                return
            reason = "an expected {:.3g} s saved over {:} increments, more than the {:.3g} s moving them cost".format(
                expectedGain, horizon, self._lastMigrationCost
            )
        else:
            reason = "no migration or element cost measured yet"

        self.journal.message(
            "After the topology change: element imbalance {:.3f} exceeds 1 + {:}, repartitioning and moving "
            "elements ({:})".format(imbalance, tolerance, reason),
            self.identification,
            2,
        )
        with performancetiming.timeit("partition"):
            owners = partitionElementsOfMesh(
                mesh, self.nProcesses, model.domainSize, self.communicator, processShares=self._processShares
            )
            owners = keepElementsWhereTheyWere(owners, inherited, self.nProcesses, self._processShares)
        self._setPartition(model, owners)
        self.moveElements()

    def elementsMustMove(self) -> bool:
        """Whether the current partition computes elements in processes that do not hold them: after
        a :meth:`rebalance` of a distributed model that changed the partition. The same in every
        process.

        Returns
        -------
        bool
            Whether :meth:`moveElements` must be called before anything else is computed.
        """

        distribution = self._model.elementDistribution
        return not distribution.replicatesElements and self._elementOwners != distribution.owners

    @performancetiming.timeit("element migration")
    def moveElements(self):
        """Collective. Move the elements of a distributed model to their processes under the current
        partition (:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.moveElementsTo`),
        and report the migration.

        The decomposition forgets the elements it computed first, so that a process holds no reference
        to an element it drops; the solver must have released its own (its degree-of-freedom indices
        and plans) or build them afresh right afterwards, and define the subdomain afresh
        (:meth:`define`), for the elements it now holds. What the moving alone costs -- without
        anything built again -- is recorded, for :meth:`rebalanceAfterTopologyChange` to weigh.
        """

        startOfMoving = perf_counter()
        self._forgetElements()

        model = self._model
        counts = model.elementDistribution.moveElementsTo(model, dict(self._elementOwners))
        self._lastMigrationCost = self.communicator.allreduce(perf_counter() - startOfMoving, op=MPI.MAX)

        allCounts = self.communicator.gather(counts, root=0)
        if allCounts is not None:
            self.journal.message(
                "Element migration: {:} element(s) changed process; element objects created {:}, dropped {:} "
                "per process".format(
                    sum(entry[2] for entry in allCounts),
                    "/".join(str(entry[0]) for entry in allCounts),
                    "/".join(str(entry[1]) for entry in allCounts),
                ),
                self.identification,
                1,
            )


def layoutFingerprint(
    model: FEModel, numbering: GlobalDofNumbering, mpcTransformation: MultiPointConstraintTransformation | None
) -> str:
    """A fingerprint of everything the processes of a domain-decomposed run must hold identically: the
    mesh -- the element numbers in mesh order, and per type of element the positions of its elements and
    the nodes of each --, the element sets and the surfaces of the mesh, each with its elements in order
    and all of them in order, the global numbering of the degrees of freedom (its digest,
    :meth:`GlobalDofNumbering.digest`, which covers the node labels in mesh order), the reference
    coordinates of the nodes and the current ones of the nodes a rigid body moves, and the multi-point
    constraints -- every slave, master and weight.

    It is made from arrays every process holds whole -- the connectivity of the mesh
    (:meth:`~edelweissfe.models.mesh.Mesh.connectivity`), its node coordinates, the numbering, the
    multi-point constraints --, not from the element or node objects, of which a process may hold only
    its own.

    Parameters
    ----------
    model
        The model tree.
    numbering
        The global numbering of the degrees of freedom, which the DofManager was checked to agree with.
    mpcTransformation
        The multi-point-constraint transformation of the equation system, or None.

    Returns
    -------
    str
        The fingerprint.
    """

    digest = hashlib.sha1()

    def add(name: str, values):
        """Add a name and an array, each preceded by its length, so that no two sequences of them
        digest alike."""

        encoded = name.encode()
        array = np.ascontiguousarray(values)
        digest.update(np.int64(len(encoded)).tobytes() + encoded)
        digest.update(np.int64(array.nbytes).tobytes() + array.tobytes())

    mesh = model.mesh
    connectivity = mesh.connectivity()
    add("elements", connectivity.numbers)
    for group in connectivity.groups:
        typeInfo = group.typeInfo
        add(
            "element type {:} {:} {:} {:}".format(
                typeInfo.fields, typeInfo.ensightType, typeInfo.nDof, typeInfo.hasKernels
            ),
            group.positions,
        )
        add("connectivity", group.nodeRows.astype(np.int64))
    for name, numbers in mesh.elementSets.items():
        add("element set " + name, np.asarray(numbers, dtype=np.int64))
    for name in mesh.surfaces:
        for face, numbers in mesh.elementNumbersOfSurface(name).items():
            add("surface {:} face {:}".format(name, face), np.asarray(numbers, dtype=np.int64))

    add("numbering of the degrees of freedom", np.frombuffer(numbering.digest().encode(), dtype=np.uint8))
    add("reference coordinates", np.asarray(mesh.nodeCoordinates, dtype=float))
    # A rigid body moves its surface nodes along with it: their current coordinates, which every process
    # computes from the reference node as it holds it.
    for name, rigidBody in model.rigidBodies.items():
        moved = list(rigidBody.referenceCoordinatesOfMovedNodes())
        add("rigid body " + name, np.asarray(moved, dtype=np.int64))
        add(
            "current coordinates",
            np.asarray([model.nodes[label].coordinates for label in moved], dtype=float).reshape(-1),
        )
    add("size", np.int64(numbering.nDof))

    if mpcTransformation is not None:
        slaves, masters, weights = mpcTransformation.slaveMasterDofWeights()
        add("multi-point constraint slaves", slaves.astype(np.int64))
        add("multi-point constraint masters", masters.astype(np.int64))
        add("multi-point constraint weights", np.asarray(weights, dtype=float))

    return digest.hexdigest()
