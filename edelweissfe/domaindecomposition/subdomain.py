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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.communicator import Communicator
from edelweissfe.domaindecomposition.loadsonsubdomain import (
    BodyLoadOnSubdomain,
    DistributedLoadOnSubdomain,
    LoadsOnSubdomain,
)
from edelweissfe.domaindecomposition.partitioning import (
    assignConstraints,
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
    InterfaceForceAssembly,
    InterfaceLoadAssembly,
    SubdomainInterface,
    ValuesFromOwners,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.solvers.base.modelpartition import ModelPartition
from edelweissfe.solvers.base.parallelelementcomputation import ElementPlan
from edelweissfe.utils.exceptions import TopologyError


@dataclass(frozen=True)
class MeasuredLoad:
    """What one process measured of its load since its increment plan was made, for
    :meth:`Subdomain.rebalance`.

    Parameters
    ----------
    elementCosts
        The cost of every element of the plan, in plan order, summed over ``nIncrements`` increments
        -- the weights of a repartition --, or None if nothing was measured.
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
    """

    elementCosts: np.ndarray | None
    nIncrements: int
    incrementsUntilNextCheck: int | None
    elementTime: float
    constraintTime: float


class Subdomain:
    """The subdomain this MPI process computes, and its exchanges with the others; see the module
    documentation.

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
        #: The measured time per increment of the constraints each process evaluates, by rank, at the
        #: last rebalancing check that measured times; zeros before.
        self._constraintCosts = np.zeros(self.nProcesses)
        #: The share of the element work each process is given beside its constraints, by rank
        #: (:func:`~edelweissfe.domaindecomposition.partitioning.elementSharesBesideConstraints`), as
        #: of the last rebalancing check; None for equal shares.
        self._processShares = None

        #: The rank of every element and of every constraint, and the element keys the partition was
        #: made for.
        self._elementOwners = None
        self._partitionedElementKeys = None
        self._constraintOwners = None

        self._ownedConstraints = {}

        self._interface = None
        self._stateSynchronization = None
        self._constraintForceExchange = None
        #: The position of every element in the model, by number: the order interface forces are
        #: summed in.
        self._elementPositions = {}

        #: What the last definition was made for; a rebalance redefines from it.
        self._model = None
        self._mpcTransformation = None

        self._forgetElements()

    def _forgetElements(self):
        """Hold no element and nothing indexed by the elements: the state before the first
        definition, and the one :meth:`moveElements` starts from, so that no element a process drops
        is kept alive here."""

        self._ownedElements = {}
        #: The loads of the elements computed here, and their assembly, for the loads last asked
        #: for; see loadsOnSubdomain.
        self._loadsOnSubdomain = None
        #: The elements, constraints and degrees of freedom this process computes, as of the last
        #: definition.
        self.partition = None
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
                    self._partition(model, measuredCosts=None, byMeasuredCosts=False)

                self._constraintOwners = self._constraintOwnersOf(model)

                self._adoptOwnership(model)

            self._defineInterface(model, ownershipChanged=topologyChanged)

        self._reportSubdomains(topologyChanged)

    def _partition(self, model: FEModel, measuredCosts: dict | None, byMeasuredCosts: bool):
        """Partition the elements of the mesh, by estimated or by measured cost. Collective.

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
            self._elementOwners = owners
        else:
            # A copy: the distribution changes its own as the mesh changes (see
            # DistributedElements.placeChildElement), and this partition must not change with it.
            self._elementOwners = dict(distribution.owners)
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

        self.partition = ModelPartition(self._ownedElements, self._ownedConstraints, self._interface.subdomainDofs)

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
        """Refuse to continue unless every process built the same mesh and degree-of-freedom layout
        (:func:`layoutFingerprint`). Collective.

        The interface exchange addresses degrees of freedom by index, and the load assembly tags a
        load contribution by its place in the element sets and surfaces of the mesh; both are only
        meaningful if every process numbered and ordered them identically, which after a refinement
        rests on every process having refined identically.

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

        fingerprints = self.communicator.allgather(layoutFingerprint(model, dofManager, self._mpcTransformation))
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

    def loadsOnSubdomain(self, distributedLoads, bodyLoads) -> LoadsOnSubdomain:
        """The given loads, restricted to the elements computed here, with their nodal forces tagged
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
            If a loaded element held here is not where the mesh describes it in its surface or set.
        """

        distributedLoads, bodyLoads = list(distributedLoads), list(bodyLoads)
        if self._loadsOnSubdomain is not None and self._loadsOnSubdomain.isFor(distributedLoads, bodyLoads):
            return self._loadsOnSubdomain

        mesh = self._model.mesh
        dofsOf = self._dofManager.idcsOfElementsInDofVector
        entryDofs, entryOrder = [], []
        loadsBefore = 0

        def computedHere(loadedSetName: str, elements, numbersInMesh: list) -> list:
            """The elements of a loaded set computed here, in set order, with their entries tagged by
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
        """The assembly of the nodal forces of loads restricted to this subdomain at the interface;
        built the first time it is asked for. Collective among neighbours when built: every process
        asks for it after the same :meth:`loadsOnSubdomain`, which is made again in every process at
        the same time.

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
        self.communicator.requireSameOnAllRanks(
            digest.hexdigest(), "the nodes and fields the constraints couple before a topology update"
        )

    @property
    def ownedDofMask(self) -> np.ndarray:
        """Whether each degree of freedom of the model is owned here: True at most at those this
        process integrates, and at each degree of freedom of the model in exactly one process. A
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

    def allgatherOwnedValues(self, vector: DofVector):
        """Make a vector, correct at the degrees of freedom owned here, the same complete vector in
        every process, in place: every entry from its owner. Collective.

        Parameters
        ----------
        vector
            A vector of the whole model.
        """

        self._interface.allgatherOwnedValues(vector)

    def gatherOwnedValuesToRoot(self, vector: DofVector):
        """Make a vector, correct at the degrees of freedom owned here, the complete vector on rank 0,
        in place: every entry from its owner. The other processes keep theirs, complete at the degrees
        of freedom they integrate only. Collective.

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
        """How this process receives the entries of vectors at the given degrees of freedom from their
        owners, point to point (:class:`~.subdomaininterface.ValuesFromOwners`). Collective.

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
        """After the subdomain was defined afresh for the same degree-of-freedom layout -- a contact
        search moved a constraint, or elements moved between the processes -- complete the vectors at
        the degrees of freedom this process integrates now but did not before, from the processes that
        owned them before. Collective.

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
        """Send the forces of the constraints evaluated here to the processes integrating their
        degrees of freedom, and add those of every constraint acting on a degree of freedom
        integrated here into ``P``, in model order
        (:class:`~.subdomaininterface.ConstraintForceExchange`). Collective among neighbours.

        The degrees of freedom are the owner's: another process' copy of a contact constraint
        couples the nodes of its last synchronization. ``P`` holds the constraint forces at the
        degrees of freedom integrated here only.

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
    def synchronizeStates(self, includeElements: bool, toEveryProcess: bool = True):
        """Give every stateful constraint -- and, if asked, every element -- of this process' model
        the state the process computing it last left it in. Collective.

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
        """Gather the element states of every process (:meth:`elementStatesOwnedHere`) to rank 0, for
        the restart checkpoint rank 0 writes. Collective.

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
        """Repartition with the measured costs if the busiest process has fallen more than
        ``loadBalanceTolerance`` behind what a partition can attain, and if that is worth it.
        Collective; only right after an increment was accepted and, where every process holds the
        whole model, every element state synchronized, since an element changing process must arrive
        with its current state.

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
            The plan of the elements whose kernels are computed here.
        load
            What this process measured.

        Returns
        -------
        bool
            Whether the partition changed; then everything derived from :attr:`partition` must be
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
        self._constraintCosts = constraintTimes
        self._processShares = elementSharesBesideConstraints(float(elementTimes.sum()), constraintTimes)
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

        measured = dict(zip(plan.elements.keys(), (costs / nIncrements).tolist()))
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
        """Record what the last repartition cost -- deciding it, partitioning, moving the elements
        and deriving everything again -- for the next :meth:`rebalance` to weigh against its gain:
        the longest of all processes, so that every process decides the same. Collective.

        Parameters
        ----------
        seconds
            The wall time of the repartition in this process.
        """

        self._lastRepartitionCost = self.communicator.allreduce(seconds, op=MPI.MAX)

    def rebalanceAfterTopologyChange(self, model: FEModel, horizon: int):
        """After a model modifier changed the mesh of a distributed model, move elements if the
        partition it inherited -- the children of a refined element are computed where their parent
        was -- is out of balance, and if that pays. Collective; right after the topology update,
        before the equation system is built again, which it is anyway: a migration here costs only
        the moving of the elements.

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

        mesh = model.mesh
        inherited = distribution.owners
        counts = np.zeros(self.nProcesses)
        for number, owner in inherited.items():
            counts[owner] += mesh.typeOf(mesh.elements[number]).hasKernels
        if self._meanElementCost is None:
            imbalance, timeBeyondTolerance = self._imbalance(counts, np.zeros(self.nProcesses))
        else:
            busyTimes = counts * self._meanElementCost + self._constraintCosts
            imbalance, timeBeyondTolerance = self._imbalance(busyTimes, self._constraintCosts)
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
        self._model = model
        self._elementOwners = owners
        self.moveElements()
        # the next definition adopts the partition the elements were moved to
        self._partitionedElementKeys = None

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
        """Move the elements of a distributed model to their processes under the current partition
        (:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.moveElementsTo`),
        and report the migration. Collective.

        The subdomain forgets the elements it computed first, so that a process holds no reference
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
    model: FEModel, dofManager: DofManager, mpcTransformation: MultiPointConstraintTransformation | None
) -> str:
    """A fingerprint of everything the processes of a domain-decomposed run must hold identically: the
    mesh -- the element numbers and the node labels of every element --, the element sets and the
    surfaces of the mesh, each with its elements in order and all of them in order, the degree of
    freedom of every node of every field, the node coordinates, the size of the system, and the
    multi-point constraints -- every slave, master and weight. It is made from the mesh, the nodes and
    the equation system, which every process holds whole, and not from the element objects, of which
    a process may hold only its own.

    Parameters
    ----------
    model
        The model tree.
    dofManager
        The degree-of-freedom layout.
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
    add("elements", np.asarray(list(mesh.elements.keys()), dtype=np.int64))
    add(
        "connectivity",
        np.concatenate(
            [np.asarray(record.nodeLabels, dtype=np.int64) for record in mesh.elements.values()]
            or [np.empty(0, dtype=np.int64)]
        ),
    )
    for name, numbers in mesh.elementSets.items():
        add("element set " + name, np.asarray(numbers, dtype=np.int64))
    for name in mesh.surfaces:
        for face, numbers in mesh.elementNumbersOfSurface(name).items():
            add("surface {:} face {:}".format(name, face), np.asarray(numbers, dtype=np.int64))

    dofsOfFieldVariables = dofManager.idcsOfFieldVariablesInDofVector
    for name, field in model.nodeFields.items():
        add("nodes of field " + name, np.asarray([node.label for node in field.nodes], dtype=np.int64))
        add(
            "degrees of freedom of field " + name,
            np.concatenate(
                [np.atleast_1d(dofsOfFieldVariables[node.fields[name]]) for node in field.nodes]
                or [np.empty(0, dtype=np.int64)]
            ).astype(np.int64),
        )
    add(
        "coordinates",
        np.concatenate([np.asarray(node.coordinates, dtype=float) for node in model.nodes.values()] or [np.empty(0)]),
    )
    add("size", np.int64(dofManager.nDof))

    if mpcTransformation is not None:
        slaves, masters, weights = mpcTransformation.slaveMasterDofWeights()
        add("multi-point constraint slaves", slaves.astype(np.int64))
        add("multi-point constraint masters", masters.astype(np.int64))
        add("multi-point constraint weights", np.asarray(weights, dtype=float))

    return digest.hexdigest()
