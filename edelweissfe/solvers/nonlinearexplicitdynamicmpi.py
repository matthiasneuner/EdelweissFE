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
"""The nonlinear explicit dynamic solver, domain-decomposed over MPI processes.

Started by an MPI launcher, every process reads the same input file and builds the same mesh --
creating either every element, or only those of its own subdomain (see
:mod:`edelweissfe.domaindecomposition.distributedelements`); this solver then has each of them
compute one *subdomain* of it -- the elements METIS assigns to it and the constraints dealt to it --
and integrate the degrees of freedom those touch.
An increment is the increment of :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED`, with
two additions. At the interface between subdomains, each process holds only its own elements'
contributions to the nodal force; the neighbours exchange the individual contributions there, and
each process sums all of them in the order of the elements in the model
(:mod:`edelweissfe.domaindecomposition.subdomaininterface`). And the forces of the constraints,
each evaluated by one process, are shared with all. That is the communication most increments
need, and its volume is the interface and the constraints, not the model.

The lumped mass and the damping are assembled once per mesh, by each process from its own elements,
completed at the interface like the forces and shared from the owners, so that they are the same
bits as without decomposition; the critical time step is the minimum over the subdomains. The
lumped operators of the whole model are therefore known everywhere, which is what lets a contact
search move a constraint onto nodes another process integrated until then.

**Where the whole model is read -- and by whom.** Field outputs, output managers, a marker deciding
a refinement, the refinement itself, and a contact search all read more than one subdomain. Each of
them receives what it reads, and only the processes reading it receive it:

* on an output increment, rank 0 -- which writes the output and the checkpoints -- receives the
  current solution at every degree of freedom, the state of every stateful constraint and the
  results of every element field output; the other processes keep theirs, current at the degrees of
  freedom they integrate (:meth:`NEDMPI.acceptIncrement`, :meth:`NEDMPI.writeIncrementOutput`);
* on an output increment the topology check follows, and at the end of a step, every process
  receives all of it, and the field outputs the markers read (:meth:`NEDMPI.wholeModelReadEverywhereNext`):
  every process refines the same mesh, interpolates the node fields onto it and builds its equation
  system from them;
* before a periodic contact search, the process evaluating the constraint receives the positions of
  the nodes the search reads, point to point, surface-sized
  (:meth:`NEDMPI.updateConstraintConnectivity`);
* after a contact search or a repartition changed which degrees of freedom a process integrates, it
  receives the solution, the velocity and the force at those it newly integrates from their previous
  owners (:meth:`NEDMPI.buildEquationSystem`).

What a process did not receive it does not read: reading the whole solution where it is not current
(:meth:`NEDMPI.requireWholeSolutionHere`), a field output gathered to rank 0 only
(:meth:`~edelweissfe.utils.fieldoutput.FieldOutputController.gatherResultsOfWholeSet`), or a constraint
copy not synchronized before a topology update, raises. Where every process holds the whole model,
every element state is received by every process on every output increment, since a repartition may
give any element to any process (:mod:`edelweissfe.domaindecomposition.statesynchronization`). A
distributed model synchronizes no element states: its element field outputs gather their results,
and its checkpoints the element states, from the processes computing them.
A contact search itself -- at a contact update and at a topology check -- runs on the process that
evaluates the constraint only, since nothing but that evaluation reads its outcome.

**Adaptive refinement.** Every process holds the whole mesh and the current solution at a topology
check, so every process runs the same refinement on the same data and arrives at the same refined
mesh -- checked, not assumed: the degree-of-freedom layout is compared across all processes after
every build. The children of a refined element are created by the process computing it, which
transfers the parent's state to them; the subdomains are then rebalanced if that pays.

**Contact and ties.** A constraint reads the contact facets of its surfaces, the nodes and the rigid
bodies, which every process holds whole -- the facets and the point masses of rigid bodies are made
by every process -- so a model with contact or ties is distributed like any other; each constraint
is evaluated by one process.

**Output.** Only rank 0 creates output managers and writes files; the others are silent. A restart
checkpoint is written after the output synchronization, so the copy of the model rank 0 writes it
from holds every element and constraint as its owner left it -- a distributed model gathers the
element states to rank 0 for it (:meth:`NEDMPI.writeIncrementOutput`): it is an ordinary checkpoint of the
whole model, and can be resumed by this solver on any number of processes, or by the serial one.
Every process resumes from it, and so starts from the same model. A conditional stop decided by an
output manager on rank 0 stops every process.

**Results do not depend on the decomposition.** Every sum that decides the solution is formed in the
order it is formed without decomposition: the element contributions at a node in model order, the
loads and the constraint forces after them in deck order. A run on any number of processes is
therefore bit-identical to :class:`NED` (and :class:`NEDParallel`) on the same input -- through
contact searches, refinements and repartitions -- and the load balancing below, whose partition
depends on measured timings, changes the speed of a run and never its result. The external work is
summed exactly (:meth:`gatherExternalWork`), so it, too, and the checkpoints recording it,
are bit-identical. Only the kinetic and internal energy of the energy table are formed per
subdomain and then added, and may differ from a serial run's in their last digits; they enter
nothing but the table.

**Load balancing.** The first partition weighs an element by its number of degrees of freedom. A
softening material costs more where it softens, so every element kernel is timed, and on an output
increment the model is repartitioned with the measured costs whenever the slowest process falls
more than ``load-balance-tolerance`` behind the mean. A distributed model then moves the elements
whose process changes (migration: the new process creates them from the mesh and receives their
state, the old one drops them; see
:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.moveElementsTo`), and
every process builds its equation system again for the elements it now holds, carrying the solution
over. ``load-balance-costs=elementNumber`` replaces the timings by the element numbers -- a
deterministic, uneven cost, for tests.

**What this solver changes.** The increment of ``NED`` runs over the
:class:`~edelweissfe.solvers.base.modelpartition.ModelPartition` of this process' subdomain, which a
:class:`~edelweissfe.domaindecomposition.subdomain.Subdomain` defines. What this solver adds is the
communication, each in an override of a method of ``NED``:

* :meth:`NEDMPI.assembleInternalForces` completes the element forces at the interface;
* :meth:`NEDMPI.assembleLumpedDiagonal` completes the lumped inertia and damping the same way, and
  shares them from the owners;
* :meth:`NEDMPI.assembleConstraintForces` sends the constraint forces to the processes integrating
  their degrees of freedom;
* :meth:`NEDMPI.assembleLoads` evaluates the loads of the elements computed here, and completes
  them at the interface like the forces;
* :meth:`NEDMPI.getCriticalTimeStepForExplicitDynamics` and :meth:`NEDMPI.energyBalanceTerms` form
  the minimum and the sums over all processes;
* :meth:`NEDMPI.acceptIncrement` synchronizes the model on output increments and rebalances it,
  rebuilding the equation system if elements moved,
  :meth:`NEDMPI.updateConstraintConnectivity` synchronizes it before a contact search,
  :meth:`NEDMPI.applyStepActionsAtStepEnd` at the end of a step;
* :meth:`NEDMPI.updateConnectivityOf` runs a contact search on the constraint's process only;
* and every step that can fail in one process alone -- the element and constraint evaluation, a
  contact search, a topology update, writing the output -- fails on all ranks together
  (:meth:`~edelweissfe.domaindecomposition.communicator.Communicator.allRanksFailTogether`).

**Limits of this prototype.** Every process holds the whole mesh, every node, every contact facet
and global-length vectors; a model under the whole-model rule
(:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsToReplicateElements`) holds every
element in every process, so that its memory per process does not shrink with the number of
processes. The topology of a refinement costs every process what it costs a serial run. Constraints
are evaluated whole, each by one process, while the others wait. An exception outside the steps
that fail on all ranks together, and an interrupt of any process, abort all of them.

Run with, for example::

    mpirun -n 8 edelweissfe input.inp

with ``*solver, solver=NEDMPI, name=...`` in the deck. ``OMP_NUM_THREADS`` sets the threads of each
process' element loop, as for ``NEDParallel``.
"""

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from mpi4py import MPI

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.communicator import Communicator
from edelweissfe.domaindecomposition.mpienvironment import (
    StepFailedOnAllRanks,
    abortAllProcesses,
    worldCommunicator,
)
from edelweissfe.domaindecomposition.subdomain import Subdomain
from edelweissfe.domaindecomposition.subdomaininterface import (
    InterfaceForceAssembly,
    ValuesFromOwners,
)
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.numerics.parallelizationutilities import getNumberOfThreads
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.modelpartition import ModelPartition
from edelweissfe.solvers.base.parallelelementcomputation import (
    ElementPlan,
    computeElementsForExplicit,
    computeLumpedDiagonalForExplicit,
)
from edelweissfe.solvers.nonlinearexplicitdynamic import (
    NED,
    ExplicitSystem,
    IncrementPlan,
    NEDSchema,
)
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class NEDMPISchema(NEDSchema):
    """The options of :class:`NEDMPI`: those of :class:`NEDSchema`, and the load balancing."""

    loadBalanceTolerance: float | None = schemaField(
        description=(
            "How far the slowest process may fall behind the mean, as a fraction, before the model is "
            "repartitioned with the measured element costs as weights. Checked on every output "
            "increment. The first partition can only estimate an element's cost -- by its number of "
            "degrees of freedom -- and a softening material costs more where it softens, so a partition "
            "balanced at the start drifts out of balance as damage localizes. 0 disables it."
        ),
        dtype=float,
        default=0.1,
        optionName="load-balance-tolerance",
    )
    loadBalanceCosts: str | None = schemaField(
        description=(
            "What an element is taken to cost when the model is repartitioned: 'measured', the measured "
            "kernel time; or 'elementNumber', its element number -- a deterministic, deliberately uneven cost, "
            "for tests that must know when the model is repartitioned and that elements move."
        ),
        dtype=str,
        default="measured",
        optionName="load-balance-costs",
    )


@dataclass(frozen=True)
class NodeFieldSlot:
    """Where degrees of freedom of a vector are published in a node field.

    Parameters
    ----------
    nodeField
        The node field.
    positions
        The positions in the node field's flattened values, as an index.
    dofs
        The degrees of freedom published there, in the same order, as an index into a vector.
    """

    nodeField: NodeField
    positions: slice | np.ndarray
    dofs: slice | np.ndarray


@dataclass(frozen=True)
class SolutionReadBySearches:
    """The part of the solution the contact searches of this process read and it does not integrate
    itself: received from the owners before every periodic search, and published into the node
    fields there.

    Parameters
    ----------
    received
        The exchange with the owners.
    slots
        Where the received degrees of freedom are published in the node fields.
    """

    received: ValuesFromOwners
    slots: list[NodeFieldSlot]


class NEDMPI(NEDParallel):
    """The nonlinear explicit dynamic solver, domain-decomposed over MPI processes.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDMPISolver"

    supportsDomainDecomposition = True

    schema = NEDMPISchema

    SolverSpecificOptions = NED.SolverSpecificOptions | {
        "load-balance-tolerance": 0.1,
        "load-balance-costs": "measured",
    }

    def __init__(self, jobInfo, journal, **kwargs):
        super().__init__(jobInfo, journal, **kwargs)

        communicator = worldCommunicator()
        #: The communicator of all processes of the launcher -- started without a launcher, of this
        #: process alone: the same code path, with every exchange a copy. The processes agree through
        #: it, e.g. on a failure in any of them (:meth:`Communicator.allRanksFailTogether`).
        self.communicator = Communicator(MPI.COMM_SELF) if communicator is None else communicator
        #: The subdomain this process computes.
        self.subdomain = Subdomain(
            self.communicator,
            journal,
            self.identification,
            self.options["load-balance-tolerance"],
        )
        #: How the element forces of the current increment plan are completed at the interface.
        self._interfaceAssembly: InterfaceForceAssembly | None = None
        #: The kernel time of every element of the current increment plan, in plan order, summed
        #: over :attr:`_nMeasuredIncrements` increments, if the subdomain balances load by measured
        #: costs; else None.
        self._elementCosts: np.ndarray | None = None
        #: The increments computed with the current increment plan.
        self._nMeasuredIncrements = 0
        #: The increments of the step done at the last topology change (or 0, at the start of the
        #: step), for the horizon a migration after a topology change is weighed over.
        self._incrementsDoneAtLastTopologyChange = 0
        #: Where the degrees of freedom integrated here are published in the node fields; see
        #: :meth:`publishNodeFields`.
        self._integratedNodeFieldSlots: list[NodeFieldSlot] = []
        #: The work at the prescribed degrees of freedom owned here of every increment since the
        #: external work was last read, in increment order: one array of products per increment,
        #: added to the external work by :meth:`gatherExternalWork`.
        self._pendingWorkAtPrescribedDofs: list[np.ndarray] = []
        #: Whether this process holds the whole solution -- ``U``, ``V`` and ``P`` at every degree of
        #: freedom, and published into the node fields -- as of the last accepted increment; else
        #: only at the degrees of freedom it integrates. See :meth:`requireWholeSolutionHere`.
        self._wholeSolutionHere = True
        #: What the periodic contact searches of this process read, planned for the current
        #: subdomain at the first search: :meth:`_planSolutionReadBySearches`; None before.
        self._solutionReadBySearches: SolutionReadBySearches | None = None
        #: Whether the periodic contact searches read the whole solution, because a searched
        #: constraint does not name the nodes it reads; decided with the plan above.
        self._searchesReadWholeSolution = False
        self._searchesPlanned = False

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: list[OutputManagerBase],
    ):
        """Report the decomposition, then start the step; see :meth:`NED.beginStep`.

        Parameters
        ----------
        step
            The step to solve.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers; on processes other than rank 0, none.
        """

        self.journal.message(
            "Domain decomposition over {:} MPI process(es), {:} thread(s) each".format(
                self.subdomain.nProcesses, getNumberOfThreads()
            ),
            self.identification,
            0,
        )
        self._incrementsDoneAtLastTopologyChange = 0
        self._pendingWorkAtPrescribedDofs = []
        return super().beginStep(step, model, fieldOutputController, outputmanagers)

    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Attempt the increment of :meth:`NED.attemptIncrement`; a step failure it raises is raised
        as having failed on all ranks together. Collective.

        Every :class:`~edelweissfe.utils.exceptions.StepFailed` an increment of ``NED`` raises is
        decided from values every process holds alike, so every process raises it in the same
        increment: the refused cutback, from a cutback request the processes agreed on
        (:meth:`~edelweissfe.domaindecomposition.communicator.Communicator.allRanksFailTogether`),
        and the diverged energy balance, from the terms summed over all processes
        (:meth:`energyBalanceTerms`). Raised as
        :class:`~edelweissfe.domaindecomposition.mpienvironment.StepFailedOnAllRanks`, they end the
        job as a failed step ends a serial one, instead of aborting it (:meth:`endStep`). A failure
        a process may meet alone must be raised through the agreement, not here.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.

        Raises
        ------
        StepFailedOnAllRanks
            If the increment failed.
        """

        try:
            super().attemptIncrement(step, model, timeStep)
        except StepFailedOnAllRanks:
            raise
        except StepFailed as failure:
            raise StepFailedOnAllRanks(str(failure)) from failure

    def endStep(self, step, model: FEModel, failure: BaseException | None = None):
        """Finish the step; see :meth:`NED.endStep`. A step that failed without the processes agreeing
        on it -- in this process alone, perhaps, while the others wait for it in their next exchange
        -- stops every process right here: unwinding further, this process would gather the results
        of the end of the step with processes that never arrive.

        Parameters
        ----------
        step
            The step that was solved.
        model
            The model tree.
        failure
            How the step failed, or None.
        """

        super().endStep(step, model, failure)
        if failure is not None and not isinstance(failure, StepFailedOnAllRanks):
            abortAllProcesses(
                "the step failed without the agreement of the other processes ({:}: {:})".format(
                    type(failure).__name__, failure
                )
            )

    # --- The subdomain ------------------------------------------------------------------------------

    def partitionModel(self, model: FEModel, topologyChanged: bool) -> ModelPartition:
        """The subdomain of this process; see :meth:`NED.partitionModel`. Collective.

        Parameters
        ----------
        model
            The model tree.
        topologyChanged
            As for :meth:`NED.partitionModel`.

        Returns
        -------
        ModelPartition
            The partition.
        """

        self.subdomain.define(model, self.theDofManager, self.mpcTransformation, topologyChanged)
        return self.subdomain.partition

    def planIncrement(self, model: FEModel) -> IncrementPlan:
        """The increment plan of :meth:`NED.planIncrement`, and how its element forces are completed
        at the interface. Collective.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        IncrementPlan
            The plan.
        """

        plan = super().planIncrement(model)
        self._integratedNodeFieldSlots = self._nodeFieldSlotsOf(model, self.partition.dofs)
        # planned again for this subdomain at the next search
        self._searchesPlanned = False
        self._solutionReadBySearches = None
        self._interfaceAssembly = self.subdomain.interfaceAssemblyFor(plan.elementPlan)
        timesKernels = self.subdomain.measuresElementCosts() and self._loadBalanceCosts() == "measured"
        self._elementCosts = np.zeros(len(plan.elementPlan.elements)) if timesKernels else None
        self._nMeasuredIncrements = 0
        return plan

    def _loadBalanceCosts(self) -> str:
        """The ``load-balance-costs`` option, checked.

        Returns
        -------
        str
            'measured' or 'elementNumber'.

        Raises
        ------
        ValueError
            For any other value.
        """

        costs = self.options["load-balance-costs"]
        if costs not in ("measured", "elementNumber"):
            raise ValueError("load-balance-costs must be 'measured' or 'elementNumber', not '{:}'".format(costs))
        return costs

    def _elementCostsForRebalancing(self) -> tuple[np.ndarray | None, int, int | None]:
        """What :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.rebalance` weighs: the
        cost of every element of the increment plan, in plan order -- the measured kernel times, or
        the element numbers (``load-balance-costs``) -- the number of increments it was summed over,
        and the increments until the next check, to weigh a repartition's gain against its cost
        (None for element numbers, which are no times).

        Returns
        -------
        tuple[np.ndarray | None, int, int | None]
            The costs, None if nothing was measured; the number of increments; the increments until
            the next check, or None.
        """

        if self._loadBalanceCosts() == "elementNumber" and self.subdomain.measuresElementCosts():
            numbers = np.array(list(self._incrementPlan.elementPlan.elements.keys()), dtype=float)
            return numbers * self._nMeasuredIncrements, self._nMeasuredIncrements, None
        return self._elementCosts, self._nMeasuredIncrements, self.options["output-frequency"]

    def assembleLumpedDiagonal(self, plan: ElementPlan, elementContribution) -> DofVector:
        """Assemble a lumped operator of the elements computed here, complete at every degree of
        freedom of the model: completed at the interface like the forces, and shared from the
        owners. Collective.

        Parameters
        ----------
        plan
            The plan of the elements computed here.
        elementContribution
            As for :meth:`NED.assembleLumpedDiagonal`.

        Returns
        -------
        DofVector
            The assembled diagonal.
        """

        vector = self.theDofManager.constructDofVector()
        vector[:] = 0.0
        contributions = computeLumpedDiagonalForExplicit(plan, elementContribution, vector)
        with performancetiming.timeit("interface forces"):
            self.subdomain.interfaceAssemblyFor(plan).assemble(contributions, vector)
            self.subdomain.allgatherOwnedValues(vector)
        return vector

    # --- The increment ------------------------------------------------------------------------------

    @performancetiming.timeit("elements")
    def assembleInternalForces(
        self, U_np: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
    ) -> tuple[DofVector, float]:
        """Evaluate the elements of the subdomain, failing on all ranks together if it fails on one, and complete their
        internal force at the interface; see :meth:`NED.assembleInternalForces`. Collective.

        Parameters
        ----------
        U_np
            The current solution vector.
        dU
            The solution increment vector.
        P
            The internal force vector; overwritten.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector, float]
            The internal force vector, and the internal energy the elements of the subdomain report.
        """

        P[:] = 0.0
        with self.communicator.allRanksFailTogether("Evaluating the elements"):
            psi, contributions = computeElementsForExplicit(
                self._incrementPlan.elementPlan, U_np, dU, P, timeStep, self._elementCosts
            )
        self._nMeasuredIncrements += 1

        # At a degree of freedom shared with another subdomain, the contributions of that
        # subdomain's elements are still missing.
        with performancetiming.timeit("interface forces"):
            self._interfaceAssembly.assemble(contributions, P)

        return P, psi

    @performancetiming.timeit("assemble loads")
    def assembleLoads(
        self,
        nodeForces: list[StepActionBase],
        distributedLoads: list[StepActionBase],
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        K: None,
        timeStep: TimeStep,
    ) -> tuple[DofVector, None]:
        """Assemble the loads of :meth:`NED.assembleLoads` in the same order, those acting on
        elements as the element forces are: each process evaluates the loads of the elements it
        computes, failing on all ranks together if it fails on one, and the contributions are completed at the interface
        and added in the order of the loads of the model; see
        :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.loadsOnSubdomain`. Collective.

        Parameters
        ----------
        nodeForces
            The concentrated (nodal) loads.
        distributedLoads
            The distributed (surface) loads.
        bodyForces
            The body loads.
        U_np
            The current solution vector.
        PExt
            The external load vector.
        K
            None: an explicit solver assembles no system matrix.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector, None]
            The updated external load vector, and ``K``.
        """

        # A concentrated load acts on degrees of freedom, not on elements: every process integrating
        # one adds it, onto the same complete net force.
        PExt = self.computeNodeForces(nodeForces, PExt, timeStep)
        if not distributedLoads and not bodyForces:
            return PExt, K

        # A load kernel -- or a load not where the mesh describes it -- may fail in one process alone,
        # which would leave its neighbours waiting in the exchange below.
        with self.communicator.allRanksFailTogether("Evaluating the loads"):
            loads = self.subdomain.loadsOnSubdomain(distributedLoads, bodyForces)
            with performancetiming.timeit("distributed loads"):
                forces = [Pe for _, Pe in self.distributedLoadsOfElements(loads.distributedLoads, U_np, K, timeStep)]
            with performancetiming.timeit("body forces"):
                forces += [Pe for _, Pe in self.bodyForcesOfElements(loads.bodyLoads, U_np, K, timeStep)]

        with performancetiming.timeit("interface loads"):
            self.subdomain.loadAssemblyFor(loads).assemble(np.concatenate(forces) if forces else np.empty(0), PExt)
        return PExt, K

    @performancetiming.timeit("assemble constraints")
    def assembleConstraintForces(
        self, constraints: dict, U_np: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
    ) -> DofVector:
        """Evaluate the constraints of the subdomain, failing on all ranks together if it fails on one, send their forces
        to the processes integrating their degrees of freedom, and add those of every constraint at the degrees of
        freedom integrated here, in model order; see :meth:`NED.assembleConstraintForces` and
        :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.addConstraintForces`. Collective.

        Parameters
        ----------
        constraints
            The constraints evaluated here, by name, in model order.
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The net force vector to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The augmented net force vector.
        """

        forces = {}
        with self.communicator.allRanksFailTogether("Evaluating the constraints"):
            for name, constraint in constraints.items():
                forces[name] = self._evaluateConstraintForce(name, constraint, U_np, dU, P, timeStep)

        self.subdomain.addConstraintForces(forces, P)
        return P

    def getCriticalTimeStepForExplicitDynamics(self, model: FEModel, U: DofVector) -> float:
        """The smallest critical time step of all subdomains; see
        :meth:`NED.getCriticalTimeStepForExplicitDynamics`. Collective.

        Parameters
        ----------
        model
            The model tree.
        U
            The solution vector.

        Returns
        -------
        float
            The critical time step.
        """

        return self.communicator.allreduceMin(super().getCriticalTimeStepForExplicitDynamics(model, U))

    def addExternalWork(self, dofs: np.ndarray, reactionTimesIncrement: np.ndarray):
        """Keep the work of this increment at the prescribed degrees of freedom owned here -- a degree
        of freedom shared by two subdomains is counted once, by its owner -- until the external work
        is next read (:meth:`gatherExternalWork`), so that the products need not be gathered every
        increment; see :meth:`NED.addExternalWork`.

        Parameters
        ----------
        dofs
            The prescribed degrees of freedom whose work is an energy.
        reactionTimesIncrement
            Per degree of freedom, the net nodal force times the prescribed increment.
        """

        self._pendingWorkAtPrescribedDofs.append(reactionTimesIncrement[self.subdomain.ownedDofMask[dofs]])

    def gatherExternalWork(self):
        """Add the work of every increment since the last call to the external work, increment by
        increment, in increment order: the products at the degrees of freedom each process owns are
        gathered to every process -- once for all those increments -- and each increment's are summed
        exactly (:func:`math.fsum`), as :meth:`NED.addExternalWork` sums them in a serial run. So the
        external work is the same, bit for bit, on any number of processes, and every process holds
        that of the whole model. Called where it is read: the energy balance, an output increment (and
        a checkpoint written with it), the end of a step. Collective.
        """

        gathered = self.communicator.allgather(self._pendingWorkAtPrescribedDofs)
        for increment in range(len(self._pendingWorkAtPrescribedDofs)):
            self._externalWork -= math.fsum(np.concatenate([ofProcess[increment] for ofProcess in gathered]).tolist())
        self._pendingWorkAtPrescribedDofs = []

    def halfMassTimesSquaredRate(self, indices: slice | np.ndarray, V: DofVector) -> float:
        """The sum of :meth:`NED.halfMassTimesSquaredRate` over the degrees of freedom owned here, so
        that the sums of all subdomains add up to that of the model.

        Parameters
        ----------
        indices
            The degrees of freedom.
        V
            The velocity vector.

        Returns
        -------
        float
            The sum over those owned here.
        """

        indices = np.arange(V.shape[0])[indices]
        return super().halfMassTimesSquaredRate(indices[self.subdomain.ownedDofMask[indices]], V)

    def energyBalanceTerms(self, psi: float, V: DofVector) -> tuple[float, float, float, list[float]]:
        """The terms of :meth:`NED.energyBalanceTerms` of the whole model. Collective.

        The internal and kinetic energies are formed per subdomain and then added in rank order, so
        they may differ from a serial run's in their last digits; they enter nothing but the energy
        table. The external work is that of the whole model (:meth:`gatherExternalWork`).

        Parameters
        ----------
        psi
            The internal energy the elements of the subdomain report.
        V
            The velocity vector.

        Returns
        -------
        tuple[float, float, float, list[float]]
            As for :meth:`NED.energyBalanceTerms`.
        """

        self.gatherExternalWork()
        Wint, Wkin, Wext, nonMechanical = super().energyBalanceTerms(psi, V)
        Wint, Wkin, *nonMechanical = self.communicator.allreduceSum([Wint, Wkin] + nonMechanical)
        return Wint, Wkin, Wext, nonMechanical

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Commit the increment; see :meth:`NED.acceptIncrement`. On an output increment, then make
        the whole model current where it is read (:meth:`wholeModelReadEverywhereNext`) -- on rank 0,
        which writes the output and a checkpoint with it, or, if the topology check at the start of
        the next increment follows, in every process -- and rebalance the subdomains; if that moved
        elements between processes, build the equation system again for the elements now held here,
        and if it changed only the subdomains, complete the vectors at the degrees of freedom a
        process integrates from now on. Collective.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.
        """

        super().acceptIncrement(step, model, timeStep)
        # the increment changed the solution at the degrees of freedom integrated here only
        self._wholeSolutionHere = False

        if self.isOutputIncrement(timeStep):
            # the output and a checkpoint written after it read the external work
            self.gatherExternalWork()
            self._synchronizeModel(
                model, timeStep, includeStates=True, toEveryProcess=self.wholeModelReadEverywhereNext()
            )

            # Right after every element state was synchronized -- or, where each process holds
            # only its own elements, accepted by the process computing it -- because an element
            # computed by another process from now on must arrive there with its current state.
            startOfRebalancing = perf_counter()
            interfaceBefore = self.subdomain.interface
            if self.subdomain.rebalance(self._incrementPlan.elementPlan, *self._elementCostsForRebalancing()):
                if self.subdomain.elementsMustMove():
                    self._moveElements(model, step)
                else:
                    self.partition = self.subdomain.partition
                    self._incrementPlan = self.planIncrement(model)
                    self.subdomain.receiveNewlyIntegratedValues(interfaceBefore, [self._U, self._V, self._P])
                self.subdomain.recordRepartitionCost(perf_counter() - startOfRebalancing)

    def wholeModelReadEverywhereNext(self) -> bool:
        """Whether every process reads the whole model before the next increment, and so the output
        synchronization of the increment just accepted makes it current in every process, rather
        than on rank 0 only.

        Rank 0 writes the output and the checkpoints, and so reads the whole model on every output
        increment. The other processes read it only where the topology check follows at the start
        of the next increment (:meth:`NED.topologyCheckDueAfter`): the refinement is replicated, and
        every process interpolates the node fields onto the refined mesh and builds its equation
        system from them, and checks that every copy of a constraint is its owner's
        (:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.requireConstraintCopiesCurrent`).

        Returns
        -------
        bool
            True if every process receives the whole model.
        """

        return self.topologyCheckDueAfter(self.prevTimeStep)

    def fieldOutputsReadEverywhereNext(self) -> set[str] | None:
        """The field outputs every process reads before the next increment, and so receives -- the
        others are gathered to rank 0 only, where the output is written: those the markers of the
        model modifiers read (:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.fieldOutputsRead`),
        if the topology check follows (:meth:`wholeModelReadEverywhereNext`); else none.

        Returns
        -------
        set[str] | None
            The names of the field outputs; None for all of them, if a modifier does not name those
            it reads.
        """

        if not self.wholeModelReadEverywhereNext():
            return set()
        readByModifiers = [modifier.fieldOutputsRead() for modifier in self._liveTopologyModifiers]
        if any(names is None for names in readByModifiers):
            return None
        return {name for names in readByModifiers for name in names}

    def advanceModelToTime(self, model: FEModel, time: float):
        """Let the elements and the constraints of this subdomain accept the state the increment
        computed (:meth:`~edelweissfe.models.femodel.FEModel.acceptStatesOf`), and advance the model
        to the end of the increment; see :meth:`NED.advanceModelToTime`. Only those: the others were
        computed by other processes, and accepting them here would overwrite the state they were last
        synchronized to with a stale trial state (a frictional contact, for example, promotes its
        current tangential force to its history).

        Parameters
        ----------
        model
            The model tree.
        time
            The time at the end of the increment.
        """

        model.time = time
        model.acceptStatesOf(self.partition.elements, self.partition.constraints)

    def _moveElements(self, model: FEModel, step):
        """Move elements between the processes, after a rebalance changed the partition of a
        distributed model, and build the equation system again for the elements now held here.
        Collective.

        Everything indexed by the elements held here -- the equation system
        (:meth:`releaseEquationSystem`) and the subdomain -- is released before the elements move,
        so that no element a process drops stays alive, and built again afterwards, as after a
        change of the topology. The solution, the velocity and the net force,
        just made complete in every process by the output synchronization, are carried over.

        The lumped operators are assembled again, from the elements now held here, and must be the
        same bits as before (:meth:`_requireSameLumpedOperators`): a serial run keeps them between
        rebuilds of its equation system, and has no rebuild where a migration happens, so a run that
        moves elements is bit-identical to it only if they are.

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved.
        """

        before = self._lumpedOperators()
        carried = self.releaseEquationSystem()
        self.subdomain.moveElements()
        self._buildSystem(self.buildEquationSystem(model, step, previous=carried))
        self._requireSameLumpedOperators(before)

    def _lumpedOperators(self) -> dict[str, np.ndarray]:
        """Plain copies of the lumped operators of the current equation system, complete at every
        degree of freedom of the model.

        Returns
        -------
        dict[str, np.ndarray]
            The vector the increment divides by, its inverse, the unfolded mass and the damping
            rate, by name.
        """

        return {
            "lumped mass": np.array(self._lumpedMass),
            "inverse lumped mass": np.array(self._Minv),
            "unfolded lumped mass": np.array(self._rawLumpedMass),
            "damping rate": np.array(self._dampingRate),
        }

    def _requireSameLumpedOperators(self, before: dict[str, np.ndarray]):
        """Refuse to continue unless the lumped operators are the same bits as ``before``.

        Elements moved between processes, and the equation system was built again for the elements
        now held here, which assembles the lumped operators again. A serial run keeps them between
        rebuilds of its equation system, and has none here: the two agree only if an element's lumped inertia and damping do not
        depend on its state. That holds for the elements of this package, and is checked here rather
        than assumed, at every migration. The operators are complete in every process, so every
        process decides the same.

        Parameters
        ----------
        before
            The operators before the elements moved (:meth:`_lumpedOperators`).

        Raises
        ------
        RuntimeError
            If any of them changed in any bit.
        """

        after = self._lumpedOperators()
        changed = [name for name in before if before[name].tobytes() != after[name].tobytes()]
        if changed:
            raise RuntimeError(
                "Elements moved between processes, and the lumped operators assembled again from the elements "
                "now held here differ from those before: {:}. An element's lumped inertia or damping depends "
                "on its state, so a run that moves elements would no longer be the serial run, which keeps "
                "them between rebuilds of its equation system. Run without load balancing (load-balance-tolerance=0).".format(
                    ", ".join(changed)
                )
            )

    def releaseEquationSystem(self) -> ExplicitSystem:
        """Release the equation system and everything built with it -- the counterpart of
        :meth:`NED.buildEquationSystem`, and of :meth:`planIncrement` -- keeping only plain copies of
        the solution, the velocity and the net force, and the critical time step.

        Everything released is indexed by the elements and constraints the system was built for:
        the DofManager (whose entity indices hold the elements themselves), every vector carrying
        those indices, the lumped operators, the partition, the increment plan, the interface
        assembly and the element timing. Before elements move to another process, so that nothing
        keeps a dropped element alive; the system is then built again, with the returned system as
        ``previous``.

        Returns
        -------
        ExplicitSystem
            The carried state: plain copies of ``U``, ``V`` and ``P``, and the critical time step;
            no ``Minv`` and no ``dU``.
        """

        carried = ExplicitSystem(
            Minv=None,
            U=np.array(self._system.U),
            dU=None,
            V=np.array(self._system.V),
            P=np.array(self._system.P),
            criticalTimeStep=self._system.criticalTimeStep,
        )
        self._system = self._Minv = self._U = self._dU = self._V = self._P = None
        self._lumpedMass = self._rawLumpedMass = self._dampingRate = None
        self._secondOrderMask = self._halfDampingRate = None
        self._reusableOperators = None
        self._constraintForces = {}
        self.theDofManager = None
        self.partition = None
        self._incrementPlan = None
        self._integratedNodeFieldSlots = []
        self._interfaceAssembly = None
        self._elementCosts = None
        self._searchesPlanned = False
        self._solutionReadBySearches = None
        return carried

    def writeIncrementOutput(self, fieldOutputController: FieldOutputController, outputManagers: list):
        """Write the output of an accepted increment, in three parts, so that no process communicates
        where another may have failed before. Collective.

        1. Each process reads its part of the output: the results of the elements it owns for the
           field outputs, and -- if a checkpoint is written now, which only rank 0, holding the output
           managers, knows, and each process holds only its own elements -- their states; failing on
           all ranks together if it fails on one.
        2. The parts are gathered: the results of every element set to rank 0 -- and to every
           process those the markers of a topology check following now read
           (:meth:`fieldOutputsReadEverywhereNext`) --, the element states to rank 0.
        3. The field outputs store the gathered results -- a field output not gathered to a process
           records there that its result is held on rank 0
           (:meth:`~edelweissfe.utils.fieldoutput.FieldOutputController.gatherResultsOfWholeSet`) --,
           and the output managers of rank 0 write --
           a checkpoint from the gathered states
           (:meth:`~edelweissfe.models.femodel.FEModel.elementStatesFromElsewhere`) --, failing on all
           ranks together if it fails on one: a conditional stop is decided by an output manager of
           rank 0, and a failure to write may happen in one process only.

        Nothing communicates in parts 1 and 3; the communicator refuses it
        (:meth:`~edelweissfe.domaindecomposition.communicator.Communicator.allRanksFailTogether`).

        Parameters
        ----------
        fieldOutputController
            The field output controller.
        outputManagers
            The output managers, restart checkpoint writers last; on processes other than rank 0,
            none.
        """

        model = fieldOutputController.model
        with performancetiming.timeit("finalize output"):
            # Where every process holds every element, rank 0 holds their states: nothing to gather.
            gathersStates = not model.elementDistribution.replicatesElements and self.communicator.bcast(
                any(manager.writesCheckpointAtNextIncrement() for manager in outputManagers), root=0
            )

            with self.communicator.allRanksFailTogether("Reading the output"):
                fieldOutputController.readResultsHere()
                statesHere = self.subdomain.elementStatesOwnedHere(model) if gathersStates else None

            fieldOutputController.gatherResultsOfWholeSet(
                readOnEveryProcess=self.fieldOutputsReadEverywhereNext(), writesOutput=self.subdomain.rank == 0
            )
            with performancetiming.timeit("gather states"):
                states = self.subdomain.gatherElementStatesToRoot(statesHere) if gathersStates else None

            with model.elementStatesFromElsewhere(states), self.communicator.allRanksFailTogether("Writing the output"):
                super().writeIncrementOutput(fieldOutputController, outputManagers)

    def applyStepActionsAtStepEnd(self, model: FEModel, stepActions: dict[str, StepActionBase]):
        """Make the whole model current in every process -- the end of a step, and the start of the
        next, read all of it -- then let the step actions finish the step. Collective.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The step actions, by type.
        """

        if self._system is not None:
            self._synchronizeModel(model, self.prevTimeStep, includeStates=True, toEveryProcess=True)
        # the end of a step reads the external work, and so does the output written after it
        self.gatherExternalWork()
        super().applyStepActionsAtStepEnd(model, stepActions)

    @performancetiming.timeit("publish node fields")
    def publishNodeFields(self, model: FEModel, U: DofVector, V: DofVector, P: DofVector):
        """Publish the degrees of freedom integrated here; see :meth:`NED.publishNodeFields`. The
        others are current only in the processes integrating them, and are published by
        :meth:`_synchronizeModel` in the processes reading them -- rank 0 on an output increment,
        every process before a topology check -- and by :meth:`_receiveSolutionReadBySearches` at the
        nodes a contact search reads.

        Parameters
        ----------
        model
            The model tree.
        U
            The solution vector.
        V
            The velocity vector.
        P
            The net force vector.
        """

        self._publishIntoNodeFields(self._integratedNodeFieldSlots, ((U, "U"), (P, "P"), (V, "V")))

        for variable in model.scalarVariables.values():
            variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

    @staticmethod
    def _publishIntoNodeFields(slots: list[NodeFieldSlot], vectors: tuple[tuple[DofVector, str], ...]):
        """Publish the degrees of freedom of the slots into the node fields.

        Parameters
        ----------
        slots
            Where the degrees of freedom are published.
        vectors
            Each vector, with the node field entry it is published into.
        """

        for slot in slots:
            nodeField = slot.nodeField
            for vector, entry in vectors:
                if entry not in nodeField:
                    nodeField.createFieldValueEntry(entry)
                values = nodeField[entry]
                # Written through a flat view, which only a contiguous array has.
                if not values.flags.c_contiguous:
                    raise RuntimeError("Node field entry {:}/{:} is not contiguous.".format(nodeField.name, entry))
                values.reshape(-1)[slot.positions] = vector.asPlainArray()[slot.dofs]

    def _nodeFieldSlotsOf(self, model: FEModel, dofs: slice | np.ndarray) -> list[NodeFieldSlot]:
        """Where the given degrees of freedom are published in the node fields.

        Parameters
        ----------
        model
            The model tree.
        dofs
            ``slice(None)`` for every degree of freedom, or sorted degree-of-freedom indices.

        Returns
        -------
        list[NodeFieldSlot]
            One slot per node field.
        """

        selected = np.zeros(self.theDofManager.nDof, dtype=bool)
        selected[dofs] = True
        slots = []
        for nodeField in model.nodeFields.values():
            indices = self.theDofManager.idcsOfNodeFieldsInDofVector[nodeField.name]
            fieldDofs = np.arange(self.theDofManager.nDof)[indices]
            positions = np.flatnonzero(selected[fieldDofs])
            if positions.shape == fieldDofs.shape:
                # Every degree of freedom of the field: published through the field's own index,
                # which copies without gathering where it is a slice.
                slots.append(NodeFieldSlot(nodeField, slice(None), indices))
            else:
                slots.append(NodeFieldSlot(nodeField, positions, fieldDofs[positions]))
        return slots

    # --- Searches and topology updates ---------------------------------------------------------------

    def updateConstraintConnectivity(self, model: FEModel) -> bool:
        """Receive the positions the contact searches of this process read, then run the periodic
        contact search; see :meth:`NED.updateConstraintConnectivity`. Collective.

        A search runs on the process owning the constraint (:meth:`updateConnectivityOf`), and reads
        the positions of every node it may couple -- the nodes the constraint names
        (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.nodesReadByConnectivityUpdate`)
        --, not only of those this process integrates. Those it does not integrate are received from
        their owners, point to point, surface-sized (:meth:`_receiveSolutionReadBySearches`). If a
        searched constraint does not name its nodes, every process receives the whole solution
        instead.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        bool
            Whether any constraint's DOF footprint changed.
        """

        self._receiveSolutionReadBySearches(model)
        return super().updateConstraintConnectivity(model)

    def _receiveSolutionReadBySearches(self, model: FEModel):
        """Receive the solution the periodic contact searches of this process read at the degrees of
        freedom it does not integrate, from their owners, and publish it into the node fields. The
        solution is the displacement of the nodes, which is what a search reads; the velocity and the
        net force there stay as they were. Collective.

        Parameters
        ----------
        model
            The model tree.
        """

        if not self._searchesPlanned:
            self._planSolutionReadBySearches(model)

        if self._searchesReadWholeSolution:
            self._synchronizeModel(model, self.prevTimeStep, includeStates=False, toEveryProcess=True)
            return

        with performancetiming.timeit("positions for searches"):
            searched = self._solutionReadBySearches
            searched.received.receive([self._U])
            self._publishIntoNodeFields(searched.slots, ((self._U, "U"),))

    def _planSolutionReadBySearches(self, model: FEModel):
        """Plan what the periodic contact searches of this process read: the degrees of freedom, at
        the nodes each searched constraint names, that this process does not integrate; or the whole
        solution, if any constraint of the model whose connectivity is searched names none. The same
        decision in every process. Collective.

        Parameters
        ----------
        model
            The model tree.
        """

        searchable = self._dynamicConnectivityConstraints
        nodesRead = {name: constraint.nodesReadByConnectivityUpdate() for name, constraint in searchable.items()}
        self._searchesPlanned = True
        self._searchesReadWholeSolution = any(nodes is None for nodes in nodesRead.values())
        if self._searchesReadWholeSolution:
            self._solutionReadBySearches = None
            return

        searchedHere = self.subdomain.constraintsSearchedHere(model, searchable)
        dofsOfFields = self.theDofManager.idcsOfFieldVariablesInDofVector
        dofs = [
            dofsOfFields[fieldVariable]
            for name in searchedHere
            for node in nodesRead[name]
            for fieldVariable in node.fields.values()
            if fieldVariable in dofsOfFields
        ]
        dofs = np.unique(np.concatenate(dofs)) if dofs else np.empty(0, dtype=np.int64)
        integrated = np.zeros(self.theDofManager.nDof, dtype=bool)
        integrated[self.partition.dofs] = True
        notIntegrated = dofs[~integrated[dofs]]
        self._solutionReadBySearches = SolutionReadBySearches(
            self.subdomain.valuesFromOwners(notIntegrated), self._nodeFieldSlotsOf(model, notIntegrated)
        )

    def updateConnectivityOf(self, model: FEModel, constraints: dict) -> bool:
        """Let those of the given constraints update their connectivity whose search runs in this
        process, failing on all ranks together; whether a constraint of any process changed. See
        :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.constraintsSearchedHere`.
        Collective.

        Parameters
        ----------
        model
            The model tree.
        constraints
            The constraints to update, by name.

        Returns
        -------
        bool
            Whether any constraint's DOF footprint changed.
        """

        searched = self.subdomain.constraintsSearchedHere(model, constraints)
        with self.communicator.allRanksFailTogether("Updating the constraint connectivity"):
            changed = super().updateConnectivityOf(model, searched)
        return self.communicator.allreduceAny(changed)

    def updateTopology(self, model: FEModel, step, offerModelModifiers: bool) -> tuple[bool, bool]:
        """Run the model modifiers and the mesh refresh of
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateTopology` in
        every process, on the same synchronized model, failing on all ranks together if it fails on one; and check that
        every process arrived at the same outcome. Collective.

        If the topology changed, the subdomains are rebalanced for the changed mesh if that pays
        (:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.rebalanceAfterTopologyChange`),
        over the increments until the end of the step or since the previous topology change,
        whichever are fewer (the latter alone if the time stepper cannot estimate the former).

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved.
        offerModelModifiers
            As for the base class.

        Returns
        -------
        tuple[bool, bool]
            Whether the topology changed, and whether a mesh-dependent consumer was refreshed.
        """

        self.subdomain.requireConstraintCopiesCurrent(model)
        with self.communicator.allRanksFailTogether("Updating the topology"):
            # the modifiers and the refresh of the mesh read the node fields at every node
            self.requireWholeSolutionHere("A topology update")
            changed = super().updateTopology(model, step, offerModelModifiers)
        self.communicator.requireSameOnAllRanks(
            tuple(bool(flag) for flag in changed), "whether the topology update changed the mesh"
        )
        if changed[0]:
            # The children of a refined element are computed where it was; move elements now if that
            # unbalances the subdomains enough to pay, inside the rebuild that follows anyway.
            incrementsDone = step.timeStepper.numberOfIncrementsDone()
            horizon = incrementsDone - self._incrementsDoneAtLastTopologyChange
            incrementsLeft = step.timeStepper.incrementsLeftEstimate()
            if incrementsLeft is not None:
                horizon = min(incrementsLeft, horizon)
            self._incrementsDoneAtLastTopologyChange = incrementsDone
            self.subdomain.rebalanceAfterTopologyChange(model, horizon)
        return changed

    def _synchronizeModel(self, model: FEModel, timeStep: TimeStep | None, includeStates: bool, toEveryProcess: bool):
        """Make the model complete before something reads all of it -- in every process, or on rank 0
        only: every degree of freedom of the vectors, published into the node fields, and optionally
        every constraint state -- and, where every process holds every element, every element state,
        in every process -- as the process computing it last left it. Collective.

        A process not receiving the whole model keeps the vectors and the node fields complete at the
        degrees of freedom it integrates only, and its copies of the constraints owned elsewhere in
        the state of their last synchronization; :meth:`requireWholeSolutionHere` and
        :meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.requireConstraintCopiesCurrent`
        refuse to read them as whole.

        Parameters
        ----------
        model
            The model tree.
        timeStep
            The last completed time step, or None before the first one.
        includeStates
            Whether the element and constraint states are synchronized as well, or only the vectors.
        toEveryProcess
            Whether every process receives the whole model, or rank 0 only.
        """

        U, V, P = self._U, self._V, self._P
        with performancetiming.timeit("gather solution"):
            for vector in (U, V, P):
                if toEveryProcess:
                    self.subdomain.allgatherOwnedValues(vector)
                else:
                    self.subdomain.gatherOwnedValuesToRoot(vector)
        wholeSolutionHere = toEveryProcess or self.subdomain.rank == 0
        if wholeSolutionHere:
            # every degree of freedom is current now, not only those integrated here
            super().publishNodeFields(model, U, V, P)
            self._wholeSolutionHere = True

        if includeStates:
            self.subdomain.synchronizeStates(includeElements=True, toEveryProcess=toEveryProcess)

        # A rigid body's surface follows its reference node, which only some processes integrated:
        # it was moved in acceptIncrement from what this process integrates, and is moved again now
        # from the complete solution.
        if timeStep is not None and wholeSolutionHere:
            self.updateRigidBodies(model, timeStep)

    def requireWholeSolutionHere(self, reader: str):
        """Refuse to let something read the whole solution -- the vectors at every degree of freedom,
        or the node fields at every node -- unless this process holds it as of the last accepted
        increment: after an output synchronization that reached it (:meth:`_synchronizeModel`), at
        the end of a step, or before the first increment. Between those, a process holds the solution
        at the degrees of freedom it integrates only, and rank 0 alone receives it on an output
        increment no topology check follows (:meth:`wholeModelReadEverywhereNext`).

        Parameters
        ----------
        reader
            What reads the whole solution, for the message.

        Raises
        ------
        RuntimeError
            If this process holds only part of it.
        """

        if not self._wholeSolutionHere:
            raise RuntimeError(
                "{:} reads the whole solution, but process {:} holds it only at the degrees of freedom it "
                "integrates: the last output synchronization gathered it to rank 0 only. A reader on every "
                "process must be known to NEDMPI.wholeModelReadEverywhereNext.".format(reader, self.subdomain.rank)
            )

    def buildEquationSystem(self, model: FEModel, step, previous: ExplicitSystem = None) -> ExplicitSystem:
        """Build the equation system; see :meth:`NED.buildEquationSystem`. Collective.

        Built afresh, it reads the solution from the node fields, which must therefore be whole here
        (:meth:`requireWholeSolutionHere`). Rebuilt with the vectors carried over -- after a contact
        search moved a constraint, or elements moved between processes -- the subdomain is defined
        afresh, and the vectors are completed at the degrees of freedom this process integrates from
        now on (:meth:`~edelweissfe.domaindecomposition.subdomain.Subdomain.receiveNewlyIntegratedValues`).

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved.
        previous
            The system being replaced, or None.

        Returns
        -------
        ExplicitSystem
            The built system.
        """

        if previous is None:
            self.requireWholeSolutionHere("Building the equation system from the node fields")
            return super().buildEquationSystem(model, step)

        interfaceBefore = self.subdomain.interface
        theSystem = super().buildEquationSystem(model, step, previous)
        self.subdomain.receiveNewlyIntegratedValues(interfaceBefore, [theSystem.U, theSystem.V, theSystem.P])
        return theSystem
