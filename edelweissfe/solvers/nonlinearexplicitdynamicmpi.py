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
and completed at the interface like the forces, so that they are the same bits as without
decomposition; the critical time step is the minimum over the subdomains.

**Vectors of the subdomain.** Each process numbers and holds the vectors -- the carried vectors, the
lumped operators -- at the degrees of freedom of its subdomain only, by a DofManager of the node
fields of its subdomain whose numbering is the global one compressed
(:meth:`NEDMPI.createDofManager`); its node fields are those of the nodes it holds -- in a
distributed model those of its subdomain and those its model entities read whole
(:mod:`~edelweissfe.domaindecomposition.distributedelements`) -- and are filled from the owners
where the whole model is read; a per-node field output and a checkpoint gather the node fields of
the whole model from the processes computing them. A model with a reason against it
(:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsForWholeModel`: every element on
every process, adaptive refinement, contact, rigid bodies, scalar variables) holds vectors of the whole
model instead, current at its subdomain degrees of freedom, with the lumped operators shared from the
owners at every degree of freedom -- which is what lets a contact search move a constraint onto
nodes another process integrated until then.

**Where the whole model is read -- and by whom.** Field outputs, output managers, a marker deciding
a refinement, the refinement itself, and a contact search all read more than one subdomain. Each of
them receives what it reads, and only the processes reading it receive it:

* on an output increment, rank 0 -- which writes the output and the checkpoints -- receives the
  current solution at every degree of freedom, the state of every stateful constraint and the
  results of every element field output; the other processes keep theirs, current at the degrees of
  freedom they integrate (:meth:`NEDMPI.acceptIncrement`, :meth:`NEDMPI.writeIncrementOutput`);
* on an output increment the topology check follows, and at the end of a step, every process
  receives all of it, and the field outputs the markers read (:meth:`NEDMPI.outputSynchronizationReachesEveryProcess`):
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
are bit-identical; so are the kinetic energy and the momentum of the journal, exact sums of the
terms each process owns, added exactly (:meth:`sumsOverModel`). Only the internal energy of the
energy table is formed per subdomain and then added in rank order, and may differ from a serial
run's in its last digits: it is the sum the elements report, one number per element, and enters
nothing but the table.

**Load balancing.** The first partition weighs an element by its number of degrees of freedom. A
softening material costs more where it softens, so every element kernel is timed, and so is every
constraint evaluation, by the process evaluating it (with its elements). On an output
increment the model is repartitioned with the measured costs whenever the busiest process -- with its
elements and its constraints -- falls more than ``load-balance-tolerance`` behind what a partition
can attain; a process evaluating a costly contact is given correspondingly fewer elements. A
distributed model then moves the elements
whose process changes (migration: the new process creates them from the mesh and receives their
state, the old one drops them; see
:meth:`~edelweissfe.domaindecomposition.distributedelements.DistributedElements.moveElementsTo`), and
every process builds its equation system again for the elements it now holds, carrying the solution
over. ``load-balance-costs=elementNumber`` replaces the timings by the element numbers -- a
deterministic, uneven cost, for tests.

**What this solver changes.** The increment of ``NED`` runs over the
:class:`~edelweissfe.solvers.base.subdomain.Subdomain` of this process, which a
:class:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition` defines. What this
solver adds is the communication, each in an override of a method of ``NED`` (the complete list is
at the end of the documentation of :mod:`~edelweissfe.solvers.nonlinearexplicitdynamic`):

* :meth:`NEDMPI.assembleInternalForces` completes the element forces at the interface;
* :meth:`NEDMPI.assembleLumpedDiagonal` completes the lumped inertia and damping the same way, and
  shares them from the owners;
* :meth:`NEDMPI.addConstraintForces` sends the constraint forces to the processes integrating
  their degrees of freedom;
* :meth:`NEDMPI.assembleLoads` evaluates the loads of the elements of the subdomain, and completes
  them at the interface like the forces;
* :meth:`NEDMPI.getCriticalTimeStepForExplicitDynamics` forms the minimum over all processes, and
  :meth:`NEDMPI.sumsOverModel` (the kinetic energy and the momentum) and
  :meth:`NEDMPI.energyBalanceTerms` (the internal energy) the sums;
* :meth:`NEDMPI.acceptIncrement` synchronizes the model on output increments and rebalances it,
  rebuilding the equation system if elements moved,
  :meth:`NEDMPI.updateConstraintConnectivity` synchronizes it before a contact search,
  :meth:`NEDMPI.applyStepActionsAtStepEnd` at the end of a step;
* :meth:`NEDMPI.updateConnectivityOf` runs a contact search on the constraint's process only;
* and every step that can fail in one process alone -- the element and constraint evaluation, a
  contact search, a topology update, writing the output -- fails on all ranks together
  (:meth:`~edelweissfe.domaindecomposition.communicator.Communicator.allRanksFailTogether`).

**Limits of this prototype.** Every process holds the whole mesh and every contact facet. A model with
a reason against vectors of the subdomain holds vectors of the whole model, and every node, in every
process; a model under the whole-model rule
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

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from mpi4py import MPI

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.communicator import Communicator
from edelweissfe.domaindecomposition.domaindecomposition import (
    DomainDecomposition,
    MeasuredLoad,
)
from edelweissfe.domaindecomposition.mpienvironment import (
    RaisedOnAllRanks,
    StepFailedOnAllRanks,
    abortAllProcesses,
    worldCommunicator,
)
from edelweissfe.domaindecomposition.subdomaininterface import (
    InterfaceForceAssembly,
    ValuesFromOwners,
)
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.modelmodifiers.base.modelmodifierbase import fieldOutputsReadByAll
from edelweissfe.models.elementdistribution import nodeFieldValuesHere
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.exactsum import ExactSum
from edelweissfe.numerics.parallelizationutilities import getNumberOfThreads
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.parallelelementcomputation import (
    ElementPlan,
    computeElementsForExplicit,
)
from edelweissfe.solvers.base.subdomain import Subdomain
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
            "How far the busiest process -- with its elements and the constraints it evaluates -- may fall "
            "behind what a partition can attain (the mean, or the constraints of one process where they cost "
            "more), as a fraction, before the model is repartitioned with the measured element and constraint "
            "costs. Checked on every output increment. The first partition can only "
            "estimate an element's cost -- by its number of degrees of freedom -- and a softening material "
            "costs more where it softens, so a partition balanced at the start drifts out of balance as "
            "damage localizes. 0 disables it."
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
    debugPoisonStaleSolution: bool = schemaField(
        description=(
            "A check, not for production runs: after every increment, set the solution, the velocity, the "
            "net force and the node fields to NaN outside the degrees of freedom a process integrates, "
            "wherever the process does not hold the whole solution. Nothing may read them there, so the "
            "result must be the same bits; a reader of data it did not receive turns it into NaN. Costs a "
            "pass over the vectors and the node fields per increment."
        ),
        dtype=bool,
        default=False,
        optionName="debug-poison-stale-solution",
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
class ReceivedSolution:
    """The solution at nodes this process reads but whose degrees of freedom it does not integrate
    itself -- the nodes a contact search of this process reads, the reference nodes of the rigid
    bodies --: received from the owners, and published into the node fields here.

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
        "debug-poison-stale-solution": False,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        super().__init__(jobInfo, journal, **kwargs)

        communicator = worldCommunicator()
        #: The communicator of all processes of the launcher -- started without a launcher, of this
        #: process alone: the same code path, with every exchange a copy. The processes agree through
        #: it, e.g. on a failure in any of them (:meth:`Communicator.allRanksFailTogether`).
        self.communicator = Communicator(MPI.COMM_SELF) if communicator is None else communicator
        #: The decomposition of the model over the processes: the subdomain this process computes,
        #: and every exchange with the others.
        self.domainDecomposition = DomainDecomposition(
            self.communicator,
            journal,
            self.identification,
            self.options["load-balance-tolerance"],
        )
        #: How the element forces of the current increment plan are completed at the interface.
        self._interfaceAssembly: InterfaceForceAssembly | None = None
        #: The kernel time of every element of the current increment plan, in plan order, the least
        #: over the :attr:`_nMeasuredIncrements` increments since the last check of the balance
        #: (:func:`computeElementsForExplicit`), if the decomposition balances load by measured costs; else
        #: None. See :meth:`_startMeasuring`.
        self._elementCosts: np.ndarray | None = None
        #: The increments computed since the last check of the balance, or since the increment plan
        #: was made.
        self._nMeasuredIncrements = 0
        #: The time this process spent computing its elements -- the element loop, on all its threads
        #: -- and evaluating its constraints, over :attr:`_nMeasuredIncrements` increments; see
        #: :meth:`_measuredLoad`.
        self._elementLoopTime = 0.0
        self._constraintEvaluationTime = 0.0
        #: The same for each constraint it evaluates, by name.
        self._constraintEvaluationTimes = {}
        #: The forces of the constraints of the subdomain, evaluated with the elements of the current
        #: increment in :meth:`assembleInternalForces` and not yet returned by
        #: :meth:`evaluateConstraintForces`, by name; None otherwise.
        self._constraintForcesOfIncrement: dict | None = None
        #: The increments of the step done at the last topology change (or 0, at the start of the
        #: step), for the horizon a migration after a topology change is weighed over.
        self._incrementsDoneAtLastTopologyChange = 0
        #: Where the subdomain degrees of freedom are published in the node fields; see
        #: :meth:`publishNodeFields`.
        self._integratedNodeFieldSlots: list[NodeFieldSlot] = []
        #: The work at the prescribed degrees of freedom this process owns of every increment since the
        #: external work was last read, in increment order: per increment the exact sum of the
        #: products, added to the external work by :meth:`gatherExternalWork`.
        self._pendingWorkAtPrescribedDofs: list[ExactSum] = []
        #: Whether this process holds the whole solution -- ``U``, ``V`` and ``P`` at every degree of
        #: freedom, and published into the node fields -- as of the last accepted increment; else
        #: only at the degrees of freedom it integrates. See :meth:`requireWholeSolutionHere`.
        self._wholeSolutionHere = True
        #: What the periodic contact searches of this process read, planned for the current
        #: subdomain at the first search: :meth:`_planSolutionReadBySearches`; None before.
        self._solutionReadBySearches: ReceivedSolution | None = None
        #: Whether the periodic contact searches read the whole solution, because a searched
        #: constraint does not name the nodes it reads; decided with the plan above.
        self._searchesReadWholeSolution = False
        self._searchesPlanned = False
        #: The solution at the reference nodes of the rigid bodies, received by every process on an
        #: output increment synchronized to rank 0 only: :meth:`_receiveRigidBodyReferences`; planned
        #: for the current subdomain at its first use, None before.
        self._rigidBodyReferences: ReceivedSolution | None = None
        #: The increments this solver accepted in the job, the zero increment of every step included:
        #: a count, not an increment number, which a step starts afresh.
        self._incrementsAccepted = 0
        #: The value of :attr:`_incrementsAccepted` when the copies every process holds were last made
        #: consistent: the model synchronized to every process, with the states of the elements and
        #: constraints (:meth:`_synchronizeModel`); 0 for the model as it was set up, the same in every
        #: process. See :meth:`_copiesConsistentInEveryProcess`.
        self._copiesConsistentAfterIncrement = 0
        #: Whether the vectors of this process -- the carried vectors, the lumped operators -- hold the
        #: degrees of freedom of its subdomain only, or every degree of freedom of the model; decided once
        #: per job, at the start of its first step, by the rule the nodes follow
        #: (:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsForWholeModel`). None before.
        self._vectorsOfSubdomain: bool | None = None
        #: The interface of the subdomain before the equation system now being rebuilt, while it is; see
        #: :meth:`carryVectorsOver`.
        self._interfaceBeforeRebuild = None
        #: Per node set and field, the rows of the set held in the vectors of the subdomain and the
        #: positions of their degrees of freedom (:meth:`_partOfNodeSetHere`); for the current system.
        self._partsOfNodeSetsHere = {}

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: list[OutputManagerBase],
    ):
        """Report the decomposition, then start the step; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.beginStep`.

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
                self.domainDecomposition.nProcesses, getNumberOfThreads()
            ),
            self.identification,
            0,
        )
        self._incrementsDoneAtLastTopologyChange = 0
        self._pendingWorkAtPrescribedDofs = []
        if self._vectorsOfSubdomain is None:
            self._decideVectors(model)
        return super().beginStep(step, model, fieldOutputController, outputmanagers)

    @property
    def vectorsOfSubdomain(self) -> bool:
        """Whether the vectors of this process hold the degrees of freedom of its subdomain only; see
        :func:`~edelweissfe.domaindecomposition.distributedelements.reasonsForWholeModel`.

        Returns
        -------
        bool
            True for vectors of the subdomain, False for vectors of the whole model.

        Raises
        ------
        RuntimeError
            If it was not decided yet: before the first step began.
        """

        if self._vectorsOfSubdomain is None:
            raise RuntimeError("whether the vectors are of the subdomain is decided when the first step begins")
        return self._vectorsOfSubdomain

    def _decideVectors(self, model: FEModel):
        """Decide, once per job, whether the vectors of each process hold its subdomain only or the whole
        model, the same in every process, and report it once: by the one rule the nodes a process holds
        follow, decided from the input with the model's element distribution
        (:func:`~edelweissfe.domaindecomposition.distributedelements.reasonsForWholeModel`). What the model
        built from the input holds is checked against it (:meth:`reasonsSeenInModel`).

        Parameters
        ----------
        model
            The model tree.

        Raises
        ------
        RuntimeError
            If the model holds something the rule should have named.
        """

        reasons = list(model.elementDistribution.reasonsForWholeModel)
        seen = self.reasonsSeenInModel(model)
        if seen and not reasons:
            raise RuntimeError(
                "The model needs every node and vectors of the whole model on every process because: {:} -- which "
                "the rule read from the input (reasonsForWholeModel) did not name; a model entity does not declare "
                "it".format("; ".join(seen))
            )
        self._vectorsOfSubdomain = not reasons
        self.domainDecomposition.vectorsOfSubdomain = self._vectorsOfSubdomain
        self.journal.message(
            (
                "Vectors of the subdomain: each process numbers and holds the degrees of freedom it integrates only"
                if not reasons
                else "Vectors of the whole model on every process because: " + "; ".join(reasons)
            ),
            self.identification,
            0,
        )

    @staticmethod
    def reasonsSeenInModel(model: FEModel) -> list[str]:
        """What the model holds that requires every node and vectors of the whole model on every process --
        the carried vectors and the lumped operators at every degree of freedom, numbered by a DofManager of
        every node: the invariant behind
        :func:`~edelweissfe.domaindecomposition.distributedelements.reasonsForWholeModel`, which decides it
        from the input; empty if nothing requires it.

        What still reads, or changes, degrees of freedom outside a subdomain during a run, until each
        receives what it reads: a model whose every process holds every element (and so the states to
        synchronize); a model modifier changing the mesh that reads every node (not adaptive refinement,
        which interpolates the nodes it creates from those a process holds,
        :attr:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.changesMeshWithNodesOfSubdomain`);
        a constraint searching its connectivity (contact), which may couple degrees
        of freedom of another subdomain at its next search; a rigid body, whose reference node every
        process reads; and scalar variables. (A multi-point constraint is not among them: its groups
        are integrated whole, and its transformation is that of the subdomain, see
        :meth:`multiPointConstraintRecords`.)

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        list[str]
            The reasons.
        """

        reasons = []
        if model.elementDistribution.replicatesElements:
            reasons.append("every process holds every element")
        modifiers = [
            name
            for name, modifier in model.modelModifiers.items()
            if modifier.initiatesTopologyChanges and not modifier.changesMeshWithNodesOfSubdomain
        ]
        if modifiers:
            reasons.append("model modifier(s) {:} change the mesh reading every node".format(", ".join(modifiers)))
        searched = [name for name, constraint in model.constraints.items() if constraint.searchesConnectivity]
        if searched:
            reasons.append("constraint(s) {:} search their connectivity (contact)".format(", ".join(searched)))
        if model.rigidBodies:
            reasons.append("rigid bodies {:}".format(", ".join(model.rigidBodies)))
        if model.scalarVariables:
            reasons.append("scalar variables")
        return reasons

    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Collective. Attempt the increment of :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.attemptIncrement`; a step failure it raises is
        raised as having failed on all ranks together.

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
        """Collective after a failure on every process. Finish the step; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.endStep`. A step
        that failed on every process together --
        a step failure or an error (:class:`~edelweissfe.domaindecomposition.mpienvironment.RaisedOnAllRanks`)
        -- ends as a serial one does, with the model of the last accepted increment
        (:meth:`_synchronizeAcceptedModel`), which the field outputs and the output managers then read
        a last time. Any other failure stops every process right after (:meth:`stepFailed`).

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
        if isinstance(failure, RaisedOnAllRanks):
            self._synchronizeAcceptedModel(model)

    def stepFailed(self, step, failure: BaseException):
        """Stop every process, unless every process raised the failure together
        (:class:`~edelweissfe.domaindecomposition.mpienvironment.RaisedOnAllRanks`). Any other
        failure -- wherever in the step, its start and its end included -- may have happened in this
        process alone, while the others wait for it in their next exchange; unwinding further, this
        process would gather the results of the end of the step with processes that never arrive, and
        a caller catching the exception (a test runner, say) would go on to another job. See
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.stepFailed`.

        Parameters
        ----------
        step
            The step that failed.
        failure
            The exception leaving it.
        """

        if not isinstance(failure, RaisedOnAllRanks):
            abortAllProcesses(
                "the step failed without the agreement of the other processes ({:}: {:})".format(
                    type(failure).__name__, failure
                )
            )

    def _synchronizeAcceptedModel(self, model: FEModel):
        """Collective. Make the model of the last accepted increment whole in every process, after an
        increment failed on every process together.

        The failed increment changed the solution, the velocity and the net force at the degrees of
        freedom each process integrates, in the vectors; the node fields still hold the last accepted
        increment there, published when it was accepted, or when the process began to integrate them
        (:meth:`_receiveNewlyIntegratedValues`) -- which is what a serial run leaves. The
        vectors are therefore taken back from the node fields at those degrees of freedom, and the model
        synchronized from them, as at the end of a step (:meth:`applyStepActionsAtStepEnd`), and with
        it the external work, which the failed increment added to as it does in a serial run.

        Parameters
        ----------
        model
            The model tree.
        """

        if self._system is not None:
            self._readFromNodeFields(self._integratedNodeFieldSlots, self._system.carried)
            self._synchronizeModel(model, self.prevTimeStep, includeStates=True, toEveryProcess=True)
        self.gatherExternalWork()

    # --- The subdomain ------------------------------------------------------------------------------

    def createDofManager(self, model: FEModel) -> DofManager:
        """Collective. The degree-of-freedom layout of a new equation system; see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.createDofManager`. Where the vectors are
        of the subdomain, the subdomain is decided first, from what every process holds
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.defineSubdomainDofs`),
        and the DofManager numbers the node fields of the subdomain only
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.subdomainNodeFields`),
        with its elements and constraints: in global order, so that its numbering is the global one,
        compressed.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        DofManager
            The DofManager.
        """

        if not self.vectorsOfSubdomain:
            return super().createDofManager(model)

        self._partsOfNodeSetsHere = {}
        decomposition = self.domainDecomposition
        numbering = decomposition.numberDegreesOfFreedom(model)
        decomposition.defineSubdomainDofs(
            model, self._collectMultiPointConstraintRecords(model, numbering.dofsOfFieldVariables(model.mesh))
        )
        return self.explicitDofManager(
            decomposition.subdomainNodeFields(model).values(),
            [],
            decomposition.ownedElements.values(),
            decomposition.ownedConstraints.values(),
            [],
        )

    def multiPointConstraintRecords(self, model: FEModel) -> list:
        """The records of the multi-point constraints of the equation system; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.multiPointConstraintRecords`.
        Where the vectors are of the subdomain, those whose slave this process integrates, by the positions
        in its vectors
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.multiPointConstraintRecordsOfSubdomain`),
        so that the transformation is the subdomain's part of the transformation of the whole model.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        list
            The records.
        """

        if not self.vectorsOfSubdomain:
            return super().multiPointConstraintRecords(model)
        return self.domainDecomposition.multiPointConstraintRecordsOfSubdomain()

    def readCarriedFromNodeFields(self, model: FEModel, carried: dict[str, DofVector]):
        """Fill the carried vectors of a new equation system from the node fields; see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.readCarriedFromNodeFields`. Vectors of
        the subdomain read the rows of their nodes.

        Parameters
        ----------
        model
            The model tree.
        carried
            The carried vectors, by node field entry; filled.
        """

        if not self.vectorsOfSubdomain:
            return super().readCarriedFromNodeFields(model, carried)
        self._readFromNodeFields(self._integratedNodeFieldSlots, carried)

    def carryVectorsOver(self, previous: ExplicitSystem, carried: dict[str, DofVector]):
        """Collective. Carry the vectors of the system being replaced over to the rebuilt one, and complete
        them at the degrees of freedom this process integrates from now on, from their previous owners; see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.carryVectorsOver`. Vectors of the
        subdomain are filled from the previous subdomain's
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.carryOverToSubdomain`).

        Parameters
        ----------
        previous
            The system being replaced.
        carried
            The carried vectors of the rebuilt system, by node field entry; filled.
        """

        before = self._interfaceBeforeRebuild
        if self.vectorsOfSubdomain:
            entries = list(carried)
            self.domainDecomposition.carryOverToSubdomain(
                before, [previous.carried[entry] for entry in entries], [carried[entry] for entry in entries]
            )
        else:
            super().carryVectorsOver(previous, carried)
            self.domainDecomposition.receiveNewlyIntegratedValues(before, list(carried.values()))

    def defineSubdomain(self, model: FEModel, topologyChanged: bool) -> Subdomain:
        """Collective. The subdomain of this process; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.defineSubdomain`.

        Parameters
        ----------
        model
            The model tree.
        topologyChanged
            As for :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.defineSubdomain`.

        Returns
        -------
        Subdomain
            The subdomain.
        """

        self.domainDecomposition.define(model, self.theDofManager, self.mpcTransformation, topologyChanged)
        return self.domainDecomposition.subdomain

    def planIncrement(self, model: FEModel) -> IncrementPlan:
        """Collective. The increment plan of :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.planIncrement`, and how its element forces are
        completed at the interface.

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
        self._integratedNodeFieldSlots = (
            self._subdomainNodeFieldSlots(model)
            if self.vectorsOfSubdomain
            else self._nodeFieldSlotsOf(model, self.subdomain.dofs)
        )
        # planned again for this subdomain at the next search
        self._forgetReceivePlans()
        self._interfaceAssembly = self.domainDecomposition.interfaceAssemblyFor(plan.elementLoop)
        timesKernels = self.domainDecomposition.measuresElementCosts() and self._loadBalanceCosts() == "measured"
        self._startMeasuring(len(plan.elementLoop.elements) if timesKernels else None)
        return plan

    def _startMeasuring(self, nElements: int | None):
        """Start measuring the load of this process afresh: the kernel times of its elements and the
        times it is busy. After every check of the balance, and for every new increment plan, so that
        what a check weighs is what the elements and the process cost since the previous one -- an
        element whose material softens since is measured as it costs now, not as it once did.

        Parameters
        ----------
        nElements
            The number of elements whose kernel times are measured; None to measure none.
        """

        self._elementCosts = None if nElements is None else np.full(nElements, np.inf)
        self._nMeasuredIncrements = 0
        self._elementLoopTime = self._constraintEvaluationTime = 0.0
        self._constraintEvaluationTimes = {}

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

    def _measuredLoad(self) -> MeasuredLoad:
        """What :meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.rebalance` weighs: the cost
        of every element of the increment plan per increment, in plan order -- the least kernel time
        measured, or the element number (``load-balance-costs``) -- the number of increments measured, the
        increments until the next check, to weigh a repartition's gain against its cost (None for
        element numbers, which are no times), and the time per increment this process was busy with
        its elements and its constraints.

        The busy times are wall times, as the time of an increment is: the time of the element loop
        on all the threads of the process -- not the sum of the kernel times, which is a multiple of it
        on several threads --, and of evaluating the constraints. For element numbers, the elements are
        busy with the sum of their numbers, and the constraints with none.

        Returns
        -------
        MeasuredLoad
            The load of this process; its element costs None if nothing was measured.
        """

        nIncrements = self._nMeasuredIncrements
        if self._loadBalanceCosts() == "elementNumber" and self.domainDecomposition.measuresElementCosts():
            numbers = np.array(list(self._incrementPlan.elementLoop.elements.keys()), dtype=float)
            return MeasuredLoad(numbers, nIncrements, None, float(numbers.sum()), 0.0, {})
        return MeasuredLoad(
            self._elementCosts,
            nIncrements,
            self.options["output-frequency"],
            self._elementLoopTime / max(nIncrements, 1),
            self._constraintEvaluationTime / max(nIncrements, 1),
            {name: seconds / max(nIncrements, 1) for name, seconds in self._constraintEvaluationTimes.items()},
        )

    def assembleLumpedDiagonal(self, elementLoop: ElementPlan, elementContribution) -> DofVector:
        """Collective. Assemble a lumped operator of the elements of the subdomain, complete at every degree
        of freedom of the vectors of this process: completed at the interface like the forces, and, for
        vectors of the whole model, shared from the owners.

        Parameters
        ----------
        elementLoop
            The plan of the elements of the subdomain, contact facets included.
        elementContribution
            As for :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.assembleLumpedDiagonal`.

        Returns
        -------
        DofVector
            The assembled diagonal.
        """

        vector, contributions = self.lumpedDiagonalAndContributions(elementLoop, elementContribution)
        with performancetiming.timeit("interface forces"):
            self.domainDecomposition.interfaceAssemblyFor(elementLoop).assemble(contributions, vector)
            # complete at every subdomain degree of freedom now; vectors of the whole model complete there too
            if not self.vectorsOfSubdomain:
                self.domainDecomposition.allgatherOwnedValues(vector)
        return vector

    def coefficientCountsOfFields(self, inertia: DofVector, damping: DofVector) -> dict[str, tuple[int, int, int]]:
        """Collective. The counts of :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.coefficientCountsOfFields`
        over the whole model: each process counts the degrees of freedom it owns, and the counts of all
        processes are added -- so that every process classifies every field alike, also one it does not
        integrate.

        Parameters
        ----------
        inertia
            The assembled lumped inertia.
        damping
            The assembled lumped damping.

        Returns
        -------
        dict[str, tuple[int, int, int]]
            Per field: its number of degrees of freedom, of those with an inertia, and of those with a
            damping.
        """

        owned = self.domainDecomposition.ownedDofMask
        fields = list(self.theDofManager.idcsOfFieldsInDofVector.items())
        countsHere = np.array(
            [
                (
                    np.count_nonzero(owned[indices]),
                    np.count_nonzero(owned[indices] & (np.asarray(inertia[indices]) != 0.0)),
                    np.count_nonzero(owned[indices] & (np.asarray(damping[indices]) != 0.0)),
                )
                for _, indices in fields
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        counts = self.communicator.allreduce(countsHere)
        return {fieldName: tuple(int(count) for count in counts[i]) for i, (fieldName, _) in enumerate(fields)}

    # --- The increment ------------------------------------------------------------------------------

    @performancetiming.timeit("elements")
    def assembleInternalForces(
        self, U_np: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
    ) -> tuple[DofVector, float]:
        """Collective. Evaluate the elements of the subdomain, and the constraints of the subdomain, failing
        on all ranks together if it fails on one, and complete the internal force at the interface; see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.assembleInternalForces`.

        The constraints are evaluated in this method (:meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.evaluateConstraintForces`), right
        after the elements and inside the same agreement, and kept for the next step of
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.assembleNetForce`, :meth:`evaluateConstraintForces`, to return: ``NED`` evaluates
        them right after the elements as well, so the result is the same. A process evaluating a
        costly constraint -- a large contact -- does so before the exchange at the interface, while
        the others are still computing the elements the load balancing gave them in its place
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.rebalance`),
        instead of every process waiting for it after the exchange.

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
        # Both timed for the load balancing -- inside the agreement, which waits for the other processes.
        with self.communicator.allRanksFailTogether("Evaluating the elements and the constraints"):
            startOfElements = perf_counter()
            psi, contributions = computeElementsForExplicit(
                self._incrementPlan.elementLoop, U_np, dU, P, timeStep, self._elementCosts
            )
            startOfConstraints = perf_counter()
            # one at a time, as NED.evaluateConstraintForces evaluates them, each timed: the shares of
            # the load balancing follow a constraint to whichever process evaluates it
            forces = {}
            for name, constraint in self.subdomain.constraints.items():
                startOfConstraint = perf_counter()
                forces[name] = self._evaluateConstraintForce(name, constraint, U_np, dU, P, timeStep)
                seconds = perf_counter() - startOfConstraint
                self._constraintEvaluationTimes[name] = self._constraintEvaluationTimes.get(name, 0.0) + seconds
            self._constraintForcesOfIncrement = forces
            self._elementLoopTime += startOfConstraints - startOfElements
            self._constraintEvaluationTime += perf_counter() - startOfConstraints
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
        """Collective. Assemble the loads as ``NED`` does (``assembleLoads``), in the same order, those
        acting on elements as the element forces are: each process evaluates the loads of the elements it computes,
        failing on all ranks together if it fails on one, and the contributions are completed at the
        interface and added in the order of the loads of the model; see
        :meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.loadsOnSubdomain`.

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
            loads = self.domainDecomposition.loadsOnSubdomain(distributedLoads, bodyForces)
            with performancetiming.timeit("distributed loads"):
                forces = [Pe for _, Pe in self.distributedLoadsOfElements(loads.distributedLoads, U_np, K, timeStep)]
            with performancetiming.timeit("body forces"):
                forces += [Pe for _, Pe in self.bodyForcesOfElements(loads.bodyLoads, U_np, K, timeStep)]

        with performancetiming.timeit("interface loads"):
            self.domainDecomposition.loadAssemblyFor(loads).assemble(
                np.concatenate(forces) if forces else np.empty(0), PExt
            )
        return PExt, K

    def evaluateConstraintForces(
        self, constraints: dict, U_np: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
    ) -> dict:
        """The forces of the constraints of the subdomain, evaluated with the elements of this increment
        in :meth:`assembleInternalForces`; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.evaluateConstraintForces`.

        Parameters
        ----------
        constraints
            The constraints of the subdomain, by name, in model order.
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The net force vector.
        timeStep
            The current time step.

        Returns
        -------
        dict[str, ConstraintForce]
            The forces of every constraint of the subdomain, by name, in model order.

        Raises
        ------
        RuntimeError
            If the constraints were not evaluated with the elements of this increment.
        """

        forces, self._constraintForcesOfIncrement = self._constraintForcesOfIncrement, None
        if forces is None or forces.keys() != constraints.keys():
            raise RuntimeError(
                "the constraints of the subdomain must be evaluated with the elements of the increment "
                "(NEDMPI.assembleInternalForces) before their forces are added"
            )
        return forces

    @performancetiming.timeit("assemble constraints")
    def addConstraintForces(self, forces: dict, P: DofVector) -> DofVector:
        """Collective. Send the forces of the constraints of the subdomain to the processes integrating
        their degrees of freedom, and add those of every constraint at the subdomain degrees of freedom,
        in model order; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.addConstraintForces` and
        :meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.addConstraintForces`.

        Parameters
        ----------
        forces
            The forces of the constraints of the subdomain, by name, in model order.
        P
            The net force vector to be augmented.

        Returns
        -------
        DofVector
            The augmented net force vector.
        """

        self.domainDecomposition.addConstraintForces(forces, P)
        return P

    def getCriticalTimeStepForExplicitDynamics(self, model: FEModel, U: DofVector) -> float:
        """Collective. The smallest critical time step of all subdomains; see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.getCriticalTimeStepForExplicitDynamics`.

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
        """Keep the work of this increment at the prescribed degrees of freedom this process owns -- a degree
        of freedom shared by two subdomains is counted once, by its owner -- until the external work
        is next read (:meth:`gatherExternalWork`), so that nothing is exchanged every increment; see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.addExternalWork`. Kept as one number per increment: the exact sum of the products
        (:class:`~edelweissfe.numerics.exactsum.ExactSum`), which the sums of the other processes
        complete exactly.

        Parameters
        ----------
        dofs
            The prescribed degrees of freedom whose work is an energy.
        reactionTimesIncrement
            Per degree of freedom, the net nodal force times the prescribed increment.
        """

        self._pendingWorkAtPrescribedDofs.append(
            ExactSum.of(reactionTimesIncrement[self.domainDecomposition.ownedDofMask[dofs]])
        )

    def gatherExternalWork(self):
        """Collective. Add the work of every increment since the last call to the external work, increment
        by increment, in increment order: the exact sums of the products at the degrees of freedom each
        process owns are gathered to every process -- once for all those increments, one integer per
        increment and process --, added exactly, and rounded once: the bits of :func:`math.fsum` of all
        the products of the increment, which :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.addExternalWork` forms in a serial run. So the
        external work is the same, bit for bit, on any number of processes, and every process holds that
        of the whole model. Called where it is read: the energy balance, an output increment (and a
        checkpoint written with it), the end of a step.
        """

        gathered = self.communicator.allgather(self._pendingWorkAtPrescribedDofs)
        for ofProcesses in zip(*gathered):
            self._externalWork -= sum(ofProcesses, ExactSum()).rounded()
        self._pendingWorkAtPrescribedDofs = []

    def sumsOverModel(self, termsAtDofs: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        """Collective. The sums of :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.sumsOverModel`
        over the whole model: each process forms the exact sum of the terms at the degrees of freedom it
        owns, the exact partial sums of all processes are gathered and added, and the total is rounded
        once -- the same bits as a serial run's, in every process, however the model is split.

        Parameters
        ----------
        termsAtDofs
            Per sum, the degrees of freedom and the term at each of them; current at least where this
            process owns them.

        Returns
        -------
        list[float]
            The correctly rounded sum of the terms of each, over the whole model.
        """

        owned = self.domainDecomposition.ownedDofMask
        ownedHere = [ExactSum.of(terms[owned[dofs]]) for dofs, terms in termsAtDofs]
        return [sum(ofProcesses, ExactSum()).rounded() for ofProcesses in zip(*self.communicator.allgather(ownedHere))]

    def energyBalanceTerms(self, psi: float, V: DofVector) -> tuple[float, float, float, list[float]]:
        """Collective. The terms of :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.energyBalanceTerms` of the whole model.

        The internal energy is formed per subdomain and then added in rank order, so it may differ from
        a serial run's in its last digits -- the one term of the table that does; it enters nothing but
        the table. The kinetic energies are exact sums over the whole model already
        (:meth:`sumsOverModel`), and the external work is that of the whole model
        (:meth:`gatherExternalWork`).

        Parameters
        ----------
        psi
            The internal energy the elements of the subdomain report.
        V
            The velocity vector.

        Returns
        -------
        tuple[float, float, float, list[float]]
            As for :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.energyBalanceTerms`.
        """

        self.gatherExternalWork()
        Wint, Wkin, Wext, nonMechanical = super().energyBalanceTerms(psi, V)
        (Wint,) = self.communicator.allreduceSum([Wint])
        return Wint, Wkin, Wext, nonMechanical

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Collective. Commit the increment; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.acceptIncrement`. On an output increment, then
        make the whole model current where it is read (:meth:`outputSynchronizationReachesEveryProcess`)
        -- on rank 0, which writes the output and a checkpoint with it, or, if the topology check at the
        start of the next increment follows, in every process -- and rebalance the subdomains; if that
        moved elements between processes, build the equation system again for the elements now local, and
        if it changed only the subdomains, complete the vectors at the degrees of freedom a process
        integrates from now on.

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
        self._incrementsAccepted += 1
        # the increment changed the solution at the subdomain degrees of freedom only
        self._wholeSolutionHere = False

        if self.isOutputIncrement(timeStep):
            # the output and a checkpoint written after it read the external work
            self.gatherExternalWork()
            self._synchronizeModel(
                model, timeStep, includeStates=True, toEveryProcess=self.outputSynchronizationReachesEveryProcess()
            )

            # Right after every element state was synchronized -- or, where each process holds
            # only its own elements, accepted by the process computing it -- because an element
            # computed by another process from now on must arrive there with its current state.
            startOfRebalancing = perf_counter()
            interfaceBefore = self.domainDecomposition.interface
            if self.domainDecomposition.rebalance(self._incrementPlan.elementLoop, self._measuredLoad()):
                if self.domainDecomposition.elementsMustMove():
                    self._moveElements(model, step)
                else:
                    self.subdomain = self.domainDecomposition.subdomain
                    self._incrementPlan = self.planIncrement(model)
                    self._receiveNewlyIntegratedValues(interfaceBefore, self._system.carried)
                self.domainDecomposition.recordRepartitionCost(perf_counter() - startOfRebalancing)
            else:
                # the next check weighs what was measured from now on (a new plan starts afresh itself)
                self._startMeasuring(None if self._elementCosts is None else self._elementCosts.shape[0])

        if self.options["debug-poison-stale-solution"] and not self._wholeSolutionHere:
            self._poisonStaleSolution(model)

    def _poisonStaleSolution(self, model: FEModel):
        """Set the solution, the velocity, the net force and the node fields to NaN outside the degrees
        of freedom this process integrates (``debug-poison-stale-solution``): there they hold what the
        last synchronization that reached this process left, which nothing may read. A contact search
        receives the positions it reads afresh (:meth:`_receiveSolutionReadBySearches`), so a position
        it reads without naming its node is NaN as well.

        Parameters
        ----------
        model
            The model tree.
        """

        if self.vectorsOfSubdomain:
            # the vectors hold nothing else; the node fields hold more nodes than the subdomain's
            for slot in self._integratedNodeFieldSlots:
                for entry in self._system.carried:
                    values = slot.nodeField[entry].reshape(-1)
                    stale = np.ones(values.shape[0], dtype=bool)
                    stale[slot.positions] = False
                    values[stale] = np.nan
            return
        outside = np.ones(self._system.U.shape[0], dtype=bool)
        outside[self.subdomain.dofs] = False
        for vector in self._system.carried.values():
            vector.asPlainArray()[outside] = np.nan
        super().publishNodeFields(model, self._system.carried)

    def outputSynchronizationReachesEveryProcess(self) -> bool:
        """Whether every process reads the whole model before the next increment, and so the output
        synchronization of the increment just accepted makes it current in every process, rather
        than on rank 0 only.

        Rank 0 writes the output and the checkpoints, and so reads the whole model on every output
        increment. The other processes read it only where the topology check follows at the start
        of the next increment (:meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.topologyCheckDueAfter`): the refinement is replicated, and
        every process interpolates the node fields onto the refined mesh and builds its equation
        system from them, and checks that every copy of a constraint is its owner's
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.requireConstraintCopiesCurrent`).

        Returns
        -------
        bool
            True if every process receives the whole model.
        """

        return self.topologyCheckDueAfter(self.prevTimeStep)

    def fieldOutputsGatheredToEveryProcess(self) -> set[str] | None:
        """The field outputs every process reads before the next increment, and so receives -- the
        others are gathered to rank 0 only, where the output is written: those the markers of the
        model modifiers read (:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.fieldOutputsRead`),
        if the topology check follows (:meth:`outputSynchronizationReachesEveryProcess`); else none.

        Returns
        -------
        set[str] | None
            The names of the field outputs; None for all of them, if a modifier does not name those
            it reads.
        """

        if not self.outputSynchronizationReachesEveryProcess():
            return set()
        return fieldOutputsReadByAll(self._liveTopologyModifiers)

    def advanceModelToTime(self, model: FEModel, time: float):
        """Let the elements and the constraints of this subdomain accept the state the increment
        computed (:meth:`~edelweissfe.models.femodel.FEModel.acceptStatesOf`), and advance the model
        to the end of the increment; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.advanceModelToTime`. Only those: the others were
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
        model.acceptStatesOf(self.subdomain.elements, self.subdomain.constraints)

    def _moveElements(self, model: FEModel, step):
        """Collective. Move elements between the processes, after a rebalance changed the partition of a
        distributed model, and build the equation system again for the elements now local.

        Every process rebuilds its equation system from its copies of the constraints owned elsewhere
        and of the rigid-body surfaces, which must therefore be consistent in every process first
        (:meth:`_copiesConsistentInEveryProcess`): if the last synchronization did not reach every
        process, the model is synchronized to every process before the elements move.

        Everything indexed by the local elements -- the equation system
        (:meth:`releaseEquationSystem`) and the subdomain -- is released before the elements move,
        so that no element a process drops stays alive, and built again afterwards, as after a
        change of the topology. The solution, the velocity and the net force are carried over, and
        each process completes them at the degrees of freedom it integrates from now on, from their
        previous owners (:meth:`buildEquationSystem`).

        The lumped operators are assembled again, from the elements now local, and must be the
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

        if not self._copiesConsistentInEveryProcess():
            # The output synchronization reached rank 0 only: the copies of the constraints owned
            # elsewhere hold their last synchronized state, from which every process would rebuild.
            self._synchronizeModel(model, self.prevTimeStep, includeStates=True, toEveryProcess=True)
        # The constraints are assigned afresh after the elements moved, and evaluated by their new
        # processes from these copies: current, as just made sure.
        self.domainDecomposition.constraintCopiesAreSynchronized()
        before = self._lumpedOperators()
        interfaceBefore = self.domainDecomposition.interface
        carried = self.releaseEquationSystem()
        self.domainDecomposition.moveElements()
        self._buildSystem(self.buildEquationSystem(model, step, previous=carried))
        self._requireSameLumpedOperators(before, interfaceBefore)

    def _lumpedOperators(self) -> dict[str, np.ndarray]:
        """Plain copies of the lumped operators of the current equation system, complete at every
        degree of freedom of its vectors.

        Returns
        -------
        dict[str, np.ndarray]
            The vector the increment divides by, its inverse, the unfolded mass and the damping
            rate, by name.
        """

        return {
            "lumped mass": np.array(self._lumpedMass),
            "inverse lumped mass": np.array(self._system.Minv),
            "unfolded lumped mass": np.array(self._rawLumpedMass),
            "damping rate": np.array(self._dampingRate),
        }

    def _requireSameLumpedOperators(self, before: dict[str, np.ndarray], interfaceBefore):
        """Collective. Refuse to continue unless the lumped operators are the same bits as ``before``.

        Elements moved between processes, and the equation system was built again for the elements
        now local, which assembles the lumped operators again. A serial run keeps them between
        rebuilds of its equation system, and has none here: the two agree only if an element's lumped inertia and damping do not
        depend on its state. That holds for the elements of this package, and is checked here rather
        than assumed, at every migration. Vectors of the subdomain compare the operators before at the
        degrees of freedom of the new subdomain, received from their previous owners where this process
        did not integrate them before; the processes then agree on what changed anywhere.

        Parameters
        ----------
        before
            The operators before the elements moved (:meth:`_lumpedOperators`).
        interfaceBefore
            The interface of the subdomain before the elements moved.

        Raises
        ------
        RuntimeError
            If any of them changed in any bit.
        """

        after = self._lumpedOperators()
        if self.vectorsOfSubdomain:
            names = list(before)
            mapped = [np.empty_like(after[name]) for name in names]
            self.domainDecomposition.carryOverToSubdomain(interfaceBefore, [before[name] for name in names], mapped)
            before = dict(zip(names, mapped))
        changedHere = [name for name in before if before[name].tobytes() != after[name].tobytes()]
        changedAnywhere = set().union(*self.communicator.allgather(changedHere))
        changed = [name for name in before if name in changedAnywhere]
        if changed:
            raise RuntimeError(
                "Elements moved between processes, and the lumped operators assembled again from the elements "
                "now local differ from those before: {:}. An element's lumped inertia or damping depends "
                "on its state, so a run that moves elements would no longer be the serial run, which keeps "
                "them between rebuilds of its equation system. Run without load balancing (load-balance-tolerance=0).".format(
                    ", ".join(changed)
                )
            )

    def releaseEquationSystem(self) -> ExplicitSystem:
        """Release the equation system and everything built with it -- the counterpart of
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.buildEquationSystem`, and of :meth:`planIncrement` -- keeping only plain copies of
        the carried vectors (the solution, the velocity and the net force), and the critical time step.

        Everything released is indexed by the elements and constraints the system was built for:
        the DofManager (whose entity indices hold the elements themselves), every vector carrying
        those indices, the lumped operators, the subdomain, the increment plan, the interface
        assembly and the element timing. Before elements move to another process, so that nothing
        keeps a dropped element alive; the system is then built again, with the returned system as
        ``previous``.

        Returns
        -------
        ExplicitSystem
            The carried state (:meth:`ExplicitSystem.plainCopyOfCarried`).
        """

        carried = self._system.plainCopyOfCarried()
        self._system = None
        self._lumpedMass = self._rawLumpedMass = self._dampingRate = None
        self._secondOrderMask = self._halfDampingRate = None
        self._reusableOperators = None
        self._constraintForces = {}
        self.theDofManager = None
        self.subdomain = None
        self._incrementPlan = None
        self._integratedNodeFieldSlots = []
        self._interfaceAssembly = None
        self._elementCosts = None
        self._partsOfNodeSetsHere = {}
        self._forgetReceivePlans()
        return carried

    def writeIncrementOutput(self, fieldOutputController: FieldOutputController, outputManagers: list):
        """Collective. Write the output of an accepted increment, in three parts, so that no process
        communicates where another may have failed before.

        1. Each process reads its part of the output: the results of the elements it owns for the
           field outputs, and -- if a checkpoint is written now, which only rank 0, holding the output
           managers, knows, and each process holds only its own elements -- their states; failing on
           all ranks together if it fails on one.
        2. The parts are gathered: the results of every element set to rank 0 -- and to every
           process those the markers of a topology check following now read
           (:meth:`fieldOutputsGatheredToEveryProcess`) --, the element states to rank 0.
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

            # A process holding the nodes of its subdomain only holds only some rows of the node fields.
            gathersNodeFields = gathersStates and model.elementDistribution.holdsNodesOfSubdomainOnly

            with self.communicator.allRanksFailTogether("Reading the output"):
                fieldOutputController.readResultsHere()
                statesHere = self.domainDecomposition.elementStatesOwnedHere(model) if gathersStates else None
                nodeFieldsHere = nodeFieldValuesHere(model) if gathersNodeFields else None

            fieldOutputController.gatherResultsOfWholeSet(
                readOnEveryProcess=self.fieldOutputsGatheredToEveryProcess(),
                writesOutput=self.domainDecomposition.rank == 0,
            )
            with performancetiming.timeit("gather states"):
                states = self.domainDecomposition.gatherElementStatesToRoot(statesHere) if gathersStates else None
                nodeFields = (
                    model.elementDistribution.nodeFieldsOfWholeModel(model, nodeFieldsHere)
                    if gathersNodeFields
                    else None
                )

            with (
                model.elementStatesFromElsewhere(states),
                model.nodeFieldsFromElsewhere(nodeFields),
                self.communicator.allRanksFailTogether("Writing the output"),
            ):
                super().writeIncrementOutput(fieldOutputController, outputManagers)

    def applyStepActionsAtStepEnd(self, model: FEModel, stepActions: dict[str, StepActionBase]):
        """Collective. Make the whole model current in every process -- the end of a step, and the start of
        the next, read all of it -- then let the step actions finish the step.

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
    def publishNodeFields(self, model: FEModel, carried: dict[str, DofVector]):
        """Publish the subdomain degrees of freedom; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.publishNodeFields`. The
        others are current only in the processes integrating them, and are published by
        :meth:`_synchronizeModel` in the processes reading them -- rank 0 on an output increment,
        every process before a topology check -- and by :meth:`_receiveSolutionReadBySearches` at the
        nodes a contact search reads.

        Parameters
        ----------
        model
            The model tree.
        carried
            The carried vectors, by node field entry (:attr:`ExplicitSystem.carriedEntries`).
        """

        self._publishIntoNodeFields(self._integratedNodeFieldSlots, carried)

        U = carried["U"]
        for variable in model.scalarVariables.values():
            variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

    @staticmethod
    def _publishIntoNodeFields(slots: list[NodeFieldSlot], vectors: dict[str, DofVector]):
        """Publish the degrees of freedom of the slots into the node fields.

        Parameters
        ----------
        slots
            Where the degrees of freedom are published.
        vectors
            The vectors, by the node field entry each is published into.
        """

        for slot in slots:
            nodeField = slot.nodeField
            for entry, vector in vectors.items():
                if entry not in nodeField:
                    nodeField.createFieldValueEntry(entry)
                values = nodeField[entry]
                # Written through a flat view, which only a contiguous array has.
                if not values.flags.c_contiguous:
                    raise RuntimeError("Node field entry {:}/{:} is not contiguous.".format(nodeField.name, entry))
                values.reshape(-1)[slot.positions] = vector.asPlainArray()[slot.dofs]

    @staticmethod
    def _readFromNodeFields(slots: list[NodeFieldSlot], vectors: dict[str, DofVector]):
        """Read the degrees of freedom of the slots from the node fields: the inverse of
        :meth:`_publishIntoNodeFields`. An entry a node field does not hold is left as it is.

        Parameters
        ----------
        slots
            Where the degrees of freedom are published.
        vectors
            The vectors, by the node field entry each is read from; written.
        """

        for slot in slots:
            for entry, vector in vectors.items():
                if entry in slot.nodeField:
                    vector.asPlainArray()[slot.dofs] = slot.nodeField[entry].reshape(-1)[slot.positions]

    def _publishWholeVector(self, model: FEModel, entry: str, whole: np.ndarray):
        """Publish a vector of the whole model, in global order
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.wholeVector`), into
        an entry of every node field: a field holding every node carrying it holds the range of its degrees
        of freedom in the global numbering, its nodes in its order; a field of the nodes held here
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.isNodeHeldHere`) their degrees
        of freedom.

        Parameters
        ----------
        model
            The model tree.
        entry
            The node field entry.
        whole
            The vector of the whole model.
        """

        numbering = self.domainDecomposition.numbering
        for field, nodeField in model.nodeFields.items():
            if entry not in nodeField:
                nodeField.createFieldValueEntry(entry)
            values = nodeField[entry]
            first = numbering.firstDofOfField[field]
            rows = model.rowsOfNodeFieldInWholeField(nodeField, numbering)
            if rows is None:
                values[:] = whole[first : first + values.size].reshape(values.shape)
            else:
                # the nodes of the field held here
                dimension = nodeField.dimension
                values[:] = whole[first + rows[:, None] * dimension + np.arange(dimension)]

    def _subdomainNodeFieldSlots(self, model: FEModel) -> list[NodeFieldSlot]:
        """Where the degrees of freedom of vectors of the subdomain are published in the node fields of the
        model: each field's range of the vectors, at the rows of the subdomain's nodes of the field.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        list[NodeFieldSlot]
            One slot per node field.
        """

        rowsOfFields = self.domainDecomposition.rowsOfSubdomainNodeFields
        slots = []
        for field, nodeField in model.nodeFields.items():
            dofs = self.theDofManager.idcsOfFieldsInDofVector[field]
            subdomainRows = rowsOfFields[field]
            if subdomainRows.shape[0] == len(nodeField.nodes):
                positions = slice(None)
            else:
                # the places of the subdomain's nodes among the nodes of the node field, both in mesh order
                placed = np.searchsorted(self._meshRowsOfNodeField(model, nodeField), subdomainRows)
                positions = (placed[:, None] * nodeField.dimension + np.arange(nodeField.dimension)).ravel()
            slots.append(NodeFieldSlot(nodeField, positions, dofs))
        return slots

    @staticmethod
    def _meshRowsOfNodeField(model: FEModel, nodeField: NodeField) -> np.ndarray:
        """The rows in the mesh of the nodes of a node field of the model, in its order -- mesh order.

        Parameters
        ----------
        model
            The model tree.
        nodeField
            The node field.

        Returns
        -------
        np.ndarray
            The rows.
        """

        return model.mesh.rowsOfNodes([node.label for node in nodeField.nodes])

    def _partOfNodeSetHere(self, nodeSet, field: str) -> tuple[np.ndarray, np.ndarray]:
        """The part of a node set whose degrees of freedom of a field the vectors of the subdomain hold: its
        rows -- the positions of its nodes in the whole set, counting those carrying the field, as a step
        action on the set lays out its values -- and the positions of their degrees of freedom in the
        vectors. From what every process holds whole -- the labels of the set, the mesh, the global
        numbering --, not from node objects, of which a process holds only some.

        Parameters
        ----------
        nodeSet
            The node set.
        field
            The field.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The rows, and the positions, one row per node and one column per component.

        Raises
        ------
        RuntimeError
            If the subdomain holds some, but not every component of the field at a node of the set.
        """

        key = (nodeSet, field)
        if key not in self._partsOfNodeSetsHere:
            numbering = self.domainDecomposition.numbering
            meshRows = self.domainDecomposition.rowsOfNodes(nodeSet.nodeLabelsOfWholeSet())
            carrying = meshRows[numbering.nodesCarryingField[field][meshRows]]
            globalDofs = numbering.dofsAtNodes(field, carrying)
            subdomainDofs = self.domainDecomposition.interface.subdomainDofs
            positions = np.minimum(np.searchsorted(subdomainDofs, globalDofs), max(subdomainDofs.shape[0] - 1, 0))
            heldComponents = subdomainDofs[positions] == globalDofs if subdomainDofs.shape[0] else globalDofs < 0
            held = heldComponents.all(axis=1)
            if not np.array_equal(held, heldComponents.any(axis=1)):
                raise RuntimeError(
                    "the subdomain holds some, but not every component of field {:} at a node of a node set".format(
                        field
                    )
                )
            self._partsOfNodeSetsHere[key] = (np.flatnonzero(held), positions[held])
        return self._partsOfNodeSetsHere[key]

    def _constrainedDofsOf(self, dirichlet: StepActionBase) -> np.ndarray:
        """The degrees of freedom a Dirichlet condition constrains, in the vectors of this process; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase._constrainedDofsOf`.
        Vectors of the subdomain hold the part of a node set whose degrees of freedom this process
        integrates (:meth:`_partOfNodeSetHere`): the condition is imposed on that part, whose rows it
        states (:attr:`~edelweissfe.stepactions.base.dirichletbase.DirichletBase.constrainedRows`).

        Parameters
        ----------
        dirichlet
            The Dirichlet condition.

        Returns
        -------
        np.ndarray
            The positions of its constrained degrees of freedom, node-major.
        """

        if not self.vectorsOfSubdomain:
            return super()._constrainedDofsOf(dirichlet)
        rows, positions = self._partOfNodeSetHere(dirichlet.nSet, dirichlet.field)
        dirichlet.constrainedRows = rows
        return positions[:, dirichlet.components].ravel()

    def computeNodeForces(self, nodeForces: list[StepActionBase], PExt: DofVector, timeStep: TimeStep) -> DofVector:
        """Add the concentrated loads; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.computeNodeForces`.
        Vectors of the subdomain add them at the part of each node set whose degrees of freedom this
        process integrates (:meth:`_partOfNodeSetHere`).

        Parameters
        ----------
        nodeForces
            The concentrated (nodal) loads.
        PExt
            The external load vector.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The updated external load vector.
        """

        if not self.vectorsOfSubdomain:
            return super().computeNodeForces(nodeForces, PExt, timeStep)
        for nodeForce in nodeForces:
            rows, positions = self._partOfNodeSetHere(nodeForce.nodeSet, nodeForce.field)
            PExt[positions.ravel()] += nodeForce.getCurrentLoad(timeStep)[rows].flatten()
        return PExt

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
        """Collective. Receive the positions the contact searches of this process read, then run the
        periodic contact search; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.updateConstraintConnectivity`.

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
        """Collective. Receive the solution the periodic contact searches of this process read at the
        degrees of freedom it does not integrate, from their owners, and publish it into the node fields.
        The solution is the displacement of the nodes, which is what a search reads; the velocity and the
        net force there stay as they were.

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
            # the solution only, of the carried vectors: what a search reads
            searched.received.receive([self._system.U])
            self._publishIntoNodeFields(searched.slots, {"U": self._system.U})

    def _planSolutionReadBySearches(self, model: FEModel):
        """Collective. Plan what the periodic contact searches of this process read: the degrees of freedom,
        at the nodes each searched constraint names, that this process does not integrate; or the whole
        solution, if any constraint of the model whose connectivity is searched names none. The same
        decision in every process.

        Parameters
        ----------
        model
            The model tree.
        """

        searchable = self._dynamicConnectivityConstraints
        nodesRead = {name: constraint.nodesReadByConnectivityUpdate() for name, constraint in searchable.items()}
        self._searchesPlanned = True
        self._searchesReadWholeSolution = any(nodes is None for nodes in nodesRead.values())
        # every process must take the same branch below: either collective
        self.communicator.requireSameOnAllRanks(
            self._searchesReadWholeSolution, "whether the contact searches read the whole solution"
        )
        if self._searchesReadWholeSolution:
            self._solutionReadBySearches = None
            return

        searchedHere = self.domainDecomposition.constraintsSearchedHere(model, searchable)
        self._solutionReadBySearches = self._receivedSolutionAt(
            model, [node for name in searchedHere for node in nodesRead[name]]
        )

    def _receivedSolutionAt(self, model: FEModel, nodes: list) -> ReceivedSolution:
        """Collective. How this process receives the solution at the given nodes, at the degrees of freedom
        of every field there it does not integrate itself.

        Parameters
        ----------
        model
            The model tree.
        nodes
            The nodes, in any order and with repetitions.

        Returns
        -------
        ReceivedSolution
            The exchange, and where it is published.
        """

        dofsOfFields = self.theDofManager.idcsOfFieldVariablesInDofVector
        dofs = [
            dofsOfFields[fieldVariable]
            for node in nodes
            for fieldVariable in node.fields.values()
            if fieldVariable in dofsOfFields
        ]
        dofs = np.unique(np.concatenate(dofs)) if dofs else np.empty(0, dtype=np.int64)
        integrated = np.zeros(self.theDofManager.nDof, dtype=bool)
        integrated[self.subdomain.dofs] = True
        notIntegrated = dofs[~integrated[dofs]]
        return ReceivedSolution(
            self.domainDecomposition.valuesFromOwners(notIntegrated), self._nodeFieldSlotsOf(model, notIntegrated)
        )

    def _forgetReceivePlans(self):
        """Forget what this process receives at chosen nodes -- for the searches, for the rigid bodies
        --: planned for the subdomain the vectors were last defined for, and planned again at their
        next use."""

        self._searchesPlanned = False
        self._solutionReadBySearches = None
        self._rigidBodyReferences = None

    def _receiveRigidBodyReferences(self, model: FEModel):
        """Collective. Receive the carried vectors -- the solution, the velocity and the net force -- at the
        reference nodes of the rigid bodies from their owners, and publish them into the node fields:
        every process moves the surfaces of the rigid bodies with them, which are nodes of the model as
        well, and so part of what every process must hold alike (the layout fingerprint of
        :meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.define` compares
        their coordinates). A few degrees of freedom per body, on an output increment synchronized to
        rank 0 only.

        Parameters
        ----------
        model
            The model tree.
        """

        if self._rigidBodyReferences is None:
            self._rigidBodyReferences = self._receivedSolutionAt(
                model, [rigidBody.rpNode for rigidBody in model.rigidBodies.values()]
            )
        carried = self._system.carried
        self._rigidBodyReferences.received.receive(list(carried.values()))
        self._publishIntoNodeFields(self._rigidBodyReferences.slots, carried)

    def updateConnectivityOf(self, model: FEModel, constraints: dict) -> bool:
        """Collective. Let those of the given constraints update their connectivity whose search runs in
        this process, failing on all ranks together; whether a constraint of any process changed. See
        :meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.constraintsSearchedHere`.

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

        searched = self.domainDecomposition.constraintsSearchedHere(model, constraints)
        with self.communicator.allRanksFailTogether("Updating the constraint connectivity"):
            changed = super().updateConnectivityOf(model, searched)
            for name, constraint in searched.items():
                self._requireCoupledNodesAmongNodesRead(name, constraint)
        return self.communicator.allreduceAny(changed)

    @staticmethod
    def _requireCoupledNodesAmongNodesRead(name: str, constraint):
        """Refuse a constraint that, after its connectivity update, couples a node it does not name
        among those its update reads
        (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.nodesReadByConnectivityUpdate`):
        its search read the position of that node too, which this process may not have received.
        Nothing to check for a constraint naming none (it receives the whole solution), nor for one
        whose update reads no positions (its footprint is not the outcome of a search).

        Parameters
        ----------
        name
            The name of the constraint.
        constraint
            The constraint.

        Raises
        ------
        RuntimeError
            If it couples a node it does not name.
        """

        nodesRead = constraint.nodesReadByConnectivityUpdate()
        if not nodesRead:
            return
        labelsRead = {node.label for node in nodesRead}
        notNamed = sorted({node.label for node in constraint.nodes if node.label not in labelsRead})
        if notNamed:
            raise RuntimeError(
                "Constraint {:} couples {:} node(s) its connectivity update does not name among the nodes it reads "
                "(nodesReadByConnectivityUpdate), e.g. {:}: a domain-decomposed search would read their positions "
                "without having received them.".format(name, len(notNamed), notNamed[:5])
            )

    def updateTopology(self, model: FEModel, step, offerModelModifiers: bool) -> tuple[bool, bool]:
        """Collective. Run the model modifiers and the mesh refresh of
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateTopology` in every
        process, on the same synchronized model, failing on all ranks together if it fails on one; and
        check that every process arrived at the same outcome.

        If the topology changed, the subdomains are rebalanced for the changed mesh if that pays
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.rebalanceAfterTopologyChange`),
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

        self.domainDecomposition.requireConstraintCopiesCurrent(model)
        distribution = model.elementDistribution
        if offerModelModifiers:
            # what the modifiers read at every node, gathered now from the node fields (whole here, as
            # checked first): nothing may communicate in the update
            with self.communicator.allRanksFailTogetherWhileCommunicating("Gathering what the modifiers read"):
                self.requireWholeSolutionHere("Gathering what the model modifiers read")
                distribution.gatherNodeFieldEntries(
                    model,
                    set().union(*(modifier.nodeFieldEntriesRead() for modifier in model.modelModifiers.values())),
                )
        with self.communicator.allRanksFailTogether("Updating the topology"):
            # the modifiers and the refresh of the mesh read the node fields at every node held here
            self.requireWholeSolutionHere("A topology update")
            changed = super().updateTopology(model, step, offerModelModifiers)
        distribution.forgetGatheredNodeFieldEntries()
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
            # It may move elements; the mesh changed, so the vectors cannot be synchronized here as
            # before a migration of acceptIncrement -- a topology update follows a synchronization to
            # every process (requireWholeSolutionHere above), which is checked.
            if not self._copiesConsistentInEveryProcess():
                raise RuntimeError(
                    "A topology update may move elements between the processes, which rebuild their systems "
                    "from the copies of the constraints and rigid-body surfaces every process holds; they "
                    "were not synchronized to every process since the last increment was accepted."
                )
            # The changed mesh has the constraints assigned afresh, possibly to other processes, which
            # evaluate them from their copies: current, as just checked.
            self.domainDecomposition.constraintCopiesAreSynchronized()
            self.domainDecomposition.rebalanceAfterTopologyChange(model, horizon)
        if changed[0] or changed[1]:
            # Where each process holds the nodes of its subdomain only, a node made here in the update whose
            # values this process could not write -- a new node on the face of an element of another
            # process, a node held from now on for a reader or a hanging node -- receives them from a
            # process holding them, before the equation system is built from the node fields.
            with self.communicator.allRanksFailTogetherWhileCommunicating("Completing the values of new nodes"):
                distribution.completeNodesAwaitingValues(model)
        return changed

    def _synchronizeModel(self, model: FEModel, timeStep: TimeStep | None, includeStates: bool, toEveryProcess: bool):
        """Collective. Make the model complete before something reads all of it -- in every process, or on
        rank 0 only: every degree of freedom of the vectors, published into the node fields, and
        optionally every constraint state -- and, where every process holds every element, every element
        state, in every process -- as the process computing it last left it.

        A process not receiving the whole model keeps the vectors and the node fields complete at the
        degrees of freedom it integrates only, and its copies of the constraints owned elsewhere in
        the state of their last synchronization; :meth:`requireWholeSolutionHere` and
        :meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.requireConstraintCopiesCurrent`
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

        carried = self._system.carried
        # Rank 0 reads the whole model from its node fields -- unless each process holds the nodes of its
        # subdomain only: then every reader of the whole model on rank 0 gathers what it reads by node from
        # the processes computing it (a per-node field output, a checkpoint; see
        # ElementDistribution.nodeValuesOfWholeSet), and nothing reads the vectors gathered here.
        if not toEveryProcess and model.elementDistribution.holdsNodesOfSubdomainOnly:
            carried = {}
        with performancetiming.timeit("gather solution"):
            for entry, vector in carried.items():
                if self.vectorsOfSubdomain:
                    # gathered for this synchronization only, into the node fields, at the nodes held here
                    whole = self.domainDecomposition.wholeVector(vector, toEveryProcess)
                    if whole is not None:
                        self._publishWholeVector(model, entry, whole)
                elif toEveryProcess:
                    self.domainDecomposition.allgatherOwnedValues(vector)
                else:
                    self.domainDecomposition.gatherOwnedValuesToRoot(vector)
        if toEveryProcess and self.vectorsOfSubdomain:
            # every node held here holds the values of the whole solution now
            model.nodeValuesWritten(set(model.nodesAwaitingValues()))
        wholeSolutionHere = toEveryProcess or (self.domainDecomposition.rank == 0 and carried)
        if wholeSolutionHere:
            # every degree of freedom is current now, not only the subdomain ones
            if not self.vectorsOfSubdomain:
                super().publishNodeFields(model, carried)
            self._wholeSolutionHere = True

        if includeStates:
            self.domainDecomposition.synchronizeStates(includeElements=True, toEveryProcess=toEveryProcess)

        # A rigid body's surface follows its reference node, which only some processes integrated:
        # it was moved in acceptIncrement from what this process integrates, and is moved again now
        # from the complete solution -- in every process, which receives the reference nodes where it
        # does not receive the whole solution.
        if not toEveryProcess and model.rigidBodies:
            self._receiveRigidBodyReferences(model)
        if timeStep is not None:
            self.updateRigidBodies(model, timeStep)
        if includeStates and toEveryProcess:
            self._copiesConsistentAfterIncrement = self._incrementsAccepted

    def _copiesConsistentInEveryProcess(self) -> bool:
        """Whether the copies every process holds of what other processes compute are the same in
        every process: whether the model was synchronized to every process, with the states of the
        elements and constraints, since this solver last accepted an increment. The same in every
        process.

        Those copies are the constraints owned elsewhere -- their state, and with it the degrees of
        freedom they couple; a synchronization to rank 0 only leaves them in the state of the last
        synchronization elsewhere -- and the surfaces of the rigid bodies, which each process moves in
        every increment from the reference node as it holds it, current only where it integrates
        that node. Moving elements rebuilds the equation system of every process from them.

        Returns
        -------
        bool
            True if they are consistent.
        """

        return self._copiesConsistentAfterIncrement == self._incrementsAccepted

    def requireWholeSolutionHere(self, reader: str):
        """Refuse to let something read the whole solution -- the vectors at every degree of freedom,
        or the node fields at every node -- unless this process holds it as of the last accepted
        increment: after an output synchronization that reached it (:meth:`_synchronizeModel`), at
        the end of a step, or before the first increment. Between those, a process holds the solution
        at the degrees of freedom it integrates only, and rank 0 alone receives it on an output
        increment no topology check follows (:meth:`outputSynchronizationReachesEveryProcess`).

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
                "process must be known to NEDMPI.outputSynchronizationReachesEveryProcess.".format(
                    reader, self.domainDecomposition.rank
                )
            )

    def buildEquationSystem(self, model: FEModel, step, previous: ExplicitSystem = None) -> ExplicitSystem:
        """Collective. Build the equation system; see :meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.buildEquationSystem`.

        Built afresh, it reads the solution from the node fields, which must therefore be whole here
        (:meth:`requireWholeSolutionHere`). Rebuilt with the vectors carried over -- after a contact
        search moved a constraint, or elements moved between processes -- the subdomain is defined
        afresh, and the vectors are completed at the degrees of freedom this process integrates from
        now on (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.receiveNewlyIntegratedValues`).

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

        # read by carryVectorsOver, which completes the carried vectors at the degrees of freedom this
        # process integrates from now on
        self._interfaceBeforeRebuild = self.domainDecomposition.interface
        try:
            theSystem = super().buildEquationSystem(model, step, previous)
        finally:
            self._interfaceBeforeRebuild = None
        # as an accepted increment publishes what it integrates (see _receiveNewlyIntegratedValues)
        self._publishIntoNodeFields(self._integratedNodeFieldSlots, theSystem.carried)
        return theSystem

    def _receiveNewlyIntegratedValues(self, interfaceBefore, carried: dict[str, DofVector]):
        """Collective. Complete the carried vectors -- the solution, the velocity and the net force -- at
        the degrees of freedom this process integrates from now on, from their previous owners
        (:meth:`~edelweissfe.domaindecomposition.domaindecomposition.DomainDecomposition.receiveNewlyIntegratedValues`),
        and publish them into the node fields, as an accepted increment publishes what it integrates: the
        node fields hold the last accepted increment at every degree of freedom a process integrates,
        which the end of a failed step reads back (:meth:`_synchronizeAcceptedModel`).

        Parameters
        ----------
        interfaceBefore
            The interface of the subdomain before it changed.
        carried
            The carried vectors, by node field entry (:attr:`ExplicitSystem.carriedEntries`).
        """

        self.domainDecomposition.receiveNewlyIntegratedValues(interfaceBefore, list(carried.values()))
        self._publishIntoNodeFields(self._integratedNodeFieldSlots, carried)
