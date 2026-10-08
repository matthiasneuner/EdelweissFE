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
"""The nonlinear explicit dynamic solver.

**Reading this module.** The step drives the solver through :meth:`NED.beginStep`, then, per
increment, :meth:`NED.prepareIncrement` (a topology update when due, and the equation system:
the lumped mass and the stable time increment), :meth:`NED.attemptIncrement` (a contact search
when due, then the increment itself) and :meth:`NED.acceptIncrement` (the converged state
committed), and finally :meth:`NED.endStep`. The increment itself, :meth:`NED.solveIncrement`, is
the central-difference method in four named steps:

1. :meth:`NED.prescribeBoundaryMotion` -- the motion of the Dirichlet degrees of freedom;
2. :meth:`NED.updateVelocities` -- :math:`v_{n+1/2} = v_{n-1/2} + \\Delta t\\, M^{-1} f_n`;
3. :meth:`NED.advanceDisplacements` -- :math:`u_{n+1} = u_n + \\Delta t\\, v_{n+1/2}`;
4. :meth:`NED.assembleNetForce` -- :math:`f_{n+1} = f^{ext} + f^{c} - f^{int}` at :math:`u_{n+1}`:
   :meth:`NED.assembleInternalForces` over the elements, the loads, and the constraint forces.

The equation system is built by :meth:`NED.buildEquationSystem`, the lumped operators by
:meth:`NED._assembleLumpedOperators`. Everything else in the module is diagnostics and validation.

**The element loop, and the solver for production runs.** :meth:`NED.assembleInternalForces` is the
element loop as a textbook writes it: element after element, gather its solution, compute its nodal
forces, add them into the global vector at its degrees of freedom; the lumped operators are assembled
the same way (:meth:`NED.assembleLumpedDiagonal`). One element at a time is also the slowest way: the
gather and the scatter of one element cost about what a fast element kernel does.
:class:`~edelweissfe.solvers.nonlinearexplicitdynamicparallel.NEDParallel` runs the same loop in bulk,
with the same result bit for bit, also on one thread -- **use it for production runs**. ``NED`` says
so at the start of a step of a model with more than 5000 elements.

An increment solves no equation system. Second-order fields are advanced by central differences
(a leapfrog, with the velocity staggered half an increment behind the displacement), first-order
fields -- a nonlocal damage field, for instance -- by forward Euler, and both divide by a **lumped**
mass, so the cost of an increment is one element pass plus vector arithmetic. There is no linear
solver, no Newton loop, and no cutback: the time increment is set by stability, not by convergence,
so a material that fails to integrate at the stable step is reported as an error rather than
answered with a smaller step.

**Inertia and damping.** An element reports two diagonals: the coefficient of each field's second
time derivative (its inertia) and of its first (its damping). The integrator divides by the one and
relaxes with the other, and *which* scheme a field gets is read off them -- an inertia means central
differences, a damping alone means forward Euler. The deck declares no field ordering, because an
inertia is exactly what a central-difference update divides by. At zero damping the update is
bit-identical to the undamped one.

The diagnostics need one thing more, and it is about units: which entries of the inertia diagonal
may be summed. That belongs to the FIELD rather than to a deck, and is recorded in
:mod:`edelweissfe.config.phenomena` as a mass, a rotational inertia, or neither. Linear momentum
sums the first kind, the energy balance the first two; conservation across a topology change is
checked per field regardless.

The case that forces it: a first-order field's forward-Euler limit falls off with the SQUARE of the
element size, so on a refined mesh it, not the mechanical field, bounds the increment. Giving it an
inertia makes it second order -- a damped wave equation, limit linear in h -- with its old
coefficient keeping its value and becoming the damping. Marmot supplies one through the ``nonlocal
micro inertia`` element property; that inertia is a time squared, so ``0.5 m_k rate^2`` is a volume
rather than an energy and the field is registered as carrying a non-mechanical inertia, which keeps
it out of both balances.

``first-order-fields``, ``second-order-fields``, ``first-order-scheme`` and ``second-order-scheme``
are removed rather than deprecated: a deck carrying any of them now fails with ``Invalid option``,
and deleting those lines is the whole of the migration.

The stable increment is computed once per mesh from the element wave speeds and scaled by
``courant-number``. It is recomputed whenever the mesh changes and deliberately *not* recomputed
when it has not: as a material softens the true limit only grows, so reusing it is the conservative
direction, and re-deriving it costs a full element pass.

**Three cadences, all in increments.** ``output-frequency`` is how often progress is logged and
field outputs are finalized (a divisor, so it must be at least 1). ``topology-check-frequency`` is
how often model modifiers may change the mesh mid-run, and must be a multiple of
``output-frequency`` because a marker reads the last finalized output; ``0`` runs the topology
update only once, before the increment loop. ``contact-update-frequency`` is how often constraints
whose connectivity is the outcome of a search re-run it; ``0`` disables that. The reasoning behind
the latter two, and what they cost in accuracy, is documented under
:doc:`/documentation/adaptivitytheory` and :doc:`/documentation/contacttheory` respectively.

**Constraints.** Multi-point constraints are enforced by folding a slave's mass and force onto its
masters (row-sum, mass-conserving) and slaving it kinematically, which leaves the critical time step
untouched. Other constraints -- contact above all -- are evaluated through
:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.applyConstraintExplicit`, which
asks for forces and no tangent, since a tangent assembled here would enter nothing. A constraint
that can act *only* through its tangent contributes nothing to an explicit increment and is
refused at validation rather than silently ignored.

**Energy balance.** Every reporting increment prints the external work accumulated at prescribed
degrees of freedom, the kinetic energy, the internal (strain) energy, and what remains unaccounted.
Its purpose is one check that nothing else performs: kinetic energy cannot exceed external work,
because the terms omitted from the balance are all non-negative, so a violation means energy is
appearing from nowhere and the time step is above the true stability limit -- which the critical
time step does not see in full, as it never sees contact penalty stiffness, and sees the nonlocal
field only where that field carries a non-mechanical inertia. Such a field's own
``0.5 m_k rate^2`` is reported on a separate line and kept out of the balance: with a coefficient in
seconds squared it has the units of a volume, not of an energy.

Two honest limits, both of which the solver states in the log rather than leaving to be
discovered. The external work accumulates only at *prescribed* degrees of freedom, so a model
driven by body forces, node forces, a distributed load or an initial velocity has none, and the
check cannot fire; and the internal energy is whatever the materials publish as a strain energy,
which not every material does. When either is identically zero -- or the external work is
negative, a structure returning work at its boundary -- the balance is not usable as a
quasi-static criterion, and the honest substitute is to integrate a reaction force against its
prescribed displacement, both of which are available as ``saveHistory`` field outputs.

**Diagnostics across a topology change.** Because the lumped mass is the operator an increment
divides by, a refinement's effect on it is reported: total mass, per-component linear momentum and
kinetic energy before and after, plus the relative mass drift accumulated over every such change,
so that many individually-tolerable changes cannot silently add up to a meaningful one. The
smallest coefficient the increment divides by is reported alongside -- an inertia at a
second-order degree of freedom, a damping at a first-order one -- since that is what bounds the
step.

**The part of the model computed here.** The increment runs over the elements, constraints and
degrees of freedom of one object, a :class:`~edelweissfe.solvers.base.modelpartition.ModelPartition`
(:attr:`NED.partition`). This solver computes the whole model in one process, so its partition is
the whole model: every element and constraint, and every degree of freedom -- the slice
``slice(None)``, so that ``V[dofs]`` is the whole vector ``V``. A domain-decomposed solver
(:class:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI`) runs the same increment over one
subdomain per process, and adds the exchange between the subdomains in methods of its own.

**Restart.** A resumed run must not repeat the half-step that starts a leapfrog: the velocity in
the checkpoint already carries the half-increment offset, so applying the startup again would apply
one half-impulse too few. The solver therefore takes the previous time increment from the restart
state rather than synthesising it, and the velocity field is checkpointed in its own right -- an
implicit solver reconstructs everything it needs from the displacement, a central-difference scheme
does not.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.phenomena import carriesKineticEnergy, carriesLinearMomentum
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.assembly import addNodalForces, hasRepeatedDofs
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.conservationchecks import (
    CONSERVATION_TOLERANCE,
    ConservationCheck,
    formatMomentumAndKineticEnergy,
    linearMomentum,
)
from edelweissfe.solvers.base.modelpartition import ModelPartition
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.solvers.base.parallelelementcomputation import ElementPlan
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import CutbackRequest, StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField

#: Fractional margin by which the kinetic energy may exceed the external work before it is
#: reported as energy creation. KE <= W_ext is exact in the continuum, but the discrete run has
#: two legitimate sources of small violation: lumping the mass matrix, and the interpolate-then-
#: re-lump of a topology change, which does not conserve kinetic energy exactly (see
#: reportTopologyChangeConservation). One per cent is far above both and far below the runaway an
#: unstable time step produces -- v5 of the anchor pry-out reached 1e+38 mm.
_ENERGY_CREATION_TOLERANCE = 1e-2


@dataclass(frozen=True)
class NEDSchema:
    """The options of the ``*solver`` datalines and of an ``>>options`` block routed to this
    solver, owned by this module and never mutated from outside it.

    Mirrors :attr:`NED.SolverSpecificOptions` one-for-one; the plain ``self.options`` dict remains the actual
    source of truth consulted at runtime (see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema`
    for why). The hyphenated option names are not valid Python identifiers, hence the
    ``optionName`` indirection.

    Every option here is a property of the ANALYSIS -- how far below the stability limit to run,
    how often to report, how often to let the mesh or a contact search change. Nothing here
    *describes* the model: which fields exist, which time derivative each carries, and what its
    coefficient means are all read from the model and from
    :mod:`edelweissfe.config.phenomena`, never asked of the deck. See
    :meth:`NED._classifyFieldsByScheme`.

    The two ``expect-*-fields`` options are the one apparent exception, and are not one: they
    describe nothing and control nothing. They state what the deck's author believes the derivation
    will produce, and :meth:`NED._checkDerivedSchemeAgainstTheDeck` stops the run if it does not.
    Omitted -- as in every deck that predates them -- they assert nothing at all.
    """

    courantNumber: float | None = schemaField(
        description="The fraction of the critical time step actually used.",
        dtype=float,
        default=0.8,
        optionName="courant-number",
    )
    outputFrequency: int | None = schemaField(
        description=(
            "The increment interval at which progress is logged and field outputs are finalized. "
            "Must be at least 1: it is a divisor, so 0 does not disable reporting."
        ),
        dtype=int,
        default=1000,
        optionName="output-frequency",
    )
    topologyCheckFrequency: int | None = schemaField(
        description=(
            "The increment interval at which model modifiers are offered a chance to change the mesh. "
            "0 runs the topology update only once, before the increment loop. Must be a multiple of "
            "output-frequency, because a marker reads the last finalized field output."
        ),
        dtype=int,
        default=0,
        optionName="topology-check-frequency",
    )
    contactUpdateFrequency: int | None = schemaField(
        description=(
            "The increment interval at which constraints whose connectivity is the outcome of a "
            "search (contact) re-run that search. 0 disables periodic updates mid-run."
        ),
        dtype=int,
        default=100,
        optionName="contact-update-frequency",
    )
    expectSecondOrderFields: list | None = schemaField(
        description=(
            "Fields the deck expects to be integrated SECOND order in time (central difference), "
            "as a comma-separated list. An ASSERTION, not a control: the scheme is always derived "
            "from the assembled inertia and damping, and this is compared against that derivation "
            "afterwards, never consulted to decide anything. A named field that turns out first "
            "order aborts the step. Optional; fields not named are not checked."
        ),
        dtype=str,
        default=None,
        optionName="expect-second-order-fields",
    )
    expectFirstOrderFields: list | None = schemaField(
        description=(
            "Fields the deck expects to be integrated FIRST order in time (forward Euler), as a "
            "comma-separated list. The counterpart of expect-second-order-fields and checked the "
            "same way. Optional; fields not named are not checked."
        ),
        dtype=str,
        default=None,
        optionName="expect-first-order-fields",
    )
    reportPerformance: bool = schemaField(
        description=(
            "Print the cumulative performance table (totals since the start of the step) on every "
            "progress report. Off by default. An explicit run is millions of increments long, so "
            "the table printed at the end of the step is otherwise the only one anyone sees."
        ),
        dtype=bool,
        default=False,
        optionName="report-performance",
    )
    lumpedQuantityConservationTolerance: float | None = schemaField(
        description=(
            "Relative-change tolerance for the per-topology-change conservation check on a "
            "row-sum-lumped quantity (mass, a first-order field's viscosity, a second-order "
            "field's non-mechanical inertia). Children of a refined element tile it and carry the "
            "same value, so this is conserved geometrically; the default is already generous "
            "relative to floating-point precision and exists to catch a genuinely wrong refinement "
            "or lumping, not to absorb ordinary quadrature noise. Raise this only when a specific, "
            "understood run trips it by a small margin (see "
            "edelweissfe.solvers.base.conservationchecks)."
        ),
        dtype=float,
        default=CONSERVATION_TOLERANCE,
        optionName="lumped-quantity-conservation-tolerance",
    )


@dataclass
class ExplicitSystem:
    """Everything an explicit increment operates on that is sized by the current equation system.

    None of these are independent state: each is indexed by the :class:`DofManager` in force when it
    was created, so the moment that system changes -- an h-adaptivity event creating nodes and
    elements, a contact constraint re-assigning its slave nodes to different master facets -- every
    one of them has to be rebuilt together, and any that is not becomes either a length mismatch or,
    worse, a silently mis-indexed vector. Bundling them makes "rebuild the system" one assignment at
    the call site instead of nine, which is what keeps the build before the increment loop and a
    rebuild inside it the same code path rather than two that drift apart.

    Parameters
    ----------
    Minv
        The inverse lumped mass. Zero on multi-point-constraint slave DOFs, which integrate no
        equation of motion of their own. The lumped mass it was built from is not carried here:
        the diagnostics need the *unfolded* mass, which is held as ``_rawLumpedMass``, and a
        second mass on this dataclass that no caller reads would only invite reading the wrong
        one.
    U
        The solution vector.
    dU
        The solution increment of the increment being computed.
    V
        The velocity vector, staggered half an increment behind ``U``.
    P
        The net nodal force -- external minus internal -- which drives the velocity update.
    criticalTimeStep
        The stable time increment for the current mesh, already scaled by the courant number.
    """

    Minv: DofVector
    U: DofVector
    dU: DofVector
    V: DofVector
    P: DofVector
    criticalTimeStep: float


@dataclass
class _ReusableExplicitOperators:
    """The lumped operators of an explicit system, and the model they were assembled from.

    A rebuild asked for by a constraint's connectivity re-assigns which nodes that constraint
    couples, and touches nothing the lumped inertia, its inverse, the mass-proportional damping
    rate or the multi-point-constraint transformation are a function of -- so those are exactly
    what they were, and :meth:`NED.buildEquationSystem` keeps them here instead of assembling
    them again.

    The recorded model description is what :meth:`NED._operatorsReusable` checks before it does,
    and anything that does not match makes it assemble from scratch instead.
    """

    elementKeys: frozenset
    multiPointConstraintKeys: frozenset
    nNodes: int
    nScalarVariables: int
    lumpedMass: np.ndarray
    rawLumpedMass: np.ndarray
    inverseLumpedMass: np.ndarray
    dampingRate: np.ndarray
    mpcTransformation: MultiPointConstraintTransformation | None


#: From how many elements with kernels on, serial NED recommends NEDParallel at the start of a step.
_ELEMENTS_WORTH_THE_PLAN = 5000


@dataclass(frozen=True)
class IncrementPlan:
    """What every increment needs prepared, once per equation system: the element loop, the
    first-order degrees of freedom, and the fold of the multi-point-constraint slave forces. Derived
    again by :meth:`NED.planIncrement` whenever the equation system or the partition changes.

    Parameters
    ----------
    elementLoop
        The element loop over the elements computed here that have kernels, as the solver runs it
        (:meth:`NED.planElementLoop`).
    firstOrderDofs
        The first-order degrees of freedom integrated here.
    mpcForceFold
        The fold of the multi-point-constraint slave forces onto their masters, as a matrix on the
        degrees of freedom integrated here; None without multi-point constraints. See
        :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.foldExplicitForceOperator`.
    """

    elementLoop: list[tuple] | ElementPlan
    firstOrderDofs: np.ndarray
    mpcForceFold: csr_matrix | None


@dataclass(frozen=True)
class ConstraintForce:
    """The nodal forces of one constraint, evaluated into a buffer of its own, and where they act.

    Built once per equation system rather than per increment -- this runs on the explicit hot path,
    tens of thousands of times -- and invalidated by exactly one event, a rebuild of the DofManager,
    which is where :attr:`NED._constraintForces` is cleared.

    Parameters
    ----------
    forces
        The force buffer, one entry per degree of freedom of the constraint.
    dofs
        The degrees of freedom they act on.
    namesDofMoreThanOnce
        Whether a degree of freedom appears in ``dofs`` more than once; see
        :func:`~edelweissfe.numerics.assembly.addNodalForces`.
    """

    forces: np.ndarray
    dofs: np.ndarray
    namesDofMoreThanOnce: bool

    def addInto(self, vector: np.ndarray):
        """Add the forces into a vector.

        Parameters
        ----------
        vector
            The net nodal force vector, as a plain array.
        """

        addNodalForces(vector, self.dofs, self.forces, self.namesDofMoreThanOnce)


class NED(NonlinearSolverBase):
    """This is the Nonlinear Explicit Dynamic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDSolver"

    supportsMPC = True

    #: This solver supports model modifiers both before the increment loop (initialOnly modifiers)
    #: and mid-run on a periodic cadence via the ``topology-check-frequency`` solver option.
    supportsModelModifiers = True

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NEDSchema

    SolverSpecificOptions = {
        "courant-number": 0.8,
        "output-frequency": 1000,
        "contact-update-frequency": 100,
        "topology-check-frequency": 0,
        "report-performance": False,
        "lumped-quantity-conservation-tolerance": CONSERVATION_TOLERANCE,
        # Lists, so _updateOptions appends the comma-separated items. Empty means "assert nothing", which is what
        # every deck that does not mention them gets.
        "expect-second-order-fields": [],
        "expect-first-order-fields": [],
    }

    #: The last completed increment (the central-difference velocity update reads its length) and the
    #: accumulated external work -- an accumulator, summed increment by increment from the reaction
    #: forces at the prescribed degrees of freedom, which nothing in a converged solution reproduces.
    checkpointedState = {
        "prevTimeStep": TimeStep,
        "_externalWork": float,
        # The energy-balance warnings are given once per step, not again after a resume.
        "_warnedAboutMissingInternalEnergy": bool,
        "_warnedAboutMissingExternalWork": bool,
        "_conservationCheck": ConservationCheck,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # Ensure mutable defaults (field lists) are isolated per solver instance.
        self.options = deepcopy(self.SolverSpecificOptions)
        self._updateOptions(kwargs, journal, strict=True)
        #: Fields integrated first order (forward Euler) and second order (central difference) in
        #: time, and their degrees of freedom. All four are DERIVED from the assembled inertia and
        #: damping in :meth:`_classifyFieldsByScheme`, never declared.
        self.firstOrderFields = []
        self.secondOrderFields = []
        self.ids_1st = None
        self.ids_2nd = None
        #: Second-order DOFs whose 0.5*m*v**2 is a mechanical energy -- the mass-carrying ones and
        #: the rotational ones -- and equally the DOFs at which prescribed work is an energy. The
        #: energy balance uses these at both ends; see phenomena.carriesKineticEnergy.
        self.ids_mechanicalEnergy = None
        #: Second-order fields whose inertia is a mass, in declaration order; the per-component
        #: momentum diagnostic walks these.
        self.linearMomentumFields = []
        #: Second-order fields whose inertia is neither a mass nor a rotational inertia, in
        #: declaration order. Each is reported on its own line of the energy table and enters
        #: neither balance; see phenomena.inertiaKind.
        self.nonMechanicalSecondOrderFields = []
        #: Mass-proportional damping rate C/M per degree of freedom. Zero wherever the elements
        #: reported no damping, which is what makes the one damped update rule below reduce
        #: exactly to the undamped central difference there.
        self._dampingRate = None
        #: Half the damping rate on second-order DOFs, zero elsewhere; see solveIncrement.
        self._halfDampingRate = None
        #: 1.0 on second-order DOFs, 0.0 elsewhere; see solveIncrement.
        self._secondOrderMask = None
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False
        #: The per-topology-change conservation check of every field's lumped total, and the
        #: drift those changes accumulate over a step; reset at every step start.
        self._conservationCheck = ConservationCheck(journal, self.identification)
        #: Work done on the model by its prescribed degrees of freedom, accumulated every
        #: increment (:meth:`addExternalWork`). Compared against the kinetic energy to detect
        #: energy creation; see _ENERGY_CREATION_TOLERANCE.
        self._externalWork = 0.0
        #: The last completed increment. The central-difference velocity update reads
        #: 0.5 * (dT + dT_prev); None makes the first increment of a cold step the half step that
        #: starts a leapfrog. A resumed step continues from the checkpointed one instead: the
        #: checkpointed velocity already carries the half-step offset.
        self.prevTimeStep = None
        #: The force buffer of every constraint evaluated here, by name; see
        #: :class:`ConstraintForce`. Cleared whenever the DofManager is rebuilt.
        self._constraintForces = {}
        #: What every increment needs prepared; see :class:`IncrementPlan` and :meth:`planIncrement`.
        #: None until the first equation system is built: an increment without a plan would
        #: evaluate no elements at all, and go on quietly producing wrong answers.
        self._incrementPlan = None
        #: The lumped operators of the current equation system, kept across a rebuild that only a
        #: constraint's connectivity asked for; see :class:`_ReusableExplicitOperators`.
        self._reusableOperators = None
        #: The elements, constraints and degrees of freedom this process computes; defined with every
        #: equation system, see :meth:`partitionModel`.
        self.partition = None

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Start the step; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.beginStep`. The equation
        system is built by the first :meth:`prepareIncrement`.

        Parameters
        ----------
        step
            The step to be solved.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """

        self.reportHowElementsAreComputed(model)
        self.validateModelCapabilities(model)

        # Against this the timing table reports what it did *not* measure, so it has to span
        # everything this method does -- the initial topology refinement and the first equation
        # system included, since both are timed categories that would otherwise be subtracted from a
        # window they never ran in and drive the residue negative.
        self._stepWallClockTic = perf_counter()

        self._externalWork = 0.0
        self.prevTimeStep = None
        self._conservationCheck.reset()
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False

        # Constraints whose DOF footprint is the outcome of a search, i.e. contact, by name. Collected
        # once, so a model without any pays nothing for the per-increment tick in the loop below.
        self._dynamicConnectivityConstraints = {
            name: constraint
            for name, constraint in model.constraints.items()
            if type(constraint).updateConnectivity is not ConstraintBase.updateConnectivity
        }

        # Modifiers that can still act once the analysis is running. Collected once, so a model whose
        # refinement is all up-front pays nothing for the periodic check below.
        self._liveTopologyModifiers = [
            modifier
            for modifier in model.modelModifiers.values()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        ]

        # Step actions before the equation system, matching NIST: nothing they do depends on it.
        self.applyStepActionsAtStepStart(model, step)

        self._system = None

    def prepareIncrement(self, step, model: FEModel, isRetry: bool):
        """The topology update, when due, and the equation system; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.prepareIncrement`.

        The topology update is due at the start of the step, and after every
        ``topology-check-frequency``-th increment. At the start of the step it runs before anything
        sized by the equation system exists, so the mesh is final before the lumped mass, the
        multi-point-constraint condensation and the critical time step are derived from it. It runs
        here, before the next increment is proposed, because a refinement may lower the stable time
        increment that increment has to use.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        isRetry
            Never True: this solver does not retry increments.
        """

        if step.timeStepper.isAtStepStart():
            # The step-start topology update: the modifiers acting at the start of the analysis (e.g.
            # hAdaptivity's initialOnly markers) and the first contact search.
            self.updateTopologyAndConnectivity(model, step)

        if self._system is None:
            self._buildSystem(self.buildEquationSystem(model, step))
            # The time stepper runs on the stable time increment from here on, which it carries (and
            # checkpoints) itself: a resumed step continues with the one it had.
            if step.timeStepper.enforcedTimeIncrement is None:
                step.timeStepper.enforceTimeIncrement(self._system.criticalTimeStep)

        # --- h-adaptivity, mid-run ----------------------------------------------------------------
        # After every topology-check-frequency-th increment, so after its output: the marker refines
        # on the last *finalized* field output, so anywhere earlier it would decide on stale results.
        # And the pairing of U with the half-step-staggered V is unambiguous only between increments.
        # The zero increment is never recorded as the last increment, so a live marker is never
        # evaluated on the initial condition.
        if not self.topologyCheckDueAfter(self.prevTimeStep):
            return

        theSystem, V = self._system, self._V
        lumpedTotalsBefore = self._perFieldLumpedTotals()
        momentumBefore = self.secondOrderMomentum(self._rawLumpedMass, V, model)
        kineticBefore = 0.5 * float(
            np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
        )

        topologyUpdate = self.updateTopologyAndConnectivity(model, step)
        meshChanged = topologyUpdate.topologyChanged or topologyUpdate.meshDependentsRefreshed
        connectivityChanged = topologyUpdate.constraintConnectivityChanged

        if meshChanged or connectivityChanged:
            # Only a mesh change needs a system built afresh. A change of the contact
            # connectivity alone carries solution, velocity and force over, as after a
            # periodic contact search.
            self._buildSystem(self.buildEquationSystem(model, step, previous=None if meshChanged else theSystem))
            theSystem = self._system

        U, V, P = self._U, self._V, self._P

        if meshChanged:
            # The net force is deliberately NOT re-evaluated on the new mesh. It
            # could be, with one extra element pass -- but that would run the
            # constitutive law off-cycle, with a zero strain increment, purely to
            # obtain a force, and the material state is what that call writes into.
            # Zeroing costs exactly one increment of force contribution to the
            # velocity update: an O(dT) error confined to the increment following an
            # event, after which it is computed normally. A bounded known error is
            # preferable to an unbounded unknown one.
            P[:] = 0.0
            self.publishNodeFields(model, U, V, P)

            self.reportTopologyChangeConservation(lumpedTotalsBefore, momentumBefore, kineticBefore, V, model)

            # Lower only. Refinement shrinks the smallest element and tightens the
            # limit, which must be honoured; softening raises it, and taking that up
            # mid-step would change the integrator's dispersion for no benefit.
            if theSystem.criticalTimeStep < step.timeStepper.enforcedTimeIncrement:
                self.journal.message(
                    "Refinement lowered the stable time increment from {:e} to "
                    "{:e}".format(step.timeStepper.enforcedTimeIncrement, theSystem.criticalTimeStep),
                    self.identification,
                    1,
                )
                step.timeStepper.enforceTimeIncrement(theSystem.criticalTimeStep)

    def topologyCheckDueAfter(self, timeStep: TimeStep | None) -> bool:
        """Whether the mid-run topology check is due after an increment: after every
        ``topology-check-frequency``-th one, if a model modifier can act once the analysis is running.
        It runs at the start of the next increment (:meth:`prepareIncrement`), and since the frequency
        is a multiple of ``output-frequency``, always right after an output increment.

        Parameters
        ----------
        timeStep
            The last completed increment, or None before the first one.

        Returns
        -------
        bool
            Whether the topology check is due.
        """

        topologyCheckFrequency = self.options["topology-check-frequency"]
        return bool(
            self._liveTopologyModifiers
            and topologyCheckFrequency
            and timeStep is not None
            and timeStep.number % topologyCheckFrequency == 0
        )

    def _buildSystem(self, theSystem):
        """Adopt a (re)built equation system, and the vectors the increments work on.

        Parameters
        ----------
        theSystem
            The equation system.
        """

        self._system = theSystem
        self._Minv = theSystem.Minv
        self._U, self._dU, self._V, self._P = theSystem.U, theSystem.dU, theSystem.V, theSystem.P

    def isOutputIncrement(self, timeStep: TimeStep) -> bool:
        """Only every ``output-frequency``-th increment: an explicit run has millions.

        Parameters
        ----------
        timeStep
            The accepted increment.

        Returns
        -------
        bool
            True to write output.
        """

        return timeStep.number % self.options["output-frequency"] == 0

    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """The periodic contact search, when due, and the central-difference update; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.attemptIncrement`.

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
        StepFailed
            If a material requests a cutback: an explicit increment is not retried smaller.
        """

        contactUpdateFrequency = self.options["contact-update-frequency"]
        topologyCheckFrequency = self.options["topology-check-frequency"]
        theSystem, Minv = self._system, self._Minv
        U, dU, V, P = self._U, self._dU, self._V, self._P

        # only print for increments matching the configured output-frequency
        if timeStep.number % self.options["output-frequency"] == 0:
            self.journal.printSeperationLine()
            self.journal.message(
                "increment {:}: {:8e}, {:8e}; time {:10e} to {:10e}".format(
                    timeStep.number,
                    timeStep.stepProgressIncrement,
                    timeStep.stepProgress,
                    timeStep.totalTime - timeStep.timeIncrement,
                    timeStep.totalTime,
                ),
                self.identification,
                level=1,
            )

            if self.options["report-performance"]:
                # The cumulative table, as at the end of the step: an explicit run is
                # millions of increments long, and without this the final table is the
                # only one anyone would ever see.
                self.journal.printPrettyTable(
                    performancetiming.makePrettyTable(wallTime=perf_counter() - self._stepWallClockTic),
                    self.identification,
                )

        # The mid-run topology check at the end of this same increment re-runs the
        # connectivity search on EVERY constraint, these included. Searching here as well
        # would build and query the same k-d tree twice within one increment: the later
        # search is the one that has to happen, because a refinement in between invalidates
        # whatever this one found. Deferring to it costs one increment of staleness -- the
        # same staleness the configured frequency already accepts, and orders of magnitude
        # below a facet dimension at an explicit time step.
        topologyCheckDueThisIncrement = bool(
            self._liveTopologyModifiers
            and topologyCheckFrequency
            and timeStep.number > 0
            and timeStep.number % topologyCheckFrequency == 0
        )

        if (
            self._dynamicConnectivityConstraints
            and contactUpdateFrequency
            and timeStep.number > 0
            and timeStep.number % contactUpdateFrequency == 0
            and not topologyCheckDueThisIncrement
        ):
            connectivityChanged = self.updateConstraintConnectivity(model)

            if connectivityChanged:
                self.journal.message("Constraint connectivity changed", self.identification, 2)
                self._buildSystem(self.buildEquationSystem(model, step, previous=theSystem))
                theSystem, Minv = self._system, self._Minv
                U, dU, V, P = self._U, self._dU, self._V, self._P

        dU[:] = 0.0
        try:
            U, V, P = self.solveIncrement(
                U,
                dU,
                V,
                P,
                Minv,
                step.actions,
                model,
                timeStep,
                self.prevTimeStep,
            )

        except CutbackRequest as e:
            # A cutback answers a CONVERGENCE failure, and an explicit scheme has no
            # convergence to fail: its time step is dictated by stability, courant *
            # dt_crit from the mesh and the wave speed. Shrinking it does nothing for a
            # material that could not integrate, and doing so was actively destructive --
            # rejecting the increment overwrites the enforced time increment with the reduced
            # value, every following increment reuses it, and
            # nothing raises it back (the critical step is enforced "lower only", and
            # SimpleTimeStepper.preventIncrementIncrease is a no-op). One failed
            # quadrature point permanently crippled the analysis: of three production runs
            # of the anchor pry-out model, two cut back to minInc and died, and the third
            # spent 291000 of 300000 increments at ~1e-14 s, covering 6e-09 s of loading.
            #
            # So the request is refused and the failure is surfaced where it happened.
            raise StepFailed(
                "A material requested a cutback in increment {:}: {:}. The explicit time "
                "step is set by stability, not by convergence, so it cannot be reduced in "
                "response -- either the material cannot integrate at the stable step, or "
                "the state reaching it is already wrong. Both need the material or the "
                "model looked at, not a smaller step.".format(timeStep.number, e)
            ) from e

        self._U, self._V, self._P = U, V, P

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Commit the increment to the model; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.acceptIncrement`.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.
        """

        U, V, P = self._U, self._V, self._P

        # The zero increment, before the first real one, is not a completed step: the
        # velocity update returns early for it, and the first real increment must be the
        # half step that starts the leapfrog.
        if timeStep.timeIncrement > 0.0:
            self.prevTimeStep = timeStep

        self.publishNodeFields(model, U, V, P)

        self.updateRigidBodies(model, timeStep)

        # Timed because it is not what it looks like. FEModel.advanceToTime is a generic
        # method shared with the implicit solvers, where it runs once per *converged*
        # increment and is amortised over a Newton loop; here it runs on every one of
        # millions of increments, and it is a serial Python loop over every element,
        # constraint and multi-point constraint in the model.
        with performancetiming.timeit("accept state"):
            self.advanceModelToTime(model, timeStep.totalTime)

    def advanceModelToTime(self, model: FEModel, time: float):
        """Let the model accept the state the increment computed, and advance it to the end of the
        increment (:meth:`~edelweissfe.models.femodel.FEModel.advanceToTime`).

        Parameters
        ----------
        model
            The model tree.
        time
            The time at the end of the increment.
        """

        model.advanceToTime(time)

    def endStep(self, step, model: FEModel, failure: BaseException | None = None):
        """Report the step's performance timing.

        Parameters
        ----------
        step
            The step that was solved.
        model
            The model tree.
        failure
            How the step failed: the exception leaving it; None if it ended normally (a
            conditional stop and the maximum number of increments included).
        """

        prettyTable = performancetiming.makePrettyTable(wallTime=perf_counter() - self._stepWallClockTic)
        self.journal.printPrettyTable(prettyTable, self.identification)
        performancetiming.reset()

    @performancetiming.timeit("increment")
    def solveIncrement(
        self,
        U: DofVector,
        dU: DofVector,
        V: DofVector,
        P: DofVector,
        Minv: DofVector,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
    ) -> tuple[DofVector, DofVector, DofVector]:
        """One increment of the central-difference method, from :math:`t_n` to
        :math:`t_{n+1} = t_n + \\Delta t`.

        On entry, ``U`` holds :math:`u_n`, ``V`` the velocity :math:`v_{n-1/2}` half an increment
        behind it, and ``P`` the net nodal force :math:`f_n = f^{ext}_n - f^{int}_n` at :math:`u_n`.
        The increment is four steps:

        1. :meth:`prescribeBoundaryMotion` -- at the Dirichlet degrees of freedom the motion is
           given, not integrated: the reaction there does work on the model, and the velocity is
           the prescribed one;
        2. :meth:`updateVelocities` -- :math:`v_{n+1/2} = v_{n-1/2} + \\Delta t_{n+1/2} M^{-1} f_n`,
           with :math:`\\Delta t_{n+1/2}` the mean of this increment and the previous one (forward
           Euler for a first-order field), then :meth:`imposeKinematicConstraints`;
        3. :meth:`advanceDisplacements` -- :math:`u_{n+1} = u_n + \\Delta t\\, v_{n+1/2}`;
        4. :meth:`assembleNetForce` -- :math:`f_{n+1}` at :math:`u_{n+1}`, which the next increment
           starts from.

        Everything runs over the degrees of freedom integrated in this process
        (:attr:`partition`): every one, in a model computed whole.

        Parameters
        ----------
        U
            The solution vector; advanced in place.
        dU
            The solution increment vector; overwritten.
        V
            The velocity vector, half an increment behind ``U``; advanced in place.
        P
            The net nodal force of the previous increment; overwritten with this one's.
        Minv
            The inverse of the lumped coefficient each degree of freedom divides by: an inertia at
            a second-order one, a damping at a first-order one.
        stepActions
            The active step actions, by type.
        model
            The model tree.
        timeStep
            The time step.
        prevTimeStep
            The previous time step; None in the first increment of a cold step, which makes it the
            half step that starts the leapfrog.

        Returns
        -------
        tuple[DofVector, DofVector, DofVector]
            The solution, the velocity and the net nodal force.
        """

        dirichlets = stepActions["dirichlet"].values()

        # Find which global DOFs the Dirichlet BCs constrain, once up front.
        self.locateConstrainedDofs(dirichlets)

        if timeStep.timeIncrement == 0.0:
            return U, V, P

        dT = timeStep.timeIncrement
        dTPrevious = 0.0 if prevTimeStep is None else prevTimeStep.timeIncrement

        with performancetiming.timeit("kinematic update"):
            prescribedVelocities = self.prescribeBoundaryMotion(dirichlets, V, P, timeStep)
            self.updateVelocities(V, P, Minv, 0.5 * (dT + dTPrevious))
            self.imposeKinematicConstraints(V, prescribedVelocities)
            self.advanceDisplacements(U, dU, V, dT)

        with performancetiming.timeit("step actions"):
            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

        P, psi = self.assembleNetForce(U, dU, P, stepActions, timeStep)

        if timeStep.number % self.options["output-frequency"] == 0:
            self.reportEnergyBalance(psi, V, timeStep)

        return U, V, P

    def prescribeBoundaryMotion(
        self, dirichlets: list, V: DofVector, P: DofVector, timeStep: TimeStep
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Prescribe the motion of the Dirichlet degrees of freedom: record the work their reactions
        do on the model, set their force to zero -- there is no equation of motion to solve there --
        and their velocity to the prescribed increment over the time increment.

        Parameters
        ----------
        dirichlets
            The Dirichlet conditions, with their degrees of freedom located.
        V
            The velocity vector; written at the prescribed degrees of freedom.
        P
            The net nodal force of the previous increment; zeroed at the prescribed degrees of
            freedom.
        timeStep
            The time step.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            Per Dirichlet condition, its degrees of freedom and their prescribed velocity, for
            :meth:`imposeKinematicConstraints`.
        """

        prescribedVelocities = []
        workedDofs = []
        workProducts = []
        for dirichlet in dirichlets:
            prescribedIncrement = dirichlet.getPrescribedIncrement(timeStep).flatten()

            # The work put into the model at a prescribed degree of freedom is the reaction force there
            # times the motion it is dragged through. At this point P holds the carried-over net nodal
            # force from the previous increment (external minus internal plus constraint
            # contributions), and the support reaction balancing it is its negative -- so this has to
            # be recorded HERE, immediately before the zeroing below, which is the only moment the
            # reaction is available.
            #
            # Only where that product is an energy. A prescribed non-local damage, say, has a
            # conjugate "force" in the units its own balance equation carries, and adding it here
            # would put cubic millimetres into a total that is then compared against a kinetic energy
            # in newton-millimetres -- the same units error the kinetic sum excludes it from, at the
            # other end of the same balance.
            if carriesKineticEnergy(dirichlet.field):
                workedDofs.append(dirichlet.constrainedDofIndices)
                workProducts.append(P[dirichlet.constrainedDofIndices] * prescribedIncrement)

            prescribedVelocity = prescribedIncrement / timeStep.timeIncrement

            P[dirichlet.constrainedDofIndices] = 0.0
            V[dirichlet.constrainedDofIndices] = prescribedVelocity
            prescribedVelocities.append((dirichlet.constrainedDofIndices, prescribedVelocity))

        self.addExternalWork(
            np.concatenate(workedDofs) if workedDofs else np.empty(0, dtype=int),
            np.concatenate(workProducts) if workProducts else np.empty(0),
        )

        return prescribedVelocities

    def addExternalWork(self, dofs: np.ndarray, reactionTimesIncrement: np.ndarray):
        """Add the work done at the prescribed degrees of freedom in one increment to the external
        work.

        The reaction is the negative of the net nodal force there, so the work is minus the sum of
        the products of force and prescribed increment. That sum is formed exactly and rounded once
        (:func:`math.fsum`): its value does not depend on the order of the products, which is what
        makes it the same, bit for bit, however a model is split over processes
        (:class:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI`).

        Parameters
        ----------
        dofs
            The prescribed degrees of freedom whose work is an energy.
        reactionTimesIncrement
            Per degree of freedom, the net nodal force times the prescribed increment.
        """

        self._externalWork -= math.fsum(reactionTimesIncrement.tolist())

    def updateVelocities(self, V: DofVector, P: DofVector, Minv: DofVector, dTAverage: float):
        """Advance the velocity of every degree of freedom integrated here by the net nodal force.

        Second order (central difference, with mass-proportional damping):

        .. math:: v_{n+1/2} = \\frac{(1 - h)\\, v_{n-1/2} + \\Delta t_{n+1/2} M^{-1} f_n}{1 + h},
            \\qquad h = \\tfrac{1}{2} \\alpha\\, \\Delta t_{n+1/2},

        with the damping rate :math:`\\alpha = C/M`. First order (forward Euler):
        :math:`\\dot{d} = C^{-1} f_n`, the inverse damping being what ``Minv`` holds there.

        Parameters
        ----------
        V
            The velocity vector; advanced in place.
        P
            The net nodal force.
        Minv
            The inverse lumped inertia (second order) or damping (first order).
        dTAverage
            The mean of this time increment and the previous one; half this one in the first
            increment of a cold step.
        """

        # The rate alpha = C/M enters as one factor on each side rather than as an extra force
        # evaluation, so a damped degree of freedom costs what an undamped one does. At alpha = 0 it
        # is bit-identical to the undamped update, which is why there is no branch -- and why
        # mechanical Rayleigh damping would need no code here. The damping is not optional for a
        # hyperbolic non-local field: undamped, the transients minted at the start of the step and at
        # every refinement never decay, and the damage variable follows every overshoot instead of
        # the mean. Written on whole vectors: h and the second-order mask are zero off the
        # second-order DOFs, which leaves those velocities unchanged.
        #
        # ``dofs`` are the degrees of freedom integrated here: in a model computed whole the slice
        # ``slice(None)``, so that every ``x[dofs]`` below is the whole vector.
        dofs = self.partition.dofs
        firstOrderDofs = self._incrementPlan.firstOrderDofs
        VPlain, PPlain, MinvPlain = V.asPlainArray(), P.asPlainArray(), Minv.asPlainArray()

        halfRateStep = self._halfDampingRate[dofs] * dTAverage

        velocity = VPlain[dofs]
        velocity *= 1.0 - halfRateStep
        velocity += MinvPlain[dofs] * PPlain[dofs] * (self._secondOrderMask[dofs] * dTAverage)
        velocity /= 1.0 + halfRateStep
        VPlain[dofs] = velocity
        VPlain[firstOrderDofs] = MinvPlain[firstOrderDofs] * PPlain[firstOrderDofs]

    def imposeKinematicConstraints(self, V: DofVector, prescribedVelocities: list[tuple[np.ndarray, np.ndarray]]):
        """Set the velocities the scheme does not integrate: the prescribed ones, and those of the
        slave degrees of freedom of multi-point constraints.

        Parameters
        ----------
        V
            The velocity vector.
        prescribedVelocities
            Per Dirichlet condition, its degrees of freedom and their prescribed velocity; see
            :meth:`prescribeBoundaryMotion`.
        """

        # A prescribed velocity is a boundary condition, not a solution, and the update overwrote
        # it: the damped one scales it by (1 - alpha dt/2)/(1 + alpha dt/2), the first-order one
        # loses it outright to Minv * 0.
        for constrainedDofIndices, prescribedVelocity in prescribedVelocities:
            V[constrainedDofIndices] = prescribedVelocity

        # Slave DOFs of multi-point constraints do not integrate their own equations of motion --
        # they ride along on their masters (Minv is zero there, so the update left them untouched);
        # their displacements follow from dU = V * dt.
        if self.mpcTransformation is not None:
            self.mpcTransformation.applySlaveKinematics(V)

    def advanceDisplacements(self, U: DofVector, dU: DofVector, V: DofVector, dT: float):
        """Advance the solution of every degree of freedom integrated here by its velocity:
        :math:`\\Delta u = \\Delta t\\, v_{n+1/2}`, :math:`u_{n+1} = u_n + \\Delta u`.

        Parameters
        ----------
        U
            The solution vector; advanced in place.
        dU
            The solution increment vector; written.
        V
            The velocity vector.
        dT
            The time increment.
        """

        dofs = self.partition.dofs
        increment = V.asPlainArray()[dofs] * dT
        dU.asPlainArray()[dofs] = increment
        U.asPlainArray()[dofs] += increment

    def assembleNetForce(
        self, U: DofVector, dU: DofVector, P: DofVector, stepActions: dict, timeStep: TimeStep
    ) -> tuple[DofVector, float]:
        """Assemble the net nodal force at the new solution: the external loads and the constraint
        forces minus the internal force of the elements, :math:`f = f^{ext} + f^{c} - f^{int}`,
        with the forces on multi-point-constraint slaves folded onto their masters.

        Parameters
        ----------
        U
            The current solution vector.
        dU
            The solution increment vector.
        P
            The net nodal force vector; overwritten.
        stepActions
            The active step actions, by type.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector, float]
            The net nodal force, and the internal energy the elements computed here report.
        """

        P, psi = self.assembleInternalForces(U, dU, P, timeStep)
        P[:] = -P[:]
        # The load assembly is shared with the implicit solvers, which also assemble the load
        # stiffness into a tangent K; an explicit increment has no tangent, hence K=None.
        P, _ = self.assembleLoads(
            stepActions["nodeforces"].values(),
            stepActions["distributedload"].values(),
            stepActions["bodyforce"].values(),
            U,
            P,
            K=None,
            timeStep=timeStep,
        )
        P = self.assembleConstraintForces(self.partition.constraints, U, dU, P, timeStep)

        # Fold the forces acting on slave DOFs onto their masters (action-reaction through the rigid
        # interpolation link); done here so the Dirichlet handling at the start of the next increment
        # operates on the already-folded vector. A slave and its masters are always integrated in the
        # same part of the model, so the fold is restricted to its DOFs.
        mpcForceFold = self._incrementPlan.mpcForceFold
        if mpcForceFold is not None:
            with performancetiming.timeit("mpc force fold"):
                PPlain = P.asPlainArray()
                dofs = self.partition.dofs
                PPlain[dofs] = mpcForceFold @ PPlain[dofs]

        return P, psi

    def reportEnergyBalance(self, psi: float, V: DofVector, timeStep: TimeStep):
        """Report the energy balance of the model -- external work, kinetic and internal energy, and
        what none of them accounts for -- and stop a run whose balance shows it has diverged.

        Called on every ``output-frequency``-th increment.

        Parameters
        ----------
        psi
            The internal energy the elements computed here report.
        V
            The velocity vector.
        timeStep
            The increment.

        Raises
        ------
        StepFailed
            If the balance is no longer a finite number.
        """

        Wint, Wkin, Wext, nonMechanical = self.energyBalanceTerms(psi, V)

        # Everything the model absorbed that no reported term accounts for: strain energy the
        # material does not publish, plastic and viscous dissipation, and damage. It must stay
        # non-negative -- a negative value means the kinetic energy has overtaken the work put
        # in, i.e. energy is being created.
        unaccounted = Wext - Wkin - Wint

        def _share(value):
            return "{:7.2f} %".format(value / Wext * 100) if Wext > 0.0 else "      -- "

        energyRows = [
            ["energy", "value", "share of W_ext"],
            ["external work W_ext", "{:+.6e}".format(Wext), _share(Wext)],
            ["kinetic", "{:+.6e}".format(Wkin), _share(Wkin)],
            ["internal (strain)", "{:+.6e}".format(Wint), _share(Wint)],
            ["unaccounted", "{:+.6e}".format(unaccounted), _share(unaccounted)],
        ]

        # One row per second-order field that is not in the balance, never summed with another
        # or into the total. For a hyperbolic non-local field the inertia is a time squared, so
        # 0.5 * m * rate**2 is a volume rather than an energy -- kept on its own line because
        # it is the energy in the ringing the damping exists to remove.
        for fieldName, energy in zip(self.nonMechanicalSecondOrderFields, nonMechanical):
            energyRows.append([fieldName, "{:+.6e}".format(energy), "not an energy"])

        self.journal.printTable(
            energyRows,
            self.identification,
            2,
        )

        # KE <= W_ext is exact in the continuum, so a violation is energy from nowhere: for an
        # explicit scheme, a time step above the true stability limit. It catches what dt_crit
        # does not see -- the nonlocal field, contact penalty stiffness, a misjudged element
        # shape. The NaN case is checked first because every ordering against a NaN is False,
        # including the `Wext > 0.0` below, so a diverged run would pass silently and keep
        # writing NaN. It fails the step rather than reporting it: nothing after it is
        # meaningful, and an explicit scheme has no cutback to answer it with.
        if not (np.isfinite(Wext) and np.isfinite(Wkin)):
            raise StepFailed(
                "THE SOLUTION HAS DIVERGED in increment {:}: the energy balance is no longer a "
                "finite number (external work {:e}, kinetic {:e}), which means the state itself "
                "is not. The time step is above the true stability limit -- and dt_crit does not "
                "see all of it: it never sees contact penalty stiffness, and it sees the nonlocal "
                "field only where that field carries a non-mechanical inertia, a first-order one "
                "having a forward-Euler limit that nothing checks. Resume from a checkpoint with "
                "a smaller courant-number.".format(timeStep.number, Wext, Wkin)
            )

        if Wext > 0.0 and Wkin > Wext * (1.0 + _ENERGY_CREATION_TOLERANCE):
            self.journal.message(
                "ENERGY IS BEING CREATED: the kinetic energy {:e} exceeds the external work "
                "{:e} by {:.1f} %. That is impossible -- the unreported terms (strain energy, "
                "dissipation, damage) can only add to the balance. The time step is above the "
                "true stability limit, which dt_crit does not see in full: it ignores the "
                "nonlocal field entirely and never sees contact penalty stiffness. Reduce "
                "courant-number, or resume from a checkpoint with a smaller one.".format(
                    Wkin, Wext, (Wkin / Wext - 1.0) * 100
                ),
                self.identification,
                0,
            )

        # Said once, loudly, because a silent zero here reads as "no strain energy yet" rather
        # than "this quantity is not being reported", and the ratio above is the usual way one
        # decides whether an explicit run is quasi-static enough.
        # The energy-creation check above is gated on Wext > 0, and _externalWork only
        # accumulates at PRESCRIBED degrees of freedom. A model whose Dirichlet conditions are
        # all fixed and whose loading comes from body forces, node forces, a distributed load
        # or an initial velocity therefore has Wext identically zero, and the check silently
        # never fires -- on precisely the impact/drop class of problem explicit dynamics
        # exists for. Say so once rather than printing a table of dashes.
        #
        # NEGATIVE is the same case and must be caught by the same branch rather than falling
        # between the two: a model that has returned more work at its prescribed degrees of
        # freedom than was put in -- a rebounding or oscillating structure -- satisfies neither
        # Wext > 0 nor Wext == 0, so gating on equality left the diagnostic disabled with
        # nothing said at all. That is the failure this check exists to prevent.
        if Wext <= 0.0 and not self._warnedAboutMissingExternalWork:
            self._warnedAboutMissingExternalWork = True
            self.journal.message(
                "The external work is {:}: only work done at prescribed degrees of freedom is "
                "accumulated, so this model is either not displacement-driven or is currently "
                "returning work at its prescribed degrees of freedom. The energy balance above "
                "is NOT usable, and in particular the energy-creation check -- the one that "
                "catches the stability contributions dt_crit does not see -- cannot fire. "
                "Watch the kinetic energy history directly instead.".format(
                    "identically zero" if Wext == 0.0 else "negative ({:e})".format(Wext)
                ),
                self.identification,
                1,
            )

        if Wint == 0.0 and not self._warnedAboutMissingInternalEnergy:
            self._warnedAboutMissingInternalEnergy = True
            self.journal.message(
                "The internal energy is identically zero: no material in this model populates a "
                "strain energy, so the internal/kinetic split above is NOT usable as a "
                "quasi-static criterion. Integrate a reaction force against its prescribed "
                "displacement instead -- both are available as saveHistory field outputs.",
                self.identification,
                1,
            )

    def energyBalanceTerms(self, psi: float, V: DofVector) -> tuple[float, float, float, list[float]]:
        """The terms of the energy balance: the internal energy, the kinetic energy, the external work,
        and the "kinetic energy" :math:`\\frac{1}{2} m v^2` of every second-order field whose inertia
        is not a mass, which is not an energy and enters no balance.

        The kinetic energy only over the degrees of freedom where :math:`\\frac{1}{2} m v^2` is an
        energy. Summing the whole vector also collected the first-order fields, whose "mass" is a
        viscosity and whose "velocity" is that field's rate -- no energy meaning, and orders of
        magnitude larger than the real term (eta ~ 1e-4 against a density ~ 1e-9): every balance
        then read as 100 % kinetic.

        Parameters
        ----------
        psi
            The internal energy the elements computed here report.
        V
            The velocity vector.

        Returns
        -------
        tuple[float, float, float, list[float]]
            The internal energy, the kinetic energy, the external work, and the term of every field
            of :attr:`nonMechanicalSecondOrderFields`, in that order.
        """

        nonMechanical = [
            self.halfMassTimesSquaredRate(self.theDofManager.idcsOfFieldsInDofVector[fieldName], V)
            for fieldName in self.nonMechanicalSecondOrderFields
        ]
        return psi, self.halfMassTimesSquaredRate(self.ids_mechanicalEnergy, V), self._externalWork, nonMechanical

    def halfMassTimesSquaredRate(self, indices: slice | np.ndarray, V: DofVector) -> float:
        """:math:`\\frac{1}{2} \\sum m v^2` over the given degrees of freedom, with the unfolded lumped
        mass: the kinetic energy, where :math:`m` is a mass.

        Parameters
        ----------
        indices
            The degrees of freedom.
        V
            The velocity vector.

        Returns
        -------
        float
            The sum.
        """

        return 0.5 * float(np.sum(self._rawLumpedMass[indices] * V[indices] ** 2))

    @performancetiming.timeit("elements")
    def assembleInternalForces(
        self,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, float]:
        """Evaluate the elements computed here and assemble their internal force,
        :math:`f^{int} = \\mathop{\\mathsf{A}}_e f^{int}_e`: the element loop of every finite element code.
        Element after element, in element order: gather its solution from the global vector, compute
        its nodal forces, and add them into ``P`` at its degrees of freedom.

        One element at a time is the plainest way to write it, and the slowest: the gather and the
        scatter cost about what a fast element kernel does. :class:`~edelweissfe.solvers.nonlinearexplicitdynamicparallel.NEDParallel`
        runs the same loop in bulk, gathering and assembling many elements at once
        (:class:`~edelweissfe.solvers.base.parallelelementcomputation.ElementPlan`), on one thread or
        several, with the same bits -- use it for production runs.

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
            The internal force vector, and the internal energy the elements report.
        """

        U, dU, P[:] = U_np.asPlainArray(), dU.asPlainArray(), 0.0
        forces = P.asPlainArray()
        time, dT = timeStep.totalTime, timeStep.timeIncrement

        psi = 0.0
        for element, dofs, namesDofMoreThanOnce in self._incrementPlan.elementLoop:
            Pe = np.zeros(dofs.shape[0])
            element.computeKernelsExplicit(Pe, U[dofs], dU[dofs], time, dT)
            addNodalForces(forces, dofs, Pe, namesDofMoreThanOnce)  # P[dofs] += Pe
            psi += element.computeInternalEnergy()

        return P, psi

    def validateModelCapabilities(self, model: FEModel):
        """Refuse the model features this solver cannot integrate, on top of the base checks.

        Two beyond :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.validateModelCapabilities`:

        A **model modifier that would act after the analysis has started without configuring
        ``topology-check-frequency``.** Modifiers that act only at simulation start are served by the
        initial topology update before the increment loop. Modifiers that act mid-run require
        ``topology-check-frequency`` to be set to a non-zero multiple of ``output-frequency``;
        otherwise they are refused to prevent silently never adapting.

        A **constraint that introduces its own scalar variables** (a Lagrange multiplier, an
        indirect-control unknown). Those DOFs carry no inertia, so their inverse lumped mass is zero
        and the explicit update leaves them untouched forever -- the constraint would appear active
        and enforce nothing. Penalty-type constraints, which act through nodal forces alone, are
        supported: see :meth:`assembleConstraintForces`.

        Parameters
        ----------
        model
            The model tree.
        """

        super().validateModelCapabilities(model)

        # A modifier that acts only at the start of the analysis is served by the single topology
        # update before the increment loop. One that acts later needs the update re-run, which this
        # solver does -- but only if it has been told how often, because a marker reads the last
        # *finalized* field output and finalization is itself on a cadence. Refusing to guess is the
        # point: a silently-never-refining run looks exactly like a converged one.
        lateModifiers = sorted(
            name
            for name, modifier in model.modelModifiers.items()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        )
        topologyCheckFrequency = self.options["topology-check-frequency"]
        outputFrequency = self.options["output-frequency"]

        # All three cadence options are validated here, together, because each is used as a modulo
        # operand further down and none of them fails usefully on its own:
        #
        #  * output-frequency is a DIVISOR, not a flag: 0 does not mean "never report", it means
        #    ZeroDivisionError from a bare traceback on the first increment. It must be >= 1.
        #  * a NEGATIVE cadence is silently accepted otherwise. It is truthy, so it passes the
        #    "is it set" gates; and when its magnitude is a multiple of output-frequency it also
        #    passes the multiple check below (-20 % 20 == 0) and then fires at exactly the same
        #    increments as its absolute value (n % -20 == 0 for the same n as n % 20 == 0). When
        #    the magnitude is NOT a multiple, the multiple-check error blames the wrong thing.
        if outputFrequency is None or outputFrequency < 1:
            raise ValueError(
                "output-frequency ({:}) must be at least 1: it is the divisor of the reporting, "
                "field-output and topology-check cadences, so 0 does not disable them, it raises "
                "ZeroDivisionError on the first increment.".format(outputFrequency)
            )

        if topologyCheckFrequency is not None and topologyCheckFrequency < 0:
            raise ValueError(
                "topology-check-frequency ({:}) cannot be negative. Use 0 to run the topology "
                "update only once, before the increment loop.".format(topologyCheckFrequency)
            )

        contactUpdateFrequency = self.options["contact-update-frequency"]
        if contactUpdateFrequency is not None and contactUpdateFrequency < 0:
            raise ValueError(
                "contact-update-frequency ({:}) cannot be negative. Use 0 to disable the periodic "
                "connectivity update mid-run.".format(contactUpdateFrequency)
            )

        if lateModifiers and not topologyCheckFrequency:
            raise ValueError(
                "The model modifier(s) {:} act after the analysis has started, but "
                "topology-check-frequency is 0, so {:} would run the topology update only once and "
                "they would never act. Set topology-check-frequency to a multiple of "
                "output-frequency ({:}).".format(", ".join(lateModifiers), self.identification, outputFrequency)
            )

        if topologyCheckFrequency:
            if topologyCheckFrequency % outputFrequency:
                raise ValueError(
                    "topology-check-frequency ({:}) must be a multiple of output-frequency ({:}): a "
                    "marker refines on the last finalized field output, and field outputs are "
                    "finalized on the output-frequency cadence, so a check that does not land on one "
                    "would decide on stale results.".format(topologyCheckFrequency, outputFrequency)
                )
            if not lateModifiers:
                self.journal.message(
                    "topology-check-frequency is set but no model modifier acts after the start of "
                    "the analysis; the topology update will run once and the periodic check will "
                    "find nothing to do.",
                    self.identification,
                    1,
                )

        for constraintName, constraint in model.constraints.items():
            nScalarVariables = constraint.getNumberOfAdditionalNeededScalarVariables()
            if nScalarVariables:
                raise NotImplementedError(
                    f"Constraint '{constraintName}' introduces {nScalarVariables} additional scalar "
                    f"variable(s), which carry no inertia and which {self.identification} therefore "
                    "has no equation of motion for. Only constraints acting through nodal forces "
                    "(penalty formulations) are supported here."
                )

    @performancetiming.timeit("publish node fields")
    def publishNodeFields(self, model: FEModel, U: DofVector, V: DofVector, P: DofVector):
        """Write the solution, velocity and force to the node fields and the scalar variables.

        Published every increment, not only on output increments, for two reasons: an h-adaptivity
        event can fall on any increment and its interpolation reads these entries, and a restart
        checkpoint written from the node fields is the only way an explicit run can resume with its
        kinetic state intact. It is an O(nDof) copy.

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

        for nodeField in model.nodeFields.values():
            indices = self.theDofManager.idcsOfNodeFieldsInDofVector[nodeField.name]
            for vector, entry in ((U, "U"), (P, "P"), (V, "V")):
                if entry not in nodeField:
                    nodeField.createFieldValueEntry(entry)
                values = nodeField[entry]
                values[:] = vector.asPlainArray()[indices].reshape(values.shape)

        for variable in model.scalarVariables.values():
            variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

    @performancetiming.timeit("constraint connectivity")
    def updateConstraintConnectivity(self, model: FEModel) -> bool:
        """Let the constraints whose connectivity is the outcome of a search re-run that search.

        Only the constraints in :attr:`_dynamicConnectivityConstraints` are ticked, and the caller
        ticks them only every ``contact-update-frequency`` increments. Both matter: a node-to-surface
        search is O(slaves x facets) in Python, which is affordable once per increment of an implicit
        analysis -- where it is amortised over a Newton loop and a linear solve -- and not affordable
        tens of thousands of times. What makes throttling defensible rather than merely cheap is that
        an explicit time step is tiny: between two searches a node moves ``V * dT * frequency``,
        orders of magnitude below a facet dimension. The caller reports the motion actually
        accumulated so that this can be checked against a given model instead of assumed.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        bool
            Whether any constraint's DOF footprint changed, i.e. whether the equation system has to
            be rebuilt.
        """

        return self.updateConnectivityOf(model, self._dynamicConnectivityConstraints)

    def _operatorsReusable(self, model: FEModel, stepActions: dict) -> bool:
        """Whether the kept lumped operators still describe this model, so that a rebuild may keep
        them and re-locate the constraints alone.

        The operators are a function of the elements -- which carry the inertia and the damping --
        of the multi-point constraints, and of the degree-of-freedom layout the two are indexed by.
        A contact search changes none of those: it re-assigns which nodes a *constraint* couples,
        which moves the constraints' own DOF footprints and nothing else. A topology change does
        move them, and never reaches here: it passes no ``previous`` system and builds afresh.

        The multi-point constraints are safe to keep for a structural reason rather than a
        coincidental one, which is worth being explicit about. They are a separate hierarchy
        (:class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`)
        with no connectivity-update protocol at all -- ``updateConnectivity`` belongs to
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`, and only those
        constraints are collected into ``_dynamicConnectivityConstraints`` and can ask for this
        rebuild -- and the dependency records they produce are a function of the node DOF indices,
        which a constraint refresh leaves untouched. Should a multi-point constraint ever gain a
        connectivity of its own, that argument fails and the transformation would have to be
        rebuilt here rather than kept.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The step's actions.

        Returns
        -------
        bool
            Whether the kept operators may be reused.
        """

        cache = self._reusableOperators
        if cache is None:
            return False

        # The inertia and the damping are assembled from the materials, so a step action that
        # changes a material property mid-step invalidates them.
        if stepActions["changematerialproperty"]:
            return False

        return (
            model.elements.keys() == cache.elementKeys
            and model.multiPointConstraints.keys() == cache.multiPointConstraintKeys
            and len(model.nodes) == cache.nNodes
            and len(model.scalarVariables) == cache.nScalarVariables
        )

    def partitionModel(self, model: FEModel, topologyChanged: bool) -> ModelPartition:
        """Decide which elements, constraints and degrees of freedom this process computes: the whole
        model.

        Called with every equation system, once its degree-of-freedom layout and its
        multi-point-constraint transformation are known, and before anything is assembled.

        Parameters
        ----------
        model
            The model tree.
        topologyChanged
            True for an equation system built afresh; False when only the constraints were
            re-located by a contact search.

        Returns
        -------
        ModelPartition
            The partition.
        """

        return ModelPartition.wholeModel(model)

    def planElementLoop(self, elements: dict) -> list[tuple]:
        """The element loop over the given elements, as :meth:`assembleInternalForces` and
        :meth:`assembleLumpedDiagonal` run it: the elements in element order, each with its degrees of
        freedom and whether it names one of them more than once (a degenerate element with repeated
        nodes; see :func:`~edelweissfe.numerics.assembly.addNodalForces`).

        Parameters
        ----------
        elements
            The elements, by number, in element order.

        Returns
        -------
        list[tuple]
            ``(element, dofs, namesDofMoreThanOnce)`` per element.
        """

        dofsOf = self.theDofManager.idcsOfHigherOrderEntitiesInDofVector
        return [(element, dofsOf[element], hasRepeatedDofs(dofsOf[element])) for element in elements.values()]

    def assembleLumpedDiagonal(self, elementLoop: list[tuple], elementContribution) -> DofVector:
        """Assemble a lumped operator -- the inertia or the damping -- of the elements of a loop:
        element after element, in element order, add its diagonal into the vector at its degrees of
        freedom.

        Parameters
        ----------
        elementLoop
            The element loop over the elements computed here, contact facets included
            (:meth:`planElementLoop`).
        elementContribution
            ``elementContribution(element, Ve)`` writes an element's diagonal into the zero ``Ve``.

        Returns
        -------
        DofVector
            The assembled diagonal.
        """

        vector = self.theDofManager.constructDofVector()
        vector[:] = 0.0
        for element, dofs, namesDofMoreThanOnce in elementLoop:
            Ve = np.zeros(dofs.shape[0])
            elementContribution(element, Ve)
            addNodalForces(vector.asPlainArray(), dofs, Ve, namesDofMoreThanOnce)
        return vector

    def elementsWithKernels(self) -> dict:
        """The elements computed here that have kernels to call -- not the contact facets --, by
        number, in element order.

        Returns
        -------
        dict
            The elements.
        """

        return {number: element for number, element in self.partition.elements.items() if element.hasKernels}

    def reportHowElementsAreComputed(self, model: FEModel):
        """Report how the elements are computed: one at a time. Recommend :class:`~edelweissfe.solvers.nonlinearexplicitdynamicparallel.NEDParallel` for a
        large model: its element loop gives the same bits, faster, even on one thread. Changes nothing.

        Parameters
        ----------
        model
            The model tree.
        """

        nElements = sum(1 for element in model.elements.values() if element.hasKernels)
        if nElements > _ELEMENTS_WORTH_THE_PLAN:
            self.journal.message(
                "{:} elements: NEDParallel computes the same result faster, also on one thread "
                "(solver=NEDParallel)".format(nElements),
                self.identification,
                0,
            )

    def planIncrement(self, model: FEModel) -> IncrementPlan:
        """Derive what an increment computes here from the current partition and equation system.

        The element loop leaves out the entities that have no kernels to call -- contact facets --
        once per plan rather than once per increment.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        IncrementPlan
            The plan.
        """

        dofs = self.partition.dofs
        integrated = np.zeros(self.theDofManager.nDof, dtype=bool)
        integrated[dofs] = True

        return IncrementPlan(
            elementLoop=self.planElementLoop(self.elementsWithKernels()),
            firstOrderDofs=self.ids_1st[integrated[self.ids_1st]],
            mpcForceFold=(
                None if self.mpcTransformation is None else self.mpcTransformation.foldExplicitForceOperator(dofs)
            ),
        )

    def _restoreLumpedOperators(self) -> tuple[DofVector, DofVector]:
        """Put the kept lumped operators back, into vectors carrying the refreshed entity mapping.

        The numbers are the ones :meth:`_assembleLumpedOperators` produced when it last ran. What
        that method additionally left on the solver -- which fields are first and which second
        order, and the DOF index sets of each -- is not restored because it was never invalidated:
        a constraint refresh leaves the fields and their indices exactly as they were, so those
        attributes still describe this system.

        Returns
        -------
        tuple[DofVector, DofVector]
            The vector the increment divides by, and its inverse.
        """

        cache = self._reusableOperators

        # The transformation is indexed by the DOF layout, which the refresh did not move.
        assert cache.mpcTransformation is None or cache.mpcTransformation.nDof == self.theDofManager.nDof
        self.mpcTransformation = cache.mpcTransformation

        M = self.theDofManager.constructDofVector()
        M[:] = cache.lumpedMass
        Minv = self.theDofManager.constructDofVector()
        Minv[:] = cache.inverseLumpedMass
        self._rawLumpedMass = self.theDofManager.constructDofVector()
        self._rawLumpedMass[:] = cache.rawLumpedMass
        self._dampingRate = self.theDofManager.constructDofVector()
        self._dampingRate[:] = cache.dampingRate
        self._lumpedMass = M

        return M, Minv

    def _assembleLumpedOperators(self, verbosity: int) -> tuple[DofVector, DofVector]:
        """Assemble everything the explicit update divides by, and everything derived from it.

        The lumped inertia and the lumped damping are summed from the elements, checked, and used
        to decide which time derivative each field carries; the multi-point-constraint
        transformation folds the slave degrees of freedom's mass onto their masters. Beyond the two
        vectors returned, this leaves the derived classification, the raw (unfolded) mass, the
        damping rate and the field index sets on the solver.

        Parameters
        ----------
        verbosity
            The journal level to report at.

        Returns
        -------
        tuple[DofVector, DofVector]
            The vector the increment divides by, and its inverse.
        """

        # The lumped operators are assembled over the elements computed here -- contact facets
        # included, which have an inertia but no kernels.
        operatorLoop = self.planElementLoop(self.partition.elements)
        M = self.assembleLumpedDiagonal(operatorLoop, lambda element, Me: element.computeLumpedInertia(Me))
        Minv = self.theDofManager.constructDofVector()  # initialize inverse lumped mass matrix

        # Each field's FIRST-derivative coefficient: zero mechanically, the non-local viscosity
        # always (whether or not that field also has an inertia -- see computeLumpedInertia()
        # above, the SECOND-derivative coefficient).
        damping = self.assembleLumpedDiagonal(operatorLoop, lambda element, Ce: element.computeLumpedDamping(Ce))

        # Checked here, because the inertia check below never sees it at a second-order DOF: there
        # the divisor stays the positive inertia and the damping enters only as the rate C/M. A
        # negative rate amplifies the transient the damping exists to remove, and alpha*dt/2 = -1
        # makes the update's denominator exactly zero. Both are silent.
        if not np.all(np.isfinite(damping)) or np.any(damping < 0.0):
            raise ValueError(
                "The assembled lumped damping is not a valid dissipation: {:} of {:} entries are "
                "negative and {:} are not finite (smallest: {:e}). A damping coefficient enters the "
                "second-order update as the rate C/M, where a negative value amplifies instead of "
                "damping, and a first-order field divides by it directly.".format(
                    int(np.count_nonzero(np.asarray(damping) < 0.0)),
                    damping.shape[0],
                    int(np.count_nonzero(~np.isfinite(np.asarray(damping)))),
                    np.nanmin(damping),
                )
            )

        # Which time derivative each field carries, read off the two vectors just assembled. An
        # inertia is what a central-difference update divides by, so a field with one is second
        # order and a field without one is not; no deck answer that disagreed could be honoured.
        self.firstOrderFields, self.secondOrderFields = self._classifyFieldsByScheme(M, damping)

        self.ids_1st = self._dofIndicesOfFields(self.firstOrderFields)
        self.ids_2nd = self._dofIndicesOfFields(self.secondOrderFields)

        self.journal.message(
            "Time integration, derived from the assembled operators: central difference for {:}; "
            "forward Euler for {:}".format(
                ", ".join(self.secondOrderFields) or "(no field)",
                ", ".join(self.firstOrderFields) or "(no field)",
            ),
            self.identification,
            verbosity,
        )

        self._checkDerivedSchemeAgainstTheDeck()

        # Which second-order fields may be summed into a linear momentum and which into the energy
        # balance. The assembled inertia cannot say -- a density, a rotational inertia and a
        # micro-inertia are all just positive numbers -- and it is a fact about the field rather
        # than about this analysis, so it comes from the registry. See phenomena.inertiaKind.
        self.linearMomentumFields = [f for f in self.secondOrderFields if carriesLinearMomentum(f)]
        self.nonMechanicalSecondOrderFields = [f for f in self.secondOrderFields if not carriesKineticEnergy(f)]
        self.ids_mechanicalEnergy = self._dofIndicesOfFields(
            [f for f in self.secondOrderFields if carriesKineticEnergy(f)]
        )

        # A first-order field integrates by forward Euler, C * rate = P, and needs the damping
        # computeLumpedDamping() reports here, not the (correctly zero) inertia. From here on this
        # vector is "the divisor", whichever of the two coefficients a degree of freedom's scheme
        # actually divides by.
        M[self.ids_1st] = damping[self.ids_1st]

        # Kept before folding, so the kinetic energy diagnostic accounts for the true velocities of
        # all nodes (including tied slaves) rather than master-placed folded mass.
        self._rawLumpedMass = M.copy()

        # compute inverses
        if np.any(M == 0.0):
            raise ValueError(
                "Zero found in the vector the increment divides by, at {:} of {:} degrees of freedom. "
                "Every FIELD is covered by the classification above, which refuses a field carrying "
                "neither coefficient, so what is left here are degrees of freedom belonging to no field "
                "-- scalar variables, which this solver has no equation of motion for.".format(
                    int(np.count_nonzero(np.asarray(M) == 0.0)), M.shape[0]
                )
            )

        # A negative lumped mass is the classical failure mode of row-summing a quadratic element's
        # consistent mass matrix, and it is worse than a zero one: the update stays finite, the run
        # continues, and those degrees of freedom integrate backwards in time. The quadratic elements
        # here blend the linear shape functions in precisely to avoid it, which is exactly why this
        # is worth stating rather than trusting.
        if np.any(M < 0.0):
            raise ValueError(
                "Negative coefficient found in {:} of {:} entries of the vector the increment "
                "divides by (smallest: {:e}). It is a lumped inertia at a second-order degree of "
                "freedom and a damping at a first-order one; either way a negative value makes the "
                "explicit update integrate backwards in time there.".format(
                    int(np.count_nonzero(M < 0.0)), M.shape[0], M.min()
                )
            )

        # Slave DOFs of multi-point constraints carry no own inertia: their mass is folded onto
        # their masters (row-sum lumping of T^T M T, mass-conserving), their Minv stays zero, and
        # their kinematics are assigned directly from the masters each increment.
        if self.mpcTransformation is not None:
            self.mpcTransformation.foldLumpedMass(M)
            # Folded with the same operator as the inertia it is divided by, so the ratio below is
            # the damping rate of the FOLDED system. Slaves of one material cancel exactly; slaves
            # of several blend by inertia, which is what the folded equation of motion has.
            self.mpcTransformation.foldLumpedMass(damping)

        Minv[M != 0.0] = 1.0 / M[M != 0.0]

        # Mass-proportional damping rate alpha = C / M, wherever an inertia is divided by and a
        # damping was assembled alongside it. Slave DOFs fold to zero inertia and integrate nothing,
        # so they stay at zero rather than dividing by it, and a DOF whose elements reported no
        # damping keeps the zero that makes the update exactly the undamped one. Second-order DOFs
        # only: a first-order one had its inertia overwritten with its damping, so C/M would be 1.0.
        self._dampingRate = self.theDofManager.constructDofVector()
        self._dampingRate[:] = 0.0
        integrating = self.ids_2nd[M[self.ids_2nd] > 0.0]
        self._dampingRate[integrating] = damping[integrating] / M[integrating]

        # kept (instead of 1/Minv) for the kinetic energy: slave DOFs have Minv = 0
        self._lumpedMass = M

        return M, Minv

    @performancetiming.timeit("build equation system")
    def buildEquationSystem(self, model: FEModel, step, previous: ExplicitSystem = None) -> ExplicitSystem:
        """Build the equation system and everything sized by it.

        Called once before the increment loop, and again from inside it whenever a constraint reports
        that its DOF footprint changed -- one method for both, so the path every model takes and the
        path only a contact model takes cannot drift apart. On that second path the DofManager is
        refreshed rather than rebuilt and the lumped operators are kept; see
        :meth:`_operatorsReusable`.

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved; its actions are needed to check the multi-point constraints
            against the prescribed Dirichlet conditions.
        previous
            The system being replaced, when this is a rebuild rather than the initial build. Its
            solution, velocity and force are carried over verbatim rather than re-read from the node
            fields, which do not hold the velocity at all. A rebuild triggered by a constraint's
            connectivity leaves the mesh -- hence the DOF layout -- untouched, and that is checked
            rather than assumed: copying between two different layouts would mis-index every vector
            silently.

        Returns
        -------
        ExplicitSystem
            The freshly built system.
        """

        isRebuild = previous is not None
        verbosity = 2 if isRebuild else 0

        # A rebuild that only a constraint's connectivity asked for moves the constraints' DOF
        # footprints and nothing else, so the DofManager is refreshed rather than rebuilt -- its
        # node numbering, its element indices and its DOF count are what they were -- and the
        # lumped operators are kept; see _operatorsReusable(). Anything else builds from scratch.
        reuseOperators = isRebuild and self._operatorsReusable(model, step.actions)

        if reuseOperators:
            self.journal.message(
                "Constraint connectivity changed: re-locating the constraints' degrees of freedom, "
                "keeping the lumped operators",
                self.identification,
                verbosity,
            )
            self.theDofManager.refreshConstraintIndices(model.constraints.values())
        else:
            self.journal.message("Creating monolithic equation system", self.identification, verbosity)
            # An explicit solver assembles no system matrix, so it never reads the sparsity (VIJ)
            # pattern; and nothing at all reads the index-to-host-object map (see below). Each of
            # them costs a pass over every element to build.
            self.theDofManager = DofManager(
                model.nodeFields.values(),
                model.scalarVariables.values(),
                model.elements.values(),
                model.constraints.values(),
                model.nodeSets.values(),
                initializeVIJPattern=False,
                determiningIndexToHostObjectMapping=False,
            )
        self.journal.message(
            "total size of eq. system: {:}".format(self.theDofManager.nDof),
            self.identification,
            verbosity,
        )

        if not isRebuild:
            self.journal.printSeperationLine()

        # self.options already reflects every >>options, name=<this solver's name>, ... block applied
        # so far, applied as each block is constructed or re-declared; there is nothing to reset or
        # re-fetch here.

        # The constraint force buffers and their index plans belong to the constraint indices that
        # were just (re)located: a refinement or a contact search changes both a constraint's DOF
        # count and where its DOFs sit, and a stale plan would scatter forces to the wrong degrees
        # of freedom silently.
        self._constraintForces = {}

        if reuseOperators:
            M, Minv = self._restoreLumpedOperators()
            # The constraints now couple other degrees of freedom, and so may be integrated
            # elsewhere.
            self.partition = self.partitionModel(model, topologyChanged=False)
        else:
            self.mpcTransformation = self.buildMPCTransformation(model, step.actions)
            self.checkMPCDirichletConflicts(self.mpcTransformation, step.actions)
            # Which elements, constraints and degrees of freedom this process computes; everything
            # from here on that loops over elements loops over those only.
            self.partition = self.partitionModel(model, topologyChanged=True)
            M, Minv = self._assembleLumpedOperators(verbosity)
            self._reusableOperators = _ReusableExplicitOperators(
                elementKeys=frozenset(model.elements.keys()),
                multiPointConstraintKeys=frozenset(model.multiPointConstraints.keys()),
                nNodes=len(model.nodes),
                nScalarVariables=len(model.scalarVariables),
                lumpedMass=np.array(M),
                rawLumpedMass=np.array(self._rawLumpedMass),
                inverseLumpedMass=np.array(Minv),
                dampingRate=np.array(self._dampingRate),
                mpcTransformation=self.mpcTransformation,
            )

        # Per-DOF factors of the velocity update, so that it runs on whole vectors.
        self._secondOrderMask = np.zeros(self._dampingRate.shape[0])
        if self.ids_2nd is not None:
            self._secondOrderMask[self.ids_2nd] = 1.0
        self._halfDampingRate = 0.5 * np.asarray(self._dampingRate) * self._secondOrderMask

        self._incrementPlan = self.planIncrement(model)

        U = self.theDofManager.constructDofVector()  # initialize displacement vector
        dU = self.theDofManager.constructDofVector()  # initialize displacement vector
        V = self.theDofManager.constructDofVector()  # initilize velocity vector
        P = self.theDofManager.constructDofVector()  # initialize reaction vector

        if not isRebuild:
            for fieldName, field in model.nodeFields.items():
                U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")
                P = self.theDofManager.writeNodeFieldToDofVector(P, field, "P")

                # The velocity entry exists only once this solver has published one, i.e. from the
                # second build onwards. Reading it back is what carries the kinetic state across an
                # h-adaptivity event: the modifier interpolated it onto the new nodes with the same
                # operator it used for U (see hadaptivity.WARM_STARTED_NODE_FIELD_ENTRIES), and there
                # is nowhere else it could come from -- a fresh vector would silently resume from
                # rest.
                if "V" in field:
                    V = self.theDofManager.writeNodeFieldToDofVector(V, field, "V")

            for variable in model.scalarVariables.values():
                U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]] = variable.value
        else:
            if previous.U.shape != U.shape:
                raise RuntimeError(
                    "The equation system was rebuilt with {:} degrees of freedom instead of {:}. Only "
                    "a constraint's connectivity is expected to trigger a rebuild here, and that "
                    "cannot add or remove degrees of freedom -- so the solution and the velocity "
                    "cannot be carried across safely.".format(U.shape[0], previous.U.shape[0])
                )

            # Carried straight over, not re-read from the node fields: the velocity is not a node
            # field, so re-reading would silently resume from rest.
            U[:] = previous.U
            V[:] = previous.V
            P[:] = previous.P

        if isRebuild:
            # The mesh is unchanged (the layout check above establishes that), so the stable time
            # increment is unchanged too, and recomputing it would cost a full element pass -- the
            # material asks for its wave speed by evaluating its own tangent at every quadrature
            # point. Reusing it is also the conservative direction: as the material softens the true
            # limit only grows, and raising the time increment mid-run would change the integrator's
            # dispersion for no benefit.
            criticalTimeStep = previous.criticalTimeStep
        else:
            criticalTimeStep = self.options.get("courant-number") * self.getCriticalTimeStepForExplicitDynamics(
                model, U
            )
            self.journal.message(
                "Critical time step for explicit dynamics: {:e}".format(criticalTimeStep), self.identification, 1
            )

        return ExplicitSystem(
            Minv=Minv,
            U=U,
            dU=dU,
            V=V,
            P=P,
            criticalTimeStep=criticalTimeStep,
        )

    @performancetiming.timeit("assemble constraints")
    def assembleConstraintForces(
        self,
        constraints: dict,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Evaluate every constraint and add its nodal forces to the net force vector.

        The explicit counterpart of
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleConstraints`, and it shares
        that method's sign convention: a constraint writes what the implicit solver calls ``PExt``,
        which is why this is called after :meth:`assembleLoads`, on a ``P`` that already holds
        ``-P_internal``.

        No tangent is requested. An explicit increment solves no linear system, so a constraint's
        stiffness enters nothing, and
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.applyConstraintExplicit`
        is the entry point that evaluates the constraint forces directly without assembling a tangent.
        A constraint that could act *only* through its tangent would contribute nothing here; that is
        precisely the class :meth:`validateModelCapabilities` refuses.

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

        PPlain = P.asPlainArray()
        for name, constraint in constraints.items():
            self._evaluateConstraintForce(name, constraint, U_np, dU, P, timeStep).addInto(PPlain)

        return P

    def _evaluateConstraintForce(
        self, name: str, constraint: ConstraintBase, U_np: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
    ) -> ConstraintForce:
        """Evaluate one constraint's nodal forces into its own buffer; see :class:`ConstraintForce`.

        Parameters
        ----------
        name
            The constraint's name.
        constraint
            The constraint.
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The net force vector, for its entity mapping.
        timeStep
            The current time step.

        Returns
        -------
        ConstraintForce
            The constraint's forces.
        """

        constraintForce = self._constraintForces.get(name)
        if constraintForce is None:
            indices = P.entitiesInDofVector[constraint]
            # A constraint may name the same DOF more than once -- a slave node that also appears in
            # its own master facet's node list. Whether it does is decided once, here, instead of on
            # every constraint of every increment; see addNodalForces.
            constraintForce = ConstraintForce(np.zeros(constraint.nDof), indices, hasRepeatedDofs(indices))
            self._constraintForces[name] = constraintForce

        # Reused, so it must be cleared: applyConstraintExplicit augments what it is handed.
        constraintForce.forces[:] = 0.0

        constraint.applyConstraintExplicit(U_np[constraint], dU[constraint], constraintForce.forces, timeStep)

        return constraintForce

    def secondOrderMomentum(self, mass: DofVector, V: DofVector, model: FEModel) -> np.ndarray:
        """The linear momentum of the fields whose inertia is a mass, per spatial component; see
        :func:`~edelweissfe.solvers.base.conservationchecks.linearMomentum`.

        Parameters
        ----------
        mass
            The lumped mass to weight with. Pass the *unfolded* mass: a multi-point-constraint slave
            carries real velocity, and folding its mass onto its masters would drop its momentum.
        V
            The velocity vector.
        model
            The model tree, for the fields' spatial dimension.

        Returns
        -------
        np.ndarray
            The momentum, one entry per spatial component.
        """

        return linearMomentum(np.asarray(mass) * np.asarray(V), self.theDofManager, self.linearMomentumFields, model)

    def _perFieldLumpedTotals(self) -> dict[str, float]:
        """The assembled lumped total of every first- and second-order field, each on its own.

        Never summed across fields. A conservation bug that halves one field's total can be
        diluted below detection by another field's total if the two differ enough in magnitude
        -- which happens routinely here, since a mechanical density and a non-local viscosity or
        non-mechanical inertia are not just different units, they are typically many orders of
        magnitude apart in value. That is true even between two fields of the SAME kind (two
        mechanical fields of very different density would have the same problem), so the fix is
        per field, not per "mechanical vs. not". The conservation check tests each field
        once with its own total, so a violation anywhere is visible regardless of what
        else is assembled alongside it.

        Returns
        -------
        dict[str, float]
            Field name to its total lumped mass, viscosity or non-mechanical inertia -- whichever
            that field's own coefficient is.
        """

        totals = {}
        for fieldName in self.firstOrderFields + self.secondOrderFields:
            indices = self.theDofManager.idcsOfFieldsInDofVector[fieldName]
            totals[fieldName] = float(np.sum(self._rawLumpedMass[indices]))
        return totals

    def _dofIndicesOfFields(self, fieldNames: list[str]) -> np.ndarray:
        """The degrees of freedom of the named fields, concatenated in the order given.

        Parameters
        ----------
        fieldNames
            Names of fields present in the current DofManager.

        Returns
        -------
        np.ndarray
            Their indices in the dof vector; empty (and of integer dtype, which matters -- numpy
            reads an index of None as np.newaxis) when no field is named.
        """

        indices = np.empty(0, dtype=int)
        for fieldName in fieldNames:
            indices = np.r_[indices, self.theDofManager.idcsOfFieldsInDofVector[fieldName]]
        return indices

    def _checkDerivedSchemeAgainstTheDeck(self):
        """Compare the derived time-integration scheme against what the deck says it expects.

        An ASSERTION, and deliberately nothing more. ``expect-second-order-fields`` and
        ``expect-first-order-fields`` are never consulted to decide how a field is integrated --
        :meth:`_classifyFieldsByScheme` has already done that from the assembled inertia and
        damping, which remain the single source of truth. This only refuses to continue when the
        two disagree.

        The distinction matters, because the deck used to *declare* the scheme and that is what was
        removed: a declaration can contradict the assembled operators, and the old one had a hole
        through which a field named in neither list was silently never integrated at all. An
        assertion cannot desynchronize from behaviour, because no behaviour reads it, and omitting
        it means "do not check" rather than "integrate nothing".

        What it buys is the one failure :meth:`_classifyFieldsByScheme` structurally cannot see. It
        refuses a field split across its own degrees of freedom, and a field carrying neither
        coefficient; it cannot refuse a field that is uniformly FIRST order when second order was
        intended. Such a run completes and reports success with different physics. That is a
        one-token slip: a non-local field is second order only if its material provides a
        micro-inertia, which for GCDP is optional material property 21 and legitimately defaults to
        zero, so forgetting it silently returns the field to the parabolic scheme -- whose own
        stability limit nothing checks.

        Raises
        ------
        ValueError
            If a field named in either list was derived as the other scheme, or names a field the
            model does not carry.
        """
        derived = {name: "second order" for name in self.secondOrderFields}
        derived.update({name: "first order" for name in self.firstOrderFields})

        for expected, declaredFields in (
            ("second order", self.options["expect-second-order-fields"]),
            ("first order", self.options["expect-first-order-fields"]),
        ):
            for fieldName in declaredFields:
                if fieldName not in derived:
                    raise ValueError(
                        "The deck expects field '{:}' to be integrated {:}, but this model carries "
                        "no such field. It carries: {:}.".format(
                            fieldName, expected, ", ".join(sorted(derived)) or "(no field)"
                        )
                    )

                if derived[fieldName] != expected:
                    raise ValueError(
                        "The deck expects field '{:}' to be integrated {:} in time, but it was "
                        "assembled {:}, so that is how it would have been integrated. The scheme "
                        "follows from the operators the elements report, never from this "
                        "declaration -- which exists precisely so that this disagreement stops the "
                        "run instead of quietly changing the answer.{:}".format(
                            fieldName,
                            expected,
                            derived[fieldName],
                            (
                                " A non-local field is second order only if its material provides a "
                                "micro-inertia; for GCDP that is optional material property 21, and "
                                "leaving it off legitimately means zero."
                                if expected == "second order"
                                else ""
                            ),
                        )
                    )

    def _classifyFieldsByScheme(self, inertia: DofVector, damping: DofVector) -> tuple[list[str], list[str]]:
        """Which fields are integrated second order in time and which first, read off the
        assembled operators rather than declared.

        The deck used to say this and had no freedom in what to say: an inertia is what a
        central-difference update divides by, so a field carrying one is second order and a field
        carrying none is not. Deriving it also closes a hole the declaration had -- a field named
        in neither list was silently never integrated at all.

        Per FIELD, not per degree of freedom: a multi-material mesh can give a micro-inertia to one
        material and not to another, so covering only some of the elements carrying a field is easy
        to do by accident, and integrating part of a field as a wave equation and the rest as a
        diffusion one is not a scheme anybody chose. Such a field is refused rather than split.

        Parameters
        ----------
        inertia
            The assembled lumped inertia, before the first-order entries are overwritten with
            their damping and before any multi-point-constraint fold -- a tied slave's own
            inertia has to still be there, or its field would look unintegrable.
        damping
            The assembled lumped damping, likewise.

        Returns
        -------
        tuple[list[str], list[str]]
            The first-order and the second-order field names, in the model's field order.

        Raises
        ------
        ValueError
            If any field carries a coefficient on some of its degrees of freedom and not on
            others, or carries neither coefficient anywhere.
        """

        firstOrderFields = []
        secondOrderFields = []

        for fieldName, indices in self.theDofManager.idcsOfFieldsInDofVector.items():
            carriesInertia = np.asarray(inertia[indices]) != 0.0
            carriesDamping = np.asarray(damping[indices]) != 0.0

            if carriesInertia.all():
                secondOrderFields.append(fieldName)
            elif carriesInertia.any():
                raise ValueError(
                    "Field {:} was assembled an inertia on {:} of its {:} degrees of freedom and "
                    "none on the rest, so it is neither second order in time nor first order. The "
                    "usual cause is a mesh whose materials disagree: for a non-local field the "
                    "micro-inertia is a material property, so a second material used by some of "
                    "the elements carrying the field and giving no micro-inertia splits it.".format(
                        fieldName, int(np.count_nonzero(carriesInertia)), carriesInertia.size
                    )
                )
            elif carriesDamping.all():
                firstOrderFields.append(fieldName)
            elif carriesDamping.any():
                raise ValueError(
                    "Field {:} carries no inertia, so it is integrated by its damping alone -- but "
                    "a damping was assembled on only {:} of its {:} degrees of freedom, and the "
                    "rest have nothing to divide by.".format(
                        fieldName, int(np.count_nonzero(carriesDamping)), carriesDamping.size
                    )
                )
            else:
                raise ValueError(
                    "Field {:} carries neither an inertia nor a damping, so there is no time "
                    "derivative for this solver to integrate it with. Its elements report both "
                    "through computeLumpedInertia and computeLumpedDamping; an element with zero "
                    "density or zero volume reports the first as zero.".format(fieldName)
                )

        return firstOrderFields, secondOrderFields

    def reportTopologyChangeConservation(
        self,
        lumpedTotalsBefore: dict[str, float],
        momentumBefore: np.ndarray,
        kineticBefore: float,
        V: DofVector,
        model: FEModel,
    ):
        """Report what a topology change did to the quantities that ought to survive it.

        Refinement interpolates the velocity onto new nodes and re-lumps every per-element
        scalar property, and the three invariants behave differently under that:

        * **Every field's own lumped total is conserved exactly**, by the same geometric
          identity regardless of what that field's coefficient physically is: the children of a
          refined element tile it and carry the same value. Checked per FIELD, not per family,
          via :class:`~edelweissfe.solvers.base.conservationchecks.ConservationCheck` -- summing across fields first, even ones
          that agree on units, would let a violation in a numerically small field hide inside a
          numerically large one. Violating any single field's total raises.
        * **Linear momentum is conserved exactly for a spatially uniform velocity field**, because
          the shape functions are a partition of unity and the child masses sum to the parent's. For
          a general field the discrepancy is second order in the velocity gradient across the parent:
          discretisation error, not a defect. Reported, not enforced. Restricted to the fields whose
          inertia is a mass, per :func:`~edelweissfe.config.phenomena.carriesLinearMomentum`: a
          viscosity or a non-mechanical inertia has no associated momentum, and a rotational inertia
          has an angular one that must not be added to a linear one. Across those fields the sum IS
          legitimate, because a real total system momentum is exactly the sum of its parts' momenta.
        * **Kinetic energy is not conserved** by interpolation plus re-lumping, and it is the most
          sensitive of the three, being quadratic in the interpolation error. Reported as a relative
          jump; more than roughly a percent is a reason to look at the transfer rather than to
          believe the physics. Restricted to the fields whose 0.5*m*v**2 is an energy -- the
          mass-carrying ones and the rotational ones, a wider set than momentum admits -- and
          reason, and summed across them for the same reason momentum is.

        Parameters
        ----------
        lumpedTotalsBefore
            Every first- and second-order field's own lumped total before the change, by field
            name; see :meth:`_perFieldLumpedTotals`.
        momentumBefore
            Per-component momentum before the change.
        kineticBefore
            Kinetic energy before the change.
        V
            The velocity vector of the rebuilt system.
        model
            The model tree.

        Raises
        ------
        RuntimeError
            If any field's own lumped total changed by more than its conservation tolerance.
        """

        lumpedTotalsAfter = self._perFieldLumpedTotals()
        momentumAfter = self.secondOrderMomentum(self._rawLumpedMass, V, model)
        kineticAfter = 0.5 * float(
            np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
        )

        tolerance = self.options["lumped-quantity-conservation-tolerance"]
        relativeChangeByField = {
            fieldName: self._conservationCheck.check(
                "lumped coefficient of field '{:}'".format(fieldName),
                before,
                lumpedTotalsAfter.get(fieldName, 0.0),
                tolerance,
            )
            for fieldName, before in lumpedTotalsBefore.items()
        }
        worstField, worstRelativeChange = (
            max(relativeChangeByField.items(), key=lambda item: item[1]) if relativeChangeByField else ("", 0.0)
        )

        # Minv = 1/M is formed wherever M != 0.0 and only negative mass is rejected, so a master
        # DOF left with a tiny positive mass yields an enormous Minv and integrates itself to
        # infinity. Report the smallest inertia actually carried by an integrating DOF, and how far
        # it sits below the median, so a collapsing mass is visible when it appears.
        integrating = self._lumpedMass[self._lumpedMass > 0.0]
        smallestMass = float(np.min(integrating)) if integrating.size else 0.0
        medianMass = float(np.median(integrating)) if integrating.size else 0.0
        massRatio = smallestMass / medianMass if medianMass > 0.0 else 0.0

        # Split by integration scheme, because the two numbers mean different things: a second-order
        # DOF divides an inertia into a central-difference update, a first-order one divides a
        # damping into a forward-Euler update, and their stability limits are different expressions.
        def smallestOf(indices):
            if not indices.size:
                return 0.0, 0.0
            entries = self._lumpedMass[indices]
            entries = entries[entries > 0.0]
            if not entries.size:
                return 0.0, 0.0
            return float(np.min(entries)), float(np.median(entries))

        smallest2nd, median2nd = smallestOf(self.ids_2nd)
        smallest1st, median1st = smallestOf(self.ids_1st)

        self.journal.message(
            "Topology change: worst-conserved field '{:}' to {:.1e} relative; smallest integrating "
            "coefficient {:.3e} ({:.1e} of median) [2nd-order inertia {:.3e} of median {:.3e}; "
            "1st-order damping {:.3e} of median {:.3e}]; {:}".format(
                worstField,
                worstRelativeChange,
                smallestMass,
                massRatio,
                smallest2nd,
                median2nd,
                smallest1st,
                median1st,
                formatMomentumAndKineticEnergy(momentumBefore, momentumAfter, kineticBefore, kineticAfter),
            ),
            self.identification,
            1,
        )

    def getCriticalTimeStepForExplicitDynamics(self, model: FEModel, U: DofVector) -> float:
        """Compute the critical time step for explicit dynamics: the smallest of the elements
        computed here.

        Parameters
        ----------
        model
            The model tree.
        U
            The solution vector.

        Returns
        -------
        float
            The critical time step for explicit dynamics.
        """
        minTimeStep = np.inf

        for element in self.partition.elements.values():
            elementTimeStep = element.computeCriticalTimeStepForExplicitDynamics(U[element])
            if elementTimeStep < minTimeStep:
                minTimeStep = elementTimeStep

        return minTimeStep
