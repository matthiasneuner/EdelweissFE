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
"""The critical time step of an explicit analysis, estimated part by part.

The central difference method is stable for :math:`\\Delta t \\le 2/\\omega_\\mathrm{max}`, the critical time step,
with :math:`\\omega_\\mathrm{max}^2` the largest eigenvalue of :math:`\\boldsymbol{M}^{-1}\\boldsymbol{K}`. The explicit
dynamic solver ``NED`` never solves that eigenvalue problem. It asks the parts of the model for an estimate of their
own critical time step -- a lower bound of it --, takes the smallest, and runs at the fraction ``courant-number`` of it.
The parts have disjoint degrees of freedom, so the smallest of their critical time steps is the model's.

A penalty constraint -- a contact, a support, a tie held by a penalty -- owns no degrees of freedom: it adds stiffness
to the ones of the parts it couples. Its estimate is therefore not a third value to take the minimum with, but an
addition to the parts' estimates at the degrees of freedom it couples: every part gives its estimate as a
:class:`CriticalTimeStepBound`, to which the row sums of the constraints' mass-scaled stiffness are added.

An estimate costs something, and how much its value can change while the analysis runs depends on the part, so every
part comes with its own policy of when its estimate is taken anew:

* :class:`CriticalTimeStepOfTheElements`: the finite elements, estimated **once per mesh**;
* :class:`CriticalTimeStepOfEntitiesWithOwnInertia`: the entities that carry their own inertia (the particles of a
  meshfree model), estimated anew **when the margin of the last estimate over the time increment in use may be used up**;
* :class:`CriticalTimeStepOfThePenaltyConstraints`: the degrees of freedom the penalty constraints couple, estimated
  **in every increment**, from the stiffness the constraints have in it.

The theory, and what is not estimated, is explained in :doc:`/theory/criticaltimestep`.
"""

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.numerics.dofvector import DofVector


class CriticalTimeStepBound(ABC):
    """The estimate of the critical time step of one part of a model, in a form the penalty constraints can be added to:
    from upper bounds of the squared frequencies :math:`\\omega^2` of the part's degrees of freedom. A penalty constraint
    adds, at every degree of freedom :math:`i` it couples, the row sum :math:`r_i` of the absolute value of its stiffness
    scaled by the lumped mass on both sides, :math:`r_i = \\sum_j |K_{ij}| / \\sqrt{M_i M_j}`; the part's estimate with
    these additions is a lower bound of the critical time step of the part and the constraints together (see
    :doc:`/theory/criticaltimestep`, "Penalty constraints").

    Attributes
    ----------
    value : float
        The critical time step of the part alone.
    dofs : numpy.ndarray
        The degrees of freedom of the part, indices into the numbering it was estimated on.
    """

    value: float
    dofs: np.ndarray

    @abstractmethod
    def criticalTimeStep(self, rowSumsOfThePenaltyConstraints: np.ndarray) -> float:
        """The critical time step of the part with the penalty constraints at its degrees of freedom: :attr:`value` where
        no constraint couples them.

        Parameters
        ----------
        rowSumsOfThePenaltyConstraints
            The row sums :math:`r_i` of the penalty constraints, one per degree of freedom of the numbering; zero where no
            constraint acts.

        Returns
        -------
        float
            The critical time step, at most :attr:`value`.
        """


class CriticalTimeStepBoundOfEachDof(CriticalTimeStepBound):
    """A part's estimate with one upper bound of the squared frequency per degree of freedom,
    :math:`\\omega_i^2`: the critical time step with the penalty constraints is :math:`2 / \\sqrt{\\max_i (\\omega_i^2 +
    r_i)}` over the degrees of freedom the constraints couple, or the part's own where that is smaller. The finite
    elements give it, with the largest squared frequency of the elements at each degree of freedom
    (:meth:`CriticalTimeStepOfTheElements.bound`).

    Parameters
    ----------
    value
        The critical time step of the part alone.
    dofs
        The degrees of freedom of the part.
    squaredFrequencies
        The upper bound of the squared frequency of each of them, in the order of ``dofs``.
    """

    def __init__(self, value: float, dofs: np.ndarray, squaredFrequencies: np.ndarray):
        self.value = value
        self.dofs = dofs
        self.squaredFrequencies = squaredFrequencies

    def criticalTimeStep(self, rowSumsOfThePenaltyConstraints: np.ndarray) -> float:
        rowSums = rowSumsOfThePenaltyConstraints[self.dofs]
        coupled = rowSums > 0.0
        if not coupled.any():
            return self.value
        return min(self.value, 2.0 / np.sqrt(np.max(self.squaredFrequencies[coupled] + rowSums[coupled])))


class CriticalTimeStepEstimate:
    """The critical time step of one part of a model, :math:`2/\\omega_\\mathrm{max}` of the part's degrees of freedom,
    as last estimated: a lower bound, not scaled by a courant number. A derived class says how the part is estimated,
    and when its estimate is taken anew (its policy).

    Parameters
    ----------
    part
        What the part is, for the journal.
    """

    def __init__(self, part: str):
        self.part = part
        self.reset()

    def reset(self):
        """Forget the estimate, at the start of every step: a material property may have changed with the step. The next
        estimate is due at once."""

        #: The last estimate; infinite before the first one, and for a part without degrees of freedom.
        self.value = np.inf
        #: How often the part was estimated in the step.
        self.numberOfEstimates = 0

    def _take(self, value: float | None):
        """Take a new estimate.

        Parameters
        ----------
        value
            The critical time step of the part; None if the model has no such part.
        """

        self.value = np.inf if value is None else value
        self.numberOfEstimates += 1


class CriticalTimeStepOfTheElements(CriticalTimeStepEstimate):
    """The critical time step of the finite elements, the smallest of the elements' own
    (``computeCriticalTimeStepForExplicitDynamics``): the largest eigenfrequency of an assembled mesh is bounded by the
    largest of its elements, so the smallest critical time step of an element bounds the mesh's.

    **Policy: once per mesh.** An element's estimate depends on its geometry, its lumped mass and the largest wave speed
    of its material, not on the state the material is in: a softening material only raises the true limit. So it is
    estimated anew only when the mesh changes (a refinement), and at the start of every step; not for a new numbering of
    the same mesh, and never within an increment loop that does not change the mesh.
    """

    def __init__(self):
        super().__init__("the finite elements")

    def reset(self):
        super().reset()
        #: The version of the mesh the estimate is of (:attr:`~edelweissfe.models.topologypipeline.TopologyPipeline.version`).
        self._mesh = None
        #: The elements and their critical time steps, as last estimated.
        self._elements = []
        self._criticalTimeStepOfElement = np.zeros(0)
        #: The bound of each degree of freedom (:meth:`bound`), and the numbering and the multi-point constraints it is
        #: of.
        self._bound = None
        self._numberingOfTheBound = None
        self._mpcTransformationOfTheBound = None

    def isDue(self, system) -> bool:
        """Whether the mesh is not the one of the last estimate.

        Parameters
        ----------
        system
            The discrete system.

        Returns
        -------
        bool
            True if the elements are to be estimated anew.
        """

        return self._mesh != system.model.topology.version

    def estimate(self, system, U: DofVector):
        """Estimate the critical time step of the elements of the discrete system
        (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elementsWithNodalInertia`).

        Parameters
        ----------
        system
            The discrete system.
        U
            The solution, which an element may need for its current geometry.
        """

        self._elements = list(system.elementsWithNodalInertia)
        self._criticalTimeStepOfElement = np.array(
            [element.computeCriticalTimeStepForExplicitDynamics(U[element]) for element in self._elements]
        )
        self._take(np.min(self._criticalTimeStepOfElement, initial=np.inf))
        self._mesh = system.model.topology.version
        self._bound = None

    def bound(self, system, mpcTransformation=None) -> CriticalTimeStepBoundOfEachDof:
        """The estimate in the form the penalty constraints add to: at each degree of freedom of the elements, the largest
        squared frequency of the elements there, :math:`\\omega_i^2 = \\max_{e \\ni i} (2 / \\Delta t_e)^2`. Formed anew
        for every numbering, from the elements' critical time steps of the last estimate.

        With multi-point constraints, a slave moves with its masters, and an element at the slave acts on the masters
        (through :math:`T^T K T`): a master takes the largest squared frequency of the elements at it and at its slaves
        (:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.foldLargestOntoMasters`).
        The row sums of the penalty constraints are folded onto the masters as well, so the two meet there.

        Parameters
        ----------
        system
            The discrete system, whose numbering the bound is of.
        mpcTransformation
            The condensation of the multi-point constraints of the discrete system; None if it has none.

        Returns
        -------
        CriticalTimeStepBoundOfEachDof
            The bound.
        """

        if (
            self._bound is None
            or self._numberingOfTheBound is not system.numbering
            or self._mpcTransformationOfTheBound is not mpcTransformation
        ):
            dofsOfElement = [np.asarray(system.locations[element]) for element in self._elements]
            dofs = np.concatenate([np.zeros(0, dtype=np.intp)] + dofsOfElement)
            squaredFrequencyOfElement = (2.0 / self._criticalTimeStepOfElement) ** 2
            squaredFrequencies = np.zeros(system.numbering.nDof)
            np.maximum.at(
                squaredFrequencies, dofs, np.repeat(squaredFrequencyOfElement, [len(d) for d in dofsOfElement])
            )
            dofs = np.unique(dofs)
            if mpcTransformation is not None:
                squaredFrequencies = mpcTransformation.foldLargestOntoMasters(squaredFrequencies)
                ofTheElements = np.zeros(system.numbering.nDof)
                ofTheElements[dofs] = 1.0
                dofs = np.flatnonzero(mpcTransformation.foldLargestOntoMasters(ofTheElements))
            self._bound = CriticalTimeStepBoundOfEachDof(self.value, dofs, squaredFrequencies[dofs])
            self._numberingOfTheBound = system.numbering
            self._mpcTransformationOfTheBound = mpcTransformation
        return self._bound


class CriticalTimeStepOfEntitiesWithOwnInertia(CriticalTimeStepEstimate):
    """The critical time step of the entities that carry their own inertia (the particles of a meshfree model), a property
    of their assembled system at their current configuration
    (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`).

    **Policy: when the margin may be used up.** It depends on the state of the entities -- their stiffness, their
    positions, their lumped inertia and damping -- and changes from increment to increment, but an estimate costs a pass
    over all of them. After an estimate :math:`v`, the solver therefore keeps, in every increment, a lower bound of what
    the estimate could be now, :math:`n` increments later,

    .. math::

        L = v \\, \\min\\left(1, \\sqrt{\\alpha}, \\beta\\right) (1 - \\delta)^n ,

    and estimates anew as soon as the increment in use is no longer a ``courant-number`` fraction of it.

    * :math:`\\alpha = \\min_i M_i / M_{0,i}` is the largest relative fall of the lumped inertia since the estimate, over
      the degrees of freedom that had one then and have one now, and :math:`\\beta` the same for the damping of the
      degrees of freedom integrated by forward Euler (a damping, no inertia). This part is exact: the estimate scales the
      stiffness with :math:`\\boldsymbol{M}^{-1/2}` on both sides, so with the stiffness held, a fall of the masses by
      :math:`\\alpha` raises :math:`\\omega_\\mathrm{max}^2` at most by :math:`1/\\alpha` and lowers :math:`2/\\omega_\\mathrm{max}`
      at most by :math:`\\sqrt{\\alpha}`; the forward Euler limit :math:`2/\\lambda_\\mathrm{max}(\\boldsymbol{C}^{-1}
      \\boldsymbol{K})` by :math:`\\beta`. The masses are known in every increment: the solver assembles them anyway.
    * :math:`\\delta` (option ``critical-time-step-decrease-per-increment``) is the largest relative fall of the estimate
      per increment that its stiffness may cause. This part is an assumption, not a theorem: the tangent of a material
      can change abruptly, and the stiffness of a particle changes with its deformation. It rests on what an increment
      is -- a fraction of the time a wave needs to cross an entity, so the configuration moves by a fraction of an entity
      -- and on measurements, and it is documented with them in :doc:`/theory/criticaltimestep`.

    The estimate is also taken anew when a degree of freedom has an inertia, or a damping alone, that it did not have at
    the last estimate (a node that a particle reaches for the first time: a new row whose mass starts from zero), after
    a new numbering, and in the first evaluation of a step. A degree of freedom that loses its inertia leaves the system
    and can only raise its critical time step.

    Where the entities set the time increment, the margin is used up after one increment, and they are estimated in
    every increment; where another part sets it, they are estimated only every so many increments, as many as their
    margin allows at the rate :math:`\\delta`.

    **Carried weights** (option ``carry-critical-time-step-weights``). The estimate of the entities is a
    Collatz-Wielandt bound with weights constant per field
    (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`). With carried
    weights, every estimate takes the weights the last one left and leaves the next ones, one step of the power
    iteration towards the Perron vector, where the bound is exact. Any positive weights bound the largest eigenvalue
    from above, but weights refined at another configuration can bound it less tightly than the weights constant per
    field they started from when the stiffness changed faster than the weights followed, e.g. where entities reach
    nodes whose weights the power iteration has made small. So an estimate with carried weights below the estimate the
    weights started from starts them anew: the entities are estimated again from the weights constant per field, and
    the larger of the two estimates is taken. Every estimate is thus at least the one the weights started from: with
    the time increment lowered only, it never falls below the smallest estimate with weights constant per field of
    the step; followed (``follow-critical-time-step``), it can fall below the estimate without carried weights of the
    same configuration, down to the one the weights started from. The weights start anew -- constant per field -- also
    with every numbering (the nodes they belong to may have changed) and every step (a material property may have
    changed), and after a resume: they are not part of the checkpoint.

    Parameters
    ----------
    decreasePerIncrement
        :math:`\\delta`, in :math:`(0, 1]`; 1 estimates in every increment.
    carryWeights
        Whether the weights of the estimate are carried from one estimate to the next.
    """

    def __init__(self, decreasePerIncrement: float, carryWeights: bool = False):
        if not 0.0 < decreasePerIncrement <= 1.0:
            raise ValueError(
                "The decrease of the critical time step per increment must lie in (0, 1], not {:}.".format(
                    decreasePerIncrement
                )
            )
        self.decreasePerIncrement = decreasePerIncrement
        self.carryWeights = carryWeights
        super().__init__("the entities that carry their own inertia")

    def reset(self):
        super().reset()
        #: The last estimate, in the form the penalty constraints add to; None before the first one, and for a model
        #: without such entities.
        self.bound = None
        #: The numbering, the number of the increment, and the lumped inertia and damping of the last estimate.
        self._numbering = None
        self._increment = None
        self._inertia = None
        self._damping = None
        #: With carried weights, the weights the last estimate left for the next one, a vector of its numbering, zero
        #: where a degree of freedom has none yet; None without carried weights, and before the first estimate on a
        #: numbering.
        self._weightsOfTheBound = None
        #: With carried weights, the critical time step of the estimate the weights started from, the first one after
        #: they were set to zero, with weights constant per field.
        self._valueTheWeightsStartedFrom = np.inf

    def estimate(self, system, lumpedInertia: DofVector, lumpedDamping: DofVector, forEachChunk, increment: int):
        """Estimate the critical time step of the entities at their current configuration; with carried weights, from
        the weights the last estimate left, started anew where it falls below the estimate they started from (see the
        class documentation, "Carried weights").

        Parameters
        ----------
        system
            The discrete system.
        lumpedInertia
            The lumped inertia of the entities at their current configuration, as the solver has just assembled it.
        lumpedDamping
            Their lumped damping, likewise.
        forEachChunk
            The solver's loop over chunks of independent entities.
        increment
            The number of the increment just accepted.
        """

        if self.carryWeights and self._numbering is not system.numbering:
            self._weightsOfTheBound = np.zeros(system.numbering.nDof)
        weights = self._weightsOfTheBound
        startsTheWeights = weights is not None and not weights.any()

        bound = system.criticalTimeStepOfEntitiesWithOwnInertia(lumpedInertia, lumpedDamping, forEachChunk, weights)
        if bound is not None and weights is not None:
            if startsTheWeights:
                self._valueTheWeightsStartedFrom = bound.value
            elif bound.value < self._valueTheWeightsStartedFrom:
                weights[:] = 0.0
                anew = system.criticalTimeStepOfEntitiesWithOwnInertia(
                    lumpedInertia, lumpedDamping, forEachChunk, weights
                )
                self._valueTheWeightsStartedFrom = anew.value
                if anew.value > bound.value:
                    bound = anew
        self.bound = bound
        self._take(None if self.bound is None else self.bound.value)
        self._numbering = system.numbering
        self._increment = increment
        self._inertia = np.array(lumpedInertia)
        self._damping = np.array(lumpedDamping)

    def lowerBound(self, increment: int, lumpedInertia: DofVector, lumpedDamping: DofVector) -> float:
        """The lower bound :math:`L` of the critical time step of the entities after the increment ``increment``, from the
        last estimate (see the class documentation). It holds for the degrees of freedom that had an inertia (or a
        damping alone) at the last estimate; whether there are others, :meth:`isDue` asks first.

        Parameters
        ----------
        increment
            The number of the increment just accepted.
        lumpedInertia
            The lumped inertia of the entities now.
        lumpedDamping
            Their lumped damping now.

        Returns
        -------
        float
            The lower bound.
        """

        massFactor, stiffnessFactor = self._fallSinceTheEstimate(increment, lumpedInertia, lumpedDamping)
        return self.value * massFactor * stiffnessFactor

    def _fallSinceTheEstimate(
        self, increment: int, lumpedInertia: DofVector, lumpedDamping: DofVector
    ) -> tuple[float, float]:
        """The factors :math:`\\min(1, \\sqrt{\\alpha}, \\beta)` and :math:`(1 - \\delta)^n` by which the critical time
        step of the entities may have fallen since the last estimate, from their masses and from their stiffness (see
        the class documentation).

        Parameters
        ----------
        increment
            The number of the increment just accepted.
        lumpedInertia
            The lumped inertia of the entities now.
        lumpedDamping
            Their lumped damping now.

        Returns
        -------
        tuple[float, float]
            The two factors, each in (0, 1].
        """

        inertia, damping = np.asarray(lumpedInertia), np.asarray(lumpedDamping)
        withInertia = (inertia > 0.0) & (self._inertia > 0.0)
        withDampingAlone = ~(inertia > 0.0) & (damping > 0.0) & ~(self._inertia > 0.0) & (self._damping > 0.0)
        alpha = np.min(inertia[withInertia] / self._inertia[withInertia], initial=1.0)
        beta = np.min(damping[withDampingAlone] / self._damping[withDampingAlone], initial=1.0)
        return min(1.0, np.sqrt(alpha), beta), (1.0 - self.decreasePerIncrement) ** (increment - self._increment)

    def isDue(
        self,
        system,
        increment: int,
        lumpedInertia: DofVector,
        lumpedDamping: DofVector,
        criticalTimeStepInUse: float,
        rowSumsOfThePenaltyConstraints: np.ndarray | None = None,
    ) -> bool:
        """Whether the entities are to be estimated anew after the increment ``increment``.

        Parameters
        ----------
        system
            The discrete system.
        increment
            The number of the increment just accepted.
        lumpedInertia
            The lumped inertia of the entities now.
        lumpedDamping
            Their lumped damping now.
        criticalTimeStepInUse
            The critical time step the time increment in use is the ``courant-number`` fraction of.
        rowSumsOfThePenaltyConstraints
            The row sums of the penalty constraints in this increment
            (:class:`CriticalTimeStepOfThePenaltyConstraints`), if the model has penalty constraints.

        Returns
        -------
        bool
            True in the first evaluation of a step, after a new numbering, when a degree of freedom has an inertia (or a
            damping alone) it did not have at the last estimate, and when the lower bound of the critical time step
            (:meth:`lowerBound`) has fallen below the one in use -- alone, or with the penalty constraints that couple
            the entities' degrees of freedom: their row sums are those of now, the entities' stiffness may have grown
            by the square of the same factor as in :meth:`lowerBound`.
        """

        if self._numbering is not system.numbering:
            return True
        inertia, damping = np.asarray(lumpedInertia), np.asarray(lumpedDamping)
        hadInertia = self._inertia > 0.0
        hadDampingAlone = ~hadInertia & (self._damping > 0.0)
        hasInertia = inertia > 0.0
        hasDampingAlone = ~hasInertia & (damping > 0.0)
        if np.any(hasInertia & ~hadInertia) or np.any(hasDampingAlone & ~hadDampingAlone):
            return True
        if self.lowerBound(increment, lumpedInertia, lumpedDamping) < criticalTimeStepInUse:
            return True
        if rowSumsOfThePenaltyConstraints is None or not np.any(rowSumsOfThePenaltyConstraints[self.bound.dofs] > 0.0):
            return False
        # With the entities' stiffness grown by 1/fall^2, their critical time step with the constraints is fall times
        # the one with the constraints' row sums scaled by fall^2.
        massFactor, stiffnessFactor = self._fallSinceTheEstimate(increment, lumpedInertia, lumpedDamping)
        fall = massFactor * stiffnessFactor
        return fall * self.bound.criticalTimeStep(fall**2 * rowSumsOfThePenaltyConstraints) < criticalTimeStepInUse

    def withThePenaltyConstraints(self, rowSumsOfThePenaltyConstraints: np.ndarray) -> float:
        """The critical time step of the entities with the penalty constraints that couple their degrees of freedom,
        from the last estimate (:meth:`CriticalTimeStepBound.criticalTimeStep`).

        Parameters
        ----------
        rowSumsOfThePenaltyConstraints
            The row sums of the penalty constraints, one per degree of freedom.

        Returns
        -------
        float
            The critical time step; infinite in a model without such entities.
        """

        return np.inf if self.bound is None else self.bound.criticalTimeStep(rowSumsOfThePenaltyConstraints)


class CriticalTimeStepOfThePenaltyConstraints(CriticalTimeStepEstimate):
    """The critical time step of the degrees of freedom the penalty constraints couple -- contacts, supports and ties
    held by a penalty --, with the parts they belong to: the estimate of each part
    (:class:`CriticalTimeStepBound`) with the row sums of the constraints added at the degrees of freedom they couple,

    .. math::

        \\omega_\\mathrm{max}^2 \\le \\max_i \\left(\\omega_i^2 + r_i\\right), \\qquad
        r_i = \\frac{1}{\\sqrt{M_i}} \\sum_j |K^\\mathrm{c}_{ij}| \\frac{1}{\\sqrt{M_j}} ,

    with :math:`\\omega_i^2` the part's bound at the degree of freedom :math:`i` and :math:`\\boldsymbol{K}^\\mathrm{c}`
    the stiffness of the constraints (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`).
    A constraint does not take the minimum with the parts: it raises the frequencies of the degrees of freedom it couples,
    so it lowers the critical time step only where it acts, and only by what it adds there.

    **Policy: in every increment.** The stiffness of a constraint follows the state of the contact -- which of its
    points are closed, and how deep --, which changes from increment to increment, and bounding it costs about as much
    as evaluating its force. It is estimated from the stiffness the constraints have after the increment, for the next
    one: a point that closes within an increment enters the estimate after it.
    """

    def __init__(self):
        super().__init__("the degrees of freedom the penalty constraints couple")

    def estimate(
        self,
        system,
        rowSumsOfThePenaltyConstraints: np.ndarray,
        elements: CriticalTimeStepOfTheElements,
        entities: CriticalTimeStepOfEntitiesWithOwnInertia,
        mpcTransformation=None,
    ):
        """Estimate the critical time step of the degrees of freedom the penalty constraints couple, with the last
        estimates of the parts.

        Parameters
        ----------
        system
            The discrete system.
        rowSumsOfThePenaltyConstraints
            The row sums :math:`r_i` of the penalty constraints, one per degree of freedom of the numbering, folded onto
            the masters of multi-point constraints.
        elements
            The estimate of the finite elements.
        entities
            The estimate of the entities that carry their own inertia.
        mpcTransformation
            The condensation of the multi-point constraints of the discrete system; None if it has none
            (:meth:`CriticalTimeStepOfTheElements.bound`).
        """

        self._take(
            min(
                elements.bound(system, mpcTransformation).criticalTimeStep(rowSumsOfThePenaltyConstraints),
                entities.withThePenaltyConstraints(rowSumsOfThePenaltyConstraints),
            )
        )
