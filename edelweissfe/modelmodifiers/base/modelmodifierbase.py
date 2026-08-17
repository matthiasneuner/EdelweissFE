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

"""Abstract base class for all ModelModifier entities in EdelweissFE."""

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import OptionSchemaProvider


class ModelModifierBase(OptionSchemaProvider, ABC):
    """Abstract base class for entities that dynamically mutate the FEModel topology,
    mesh, or state variables during analysis steps.
    """

    def __init__(self, name: str, model: FEModel, journal: Journal, **kwargs):
        self._name = name
        self._model = model
        self._journal = journal

    @property
    def name(self) -> str:
        """Name of the model modifier."""
        return self._name

    @abstractmethod
    def plan(self, model: FEModel, change, step, timeStep: float):
        """Decide what, if anything, this modifier wants to change about the model -- without
        changing it.

        This half is allowed to read solution state: markers, node fields, the current time, the
        step. It must return a **serializable** description of the decision, which
        :meth:`apply` can carry out on its own, or ``None`` if there is nothing to do.

        A restart replay never calls this. It reads the recorded plan and calls :meth:`apply`
        directly, which is what makes a resumed run reproduce the original topology exactly instead
        of re-deciding from state that may not be identical.

        Parameters
        ----------
        model
            The FEModel object. Read only -- do not mutate it here.
        change
            The net :class:`~edelweissfe.models.modelchange.ModelChange` since this modifier last
            planned within the current topology update, or ``None`` on the first round. **Return
            ``None`` when it does not touch this modifier's domain** -- that is what lets the
            pipeline reach a fixed point instead of looping (see
            :meth:`~edelweissfe.models.femodel.FEModel.updateTopology`).
        step
            The current step.
        timeStep
            The current timeStep.

        Returns
        -------
        object | None
            A plan for :meth:`apply`, or ``None``.
        """

    @abstractmethod
    def apply(self, model: FEModel, plan):
        """Carry out ``plan``, mutating the model.

        **Must not read solution state.** A pure function of ``(model, plan)`` -- everything the
        decision depended on belongs in the plan. This is the single code path shared by a live run
        and a restart replay, and the reason a replayed mesh is numbered identically: there is no
        second implementation to drift apart from this one.

        Runs inside an open topology window (see
        :meth:`~edelweissfe.models.femodel.FEModel.topologyChanges`), so it may create and delete
        elements -- through :meth:`~edelweissfe.models.femodel.FEModel.reserveElementNumbers` and
        :meth:`~edelweissfe.models.femodel.FEModel.createElement`, never by writing
        ``model.elements`` directly.

        Parameters
        ----------
        model
            The FEModel object, mutated in place.
        plan
            A plan previously returned by :meth:`plan` (live) or restored from the recorded
            topology history (replay).

        Returns
        -------
        ModelChange
            The changeset this mutation produced.
        """

    def updateModel(self, model: FEModel, step, timeStep: float) -> bool:
        """Plan, then apply, in one call.

        A convenience for callers outside the pipeline (and for tests). The pipeline itself --
        :meth:`~edelweissfe.models.femodel.FEModel.updateTopology` -- calls :meth:`plan` and
        :meth:`apply` separately, so that it can run the modifiers to a fixed point and record each
        plan for restart.

        Returns
        -------
        bool
            True if the topology changed.
        """

        plan = self.plan(model, None, step, timeStep)
        if plan is None:
            return False
        self.apply(model, plan)
        return True

    def onStepStart(self, model: FEModel, step):
        """Optional lifecycle hook called at the start of an analysis step."""

    def onIncrementEnd(self, model: FEModel, step, timeStep: float):
        """Optional lifecycle hook called after an increment converges."""

    #: Element labels whose state this modifier restored itself in :meth:`setRestartData`
    #: (keyed by something stable, e.g. an octree id). FEModel.readRestart skips these in its own
    #: number-keyed element-state restore, because element numbers are not reproducible across a
    #: replay -- see :meth:`~edelweissfe.models.femodel.FEModel.readRestart`.
    restoredElementLabels: frozenset = frozenset()

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """Return this modifier's history needed to reproduce its effect on the model topology
        (e.g. AMR's log of past refinement decisions), to be serialized by
        :meth:`~edelweissfe.models.femodel.FEModel.writeRestart`, or ``None`` if the modifier never
        changed the topology and has nothing pending.

        The default implementation returns ``None``, correct for any modifier that never mutates
        topology at all.

        Returns
        -------
        dict[str, np.ndarray] | None
            A flat mapping of array name to array, or ``None``.
        """

        return None

    def setRestartData(self, model: FEModel, data: dict[str, np.ndarray]):
        """Reproduce this modifier's effect on ``model``'s topology from a restart checkpoint.

        Unlike :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.setRestartData`
        (passive: only restores internal history, never touches the model), this is expected to
        *mutate* ``model`` -- e.g. AMR replaying past refinements to recreate the elements/nodes a
        plain rebuild from the ``.inp`` file cannot reproduce. That is why it takes ``model``
        explicitly rather than relying on a modifier-held reference.

        Called by :meth:`~edelweissfe.models.femodel.FEModel.readRestart` *before* it restores node
        fields, element state variables, and scalar variables -- any element this call materializes
        must already exist by the time that follow-up restore runs, since it addresses elements by
        label.

        Parameters
        ----------
        model
            The FEModel object, to be mutated as needed.
        data
            The mapping previously returned by :meth:`getRestartData`.
        """

        raise NotImplementedError("This model modifier does not carry restartable topology history.")
