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
    def updateModel(self, model: FEModel, step, timeStep: float) -> bool:
        """Invoked by the solver at designated lifecycle hooks (e.g. start of increment).

        Parameters
        ----------
        model
            The FEModel object.
        step
            The current step.
        timeStep
            The current timeStep.

        Returns
        -------
        bool
            True if the model topology, element/node count, or DOF system changed,
            signaling the solver to rebuild equation system structures (DofManager,
            CSR matrices, solution vectors, and MPC transformations).
        """

    def onStepStart(self, model: FEModel, step):
        """Optional lifecycle hook called at the start of an analysis step."""

    def onIncrementEnd(self, model: FEModel, step, timeStep: float):
        """Optional lifecycle hook called after an increment converges."""

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
