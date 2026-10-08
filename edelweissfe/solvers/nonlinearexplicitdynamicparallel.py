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
# Created on Mon Sep 24 13:52:01 2018

# @author: matthias
"""The nonlinear explicit dynamic solver, with the element loop in bulk, on one thread or several.

:class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED` with one difference: its element loop. ``NED``
computes the elements one at a time -- gather, evaluate, scatter, element after element --, the plainest
form of the loop and the slowest, since the gather and the scatter of one element cost about what a fast
element kernel does. This solver runs the same loop in bulk: the elements are cut into chunks, each chunk
gathers its solution in one indexed read, and the forces of all elements are assembled at once, summed
at every degree of freedom in element order (:class:`~edelweissfe.solvers.base.parallelelementcomputation.ElementPlan`).
The chunks run on ``OMP_NUM_THREADS`` threads (with the GIL disabled; see
:doc:`/documentation/parallelization`). The result is the same, bit for bit, as ``NED``'s on any number of
threads -- the forces, and so the solution; the internal energy of the energy table is summed per chunk on
more than one thread and may differ in its last digits. **Use this solver for production runs**, also on
one thread.
"""

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
    reportThreadAvailability,
)
from edelweissfe.solvers.base.parallelelementcomputation import (
    ElementPlan,
    computeElementsForExplicit,
    computeLumpedDiagonalForExplicit,
    planElements,
)
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.timesteppers.timestep import TimeStep


class NEDParallel(NED):
    """The nonlinear explicit dynamic solver, with the element loop in bulk, on one thread or several.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDPSolver"

    def reportHowElementsAreComputed(self, model: FEModel):
        """Report the threads available to the element loop.

        Parameters
        ----------
        model
            The model tree.
        """

        reportThreadAvailability(getNumberOfThreads(), self.journal, self.identification)

    def elementLoopThreads(self) -> int:
        """The number of threads the element loop runs on: ``OMP_NUM_THREADS``, if the interpreter
        runs without the GIL, one otherwise.

        Returns
        -------
        int
            The number of threads.
        """

        return getNumberOfThreads() if isFreeThreadingSupported() else 1

    def planElementLoop(self, elements: dict) -> ElementPlan:
        """The element loop of :meth:`NED.planElementLoop`, in bulk: the chunks of elements, how they
        gather their degrees of freedom and how their contributions are assembled
        (:func:`~edelweissfe.solvers.base.parallelelementcomputation.planElements`).

        Parameters
        ----------
        elements
            The elements, by number, in element order.

        Returns
        -------
        ElementPlan
            The plan.
        """

        return planElements(
            elements,
            self.theDofManager.idcsOfHigherOrderEntitiesInDofVector,
            self.partition.dofs,
            self.theDofManager.nDof,
            self.elementLoopThreads(),
        )

    @performancetiming.timeit("elements")
    def assembleInternalForces(
        self, U_np: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
    ) -> tuple[DofVector, float]:
        """The element loop of :meth:`NED.assembleInternalForces`, in bulk: see
        :func:`~edelweissfe.solvers.base.parallelelementcomputation.computeElementsForExplicit`.

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

        P[:] = 0.0
        psi, _ = computeElementsForExplicit(self._incrementPlan.elementLoop, U_np, dU, P, timeStep)
        return P, psi

    def assembleLumpedDiagonal(self, elementLoop: ElementPlan, elementContribution) -> DofVector:
        """Assemble a lumped operator of :meth:`NED.assembleLumpedDiagonal`, in bulk.

        Parameters
        ----------
        elementLoop
            The plan of the elements computed here, contact facets included (:meth:`planElementLoop`).
        elementContribution
            As for :meth:`NED.assembleLumpedDiagonal`.

        Returns
        -------
        DofVector
            The assembled diagonal.
        """

        return self.lumpedDiagonalAndContributions(elementLoop, elementContribution)[0]

    def lumpedDiagonalAndContributions(
        self, elementLoop: ElementPlan, elementContribution
    ) -> tuple[DofVector, np.ndarray]:
        """Assemble a lumped operator with a plan, and keep what it was assembled from.

        Parameters
        ----------
        elementLoop
            The plan of the elements.
        elementContribution
            As for :meth:`NED.assembleLumpedDiagonal`.

        Returns
        -------
        tuple[DofVector, np.ndarray]
            The assembled diagonal, and the contribution buffer it was assembled from
            (:func:`~edelweissfe.solvers.base.parallelelementcomputation.computeLumpedDiagonalForExplicit`).
        """

        vector = self.theDofManager.constructDofVector()
        vector[:] = 0.0
        contributions = computeLumpedDiagonalForExplicit(elementLoop, elementContribution, vector)
        return vector, contributions
