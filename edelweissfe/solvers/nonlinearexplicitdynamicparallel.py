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
"""The nonlinear explicit dynamic solver, with the element loop on several threads.

:class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED` with one difference: its element loop
runs on ``OMP_NUM_THREADS`` threads (with the GIL disabled; see
:doc:`/documentation/parallelization`). The loop is the same one -- the same chunks, the same
assembly in element order -- so the result is bit-identical to ``NED``.
"""

from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
    reportThreadAvailability,
)
from edelweissfe.solvers.nonlinearexplicitdynamic import NED


class NEDParallel(NED):
    """The nonlinear explicit dynamic solver, with the element loop on several threads.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDPSolver"

    def beginStep(self, step, model, fieldOutputController, outputmanagers):
        """Report the threads available, then start the step; see :meth:`NED.beginStep`.

        Parameters
        ----------
        step
            The step to solve.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """

        reportThreadAvailability(getNumberOfThreads(), self.journal, self.identification)
        return super().beginStep(step, model, fieldOutputController, outputmanagers)

    def elementLoopThreads(self) -> int:
        """The number of threads the element loop runs on: ``OMP_NUM_THREADS``, if the interpreter
        runs without the GIL, one otherwise.

        Returns
        -------
        int
            The number of threads.
        """

        return getNumberOfThreads() if isFreeThreadingSupported() else 1
