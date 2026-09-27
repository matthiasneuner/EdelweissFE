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
"""The initial acceleration of the implicit dynamic solver (``NID``): the reduced equilibrium
:math:`M_{ff} a_f = R_f` on the free dynamic degrees of freedom.

The fixture has everything the reduction has to get right at once: a dynamic field (the
displacement), a quasi-static one (the AT2 phase field of a ``GC3D20R``), Dirichlet degrees of
freedom (the lateral displacements), and hanging-node multi-point constraints (one of the two
elements is refined at the start). A uniform body force ``b`` along the bar, free in that direction,
must then accelerate every node by exactly ``b / density`` -- slaves included -- and nothing else.
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.utils.inputfileparser import parseInputFile

DENSITY = 2.0
BODY_FORCE = 3.0

DECK = f"""
*material, name=AT2PHASEFIELD, id=at2
**E    nu   Gc   l    density        eta
100.0, 0.2, 1.0, 0.5, {DENSITY!r}, 1e-3

*section, name=section1, material=at2, type=solid
gen_all

*job, name=nidInitialAcceleration, domain=3d
*solver, solver=NID, name=theSolver

*modelGenerator, generator=boxGen, name=gen
nX=2
nY=1
nZ=1
lX=2.0
lY=1.0
lZ=1.0
elType=GC3D20R

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=gen_left, initialOnly=True
refineElSet=gen_all
maxLevel=1

*step, solver=theSolver
stepLength=1.0, startInc=0.1, maxInc=0.1, minInc=1e-6, maxNumInc=1, maxIter=10
>>dirichlet, name=lateral, nSet=all, field=displacement, 2=0.0, 3=0.0
>>bodyforce, name=gravity, elSet=all, forceVector='{BODY_FORCE!r}, 0.0, 0.0', f(t)='1'
"""


class _StopAfterInitialAcceleration(Exception):
    """Ends the run once the initial acceleration is committed: the bar is free along its axis,
    so the Newton solve that would follow is singular, and it is not what is tested here."""


def _initialAcceleration(tmp_path, monkeypatch):
    """Run the fixture up to its initial acceleration; return the solver and the mass solve."""

    solves = []
    trueSolve = NonlinearImplicitDynamic._solveWithMassMatrix
    trueCompute = NonlinearImplicitDynamic._computeInitialAcceleration

    def recordingSolve(self, Mff, Rf):
        af = trueSolve(self, Mff, Rf)
        solves.append((Mff, Rf, af))
        return af

    def stoppingCompute(self, *args):
        trueCompute(self, *args)
        raise _StopAfterInitialAcceleration(self)

    monkeypatch.setattr(NonlinearImplicitDynamic, "_solveWithMassMatrix", recordingSolve)
    monkeypatch.setattr(NonlinearImplicitDynamic, "_computeInitialAcceleration", stoppingCompute)

    path = tmp_path / "nid_initial_acceleration.inp"
    path.write_text(DECK)
    with pytest.raises(_StopAfterInitialAcceleration) as stopped:
        finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return stopped.value.args[0], solves


def test_a_uniform_body_force_accelerates_a_free_bar_by_force_over_density(tmp_path, monkeypatch):
    solver, solves = _initialAcceleration(tmp_path, monkeypatch)

    assert (
        solver.mpcTransformation is not None and solver.mpcTransformation.nEliminatedDof > 0
    ), "no hanging node: the fixture does not exercise the condensation"
    ((Mff, Rf, af),) = solves
    np.testing.assert_allclose(Mff @ af, Rf, rtol=0.0, atol=1e-12 * np.abs(Rf).max())

    dofManager = solver.theDofManager
    A = np.asarray(solver._newmarkSystem.A)
    displacement = dofManager.idcsOfFieldsInDofVector["displacement"]
    Ax, Ay, Az = (A[displacement][component::3] for component in range(3))
    np.testing.assert_allclose(Ax, BODY_FORCE / DENSITY, rtol=1e-9)
    np.testing.assert_array_equal(Ay, 0.0)
    np.testing.assert_array_equal(Az, 0.0)

    # the phase field is quasi-static: it has no acceleration equation, and gets none
    np.testing.assert_array_equal(A[dofManager.idcsOfFieldsInDofVector["nonlocal damage"]], 0.0)


def test_refuses_an_unconverged_initial_acceleration(tmp_path, monkeypatch):
    monkeypatch.setattr(NonlinearImplicitDynamic, "_MASS_SOLVE_TOLERANCE", 1e-300)
    with pytest.raises(RuntimeError, match="initial acceleration did not converge"):
        _initialAcceleration(tmp_path, monkeypatch)
