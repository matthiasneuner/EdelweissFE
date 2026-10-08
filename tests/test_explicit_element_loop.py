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
"""The plain element loop of serial NED and the chunked one of NEDParallel give the same bits.

``NED`` computes the elements one at a time -- gather, evaluate, scatter, in element order --;
``NEDParallel`` (and ``NEDMPI``) in chunks, assembling at every degree of freedom in element order with
:class:`~edelweissfe.solvers.base.parallelelementcomputation.ElementPlan`, on one thread or several. Each
test case is run with ``NEDParallel``, and on every increment both loops evaluate the same elements in
the same state: the forces -- and the lumped inertia and damping of every equation system -- must be
the same bits; the internal energy, too, on one thread (on several it is summed per chunk).
"""

import ast
import os
import re
import shutil
import subprocess
import sys

import pytest

import edelweissfe

_TESTFILES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles")

#: Python elements (CPE4, C3D8 with contact facets, a tie) and Marmot elements (CPE4, C3D20R refined live).
_DECKS = [
    ("edelweiss-only", "NED"),
    ("edelweiss-only", "NEDContact"),
    ("edelweiss-only", "TieNED"),
    ("marmot", "NED"),
    ("marmot", "NEDLiveAMR"),
]

_BOTH_LOOPS_SCRIPT = """
import contextlib, io, sys
import numpy as np
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.utils.inputfileparser import parseInputFile

counts = {"increments": 0, "forces differ": 0, "energy differs": 0, "operators": 0, "operators differ": 0}
assembleInternalForces = NEDParallel.assembleInternalForces
assembleLumpedDiagonal = NEDParallel.assembleLumpedDiagonal


def bothElementLoops(self, U, dU, P, timeStep):
    plain = self.theDofManager.constructDofVector()
    plain, plainEnergy = NED.assembleInternalForces(self, U, dU, plain, timeStep)
    P, energy = assembleInternalForces(self, U, dU, P, timeStep)
    counts["increments"] += 1
    counts["forces differ"] += int(np.asarray(plain).tobytes() != np.asarray(P).tobytes())
    counts["energy differs"] += int(plainEnergy != energy)
    return P, energy


def bothLumpedLoops(self, elementContribution):
    plain = NED.assembleLumpedDiagonal(self, elementContribution)
    vector = assembleLumpedDiagonal(self, elementContribution)
    counts["operators"] += 1
    counts["operators differ"] += int(np.asarray(plain).tobytes() != np.asarray(vector).tobytes())
    return vector


NEDParallel.assembleInternalForces = bothElementLoops
NEDParallel.assembleLumpedDiagonal = bothLumpedLoops
with contextlib.redirect_stdout(io.StringIO()):
    finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
print("COUNTS", counts)
"""


@pytest.mark.parametrize("nThreads", [1, 4])
@pytest.mark.parametrize("suite, deck", _DECKS, ids=["/".join(deck) for deck in _DECKS])
def test_the_plain_element_loop_gives_the_bits_of_the_plan(tmp_path, suite, deck, nThreads):
    if suite == "marmot":
        pytest.importorskip("edelweissfe.elements.marmotelement.element")
    if (suite, deck) == ("edelweiss-only", "NED") and nThreads > 1:
        # its pure-Python von Mises material shares state between threads: NEDParallel on several
        # threads does not even reproduce itself there (a known defect of the material, not of a loop)
        pytest.skip("the pure-Python von Mises material races on several threads")
    shutil.copytree(os.path.join(_TESTFILES, suite, deck), tmp_path / deck)
    directory = tmp_path / deck
    inputFile = (directory / "test.inp").read_text()
    (directory / "test.inp").write_text(re.sub(r"solver=NED(Parallel)?\b", "solver=NEDParallel", inputFile))
    (directory / "run.py").write_text(_BOTH_LOOPS_SCRIPT)

    environment = dict(
        os.environ,
        PYTHONPATH=os.path.dirname(os.path.dirname(edelweissfe.__file__)),
        OMP_NUM_THREADS=str(nThreads),
    )
    output = subprocess.run(
        [sys.executable, "run.py"], cwd=directory, env=environment, capture_output=True, text=True, timeout=600
    )
    counts = [line for line in output.stdout.splitlines() if line.startswith("COUNTS")]
    assert len(counts) == 1, output.stdout + output.stderr
    counts = ast.literal_eval(counts[0][len("COUNTS ") :])

    assert counts["increments"] > 0 and counts["operators"] >= 2, counts
    assert counts["forces differ"] == 0 and counts["operators differ"] == 0, counts
    if nThreads == 1:
        assert counts["energy differs"] == 0, counts
