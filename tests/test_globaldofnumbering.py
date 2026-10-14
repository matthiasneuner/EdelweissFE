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
"""The global numbering of the degrees of freedom, written down from the mesh
(:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`), is the DofManager's.

A domain-decomposed run (``NEDMPI``) checks it against the DofManager whenever the equation system is
built (``Subdomain._checkReplicatedLayout``); these decks, run with ``NEDMPI`` on one process, build it
again after adaptive refinements and contact or tie surfaces regenerated, and every check must pass.
The model the run ends with is checked once more here, against a DofManager built afresh.
"""

import contextlib
import io
import os
import re
import shutil

import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.numerics.globaldofnumbering import GlobalDofNumbering
from edelweissfe.utils.inputfileparser import parseInputFile

_TESTFILES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testfiles")

#: (directory, number of builds of the equation system at least): contact, ties, a rigid body, live
#: adaptive refinement, and refinement with a tie.
_DECKS = [
    ("edelweiss-only/NEDContact", 1),
    ("edelweiss-only/NEDSurfaceContact", 1),
    ("edelweiss-only/NEDSurfaceToDiscreteRigidBodyContact", 1),
    ("edelweiss-only/TieNED", 1),
    ("marmot/NEDLiveAMR", 2),
    ("mpi/marmot/NEDLiveAMRTieDistributed", 2),
]


@pytest.mark.parametrize("deck,builds", _DECKS, ids=[deck for deck, _ in _DECKS])
def test_the_numbering_of_the_mesh_agrees_with_the_dof_manager(tmp_path, monkeypatch, deck, builds):
    source = os.path.join(_TESTFILES, deck)
    if not os.path.isfile(os.path.join(source, "test.inp")):
        pytest.skip("deck {:} not found".format(deck))
    shutil.copytree(source, tmp_path / "deck")
    inputFile = tmp_path / "deck" / "test.inp"
    inputFile.write_text(re.sub(r"(\*solver,\s*solver=)NED(Parallel)?\b", r"\1NEDMPI", inputFile.read_text()))

    checks = []
    check = GlobalDofNumbering.requireAgreementWith

    def counted(numbering, *args):
        checks.append(numbering.nDof)
        return check(numbering, *args)

    monkeypatch.setattr(GlobalDofNumbering, "requireAgreementWith", counted)
    monkeypatch.chdir(tmp_path / "deck")
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)

    assert len(checks) >= builds
    dofManager = DofManager(list(model.nodeFields.values()), list(model.scalarVariables.values()))
    GlobalDofNumbering.ofModel(model).requireAgreementWith(dofManager, model.nodeFields, model.scalarVariables)
