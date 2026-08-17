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
"""Round-trip test for HAdaptivity.getRestartData/setRestartData directly (unlike
tests/test_restart_integration.py's AMR case, which drives the same feature end-to-end through a
real solve). Builds two independent models from the same .inp via fillFEModelFromInputFile
(mirroring the driver's own setup sequence, minus solvers/steps, which this doesn't need), drives
one real refinement on the first via the live updateModel path, and asserts that replaying the
captured getRestartData() onto the second -- freshly rebuilt, unrefined -- reproduces the same
topology.

Uses an `initialOnly` marker (edelweissfe.adaptivity.marking) so refinement triggers
deterministically on the very first updateModel call, regardless of field state -- avoids
depending on a real solve's marker-evaluation timing, which the full-solve integration test in
test_restart_integration.py already covers separately.
"""

from pathlib import Path

from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputfileparser import parseInputFile

_INP = """
*material, name=linearelastic, id=mat
18000, 0.0

*section, name=sec, material=mat, type=solid
fixed_all

*modelGenerator, generator=boxGen, name=fixed
nX      =2
nY      =2
nZ      =2
x0      =0
y0      =0
z0      =0
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=fixed_top, initialOnly=True
maxLevel=1

*job, name=hadaptivityRestartTest, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=fixed_all, field=displacement, result=U, name=dispFixed

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixBack, nSet=fixed_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""


def _buildModel(tmp_path: Path, name: str) -> FEModel:
    inpPath = tmp_path / name
    inpPath.write_text(_INP)
    inputfile = parseInputFile(str(inpPath))

    journal = Journal(verbose=False)
    job = inputfile["job"][0]
    from edelweissfe.config.phenomena import domainMapping

    model = FEModel(domainMapping[job["domain"]])
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(job.get("startTime", 0.0))

    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])

    return model


def _hangingRecordsByLabel(hangingConstraint) -> dict:
    """``_records`` is a ``list[(Node, [(Node, weight), ...])]`` -- Node identity differs between
    two independently-built models even when labels match, so compare by label instead."""
    return {
        slave.label: sorted((master.label, weight) for master, weight in masters)
        for slave, masters in hangingConstraint._records
    }


def test_restart_data_roundtrip_reproduces_committed_refinement(tmp_path):
    modelA = _buildModel(tmp_path, "a.inp")
    amrA = modelA.modelModifiers["amr"]

    with modelA.topologyChanges():  # the solver opens this per increment; see FEModel.topologyChanges
        refined = amrA.updateModel(modelA, step=None, timeStep=0.0)
    assert refined, "the initialOnly marker should have triggered a refinement on the first call"
    assert amrA._committedOccasions, "the committed occasion log should now have one entry"

    restartData = amrA.getRestartData()
    assert restartData is not None

    modelB = _buildModel(tmp_path, "b.inp")
    amrB = modelB.modelModifiers["amr"]
    assert len(modelB.elements) < len(modelA.elements), "model B must start unrefined"

    with modelB.topologyChanges():  # FEModel.readRestart opens this around the replay
        amrB.setRestartData(modelB, restartData)

    assert set(modelB.elements.keys()) == set(modelA.elements.keys())
    assert set(modelB.nodes.keys()) == set(modelA.nodes.keys())
    assert amrB._committedOccasions == amrA._committedOccasions
    assert _hangingRecordsByLabel(amrB._hanging) == _hangingRecordsByLabel(amrA._hanging)
    # A checkpoint only exists after at least one increment converged, so this can never truly be
    # modelB's first updateModel call -- otherwise the live path would re-evaluate initialOnly
    # markers redundantly on its next real call.
    assert amrB._isFirstCall is False


def test_restart_data_roundtrip_reproduces_pending_marks(tmp_path):
    modelA = _buildModel(tmp_path, "a.inp")
    amrA = modelA.modelModifiers["amr"]
    with modelA.topologyChanges():  # the solver opens this per increment; see FEModel.topologyChanges
        amrA.updateModel(modelA, step=None, timeStep=0.0)

    # Simulate a second marking round that hasn't reached minMarkedElements yet: still-active,
    # not-yet-refined elements sitting in the batching buffer at checkpoint time.
    stillActive = [el for eid, el in amrA._eidToEl.items() if amrA._mesh.elements[eid]["active"]]
    pendingLabels = {el.elNumber for el in stillActive[:1]}
    amrA._pendingMarkedElements = {el for el in stillActive if el.elNumber in pendingLabels}

    restartData = amrA.getRestartData()
    assert restartData is not None

    modelB = _buildModel(tmp_path, "b.inp")
    amrB = modelB.modelModifiers["amr"]
    with modelB.topologyChanges():  # FEModel.readRestart opens this around the replay
        amrB.setRestartData(modelB, restartData)

    assert {el.elNumber for el in amrB._pendingMarkedElements} == pendingLabels
