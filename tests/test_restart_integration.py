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
"""P4 of PLAN_RESTART.md: an end-to-end restart test driving the real ``.inp``/driver/solver stack
(unlike ``tests/test_femodel_restart.py``, which only exercises ``FEModel.writeRestart``/
``readRestart`` directly).

Mirrors EdelweissMeshfree's ``examples/114_marmot_micropolar_snni_quad_restart_test/`` two-invocation
shape (run once uninterrupted for a reference, run again truncated + resumed, diff the final
result), adapted for EdelweissFE's ``*restart``/``*output, type=restart`` keywords instead of
Meshfree's ``solveStep(...)`` kwargs. Uses ``VonMises`` (a Marmot-backed material with real
per-quadrature-point plastic history), not ``LinearElastic``: a history-free material would not
exercise ``getStateVars``/``setStateVars`` round-tripping at all, so it could not catch a state
transfer bug.

Parametrized over ``linsolver`` (``pardiso``, the shipped default, and ``blockamg``, the
block-AMG-preconditioned outer solver from ``perf/linsolve-investigation``): restart's
reconstruct-then-overwrite model construction never touches the linear solver at all -- every
``NIST`` step builds a fresh one in ``solveStep`` regardless of whether the run is resumed -- so
this is less "does restart know about blockamg" (it doesn't need to) and more a pinned regression
proving that claim, on the one solver stack this branch actually adds.

This two-invocation shape does not fit ``run_tests_edelweissfe``'s single ``test.inp``/``U.ref``
model, so it lives here as a real pytest test instead (this repo's ``tests/`` already hosts one,
see ``PLAN_INPUT_SYSTEM.md``), driving ``finiteElementSimulation`` directly rather than the ``.inp``
files under ``testfiles/`` -- both invocations of the resumed run must share byte-identical step
definitions with the uninterrupted reference (see the module-level template), which a
``testfiles/marmot/RestartTest/test.inp`` alone could not enforce without duplicating it into two
near-identical input files anyway.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile


def _materialAndMesh(linsolver: str) -> str:
    # "pardiso" is the shipped NIST default (edelweissfe/solvers/nonlinearimplicitstatic.py) and
    # needs no explicit option; only deviate from it for "blockamg".
    linsolverOption = "" if linsolver == "pardiso" else f"linsolver={linsolver}\n"
    return f"""
*material, name=VonMises, id=myMaterial
210000, 0.3, 550, 1000, 200, 1400

*section, name=section1, material=myMaterial, type=plane, thickness=1
all

*job, name=restartTestJob, domain=2d
*solver, solver=NIST, name=theSolver
{linsolverOption}
*modelGenerator, generator=planeRectQuad, name=gen
l=10
h=10
nX=2
nY=2
elType=CPE4
"""


_STEP_1 = """
*step, solver=theSolver
maxInc=1e0, minInc=1e-2, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=bottom, nSet=gen_leftBottom, field=displacement, 2=0
"""


def _step2(maxNumInc) -> str:
    return f"""
*step, solver=theSolver
maxInc=5e-2, minInc=1e-4, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>dirichlet, name=right, nSet=gen_right, field=displacement, 2=0.2
"""


def _runInputFile(path: Path):
    inputfile = parseInputFile(str(path))
    model, fieldOutputController = finiteElementSimulation(inputfile, verbose=False, suppressPlots=True)
    return model


def _mostRecentCheckpoint(tmp_path: Path) -> Path:
    checkpoints = list(tmp_path.glob("ckpt_*.h5"))
    assert checkpoints, "the truncated run did not write any restart checkpoint"

    def checkpointTime(path):
        with h5py.File(path, "r") as f:
            return f.attrs["time"]

    return max(checkpoints, key=checkpointTime)


@pytest.mark.parametrize("linsolver", ["pardiso", "blockamg"])
def test_restart_resume_matches_uninterrupted_reference(tmp_path, linsolver):
    materialAndMesh = _materialAndMesh(linsolver)

    fullPath = tmp_path / "full.inp"
    fullPath.write_text(materialAndMesh + _STEP_1 + _step2(maxNumInc=10000))
    referenceModel = _runInputFile(fullPath)
    referenceU = referenceModel.nodeFields["displacement"]["U"].copy()

    # Absolute checkpoint base name/path: the output manager and the driver's *restart, readFrom=
    # both resolve their file name relative to the process's current working directory, not this
    # test's tmp_path, so a bare "ckpt" would scatter checkpoint files outside tmp_path.
    checkpointBaseName = tmp_path / "ckpt"

    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        materialAndMesh + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n"
        + _STEP_1
        + _step2(maxNumInc=3)  # deliberately too low: truncates the job before step 2 finishes
    )
    _runInputFile(truncatedPath)

    checkpoint = _mostRecentCheckpoint(tmp_path)

    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(materialAndMesh + f"\n*restart, readFrom={checkpoint}\n" + _STEP_1 + _step2(maxNumInc=10000))
    resumedModel = _runInputFile(resumePath)
    resumedU = resumedModel.nodeFields["displacement"]["U"]

    np.testing.assert_allclose(resumedU, referenceU, atol=1e-10)


def test_restart_ensight_output_continues_across_resume(tmp_path):
    """Found via a real end-to-end validation run (AnchorPryOut): a resumed run's Ensight output
    used to restart its transient sequence numbering from zero -- since a fresh OutputManager
    instance has no way to know what a previous, now-dead process already wrote -- silently
    orphaning the pre-resume portion (the .case file only ever describes what *this* process
    wrote) and misordering the on-disk frame files (the resumed run's frame 0 landing chronologically
    after the truncated run's later frames). This pins the fix: the .case file's declared step
    list, after resuming, must cover the *entire* run (pre- and post-checkpoint), not just the
    resumed portion, and must be in correct chronological order.
    """
    materialAndMesh = _materialAndMesh("pardiso")
    ensightName = tmp_path / "esTest"
    ensightBlock = (
        "\n*fieldOutput\n"
        ">>perNode, elSet=all, field=displacement, result=U, name=displacement\n"
        f"\n*output, type=ensight, name={ensightName}\n"
        ">>perNode, fieldOutput=displacement\n"
        ">>configuration, overwrite=yes\n"
    )

    fullPath = tmp_path / "full.inp"
    fullPath.write_text(materialAndMesh + ensightBlock + _STEP_1 + _step2(maxNumInc=10000))
    _runInputFile(fullPath)
    fullStepCount = _ensightStepCount(ensightName)

    checkpointBaseName = tmp_path / "ckpt"
    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        materialAndMesh + ensightBlock + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n" + _STEP_1 + _step2(maxNumInc=3)
    )
    _runInputFile(truncatedPath)
    truncatedStepCount = _ensightStepCount(ensightName)
    assert 0 < truncatedStepCount < fullStepCount, "the truncated run should stop partway through"

    checkpoint = _mostRecentCheckpoint(tmp_path)
    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(
        materialAndMesh + ensightBlock + f"\n*restart, readFrom={checkpoint}\n" + _STEP_1 + _step2(maxNumInc=10000)
    )
    _runInputFile(resumePath)
    resumedStepCount, resumedTimeValues = _ensightStepCountAndTimes(ensightName)

    assert resumedStepCount == fullStepCount, "resuming must cover the whole run, not just the resumed portion"
    assert list(resumedTimeValues) == sorted(resumedTimeValues), "time values must stay in chronological order"

    # every declared frame file must actually exist, in the range the .case file promises
    for i in range(resumedStepCount):
        assert (ensightName.parent / f"{ensightName.name}" / f"displacement.var_{i:04d}").exists()

    # the GEOMETRY section must still reference the model: since this mesh never changes, the
    # resumed run's own writeOutput never calls writeGeometryTrendChunk (mesh_changed stays False
    # the whole time), so geometryTrends -- which the "model:" line is written from -- would stay
    # empty unless it was restored too, even though timeAndFileSets/the sequence itself is correct.
    caseText = (ensightName.parent / f"{ensightName.name}.case").read_text()
    assert "model:" in caseText, "GEOMETRY section lost its model: reference after resume"


def _ensightStepCount(ensightName: Path) -> int:
    return _ensightStepCountAndTimes(ensightName)[0]


def _ensightStepCountAndTimes(ensightName: Path) -> tuple[int, list]:
    """The .case file's TIME block declares one sub-block per time/file set (geometry, which only
    grows when the mesh changes, and variables, which grow every write) -- parse per-set and
    return the variable set's (always >= geometry's, and the one file numbering/comparisons here
    actually care about), not just whichever "number of steps:" line comes first."""

    caseFile = ensightName.parent / f"{ensightName.name}.case"
    lines = caseFile.read_text().splitlines()

    setsByNumber = {}
    currentSetNumber = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("time set:"):
            currentSetNumber = int(stripped.split(":")[1].strip().split()[0])
            setsByNumber[currentSetNumber] = {"numberOfSteps": None, "timeValues": []}
        elif stripped.startswith("number of steps:"):
            setsByNumber[currentSetNumber]["numberOfSteps"] = int(stripped.split(":")[1].strip())
        elif stripped.startswith("time values:"):
            setsByNumber[currentSetNumber]["timeValues"].append(float(stripped.split(":")[1].strip()))
        elif stripped.startswith(("filename start number:", "filename increment:")):
            pass
        elif stripped.startswith(("GEOMETRY", "VARIABLE", "FORMAT")):
            currentSetNumber = None
        elif currentSetNumber is not None and stripped:
            setsByNumber[currentSetNumber]["timeValues"].append(float(stripped))

    variableSet = max(setsByNumber.values(), key=lambda s: s["numberOfSteps"])
    return variableSet["numberOfSteps"], variableSet["timeValues"]


_AMR_MATERIAL_AND_MESH = """
*material, name=linearelastic, id=mat
30000, 0.15

*section, name=sec, material=mat, type=solid
all

*modelGenerator, generator=boxGen, name=gen
nX      =1
nY      =1
nZ      =1
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=fieldOutput, fieldOutput=stressForAMR, operator='>', threshold=300.0
maxLevel=1

*job, name=amrRestartTestJob, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=U
>>perElement, elSet=all, result=stress, quadraturePoint=0:27, name=stressForAMR, f(x)='np.abs(x)'
"""


def _amrStep(maxNumInc) -> str:
    # A single 1x1x1 C3D20 block, pushed on its right face against a fixed left face -- stress
    # crosses the marker's threshold partway through, refining 1 -> 8 active elements mid-step (not
    # at model setup, so this actually exercises replay -- unlike an initialOnly marker, which would
    # refine identically on every fresh rebuild regardless of whether restart's replay mechanism
    # works at all; see tests/test_hadaptivity_restart.py for the initialOnly-marker, lower-level
    # round-trip instead).
    return f"""
*step, solver=theSolver
maxInc=0.02, minInc=1e-6, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLeft, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=fixBottomLeftFront, nSet=gen_bottomLeftFront, field=displacement, 2=0.0, 3=0.0
>>dirichlet, name=fixBottomLeftBack, nSet=gen_bottomLeftBack, field=displacement, 2=0.0
>>dirichlet, name=pushRight, nSet=gen_right, field=displacement, 1=-0.1
"""


def test_restart_resume_matches_uninterrupted_reference_with_amr(tmp_path):
    fullPath = tmp_path / "full.inp"
    fullPath.write_text(_AMR_MATERIAL_AND_MESH + _amrStep(maxNumInc=10000))
    referenceModel = _runInputFile(fullPath)
    referenceU = referenceModel.nodeFields["displacement"]["U"].copy()
    assert len(referenceModel.elements) > 1, "the stress marker never triggered a refinement"

    checkpointBaseName = tmp_path / "ckpt"

    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        _AMR_MATERIAL_AND_MESH + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n"
        + _amrStep(maxNumInc=5)  # deliberately too low: truncates before the step finishes
    )
    truncatedModel = _runInputFile(truncatedPath)
    # the truncation point must be past the refinement, or this test would not exercise replay at
    # all -- resuming an unrefined checkpoint doesn't touch the new code paths this is meant to pin.
    assert len(truncatedModel.elements) > 1, "truncated too early: refinement had not happened yet"
    assert truncatedModel.time < 1.0, "the step already finished within maxNumInc -- not a real truncation"

    checkpoint = _mostRecentCheckpoint(tmp_path)

    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(_AMR_MATERIAL_AND_MESH + f"\n*restart, readFrom={checkpoint}\n" + _amrStep(maxNumInc=10000))
    resumedModel = _runInputFile(resumePath)
    resumedU = resumedModel.nodeFields["displacement"]["U"]

    assert set(resumedModel.elements.keys()) == set(referenceModel.elements.keys())
    assert set(resumedModel.nodes.keys()) == set(referenceModel.nodes.keys())
    np.testing.assert_allclose(resumedU, referenceU, atol=1e-10)


_AMR_PLASTIC_MATERIAL_AND_MESH = """
*material, name=VonMises, id=mat
210000, 0.3, 100, 1000, 200, 1400

*section, name=sec, material=mat, type=solid
all

*modelGenerator, generator=boxGen, name=gen
nX      =1
nY      =1
nZ      =1
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=fieldOutput, fieldOutput=stressForAMR, operator=\'>\', threshold=100.0
maxLevel=1

*job, name=amrPlasticRestartTestJob, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=U
>>perElement, elSet=all, result=stress, quadraturePoint=0:27, name=stressForAMR, f(x)=\'np.abs(x)\'
"""


def _amrPlasticStep(maxNumInc) -> str:
    return f"""
*step, solver=theSolver
maxInc=0.02, minInc=1e-6, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLeft, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=fixBottomLeftFront, nSet=gen_bottomLeftFront, field=displacement, 2=0.0, 3=0.0
>>dirichlet, name=fixBottomLeftBack, nSet=gen_bottomLeftBack, field=displacement, 2=0.0
>>dirichlet, name=pushRight, nSet=gen_right, field=displacement, 1=-0.2
"""


def test_restart_resume_matches_uninterrupted_reference_with_amr_and_plastic_state(tmp_path):
    """AMR + a history-dependent material (VonMises): refined child elements must retain their
    accumulated plastic state across a restart. Regresses the bug where AMR-replayed children were
    restored virgin -- their element numbers are not reproducible across replay (facet elements
    claim labels between refinements), so FEModel.readRestart's number-keyed state restore missed
    them -- causing return-mapping failures / a diverged solution deep in a run. The plain AMR
    restart test uses linearelastic and so has no material history to lose; this one does."""
    fullPath = tmp_path / "full.inp"
    fullPath.write_text(_AMR_PLASTIC_MATERIAL_AND_MESH + _amrPlasticStep(maxNumInc=10000))
    referenceModel = _runInputFile(fullPath)
    referenceU = referenceModel.nodeFields["displacement"]["U"].copy()
    assert len(referenceModel.elements) > 1, "the stress marker never triggered a refinement"

    checkpointBaseName = tmp_path / "ckpt"
    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        _AMR_PLASTIC_MATERIAL_AND_MESH + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n" + _amrPlasticStep(maxNumInc=6)
    )
    truncatedModel = _runInputFile(truncatedPath)
    assert len(truncatedModel.elements) > 1, "truncated too early: refinement had not happened yet"
    assert truncatedModel.time < 1.0, "the step already finished -- not a real truncation"

    checkpoint = _mostRecentCheckpoint(tmp_path)
    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(
        _AMR_PLASTIC_MATERIAL_AND_MESH + f"\n*restart, readFrom={checkpoint}\n" + _amrPlasticStep(maxNumInc=10000)
    )
    resumedModel = _runInputFile(resumePath)
    resumedU = resumedModel.nodeFields["displacement"]["U"]

    assert set(resumedModel.elements.keys()) == set(referenceModel.elements.keys())
    np.testing.assert_allclose(resumedU, referenceU, atol=1e-10)


_AMR_STATE_UNIT_INP = """
*material, name=VonMises, id=mat
210000, 0.3, 100, 1000, 200, 1400

*section, name=sec, material=mat, type=solid
all

*modelGenerator, generator=boxGen, name=gen
nX      =2
nY      =2
nZ      =2
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=gen_top, initialOnly=True
maxLevel=1

*job, name=amrStateUnitJob, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=U
"""


def _buildDirect(tmp_path, name):
    from edelweissfe.config.phenomena import domainMapping
    from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel

    p = tmp_path / name
    p.write_text(_AMR_STATE_UNIT_INP)
    inputfile = parseInputFile(str(p))
    journal = Journal(verbose=False)
    job = inputfile["job"][0]
    model = FEModel(domainMapping[job["domain"]])
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(job.get("startTime", 0.0))
    for nf in model.nodeFields.values():
        nf.createFieldValueEntry("U")
        nf.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])
    return model


def test_refined_child_state_survives_restart(tmp_path):
    """A refined child's material history must come back on resume.

    This is the original bug in its simplest form: children were restored virgin because their
    element numbers were not reproducible. It is now checked directly -- replay the recorded
    history, confirm the mesh is reproduced, then restore state by number exactly as
    FEModel.readRestart does -- rather than by asserting on the hotfix's bookkeeping.

    Replaces two tests that pinned the retired mechanism: one required getRestartData to checkpoint
    child state by octree eid, the other required the modifier to publish restoredElementLabels so
    FEModel could skip those elements. Both mechanisms are gone; what they protected is what this
    asserts.
    """

    modelA = _buildDirect(tmp_path, "a.inp")
    assert modelA.updateTopology(step=None, timeStep=0.0), "initialOnly marker should refine"
    amrA = modelA.modelModifiers["amr"]

    # give the refined children a non-trivial, distinguishable history
    expected = {}
    for offset, element in enumerate(amrA._eidToEl.values()):
        stateVars = element.getStateVars()
        if stateVars.size:
            stateVars[:] = 0.125 + offset
            expected[element.elNumber] = np.array(stateVars, copy=True)
    assert expected, "test needs a stateful element (VonMises should have per-QP plastic state)"

    modelB = _buildDirect(tmp_path, "b.inp")
    modelB.replayTopologyHistory(modelA.topologyHistory)
    assert modelB.topologyFingerprint() == modelA.topologyFingerprint(), "replay must reproduce the mesh"

    # restore exactly as FEModel.readRestart does: by element number, no skip set, nothing swallowed
    for elNumber, stateVars in expected.items():
        modelB.elements[elNumber].setStateVars(stateVars)
    for elNumber, stateVars in expected.items():
        np.testing.assert_allclose(modelB.elements[elNumber].getStateVars(), stateVars)


# A tie interleaves facet minting with AMR's own: every refinement of the tied surface makes the tie
# regenerate that surface's facets, so the two claim element numbers alternately. That interleaving
# is what a batched replay does not reproduce, and it is invisible in the plain AMR test above --
# which asserts matching element numbers and passes, because nothing else mints there.
_AMR_TIE_MATERIAL_AND_MESH = """
*material, name=linearelastic, id=mat
30000, 0.15

*section, name=sec, material=mat, type=solid
lower_all
upper_all

*modelGenerator, generator=boxGen, name=lower
nX      =1
nY      =1
nZ      =1
x0      =0
y0      =0
z0      =0
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelGenerator, generator=boxGen, name=upper
nX      =1
nY      =1
nZ      =1
x0      =0
y0      =0
z0      =1
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelGenerator, generator=surfaceElementGenerator, name=genMaster
surface = lower_front
name    = masterSurf

*modelGenerator, generator=surfaceElementGenerator, name=genSlave
surface = upper_back
name    = slaveSurf

*constraint, name=tie, type=tie
slaveSurface  = slaveSurf_facets
masterSurface = masterSurf_facets

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=fieldOutput, fieldOutput=stressForAMR, operator='>', threshold=300.0
refineElSet=lower_all
maxLevel=2

*job, name=amrTieRestartTestJob, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=U
>>perElement, elSet=lower_all, result=stress, quadraturePoint=0:27, name=stressForAMR, f(x)='np.abs(x)'
"""


def _amrTieStep(maxNumInc) -> str:
    return f"""
*step, solver=theSolver
maxInc=0.02, minInc=1e-6, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixFar, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pushFar, nSet=upper_front, field=displacement, 3=-0.05
"""


def test_restart_with_amr_and_a_tie_reproduces_the_topology_exactly(tmp_path):
    """The invariant the whole topology pipeline exists to establish: a resumed run rebuilds the
    same mesh, with the same element numbers, as the run it resumed.

    Uses topologyFingerprint, which covers element numbers, types, connectivity and node
    coordinates -- and deliberately not solution state, so a failure here means the *mesh*
    diverged.
    """

    fullPath = tmp_path / "full.inp"
    fullPath.write_text(_AMR_TIE_MATERIAL_AND_MESH + _amrTieStep(maxNumInc=10000))
    referenceModel = _runInputFile(fullPath)
    # Guard against a vacuous pass: the divergence this pins comes from AMR and the tie's facet
    # regeneration claiming element numbers ALTERNATELY across several refinement occasions, and
    # from the replay materialising level-wise rather than occasion-wise. One occasion at one level
    # exercises neither.
    occasions = referenceModel.modelModifiers["amr"]._committedOccasionEids
    levels = {
        referenceModel.modelModifiers["amr"]._mesh.elements[eid]["level"]
        for eid in referenceModel.modelModifiers["amr"]._eidToEl
    }
    assert len(occasions) >= 2, "only {:} refinement occasion(s): the interleaving is not exercised".format(
        len(occasions)
    )
    assert levels >= {1, 2}, "only levels {:}: the level-wise replay batching is not exercised".format(sorted(levels))
    assert len(referenceModel.elements) > 4, "the stress marker never triggered a refinement"

    checkpointBaseName = tmp_path / "ckpt"
    truncPath = tmp_path / "trunc.inp"
    truncPath.write_text(
        _AMR_TIE_MATERIAL_AND_MESH + "\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n" + _amrTieStep(maxNumInc=5)
    )
    truncatedModel = _runInputFile(truncPath)
    assert len(truncatedModel.elements) > 4, "truncated too early: refinement had not happened yet"
    assert truncatedModel.time < 1.0, "the step already finished within maxNumInc -- not a truncation"

    checkpoint = _mostRecentCheckpoint(tmp_path)
    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(
        _AMR_TIE_MATERIAL_AND_MESH + f"\n*restart, readFrom={checkpoint}\n" + _amrTieStep(maxNumInc=10000)
    )
    resumedModel = _runInputFile(resumePath)

    assert (
        resumedModel.topologyFingerprint() == referenceModel.topologyFingerprint()
    ), "resumed topology differs from the uninterrupted run: " "{:} elements vs {:}, numbers {:} vs {:}".format(
        len(resumedModel.elements),
        len(referenceModel.elements),
        sorted(resumedModel.elements)[:12],
        sorted(referenceModel.elements)[:12],
    )
