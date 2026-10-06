#!/usr/bin/env python3
"""Check how each test case of ``testfiles/mpi`` held its model when it ran over several MPI processes.

A test case either runs **distributed** -- each process creates only the elements it computes -- or
with the **whole model** in every process, by the rule of
:func:`edelweissfe.domaindecomposition.distributedelements.reasonsForTheWholeModel`. Both give the
same result, so comparing results cannot tell which one ran; this script runs every test case (in a
copy) and checks, from the model each process ends with, that it ran in the mode expected below:

* distributed: some process created fewer elements than the mesh has, and every element of the
  mesh was computed by exactly one process;
* whole model: every process created every element.

The test cases expected to **migrate** -- to rebalance a distributed model, moving elements between
processes -- are checked further: some element must have been received by another process than the
one computing it before (counted by the distribution over the run, so that it also counts the
children of a refinement, which no first partition knew), and no process may still hold an object of
an element it dropped (the element objects alive in the process, counted by the garbage collector,
are exactly those of the model).

Run it under the MPI launcher, from anywhere::

    mpirun -n 2 python testfiles/mpi/check_element_distribution.py

It prints one line per test case and exits with 1 if any test case ran in an unexpected mode.
"""

import contextlib
import gc
import io
import os
import shutil
import sys
import tempfile

from edelweissfe.domaindecomposition.mpienvironment import worldCommunicator
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

#: The test cases expected to run distributed; every other test case is expected to hold the whole
#: model (it has contact, a tie, or contact facets).
EXPECTED_DISTRIBUTED = {
    "edelweiss-only/NED",
    "edelweiss-only/NEDContact",
    "edelweiss-only/NEDRestart1Write",
    "edelweiss-only/NEDRestart2Resume",
    "edelweiss-only/NEDSurfaceContact",
    "edelweiss-only/NEDSurfaceToDiscreteRigidBodyContact",
    "edelweiss-only/TieNED",
    "marmot/GCDPNEDExplicit",
    "marmot/GCDPNEDExplicitAMR",
    "marmot/GCDPNEDExplicitHyperbolic",
    "marmot/GCDPNEDExplicitHyperbolicAMR",
    "marmot/NED",
    "marmot/NEDInitialStressPressure",
    "marmot/NEDLiveAMR",
    "marmot/NEDLiveAMRRebalanceDistributed",
    "marmot/NEDLiveAMRRecoveryErrorDistributed",
    "marmot/NEDLiveAMRRestartDistributed1Write",
    "marmot/NEDLiveAMRRestartDistributed2Resume",
    "marmot/NEDLiveAMRStepEndsOffInterval",
    "marmot/NEDParallel",
    "marmot/NEDRestartDistributed1Write",
    "marmot/NEDRestartDistributed2Resume",
    "marmot/NEDRebalanceDistributed",
}

#: The test cases expected to move elements between processes (a distributed model rebalanced).
EXPECTED_MIGRATING = {
    "marmot/NEDLiveAMRRebalanceDistributed",
    "marmot/NEDLiveAMRRecoveryErrorDistributed",
    "marmot/NEDLiveAMRRestartDistributed1Write",
    "marmot/NEDRebalanceDistributed",
}


def modeOf(model, communicator) -> str:
    """The mode a model was held in, from the elements every process created."""

    created = communicator.allgather(set(model.elements))
    computed = communicator.allgather(
        [element.elNumber for element in model.elementDistribution.elementsReportedHere(model.elements.values())]
    )
    meshElements = set(model.mesh.elements)
    if all(elements == meshElements for elements in created):
        return "whole model"
    everyElementComputedOnce = sorted(number for numbers in computed for number in numbers) == sorted(meshElements)
    return "distributed" if everyElementComputedOnce else "inconsistent"


def migrationOf(model, communicator) -> str:
    """How many elements of a distributed model moved between the processes over the run, and whether
    every process holds exactly the element objects of its model."""

    moved = sum(communicator.allgather(model.elementDistribution.nElementsReceived))

    gc.collect()
    elementClasses = {type(element) for element in model.elements.values()}
    alive = sum(type(candidate) in elementClasses for candidate in gc.get_objects())
    heldExactly = all(communicator.allgather(alive == len(model.elements)))
    return "{:} moved, {:}".format(moved, "no dropped element alive" if heldExactly else "DROPPED ELEMENTS ALIVE")


def main() -> int:
    communicator = worldCommunicator()
    if communicator is None:
        print("run this under an MPI launcher, with more than one process")
        return 1
    rank = communicator.Get_rank()

    here = os.path.dirname(os.path.abspath(__file__))
    cases = sorted(
        "{:}/{:}".format(suite, case)
        for suite in ("edelweiss-only", "marmot")
        for case in os.listdir(os.path.join(here, suite))
        if os.path.isfile(os.path.join(here, suite, case, "test.inp"))
    )

    workDirectory = communicator.bcast(tempfile.mkdtemp() if rank == 0 else None, root=0)
    if rank == 0:
        for suite in ("edelweiss-only", "marmot"):
            shutil.copytree(os.path.join(here, suite), os.path.join(workDirectory, suite))
    communicator.Barrier()

    failures = 0
    for case in cases:
        os.chdir(os.path.join(workDirectory, case))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
        except NotImplementedError as exception:
            # e.g. a private Marmot material missing from this build
            if rank == 0:
                print("{:<50} SKIPPED ({:})".format(case, exception))
            continue
        mode = modeOf(model, communicator)
        expected = "distributed" if case in EXPECTED_DISTRIBUTED else "whole model"
        failed = mode != expected
        migration = ""
        if case in EXPECTED_MIGRATING and not failed:
            migration = migrationOf(model, communicator)
            failed = migration.startswith("0 moved") or "ALIVE" in migration
        if rank == 0:
            print(
                "{:<50} {:<12} {:}{:}".format(
                    case,
                    mode,
                    "EXPECTED " + expected if mode != expected else "FAILED" if failed else "OK",
                    " ({:})".format(migration) if migration else "",
                )
            )
        failures += failed

    communicator.Barrier()
    if rank == 0:
        shutil.rmtree(workDirectory)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
