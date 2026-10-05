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
"""Whether this process is one of several MPI processes, and the communicator connecting them.

``mpi4py`` is imported only when the process was started by an MPI launcher, recognised by the
environment variables the common launchers set (Open MPI's ``mpirun``, the Hydra launcher of MPICH
and Intel MPI, MVAPICH, and Slurm's ``srun`` with PMI or PMIx). A serial run never imports it, so an
installation without MPI is unaffected. ``EDELWEISSFE_MPI=0`` forces a serial run under a launcher,
``EDELWEISSFE_MPI=1`` forces the import without one.

When there is more than one process, an uncaught exception in any of them aborts all of them: the
others would otherwise wait forever in their next collective operation for a process that is gone.
"""

import os
import sys
from functools import cache

#: Environment variables at least one of which every supported MPI launcher sets in the processes
#: it starts.
_LAUNCHER_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_RANK",
    "MPI_LOCALNRANKS",
    "MV2_COMM_WORLD_SIZE",
)


def _startedByMPILauncher() -> bool:
    """Whether this process was started by an MPI launcher, or ``EDELWEISSFE_MPI`` says so."""

    forced = os.environ.get("EDELWEISSFE_MPI")
    if forced is not None:
        return forced.strip() not in ("0", "", "false", "False", "no")
    return any(variable in os.environ for variable in _LAUNCHER_VARIABLES)


def _abortAllProcessesOnUncaughtException(communicator):
    """Replace the exception hook so that an uncaught exception aborts every process.

    Parameters
    ----------
    communicator
        The communicator whose processes are aborted.
    """

    previousHook = sys.excepthook

    def abortingHook(exceptionType, exception, traceback):
        previousHook(exceptionType, exception, traceback)
        sys.stderr.write(
            "EdelweissFE: uncaught exception in MPI process {:} of {:}; aborting all processes.\n".format(
                communicator.Get_rank(), communicator.Get_size()
            )
        )
        sys.stderr.flush()
        communicator.Abort(1)

    sys.excepthook = abortingHook


@cache
def worldCommunicator():
    """The communicator of all processes of this job, or None for a serial run.

    Returns
    -------
    mpi4py.MPI.Comm | None
        ``MPI.COMM_WORLD`` if this process is one of more than one; None otherwise.
    """

    if not _startedByMPILauncher():
        return None

    from mpi4py import MPI

    communicator = MPI.COMM_WORLD
    if communicator.Get_size() == 1:
        return None

    _abortAllProcessesOnUncaughtException(communicator)
    return communicator


def numberOfProcesses() -> int:
    """The number of processes of this job; 1 for a serial run.

    Returns
    -------
    int
        The number of processes.
    """

    communicator = worldCommunicator()
    return 1 if communicator is None else communicator.Get_size()


def processRank() -> int:
    """The rank of this process; 0 for a serial run.

    Returns
    -------
    int
        The rank.
    """

    communicator = worldCommunicator()
    return 0 if communicator is None else communicator.Get_rank()


def isRootProcess() -> bool:
    """Whether this process is the one that writes output and reports progress.

    Returns
    -------
    bool
        True for rank 0, and for a serial run.
    """

    return processRank() == 0
