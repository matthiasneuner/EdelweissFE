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
"""The communicator of the processes sharing a model, which refuses to communicate where a process
may be missing.

A step that may fail in one process alone runs in a context in which the processes agree on its
outcome afterwards (:meth:`Communicator.allRanksFailTogether`). Inside such a
context no process may communicate: a process that failed before reaching a collective operation
skips it, and the others wait in it for that process forever, or meet it in a different collective
operation. What reads more than one process is therefore split into the part each process does
alone -- inside the context -- and the communication, after the agreement.

This class makes that rule a mechanism instead of a convention: it forwards the operations of an
``mpi4py`` communicator that this package uses, and every collective and point-to-point operation
raises while a context without communication is open (:meth:`Communicator.withoutCommunication`).
Since every process raises at the same operation -- or the process that alone tried to communicate
raises there --, the agreement that closes the context reports it as a failure on all ranks.
"""

from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.mpienvironment import StepFailedOnAllRanks
from edelweissfe.utils.exceptions import ConditionalStop, CutbackRequest

#: How :meth:`Communicator.allRanksFailTogether` ranks what went wrong: the most severe outcome of any
#: process is the one every process raises.
_NO_FAILURE, _CUTBACK, _CONDITIONAL_STOP, _FAILURE = 0, 1, 2, 3


class Communicator:
    """The communicator of the processes sharing a model; see the module documentation. Besides the
    operations of ``mpi4py``, it lets the processes agree: on a failure in any of them
    (:meth:`allRanksFailTogether`), on sums and extremes formed the same in every process, and that
    they hold the same value.

    The methods carry the names of the ``mpi4py`` communicator they forward to, so that this is the
    communicator wherever one is expected.

    Parameters
    ----------
    mpiCommunicator
        The ``mpi4py`` communicator, e.g. ``MPI.COMM_WORLD``.
    """

    def __init__(self, mpiCommunicator):
        #: The ``mpi4py`` communicator every operation is forwarded to.
        self.mpiCommunicator = mpiCommunicator
        #: What is done in the open context without communication, or None if there is none.
        self._operationWithoutCommunication = None

    @contextmanager
    def withoutCommunication(self, operation: str):
        """A context in which every communication raises.

        Parameters
        ----------
        operation
            What is done in the context, for the message.

        Yields
        ------
        None
        """

        enclosing = self._operationWithoutCommunication
        self._operationWithoutCommunication = operation
        try:
            yield
        finally:
            self._operationWithoutCommunication = enclosing

    def _requireCommunicationAllowed(self, communication: str):
        """Raise if a context without communication is open.

        Parameters
        ----------
        communication
            The operation asked for, for the message.

        Raises
        ------
        RuntimeError
            Inside :meth:`withoutCommunication`.
        """

        if self._operationWithoutCommunication is not None:
            raise RuntimeError(
                "{:} was called while {:} -- a step in which no process may communicate, since a process "
                "failing before it would skip the communication and leave the others waiting. Communicate "
                "after the processes agreed on the outcome.".format(
                    communication, self._operationWithoutCommunication.lower()
                )
            )

    # --- Who --------------------------------------------------------------------------------------

    def Get_rank(self) -> int:
        """The rank of this process.

        Returns
        -------
        int
            The rank.
        """

        return self.mpiCommunicator.Get_rank()

    def Get_size(self) -> int:
        """The number of processes.

        Returns
        -------
        int
            The number of processes.
        """

        return self.mpiCommunicator.Get_size()

    def Abort(self, errorcode: int = 1):
        """Abort every process; allowed anywhere.

        Parameters
        ----------
        errorcode
            The exit code.
        """

        self.mpiCommunicator.Abort(errorcode)

    # --- Collective operations ----------------------------------------------------------------------

    def Barrier(self):
        """Wait for every process; see ``mpi4py``. Collective."""

        self._requireCommunicationAllowed("Barrier")
        self.mpiCommunicator.Barrier()

    def bcast(self, obj, root: int = 0):
        """Broadcast a Python object from ``root``; see ``mpi4py``. Collective.

        Parameters
        ----------
        obj
            The object, read on ``root``.
        root
            The rank broadcasting.

        Returns
        -------
        object
            The object of ``root``.
        """

        self._requireCommunicationAllowed("bcast")
        return self.mpiCommunicator.bcast(obj, root=root)

    def Bcast(self, buffer, root: int = 0):
        """Broadcast a buffer from ``root``, in place; see ``mpi4py``. Collective.

        Parameters
        ----------
        buffer
            The buffer.
        root
            The rank broadcasting.
        """

        self._requireCommunicationAllowed("Bcast")
        self.mpiCommunicator.Bcast(buffer, root=root)

    def gather(self, obj, root: int = 0):
        """Gather a Python object of every process to ``root``; see ``mpi4py``. Collective.

        Parameters
        ----------
        obj
            This process' object.
        root
            The rank gathering.

        Returns
        -------
        list | None
            The objects, by rank, on ``root``; None elsewhere.
        """

        self._requireCommunicationAllowed("gather")
        return self.mpiCommunicator.gather(obj, root=root)

    def allgather(self, obj) -> list:
        """Gather a Python object of every process to every process; see ``mpi4py``. Collective.

        Parameters
        ----------
        obj
            This process' object.

        Returns
        -------
        list
            The objects, by rank.
        """

        self._requireCommunicationAllowed("allgather")
        return self.mpiCommunicator.allgather(obj)

    def Allgatherv(self, sendbuf, recvbuf):
        """Gather buffers of varying size of every process to every process; see ``mpi4py``.
        Collective.

        Parameters
        ----------
        sendbuf
            This process' buffer.
        recvbuf
            The receive buffer specification.
        """

        self._requireCommunicationAllowed("Allgatherv")
        self.mpiCommunicator.Allgatherv(sendbuf, recvbuf)

    def allreduce(self, obj, op=MPI.SUM):
        """Reduce a Python object over every process; see ``mpi4py``. Collective.

        Parameters
        ----------
        obj
            This process' object.
        op
            The reduction; a sum by default.

        Returns
        -------
        object
            The reduced object.
        """

        self._requireCommunicationAllowed("allreduce")
        return self.mpiCommunicator.allreduce(obj, op=op)

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        """Reduce a buffer over every process; see ``mpi4py``. Collective.

        Parameters
        ----------
        sendbuf
            This process' buffer, or ``MPI.IN_PLACE``.
        recvbuf
            The result buffer.
        op
            The reduction; a sum by default.
        """

        self._requireCommunicationAllowed("Allreduce")
        self.mpiCommunicator.Allreduce(sendbuf, recvbuf, op=op)

    def alltoall(self, objects: list) -> list:
        """Send a Python object to every process, and receive one from each; see ``mpi4py``.
        Collective.

        Parameters
        ----------
        objects
            The object for every process, by rank.

        Returns
        -------
        list
            The object of every process, by rank.
        """

        self._requireCommunicationAllowed("alltoall")
        return self.mpiCommunicator.alltoall(objects)

    # --- Point-to-point operations ------------------------------------------------------------------

    def Isend(self, buffer, dest: int, tag: int):
        """Start sending a buffer to a process; see ``mpi4py``.

        Parameters
        ----------
        buffer
            The buffer.
        dest
            The receiving rank.
        tag
            The message tag.

        Returns
        -------
        mpi4py.MPI.Request
            The request.
        """

        self._requireCommunicationAllowed("Isend")
        return self.mpiCommunicator.Isend(buffer, dest=dest, tag=tag)

    def Irecv(self, buffer, source: int, tag: int):
        """Start receiving a buffer from a process; see ``mpi4py``.

        Parameters
        ----------
        buffer
            The buffer.
        source
            The sending rank.
        tag
            The message tag.

        Returns
        -------
        mpi4py.MPI.Request
            The request.
        """

        self._requireCommunicationAllowed("Irecv")
        return self.mpiCommunicator.Irecv(buffer, source=source, tag=tag)

    # --- Agreeing among the processes ---------------------------------------------------------------

    def allreduceSum(self, values: list[float]) -> list[float]:
        """The sums of values over all processes, added in ascending rank order, so that they are the
        same bits in every process. Collective.

        Parameters
        ----------
        values
            This process' contributions.

        Returns
        -------
        list[float]
            The sum of every entry over all processes.
        """

        gathered = self.allgather(list(values))
        totals = list(gathered[0])
        for contributions in gathered[1:]:
            totals = [total + value for total, value in zip(totals, contributions)]
        return totals

    def allreduceMin(self, value: float) -> float:
        """The minimum of a value over all processes. Collective.

        Parameters
        ----------
        value
            This process' value.

        Returns
        -------
        float
            The minimum.
        """

        return self.allreduce(value, op=MPI.MIN)

    def allreduceAny(self, flag: bool) -> bool:
        """Whether a flag is set in any process. Collective.

        Parameters
        ----------
        flag
            This process' flag.

        Returns
        -------
        bool
            Whether any process set it.
        """

        return bool(self.allreduce(bool(flag), op=MPI.LOR))

    def requireSameOnAllRanks(self, value, description: str):
        """Refuse to continue unless every process holds the same value. Collective.

        Parameters
        ----------
        value
            This process' value; anything comparable for equality.
        description
            What the value says, for the message.

        Raises
        ------
        RuntimeError
            In every process, if two processes hold different values.
        """

        values = self.allgather(value)
        if any(other != values[0] for other in values[1:]):
            raise RuntimeError(
                "The processes disagree on {:} ({:}); they must compute the replicated parts of the model "
                "identically.".format(description, values)
            )

    @contextmanager
    def allRanksFailTogether(self, operation: str):
        """A context in which an exception raised in one process is raised in every process.
        Collective.

        A process raising alone would leave the others waiting for it in the next exchange forever.
        Every process reports what went wrong in it, and every process raises the most severe: a
        :class:`~edelweissfe.utils.exceptions.ConditionalStop` or a
        :class:`~edelweissfe.utils.exceptions.CutbackRequest` anywhere is raised as such everywhere
        -- a cutback with the smallest size any process requested -- so that every process takes the
        same path out of the step. Nothing inside the context may communicate: a process that
        raised would skip the communication, and leave the others waiting in it. That is enforced,
        not assumed -- this communicator raises at any communication inside the context
        (:meth:`withoutCommunication`),
        which then fails on all ranks like any other failure. A step that needs to communicate is
        split: each process does its own part in the context, and communicates after it (see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.writeIncrementOutput`).

        Parameters
        ----------
        operation
            What is done in the context, for the message.

        Yields
        ------
        None
        """

        failure = None
        try:
            with self.withoutCommunication(operation):
                yield
        except Exception as exception:
            failure = exception

        if failure is None:
            outcome = _NO_FAILURE
        elif isinstance(failure, CutbackRequest):
            outcome = _CUTBACK
        elif isinstance(failure, ConditionalStop):
            outcome = _CONDITIONAL_STOP
        else:
            outcome = _FAILURE

        status = np.array([outcome], dtype=np.int32)
        # Where the processes wait for the slowest one: timed on its own, it is the load imbalance.
        with performancetiming.timeit("subdomain wait"):
            self.Allreduce(MPI.IN_PLACE, status, op=MPI.MAX)

        if status[0] == _NO_FAILURE:
            return
        if status[0] == _CONDITIONAL_STOP:
            raise ConditionalStop() from failure

        reports = self.allgather(
            None
            if failure is None
            else (
                "{:}: {:}".format(type(failure).__name__, failure),
                failure.cutbackSize if outcome == _CUTBACK else None,
            )
        )
        message = "; ".join(
            "process {:}: {:}".format(rank, report[0]) for rank, report in enumerate(reports) if report is not None
        )
        if status[0] == _CUTBACK:
            cutbackSize = min(report[1] for report in reports if report is not None and report[1] is not None)
            raise CutbackRequest(message, cutbackSize) from failure
        raise StepFailedOnAllRanks("{:} failed in {:}".format(operation, message)) from failure
