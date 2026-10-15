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

Setting up the model cannot be split so: the processes communicate in the middle of it -- the
partition of the mesh is broadcast before the elements are made. Its agreement
(:meth:`Communicator.allRanksFailTogetherWhileCommunicating`) lets the processes communicate, and
every process first agrees, at each collective operation, that no process has failed so far. That
holds as long as a process that failed does not communicate any more on its way out of the context:
it then meets the others, waiting at their next operation, with its failure, at the agreement that
closes its context. So a process does not communicate inside that context while it handles an
exception -- in an ``except`` clause, a ``finally`` clause or an ``__exit__`` during unwinding --:
the communication raises instead, without communicating, and the agreement reports it as a failure.
A handler that recovers from an exception and then communicates counts as a failure there, too.

The ``mpi4py`` communicator itself is private to this class, so that every operation passes these
checks.
"""

import builtins
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

import numpy as np
from mpi4py import MPI

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.domaindecomposition.mpienvironment import (
    ErrorOnAllRanks,
    RaisedOnAllRanks,
    StepFailedOnAllRanks,
)
from edelweissfe.utils.exceptions import ConditionalStop, CutbackRequest, StepFailed

#: How :meth:`Communicator.allRanksFailTogether` ranks what went wrong: the most severe outcome of any
#: process is the one every process raises. A step failure is what a serial run ends the step with
#: (:class:`~edelweissfe.utils.exceptions.StepFailed`); an error is any other exception, which ends a
#: serial run with a non-zero exit status.
_NO_FAILURE, _CUTBACK, _CONDITIONAL_STOP, _STEP_FAILURE, _ERROR = 0, 1, 2, 3, 4


class Communicator:
    """The communicator of the processes sharing a model; see the module documentation. Besides the
    operations of ``mpi4py``, it lets the processes agree: on a failure in any of them
    (:meth:`allRanksFailTogether`), on sums and extremes formed the same in every process, and that
    they hold the same value.

    The methods carry the names of the ``mpi4py`` communicator they forward to, so that this is the
    communicator wherever one is expected; they take and return the ``mpi4py`` types (buffers, buffer
    specifications, reductions such as ``MPI.MAX``, ``MPI.Request``) unchanged.

    Parameters
    ----------
    mpiCommunicator
        The ``mpi4py`` communicator, e.g. ``MPI.COMM_WORLD``.
    """

    def __init__(self, mpiCommunicator):
        #: The ``mpi4py`` communicator every operation is forwarded to.
        self._mpiCommunicator = mpiCommunicator
        #: What is done in the open context without communication, or None if there is none.
        self._operationWithoutCommunication = None
        #: What is done in the open agreement in which the processes communicate, or None if there is
        #: none; see :meth:`allRanksFailTogetherWhileCommunicating`.
        self._agreementWhileCommunicating = None
        #: The outcome the processes agreed on inside that agreement, once this process learned of a
        #: failure there; None before.
        self._failureAgreedOn = None
        #: The exception being handled where that agreement was entered, if any: not one handled inside.
        self._exceptionHandledOutside = None

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

    def _requireCommunicationAllowed(self, communication: str, collective: bool = True):
        """Raise if a context without communication is open; inside an agreement in which the
        processes communicate (:meth:`allRanksFailTogetherWhileCommunicating`), first agree that no
        process has failed so far, and raise if one has.

        Parameters
        ----------
        communication
            The operation asked for, for the message.
        collective
            Whether every process of the communicator takes part in the operation.

        Raises
        ------
        RuntimeError
            Inside :meth:`withoutCommunication`; and, inside
            :meth:`allRanksFailTogetherWhileCommunicating`, for an operation not every process takes
            part in, or while this process handles an exception raised inside.
        _FailedInAnotherProcess
            Inside :meth:`allRanksFailTogetherWhileCommunicating`, if a process has failed.
        """

        if self._operationWithoutCommunication is not None:
            raise RuntimeError(
                "{:} was called while {:} -- a step in which no process may communicate, since a process "
                "failing before it would skip the communication and leave the others waiting. Communicate "
                "after the processes agreed on the outcome.".format(
                    communication, self._operationWithoutCommunication.lower()
                )
            )
        if self._agreementWhileCommunicating is None:
            return
        if self._failureAgreedOn is not None:
            # this process already learned of the failure, and leaves the context without communicating
            raise _FailedInAnotherProcess(self._failureAgreedOn)
        handled = sys.exception()
        if handled is not None and handled is not self._exceptionHandledOutside:
            # A process that failed communicates on its way out -- in an except or finally clause, an
            # __exit__ --, while the others are at their next operation: agreeing here, it would tell
            # them that nothing failed, and the operations would not match. It raises instead.
            raise RuntimeError(
                "{:} was called while {:}, during the handling of {:}: {:} -- a process communicates "
                "there only while no exception is being handled, since one that failed would meet the "
                "others in an operation they never reach.".format(
                    communication, self._agreementWhileCommunicating.lower(), type(handled).__name__, handled
                )
            )
        if not collective:
            raise RuntimeError(
                "{:} was called while {:} -- a step in which the processes agree on failures before every "
                "collective operation, which a point-to-point operation is not.".format(
                    communication, self._agreementWhileCommunicating.lower()
                )
            )
        outcome = self._agreedOutcome(_NO_FAILURE)
        if outcome != _NO_FAILURE:
            self._failureAgreedOn = outcome
            raise _FailedInAnotherProcess(outcome)

    # --- Who --------------------------------------------------------------------------------------

    def Get_rank(self) -> int:
        """The rank of this process.

        Returns
        -------
        int
            The rank.
        """

        return self._mpiCommunicator.Get_rank()

    def Get_size(self) -> int:
        """The number of processes.

        Returns
        -------
        int
            The number of processes.
        """

        return self._mpiCommunicator.Get_size()

    def Abort(self, errorcode: int = 1):
        """Abort every process; allowed anywhere.

        Parameters
        ----------
        errorcode
            The exit code.
        """

        self._mpiCommunicator.Abort(errorcode)

    # --- Collective operations ----------------------------------------------------------------------

    def Barrier(self):
        """Collective. Wait for every process; see ``mpi4py``."""

        self._requireCommunicationAllowed("Barrier")
        self._mpiCommunicator.Barrier()

    def bcast(self, obj, root: int = 0):
        """Collective. Broadcast a Python object from ``root``; see ``mpi4py``.

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
        return self._mpiCommunicator.bcast(obj, root=root)

    def Bcast(self, buffer, root: int = 0):
        """Collective. Broadcast a buffer from ``root``, in place; see ``mpi4py``.

        Parameters
        ----------
        buffer
            The buffer.
        root
            The rank broadcasting.
        """

        self._requireCommunicationAllowed("Bcast")
        self._mpiCommunicator.Bcast(buffer, root=root)

    def gather(self, obj, root: int = 0):
        """Collective. Gather a Python object of every process to ``root``; see ``mpi4py``.

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
        return self._mpiCommunicator.gather(obj, root=root)

    def allgather(self, obj) -> list:
        """Collective. Gather a Python object of every process to every process; see ``mpi4py``.

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
        return self._mpiCommunicator.allgather(obj)

    def Allgatherv(self, sendbuf, recvbuf):
        """Collective. Gather buffers of varying size of every process to every process; see ``mpi4py``.

        Parameters
        ----------
        sendbuf
            This process' buffer.
        recvbuf
            The receive buffer specification.
        """

        self._requireCommunicationAllowed("Allgatherv")
        self._mpiCommunicator.Allgatherv(sendbuf, recvbuf)

    def Gatherv(self, sendbuf, recvbuf, root: int = 0):
        """Collective. Gather buffers of varying size of every process to ``root``; see ``mpi4py``.

        Parameters
        ----------
        sendbuf
            This process' buffer.
        recvbuf
            The receive buffer specification on ``root``; None elsewhere.
        root
            The rank gathering.
        """

        self._requireCommunicationAllowed("Gatherv")
        self._mpiCommunicator.Gatherv(sendbuf, recvbuf, root=root)

    def allreduce(self, obj, op=MPI.SUM):
        """Collective. Reduce a Python object over every process; see ``mpi4py``.

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
        return self._mpiCommunicator.allreduce(obj, op=op)

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        """Collective. Reduce a buffer over every process; see ``mpi4py``.

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
        self._mpiCommunicator.Allreduce(sendbuf, recvbuf, op=op)

    def alltoall(self, objects: list) -> list:
        """Collective. Send a Python object to every process, and receive one from each; see ``mpi4py``.

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
        return self._mpiCommunicator.alltoall(objects)

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

        self._requireCommunicationAllowed("Isend", collective=False)
        return self._mpiCommunicator.Isend(buffer, dest=dest, tag=tag)

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

        self._requireCommunicationAllowed("Irecv", collective=False)
        return self._mpiCommunicator.Irecv(buffer, source=source, tag=tag)

    # --- Agreeing among the processes ---------------------------------------------------------------

    def allreduceSum(self, values: list[float]) -> list[float]:
        """Collective. The sums of values over all processes, added in ascending rank order, so that they
        are the same bits in every process.

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
        """Collective. The minimum of a value over all processes.

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
        """Collective. Whether a flag is set in any process.

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
        """Collective. Refuse to continue unless every process holds the same value.

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
        """Collective. A context in which an exception raised in one process is raised in every process.

        A process raising alone would leave the others waiting for it in the next exchange forever.
        Every process reports what went wrong in it, and every process raises the most severe: a
        :class:`~edelweissfe.utils.exceptions.ConditionalStop` or a
        :class:`~edelweissfe.utils.exceptions.CutbackRequest` anywhere is raised as such everywhere
        -- a cutback with the smallest size any process requested --, a
        :class:`~edelweissfe.utils.exceptions.StepFailed` as
        :class:`~edelweissfe.domaindecomposition.mpienvironment.StepFailedOnAllRanks`, and any other
        exception -- an error, which a serial run does not end as a failed step, but with a non-zero
        exit status -- as :class:`~edelweissfe.domaindecomposition.mpienvironment.ErrorOnAllRanks`;
        so that every process takes the same path out of the step, the path a serial run takes.

        Nothing inside the context may communicate: a process that raised would skip the
        communication, and leave the others waiting in it. That is enforced, not assumed -- this
        communicator raises at any communication inside the context (:meth:`withoutCommunication`),
        which then fails on all ranks like any other error. A step that needs to communicate is
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

        if self._agreementWhileCommunicating is not None:
            raise RuntimeError(
                "{:} cannot agree on failures inside another agreement ({:}).".format(
                    operation, self._agreementWhileCommunicating.lower()
                )
            )

        failure = None
        try:
            with self.withoutCommunication(operation):
                yield
        except Exception as exception:
            failure = exception

        outcome = _outcomeOf(failure)
        # Where the processes wait for the slowest one: timed on its own, it is the load imbalance.
        with performancetiming.timeit("waiting for all ranks"):
            # refused inside an enclosing context without communication, like any other operation
            self._requireCommunicationAllowed("Allreduce")
            agreed = self._agreedOutcome(outcome)
        self._raiseAgreedOutcome(operation, agreed, outcome, failure)

    @contextmanager
    def allRanksFailTogetherWhileCommunicating(self, operation: str):
        """Collective. A context in which an exception raised in one process is raised in every process, as
        in :meth:`allRanksFailTogether`, but in which the processes may communicate -- through operations
        every process takes part in.

        Before every such operation inside the context, the processes agree that none of them has
        failed so far: a process that failed is not at that operation but at the end of its context,
        where it agrees on its failure -- the same agreement. The others learn of the failure there,
        and leave the context without the operation, so that no process waits for another. An
        operation not every process takes part in (``Isend``, ``Irecv``) raises inside it. The
        agreement before each operation costs a reduction of one integer: meant for setting up the
        model, which communicates a few times, not for an increment.

        Parameters
        ----------
        operation
            What is done in the context, for the message.

        Yields
        ------
        None
        """

        if self._agreementWhileCommunicating is not None or self._operationWithoutCommunication is not None:
            raise RuntimeError("{:} cannot agree on failures inside another agreement.".format(operation))

        failure = None
        self._agreementWhileCommunicating = operation
        self._exceptionHandledOutside = sys.exception()
        try:
            yield
        except (Exception, _FailedInAnotherProcess) as exception:
            failure = exception
        finally:
            self._agreementWhileCommunicating = self._failureAgreedOn = self._exceptionHandledOutside = None

        if isinstance(failure, _FailedInAnotherProcess):
            # the agreement was reached before an operation, where this process learned of the failure
            agreed, outcome, failure = failure.agreedOutcome, _NO_FAILURE, None
        else:
            outcome = _outcomeOf(failure)
            agreed = self._agreedOutcome(outcome)
        self._raiseAgreedOutcome(operation, agreed, outcome, failure)

    def _agreedOutcome(self, outcome: int) -> int:
        """Collective. The most severe outcome of any process.

        Parameters
        ----------
        outcome
            This process' outcome.

        Returns
        -------
        int
            The most severe outcome.
        """

        status = np.array([outcome], dtype=np.int32)
        self._mpiCommunicator.Allreduce(MPI.IN_PLACE, status, op=MPI.MAX)
        return int(status[0])

    def _raiseAgreedOutcome(self, operation: str, agreed: int, outcome: int, failure: Exception | None):
        """Collective if a process failed. Raise the outcome the processes agreed on, in every process
        alike, with what went wrong in each of them.

        Parameters
        ----------
        operation
            What was done, for the message.
        agreed
            The most severe outcome of any process.
        outcome
            This process' outcome.
        failure
            The exception this process raised, or None.
        """

        if agreed == _NO_FAILURE:
            return
        if agreed == _CONDITIONAL_STOP:
            raise ConditionalStop() from failure

        reports = self.allgather(None if failure is None else _reportOf(failure, outcome))
        failed = [report for report in reports if report is not None]
        if (
            failure is not None
            and all(report.alreadyAgreed for report in failed)
            and len({report.text for report in failed}) == 1
        ):
            # agreed on already, in every process that raised it: raised on as it is, not wrapped again
            raise failure
        message = "; ".join(
            "process {:}: {:}".format(rank, report.text) for rank, report in enumerate(reports) if report is not None
        )
        if agreed == _CUTBACK:
            cutbackSize = min(report.cutbackSize for report in failed if report.cutbackSize is not None)
            raise CutbackRequest(message, cutbackSize) from failure
        if agreed == _ERROR:
            errorTypes = {report.builtinErrorType for report in failed if report.outcome == _ERROR}
            raise _errorOnAllRanks(errorTypes, "{:} failed in {:}".format(operation, message)) from failure
        raise StepFailedOnAllRanks("{:} failed in {:}".format(operation, message)) from failure


@dataclass(frozen=True)
class _FailureReport:
    """What one process tells the others of its failure in an agreement.

    Parameters
    ----------
    text
        What went wrong, for the message.
    outcome
        What it means for the agreement (:func:`_outcomeOf`).
    cutbackSize
        The cutback size requested, for a cutback request; else None.
    builtinErrorType
        For an error, the name of its type if that is a built-in exception type; else None.
    alreadyAgreed
        Whether the failure was agreed on already, in an agreement inside this one.
    """

    text: str
    outcome: int
    cutbackSize: float | None
    builtinErrorType: str | None
    alreadyAgreed: bool


def _reportOf(failure: Exception, outcome: int) -> _FailureReport:
    """The report of a process' failure.

    Parameters
    ----------
    failure
        The exception.
    outcome
        What it means for the agreement.

    Returns
    -------
    _FailureReport
        The report.
    """

    alreadyAgreed = isinstance(failure, RaisedOnAllRanks)
    errorType = type(failure).__name__
    return _FailureReport(
        text=str(failure) if alreadyAgreed else "{:}: {:}".format(errorType, failure),
        outcome=outcome,
        cutbackSize=failure.cutbackSize if outcome == _CUTBACK else None,
        builtinErrorType=errorType if vars(builtins).get(errorType) is type(failure) else None,
        alreadyAgreed=alreadyAgreed,
    )


def _errorOnAllRanks(errorTypes: set, message: str) -> ErrorOnAllRanks:
    """The error every process raises: an :class:`ErrorOnAllRanks` that is also of the built-in type
    every failing process raised, if they raised one and the same -- so that a caller catching that
    type (a test runner skipping a test case on a ``NotImplementedError``) catches it in every process,
    as it would the error of a serial run --; a plain :class:`ErrorOnAllRanks` otherwise. The same in
    every process.

    Parameters
    ----------
    errorTypes
        The names of the built-in types of the errors raised, None for an error of another type.
    message
        The message.

    Returns
    -------
    ErrorOnAllRanks
        The error.
    """

    if len(errorTypes) == 1 and None not in errorTypes:
        try:
            return _errorOnAllRanksOfType(vars(builtins)[next(iter(errorTypes))])(message)
        except TypeError:
            # a built-in type that cannot be combined or constructed so (its layout, its arguments)
            pass
    return ErrorOnAllRanks(message)


@cache
def _errorOnAllRanksOfType(errorType: type) -> type:
    """The class of an :class:`ErrorOnAllRanks` that is also an ``errorType``.

    Parameters
    ----------
    errorType
        A built-in exception type.

    Returns
    -------
    type
        The class.
    """

    return type(
        "ErrorOnAllRanks",
        (ErrorOnAllRanks, errorType),
        # the message as given, also where the built-in type quotes it (KeyError)
        {"__module__": ErrorOnAllRanks.__module__, "__str__": BaseException.__str__},
    )


def _outcomeOf(failure: Exception | None) -> int:
    """What a process' exception means for the agreement on it.

    Parameters
    ----------
    failure
        The exception, or None.

    Returns
    -------
    int
        The outcome: no failure, a cutback, a conditional stop, a step failure, or an error.
    """

    if failure is None:
        return _NO_FAILURE
    if isinstance(failure, CutbackRequest):
        return _CUTBACK
    if isinstance(failure, ConditionalStop):
        return _CONDITIONAL_STOP
    if isinstance(failure, StepFailed):
        return _STEP_FAILURE
    return _ERROR


class _FailedInAnotherProcess(BaseException):
    """Another process failed inside :meth:`Communicator.allRanksFailTogetherWhileCommunicating`: this
    process learned of it in the agreement before an operation, and leaves the context -- without
    communicating again there, and past any ``except Exception`` on the way, which would otherwise
    let it go on to an operation the others never reach.

    Parameters
    ----------
    agreedOutcome
        The most severe outcome of any process, agreed on.
    """

    def __init__(self, agreedOutcome: int):
        super().__init__("another process failed")
        self.agreedOutcome = agreedOutcome
