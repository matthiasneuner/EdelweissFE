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

"""
FieldOutputs store all kind of analysis results,
and are defined via the keyword ``*fieldOutput``.
All fieldOutputs are accessable to all outputmanagers at the end of each
increment, step and job.
Furthermore, they can be exported to ``*.csv`` files at the end of the analysis job.

ATTENTION:
    If the results are exported to a .csv file with enabled "saveHistory",
    the time History is automatically appended to the .csv file"
"""

import os
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.orderedset import OrderedSet
from edelweissfe.utils.elementresultcollector import ElementResultCollector
from edelweissfe.utils.schema import schemaField, subKeywordField


@dataclass(frozen=True)
class FieldOutputPerNodeSchema:
    """The options of a single ``>>perNode`` field-output block."""

    name: str | None = schemaField(description="Name of the field output.", dtype=str, default=None, required=True)
    field: str | None = schemaField(description="Field of the result.", dtype=str, default=None, required=True)
    result: str | None = schemaField(description="Result name.", dtype=str, default=None, required=True)
    elSet: str | None = schemaField(
        description="Element set: the output covers its nodes (none for an empty set); without a set, the whole field.",
        dtype=str,
        default=None,
    )
    nSet: str | None = schemaField(
        description="Node set: the output covers its nodes (none for an empty set); without a set, the whole field.",
        dtype=str,
        default=None,
    )
    rigidBody: str | None = schemaField(
        description="Rigid body (as registered in model.rigidBodies).", dtype=str, default=None
    )
    saveHistory: bool = schemaField(
        description="Save complete History or only last (increment) result", dtype=bool, default=False
    )
    f_x: str | None = schemaField(
        description="Function to apply in each increment.", dtype=str, default=None, optionName="f(x)"
    )
    f_export_x: str | None = schemaField(
        description="Function to apply on final result (table).", dtype=str, default=None, optionName="f_export(x)"
    )
    export: str | None = schemaField(
        description="Export the field output to a file at the end of the job.", dtype=str, default=None
    )


@dataclass(frozen=True)
class FieldOutputPerElementSchema:
    """The options of a single ``>>perElement`` field-output block."""

    name: str | None = schemaField(description="Name of the field output.", dtype=str, default=None, required=True)
    elSet: str | None = schemaField(description="Element set.", dtype=str, default=None, required=True)
    result: str | None = schemaField(description="Result name.", dtype=str, default=None, required=True)
    quadraturePoint: str | None = schemaField(description="Integer or slice.", dtype=str, default=None, required=True)
    saveHistory: bool = schemaField(
        description="Save complete History or only last (increment) result", dtype=bool, default=False
    )
    f_x: str | None = schemaField(
        description="Function to apply in each increment.", dtype=str, default=None, optionName="f(x)"
    )
    f_export_x: str | None = schemaField(
        description="Function to apply on final result (table).", dtype=str, default=None, optionName="f_export(x)"
    )
    export: str | None = schemaField(
        description="Export the field output to a file at the end of the job.", dtype=str, default=None
    )


@dataclass(frozen=True)
class FieldOutputFromExpressionSchema:
    """The options of a single ``>>fromExpression`` field-output block."""

    name: str | None = schemaField(description="Name of the field output.", dtype=str, default=None, required=True)
    elSet: str | None = schemaField(description="Element set.", dtype=str, default=None)
    nSet: str | None = schemaField(description="Node set.", dtype=str, default=None)
    expression: str | None = schemaField(
        description="Expression for retrieving field output.", dtype=str, default=None, required=True
    )
    saveHistory: bool = schemaField(
        description="Save complete History or only last (increment) result", dtype=bool, default=False
    )
    f_x: str | None = schemaField(
        description="Function to apply in each increment.", dtype=str, default=None, optionName="f(x)"
    )
    f_export_x: str | None = schemaField(
        description="Function to apply on final result (table).", dtype=str, default=None, optionName="f_export(x)"
    )
    export: str | None = schemaField(
        description="Export the field output to a file at the end of the job.", dtype=str, default=None
    )


@dataclass(frozen=True)
class FieldOutputSchema:
    """The sub-keyword blocks of the ``*fieldOutput`` keyword.

    ``*fieldOutput`` itself declares no line options of its own (see
    ``edelweissfe.keywords.fieldoutput``'s module docstring) -- its entire grammar is these three
    repeatable ``>>`` blocks. Documentation/parser-validation only: ``_FieldOutputBase`` and friends
    still construct from the raw parsed dict via ``abqmodelconstructor``/``inputfilehelpers``,
    unchanged.
    """

    perNode: tuple[FieldOutputPerNodeSchema, ...] = subKeywordField(
        description="Create node-based field output.", schema=FieldOutputPerNodeSchema
    )
    perElement: tuple[FieldOutputPerElementSchema, ...] = subKeywordField(
        description="Create element-based field output.", schema=FieldOutputPerElementSchema
    )
    fromExpression: tuple[FieldOutputFromExpressionSchema, ...] = subKeywordField(
        description="Create field output from expression.", schema=FieldOutputFromExpressionSchema
    )


class _FieldOutputBase:
    """
    Entity of a fieldOutput request.
    Carries the history or the latest result.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    model
        A dictionary containing the model tree.
    journal
        The journal object for logging.
    saveHistory
        Save the complete history or only the last result.
    f_x
        Apply a math function on the results.
    export
        Export the results to a file.
    fExport_x
        Apply a math function on the results before exporting.
    reshape_to_dimensions
        Reshape the result to a specific number of dimensions.
    """

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
        reshape_to_dimensions: int = None,
    ):
        self.timeTotal = 0.0
        self.name = name
        self.model = model
        self.journal = journal
        self.appendResults = saveHistory
        self.result = [] if self.appendResults else None
        self.f = f_x
        self.f_export = fExport_x
        self.timeHistory = []
        self.export = export
        #: The size of the export file the run has written so far. Before its first write, a
        #: process truncates the file to it: a cold run starts the file afresh, and a resumed run
        #: drops whatever the interrupted run wrote after its checkpoint.
        self._exportedBytes = 0
        self._exportFileTruncated = False
        self._reshape_to_dimensions = reshape_to_dimensions
        #: Whether the next :meth:`finalizeIncrement` stores a result here; see
        #: :meth:`gatherResultsOfWholeSet`.
        self._storesNextResultHere = True
        #: Whether the last result, and whether the history, is held by another process -- the one
        #: writing the output -- and not here: a domain-decomposed run gathered it there only.
        self._lastResultHeldElsewhere = False
        self._historyHeldElsewhere = False
        #: Whether the last result was gathered to every process (or read whole here): the same in every
        #: process, unlike whether it is held here; see :meth:`finalizeStep`.
        self._lastResultReachedEveryProcess = True

    def getLastResult(
        self,
    ) -> np.ndarray:
        """Get the last result, no matter if the history is stored or not.

        Returns
        -------
        np.ndarray
            The result array.

        Raises
        ------
        RuntimeError
            If the last result is held by another process only (:meth:`gatherResultsOfWholeSet`).
        """

        if self._lastResultHeldElsewhere:
            raise RuntimeError(self._heldElsewhereMessage("last result"))
        return self.result[-1] if self.appendResults else self.result

    def _heldElsewhereMessage(self, what: str) -> str:
        """Why a result of this field output cannot be read in this process.

        Parameters
        ----------
        what
            What was to be read.

        Returns
        -------
        str
            The message.
        """

        return (
            "fieldOutput {:}: its {:} is held by the process writing the output only, to which a "
            "domain-decomposed run gathers a field output that nothing in the other processes reads. A reader "
            "in every process must name the field outputs it reads (e.g. MarkerBase.fieldOutputsRead).".format(
                self.name, what
            )
        )

    def getResultHistory(
        self,
    ) -> np.ndarray:
        """Get the history.
        Throws an exception if the history is not stored, or if the stored entries do not all
        share the same shape (e.g. because the mesh changed under adaptive refinement, in which
        case a rectangular per-node history is not well-defined).

        Returns
        -------
        np.ndarray
            The result history.
        """

        if not self.appendResults:
            raise Exception(
                "fieldOutput {:} does not save any history; please define it with saveHistory=True!".format(self.name)
            )
        if self._historyHeldElsewhere:
            raise RuntimeError(self._heldElsewhereMessage("history"))

        if self.result:
            firstShape = np.shape(self.result[0])
            if any(np.shape(r) != firstShape for r in self.result):
                raise Exception(
                    "fieldOutput {:} has a per-node/per-element history with varying shapes across "
                    "increments (e.g. due to adaptive mesh refinement); a rectangular history is not "
                    "well-defined. Use a reducing f(x) (as for RF in examples/WinklerL/WinklerL_blockamg_amr.inp) "
                    "instead of saveHistory=True in this case.".format(self.name)
                )

        return np.asarray(self.result)

    def getTimeHistory(
        self,
    ) -> np.ndarray:
        """Get the time history.

        Returns
        -------
        np.ndarray
            The time history.
        """

        return np.asarray(self.timeHistory)

    def _applyResultsPipleline(self, result):
        """Apply the pipeline of operations onto the results.
        Called by inheriting classes.

        Parameters
        ----------
        model
            The model tree.
        """

        self.timeHistory.append(self.model.time)

        if self._reshape_to_dimensions is not None:
            result = np.reshape(result, (-1, self._reshape_to_dimensions))

        if self.f:
            result = self.f(result)

        if self.appendResults:
            self.result.append(result.copy())
        else:
            self.result = result
        self._lastResultHeldElsewhere = False

    def _recordResultHeldElsewhere(self):
        """Record that the result of this increment is stored by another process, not here: the time
        is recorded as for a stored result -- so that every process decides alike whether the end of a
        step stores another one (:meth:`finalizeStep`) --, and reading the last result, or the history,
        raises until a result is stored here again (or the history restored from a checkpoint)."""

        self.timeHistory.append(self.model.time)
        self._lastResultHeldElsewhere = True
        self._historyHeldElsewhere = self._historyHeldElsewhere or self.appendResults

    def _resultTableForExport(self) -> np.ndarray:
        """Assemble the result table that ``f_export`` is applied to.

        Normally this is the whole stored history, so that a table-wide ``f_export`` such as
        ``x[:,:,1]`` sees the increment axis it expects. Under adaptive mesh refinement the
        per-increment shape changes, and the history as a whole is no longer representable as a
        single rectangular array; in that case only the last increment is tabulated, as a table
        of one row. This keeps the dimensionality (and hence ``f_export``) valid while exporting
        the current increment, which is all ``writeLastResult`` consumes.

        Returns
        -------
        np.ndarray
            The result table, with the increment as leading axis.
        """

        if not self.appendResults:
            return np.asarray(self.result)

        lastShape = np.shape(self.result[-1])
        if any(np.shape(entry) != lastShape for entry in self.result):
            return np.asarray(self.result[-1:])

        return np.asarray(self.result)

    def disableFileExport(self):
        """Stop this field output from writing its ``.csv`` export; its results are still recorded.

        For the processes of a domain-decomposed run other than rank 0, which compute the same field
        outputs and must not write the same file concurrently.
        """

        self.export = None

    def writeLastResult(self):
        """Update file output.

        Parameters
        ----------
        model
            The model tree.
        """
        res = self._resultTableForExport()
        if self.f_export:
            res = self.f_export(res)

        if res.ndim > 2:
            self.journal.message("Reshaping fieldOutput result for export in .csv file", self.name)
            res = res.reshape((res.shape[0], -1))

        fileName = f"{self.export}.csv"
        if not self._exportFileTruncated:
            open(fileName, "a").close()
            if os.path.getsize(fileName) > self._exportedBytes:
                os.truncate(fileName, self._exportedBytes)
            self._exportFileTruncated = True

        with open(fileName, "a") as f:
            np.savetxt(
                f,
                np.hstack(
                    (
                        self.timeHistory[-1],
                        res[-1,],
                    )
                ).reshape((1, -1)),
            )
            self._exportedBytes = f.tell()

    def initializeJob(self):
        """Initalize everything. Will also update the results
        based on the proved start time and solution."""

        self.updateResults(self.model)

    def getRestartData(self) -> dict[str, np.ndarray]:
        """The history this field output carries from one increment to the next -- the times, the
        results (flattened, since refinement changes their shape), and how much of the export file
        belongs to the run.

        Returns
        -------
        dict[str, numpy.ndarray]
            The state.
        """

        state = {"timeHistory": np.array(self.timeHistory, dtype=float), "exportedBytes": np.array(self._exportedBytes)}
        if self.appendResults:
            state["resultShapes"] = np.array([np.shape(r) for r in self.result], dtype=np.int64)
            state["resultValues"] = np.concatenate([np.ravel(r) for r in self.result]) if self.result else np.empty(0)
        elif self.result is not None:
            state["result"] = np.asarray(self.result)
        return state

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the state :meth:`getRestartData` returned.

        Parameters
        ----------
        data
            The state.
        """

        self.timeHistory = [float(t) for t in data["timeHistory"]]
        self._exportedBytes = int(data["exportedBytes"])
        self._lastResultHeldElsewhere = self._historyHeldElsewhere = False
        self._lastResultReachedEveryProcess = True
        if self.appendResults:
            self.result, offset = [], 0
            for shape in data["resultShapes"]:
                size = int(np.prod(shape))
                self.result.append(np.array(data["resultValues"][offset : offset + size]).reshape(shape))
                offset += size
        else:
            self.result = np.array(data["result"]) if "result" in data else None

    def initializeStep(self, step):
        """Write the current (just-updated) result as the first row of this step.

        Only at the start of a step: a step resumed from a checkpoint already exported this state,
        as the last row before the checkpoint was written.
        """
        if not step.timeStepper.isAtStepStart():
            return

        if self.export:
            self.writeLastResult()

    def finalizeIncrement(
        self,
    ):
        """Finalize an increment, i.e. store the current results -- unless the result of this increment
        was gathered to another process only (:meth:`gatherResultsOfWholeSet`)."""

        if not self._storesNextResultHere:
            self._storesNextResultHere = True
            self._recordResultHeldElsewhere()
            return

        self.updateResults(self.model)

        if self.export:
            self.writeLastResult()

    def readResultsHere(self):
        """Read the part of the current result this process holds, without communicating: the first
        half of :meth:`updateResults` for a result gathered from several processes, which a
        domain-decomposed solver calls apart from the second, :meth:`gatherResultsOfWholeSet` (see
        :meth:`FieldOutputController.readResultsHere`). Nothing, for a result every process holds
        whole: every result but that of an element set.
        """

    def gatherResultsOfWholeSet(self, toEveryProcess: bool = True, storedHere: bool = True):
        """Complete the part read by :meth:`readResultsHere` to the result of the whole set, from the
        processes holding the rest -- in every process, or in the process writing the output only --;
        the next :meth:`finalizeIncrement` stores it, if it is stored here at all. Collective where the
        elements are distributed over several processes. For a result every process holds whole,
        nothing is gathered.

        Parameters
        ----------
        toEveryProcess
            Whether every process receives the result of the whole set, or rank 0 only.
        storedHere
            Whether the next :meth:`finalizeIncrement` stores the result here; if not, it records that
            the result is held by the process writing the output.
        """

        self._storesNextResultHere = storedHere
        self._lastResultReachedEveryProcess = toEveryProcess

    def finalizeStep(
        self,
    ):
        """Bring the stored result up to date with the model's final state, if the step advanced
        past the increment that stored the current one.

        The output managers write one final frame at the end of a step, from the result stored
        here -- see the Ensight export's ``finalizeStep``, which writes whenever time has advanced
        since its last frame. If the topology changed after that result was stored, which is
        exactly what live h-adaptivity does between two output increments, the result belongs to a
        different mesh than the one being written and the export rejects it::

            Variable displacement result size (32) does not match the number of nodes (141)

        Guarded on time having advanced, by the same rule the managers use, so the two decisions
        cannot disagree: whenever a manager writes a final frame the result behind it has just been
        refreshed, and a step ending on an increment that already stored a result does not store a
        second one at the same time and duplicate the last point of a history export.

        A step ending on an output increment whose result was gathered to the process writing the
        output only (:meth:`gatherResultsOfWholeSet`) leaves the other processes without it, while the
        next step may start by reading it in every process -- a marker of the step-start topology
        update, a ``setField`` step action. So that result is read again, from the model the end of the
        step made whole in every process, and replaces the last one, in every process alike (the bits
        are those stored on rank 0): the end of a step leaves every field output's last result in every
        process. The history of a process that did not store every result stays incomplete.
        """
        if not self.timeHistory or self.model.time - self.timeHistory[-1] > 1e-12:
            self.updateResults(self.model)
        elif not self._lastResultReachedEveryProcess:
            self.timeHistory.pop()
            if self.appendResults and not self._lastResultHeldElsewhere:
                self.result.pop()
            self.updateResults(self.model)
            self._lastResultReachedEveryProcess = True

    def finalizeJob(
        self,
    ):
        pass

    def setResults(self, values: np.ndarray):
        """Modifies a result at it's origin, if possible.
        Throws an exception if not possible.

        Parameters
        ----------
        values
            The values.
        """
        raise Exception("setting field output currently not implemented for this type of output!")

    def __eq__(self, other):
        if type(other) is str:
            return other == self.name
        return self.getLastResult() == other

    def __ne__(self, other):
        return self.getLastResult() != other

    def __lt__(self, other):
        return self.getLastResult() < other

    def __le__(self, other):
        return self.getLastResult() <= other

    def __gt__(self, other):
        return self.getLastResult() > other

    def __ge__(self, other):
        return self.getLastResult() >= other

    def __getitem__(self, index):
        return self.getLastResult()[index]

    def __add__(self, other):
        return self.getLastResult() + other

    def __sub__(self, other):
        return self.getLastResult() - other


class NodeFieldOutput(_FieldOutputBase):
    """
    This is a Node based FieldOutput.
    It operates on NodeFields.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    nodeField
        The NodeField, on which this FieldOutput operates.
    result
        The name of the result entry in the NodeField.
    model
        The model tree instance.
    journal
        The journal object for logging.
    saveHistory
        Save the complete history or only the last result.
    f_x
        Apply a math function on the results.
    export
        Export the results to a file.
    fExport_x
        Apply a math function on the results before exporting.
    """

    def __init__(
        self,
        name: str,
        nodeField,
        result: str,
        model: FEModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
    ):
        self.entry = result
        self.fieldName = nodeField.name
        self._nodeField = nodeField
        self.associatedSet = nodeField.associatedSet

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x)

    def updateResults(self, model: FEModel):
        """Update the field output.
        Will use the current solution and reaction vector if result is a nodal result.

        Parameters
        ----------
        model
            The model tree.
        """

        result = self._nodeField[self.entry]

        return super()._applyResultsPipleline(result)


class RigidBodyFieldOutput(_FieldOutputBase):
    """
    This is a FieldOutput operating directly on a RigidBody.

    A rigid body's visualization (e.g., surface) nodes are not independent
    degrees of freedom -- they are fully determined by the rigid body's
    reference point, which is itself part of an ordinary NodeField. This
    FieldOutput therefore does not operate on a NodeField at all: it queries
    :meth:`~edelweissfe.rigidbodies.rigidbody.RigidBody.getVisualizationField`,
    which computes the requested field directly from the rigid body's
    (already solved) kinematics.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    rigidBody
        The RigidBody on which this FieldOutput operates.
    field
        The name of the field to retrieve from the rigid body (e.g., "displacement").
    model
        The model tree instance.
    journal
        The journal object for logging.
    saveHistory
        Save the complete history or only the last result.
    f_x
        Apply a math function on the results.
    export
        Export the results to a file.
    fExport_x
        Apply a math function on the results before exporting.
    """

    def __init__(
        self,
        name: str,
        rigidBody,
        field: str,
        model: FEModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
    ):
        self._rigidBody = rigidBody
        self._field = field
        self.associatedSet = rigidBody

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x)

    def updateResults(self, model: FEModel):
        """Update the field output from the rigid body's current kinematics.

        Parameters
        ----------
        model
            The model tree.
        """

        result = self._rigidBody.getVisualizationField(self._field)

        return super()._applyResultsPipleline(result)


class ElementFieldOutput(_FieldOutputBase):
    """
    This is a Element based FieldOutput.
    It operates on ElementSets.

    Its result is a result of the whole element set, one row per element in set order. Where a
    domain-decomposed run created only part of the set in this process, the rows of the other
    elements are gathered from the processes computing them
    (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.resultsOfWholeSet`), so that
    the result is the same in every process.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    elSet
        The ElementSet on which this FieldOutput operates.
    resultName
        The name of the result entry in the :class:`ElementBase`.
    model
        The model tree instance.
    journal
        The journal object for logging.
    saveHistory
        Save the complete history or only the last result.
    f_x
        Apply a math function on the results.
    export
        Export the results to a file.
    fExport_x
        Apply a math function on the results before exporting.
    quadraturePoints
        The list of quadrature points for which the results should be extracted.
    """

    def __init__(
        self,
        name: str,
        elSet: ElementSet,
        resultName: str,
        model: FEModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
        quadraturePoints: Union[int, slice, list[int]] = 0,
    ):
        self.associatedSet = elSet
        self.resultName = resultName
        self.quadraturePoints = quadraturePoints
        #: Whether :meth:`readResultsHere` read results that are not yet gathered, and those results
        #: of the elements of the set owned here (None if none are owned here).
        self._resultsHereRead = False
        self._resultsHere = None
        #: The results of the whole set, gathered by :meth:`gatherResultsOfWholeSet` and not yet
        #: stored; None if nothing was gathered since.
        self._resultsOfWholeSet = None

        self._collectFromOwnedElements(model)

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x)

    def _collectFromOwnedElements(self, model: FEModel):
        """Set up the element result collector for the elements of the set this process reports --
        all of them, unless a domain-decomposed run computes some of them elsewhere."""

        self._seenSetVersion = self.associatedSet._version
        self._seenOwnershipVersion = model.elementDistribution.ownershipVersion
        ownedHere = model.elementDistribution.ownedElements(self.associatedSet.localElements())
        # Numbers, not the elements: an element moving to another process must not be kept alive here.
        self._numbersOwnedHere = [element.elNumber for element in ownedHere]
        self.elementResultCollector = (
            ElementResultCollector(ownedHere, self.quadraturePoints, self.resultName) if ownedHere else None
        )

    def _rebuildCollectorIfSetChanged(self):
        """Rebuild the element result collector -- which pins a fixed snapshot of the element
        list at construction -- if the associated ElementSet was mutated in-place (e.g. AMR
        replacing a refined parent element with its children), or elements moved between the
        processes of a domain-decomposed run, since the last check. Unlike a plain iteration over
        the set, this pinned snapshot does not see new elements on its own, and it holds pointers
        into the state of the elements it was made for, without keeping them alive: after elements
        moved, a pointer would read an element this process no longer computes, or no longer holds.
        Every read of the collector therefore comes after this check."""
        if (
            self.associatedSet._version != self._seenSetVersion
            or self.model.elementDistribution.ownershipVersion != self._seenOwnershipVersion
        ):
            self._collectFromOwnedElements(self.model)

    def readResultsHere(self):
        """Read the results of the elements of the set owned here; see
        :meth:`_FieldOutputBase.readResultsHere`."""

        self._rebuildCollectorIfSetChanged()
        self._resultsHere = self.elementResultCollector.getCurrentResults() if self.elementResultCollector else None
        self._resultsHereRead = True

    def gatherResultsOfWholeSet(self, toEveryProcess: bool = True, storedHere: bool = True):
        """Gather the results of the whole set from those read here and those owned elsewhere
        (:meth:`~edelweissfe.models.elementdistribution.ElementDistribution.resultsOfWholeSet`);
        see :meth:`_FieldOutputBase.gatherResultsOfWholeSet`. Collective where the elements are
        distributed.

        Parameters
        ----------
        toEveryProcess
            Whether every process receives the result of the whole set, or rank 0 only.
        storedHere
            Whether the next :meth:`finalizeIncrement` stores the result here.
        """

        super().gatherResultsOfWholeSet(toEveryProcess, storedHere)

        if not self._resultsHereRead:
            raise RuntimeError(
                "field output {:}: the results of the whole set are gathered from those read here, "
                "which were not read (readResultsHere)".format(self.name)
            )
        resultsHere, self._resultsHere, self._resultsHereRead = self._resultsHere, None, False
        resultsOfWholeSet = self.model.elementDistribution.resultsOfWholeSet(
            self.associatedSet, self._numbersOwnedHere, resultsHere, toEveryProcess
        )
        # Kept only if stored here: where every process holds every element, the whole result is here
        # even if it was gathered for the process writing the output, and kept, it would be stored by
        # the next update instead of the result read then (the end of a step, say).
        self._resultsOfWholeSet = resultsOfWholeSet if storedHere else None

    def updateResults(self, model: FEModel):
        """Update the field output: read the results of the elements owned here, gather those of the
        whole set, and store them -- or store the results gathered already, if a domain-decomposed
        solver gathered them apart (:meth:`gatherResultsOfWholeSet`).

        Parameters
        ----------
        model
            The model tree.
        """

        if self._resultsOfWholeSet is None:
            self.readResultsHere()
            self.gatherResultsOfWholeSet()
        result, self._resultsOfWholeSet = self._resultsOfWholeSet, None

        super()._applyResultsPipleline(result)

    def setResults(self, values: np.ndarray):
        """Modifies a result at it's origin, if possible.
        Throws an exception if not possible.

        Parameters
        ----------
        values
            The values.
        """
        if self.f:
            raise Exception("cannot set field output for modified results (f(x) != None) !")

        self._rebuildCollectorIfSetChanged()
        for i, el in enumerate(self.associatedSet):
            for j, g in enumerate(self.quadraturePoints):
                theArray = el.getResultArray(self.resultName, g, True)
                theArray[:] = values[i, j, :]
                el.acceptLastState()


class ExpressionFieldOutput(_FieldOutputBase):
    """
    This is a Node based FieldOutput.
    It operates on NodeFields.

    Parameters
    ----------
    associatedSet
        The associated set of nodes or elements.
    theExpression
        The expression to be evaluated.
    name
        The name of this FieldOutput.
    nodeField
        The NodeField, on which this FieldOutput operates.
    result
        The name of the result entry in the NodeField.
    model
        The model tree instance.
    journal
        The journal object for logging.
    saveHistory
        Save the complete history or only the last result.
    f_x
        Apply a math function on the results.
    export
        Export the results to a file.
    fExport_x
        Apply a math function on the results before exporting.
    """

    def __init__(
        self,
        associatedSet: OrderedSet | None,
        theExpression,
        name: str,
        model: FEModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
    ):

        self.associatedSet = associatedSet
        self.theExpression = theExpression

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x)

    def updateResults(self, model: FEModel):
        """Update the field output.
        Will use the current solution and reaction vector if result is a nodal result.

        Parameters
        ----------
        model
            The model tree.
        """

        if self.associatedSet:
            result = np.reshape(np.asarray(self.theExpression()), (len(self.associatedSet), -1))
        else:
            result = np.reshape(np.asarray(self.theExpression()), (1, -1))

        return super()._applyResultsPipleline(result)


class FieldOutputController:
    """
    The central module for managing field outputs, which can be used by output managers.
    """

    def __init__(self, model: FEModel, journal: Journal):
        self.model = model
        self.journal = journal
        self.fieldOutputs = {}

    def addExpressionFieldOutput(
        self,
        associatedSet: set,
        theExpression: Callable,
        name: str,
        saveHistory=False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
    ):
        """Add a new FieldOutput entry to be computed during the simulation

        Parameters
        ----------
        associatedSet
            The associated set of nodes or elements.
        theExpression
            The expression to be evaluated.
        name
            The name of this FieldOutput.
        saveHistory
            Save the complete history or only the last result.
        f_x
            Apply a math function on the results.
        export
            Export the results to a file.
        fExport_x
            Apply a math function on the results before exporting.
        """

        if name in self.fieldOutputs:
            raise Exception("FieldOutput {:} already exists!".format(name))

        self.fieldOutputs[name] = ExpressionFieldOutput(
            associatedSet,
            theExpression,
            name,
            self.model,
            self.journal,
            saveHistory,
            f_x=f_x,
            export=export,
            fExport_x=fExport_x,
        )

    def addPerNodeFieldOutput(
        self,
        name: str,
        nodeField: NodeField,
        result: str = None,
        saveHistory=False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
    ):
        """Add a new FieldOutput entry to be computed during the simulation

        Parameters
        ----------
        name
            The name of this FieldOutput.
        nodeField
            The :class:`NodeField` on which this FieldOutput should operate.
        result
            The name of the result entry in the :class:`NodeField`.
        journal
            The :class:`Journal` instance for logging purposes.
        saveHistory
            Save the complete history or only the last result.
        f_x
            Apply a math function on the results.
        export
            Export the results to a file.
        fExport_x
            Apply a math function on the results before exporting.
        """

        if not result:
            result = name

        if name in self.fieldOutputs:
            raise Exception("FieldOutput {:} already exists!".format(name))

        self.fieldOutputs[name] = NodeFieldOutput(
            name,
            nodeField,
            result,
            self.model,
            self.journal,
            saveHistory,
            f_x,
            export,
            fExport_x,
        )

    def addRigidBodyFieldOutput(
        self,
        name: str,
        rigidBody,
        field: str,
        saveHistory=False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
    ):
        """Add a new FieldOutput entry operating directly on a RigidBody.

        Parameters
        ----------
        name
            The name of this FieldOutput.
        rigidBody
            The :class:`~edelweissfe.rigidbodies.rigidbody.RigidBody` on which this FieldOutput should operate.
        field
            The name of the field to retrieve from the rigid body (e.g., "displacement").
        saveHistory
            Save the complete history or only the last result.
        f_x
            Apply a math function on the results.
        export
            Export the results to a file.
        fExport_x
            Apply a math function on the results before exporting.
        """

        if name in self.fieldOutputs:
            raise Exception("FieldOutput {:} already exists!".format(name))

        self.fieldOutputs[name] = RigidBodyFieldOutput(
            name,
            rigidBody,
            field,
            self.model,
            self.journal,
            saveHistory,
            f_x,
            export,
            fExport_x,
        )

    def addPerElementFieldOutput(
        self,
        name: str,
        elSet: ElementSet,
        result: str = None,
        saveHistory=False,
        f_x: Callable = None,
        export: str = None,
        fExport_x: Callable = None,
        quadraturePoints=Union[int, slice, list[int]],
    ):
        """Add a new FieldOutput entry to be computed during the simulation

        Parameters
        ----------
        name
            The name of this FieldOutput.
        elSet
            The :class:`ElementSet` on which this FieldOutput should operate.
        result
            The name of the result, which is provided by the Elements in the :class:`ElementSet`.
        journal
            The :class:`Journal` instance for logging purposes.
        saveHistory
            Save the complete history or only the last result.
        f_x
            Apply a math function on the results.
        export
            Export the results to a file.
        fExport_x
            Apply a math function on the results before exporting.
        quadraturePoints
            The indices of quadrature points for which the results should be extracted.
        """

        if not result:
            result = name

        if name in self.fieldOutputs:
            raise Exception("FieldOutput {:} already exists!".format(name))

        self.fieldOutputs[name] = ElementFieldOutput(
            name,
            elSet,
            result,
            self.model,
            self.journal,
            saveHistory,
            f_x,
            export,
            fExport_x,
            quadraturePoints,
        )

    def initializeJob(self):
        for fieldOutput in self.fieldOutputs.values():
            fieldOutput.initializeJob()

    def disableFileExport(self):
        """Stop every field output from writing its ``.csv`` export; see
        :meth:`_FieldOutputBase.disableFileExport`."""

        for fieldOutput in self.fieldOutputs.values():
            fieldOutput.disableFileExport()

    def finalizeIncrement(
        self,
    ):
        """Finalize all field outputs at the end of an increment."""

        for output in self.fieldOutputs.values():
            output.finalizeIncrement()

    def readResultsHere(self):
        """Let every field output read the part of its current result this process holds, without
        communicating (:meth:`_FieldOutputBase.readResultsHere`).

        :meth:`finalizeIncrement` reads, gathers and stores every result in one pass. A
        domain-decomposed solver splits it instead: each process reads its part -- which may fail in
        one process alone, and so is done where the processes agree on failures afterwards --, then
        the parts are gathered (:meth:`gatherResultsOfWholeSet`), and only then :meth:`finalizeIncrement`
        stores them, communicating no more (see
        :meth:`~edelweissfe.solvers.nonlinearexplicitdynamicmpi.NEDMPI.writeIncrementOutput`).
        """

        for output in self.fieldOutputs.values():
            output.readResultsHere()

    def gatherResultsOfWholeSet(self, readOnEveryProcess: set[str] | None = None, writesOutput: bool = True):
        """Let every field output complete the part read by :meth:`readResultsHere` to the result of
        its whole set, for the next :meth:`finalizeIncrement`. Collective where the elements are
        distributed over several processes: every process gathers the field outputs in the same
        order.

        A field output read in every process is gathered to every process and stored everywhere;
        any other is gathered to rank 0 and stored only by the process writing the output. Where it
        is not stored, it records that its result is held elsewhere, and reading it there raises
        (:meth:`_FieldOutputBase.getLastResult`).

        Parameters
        ----------
        readOnEveryProcess
            The names of the field outputs read in every process; None for all of them.
        writesOutput
            Whether this process writes the output, and so stores every result.
        """

        for name, output in self.fieldOutputs.items():
            everywhere = readOnEveryProcess is None or name in readOnEveryProcess
            output.gatherResultsOfWholeSet(toEveryProcess=everywhere, storedHere=everywhere or writesOutput)

    def finalizeStep(
        self,
    ):
        """Finalize all field outputs at the end of a step."""

        for output in self.fieldOutputs.values():
            output.finalizeStep()

    def initializeStep(self, step):
        """Initalize an step.

        Parameters
        ----------
        step
            The step information.
        """

        for output in self.fieldOutputs.values():
            output.initializeStep(step)

    def finalizeJob(
        self,
    ):
        """Finalize all field outputs at the end of a job."""
        for output in self.fieldOutputs.values():
            output.finalizeJob()
