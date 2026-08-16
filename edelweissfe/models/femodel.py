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
# Created on Fri Jan 27 19:53:45 2017

# @author: Matthias Neuner

import textwrap
from operator import attrgetter

import h5py
import numpy as np

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.modelchange import ModelChange, coalesce
from edelweissfe.variables.fieldvariable import FieldVariable
from edelweissfe.variables.scalarvariable import ScalarVariable


class FEModel:
    """This is is a standard finite element model tree.
    It takes care of the correct number of variables,
    for nodes and scalar degrees of freedem, and it manages the fields.


    Parameters
    ----------
    dimension
        The dimension of the model.
    """

    identification = "FEModel"

    def __init__(self, dimension: int):
        self.time = 0.0  #: Current time of the model.
        self.nodes = {}  #: Nodes in the model.
        self.elements = {}  #: Elements in the model.
        self.nodeSets = {}  #: NodeSets in the model.
        self.nodeFields = {}  #: NodeFields in the model.
        self.elementSets = {}  #: ElementSets in the model.
        self.sections = {}  #: Sections in the model.
        self.surfaces = {}  #: Surface definitions in the model.
        self.constraints = {}  #: Constraints in the model.
        self.constraintSets = {}  #: ConstraintsSets in the model.
        self.multiPointConstraints = {}  #: Multi-point (DOF-elimination) constraints in the model.
        self.modelModifiers = {}  #: Model modifiers (dynamic topology / mesh mutation entities) in the model.
        self._modelChangeObservers = []  #: Observers notified when the model is mutated (e.g. AMR).
        self.topologyVersion = 0  #: Bumped on every structural mutation; drives pull-based reconcile.
        self._changeLog = []  #: Recorded :class:`ModelChange` per mutation, newest last.
        self.contactFacetRecipes = {}  #: facet elSet name -> (surfaceName, prefix, triangulation).
        self.materials = {}  #: Materials in the model.
        self.analyticalFields = {}  #: AnalyticalFields in the model.
        self.scalarVariables = {}  #: ScalarVariables in the model.
        self.additionalParameters = {}  #: Additional information.
        self.rigidBodies = {}  #: RigidBodies in the model.
        self.domainSize = dimension  #: Spatial dimension of the model
        self.fieldOutputController = None  #: Set once by the driver; lets in-model entities (e.g. AMR markers) look up a named *fieldOutput by value, not just by declaration.

    def registerObserver(self, observer):
        """Register a :class:`~edelweissfe.models.modelchangeobserver.ModelChangeObserver` to be
        notified when the model is mutated (e.g. by adaptive mesh refinement)."""
        if not any(observer is obs for obs in self._modelChangeObservers):
            self._modelChangeObservers.append(observer)

    def unregisterObserver(self, observer):
        self._modelChangeObservers = [obs for obs in self._modelChangeObservers if obs is not observer]

    def notifyModelChanged(self, changeType, change: ModelChange = None):
        """Record a model mutation (bumping :attr:`topologyVersion`, so a pull-based consumer can
        catch up later via :meth:`changesSince`) and notify all registered push observers.

        Parameters
        ----------
        changeType
            The :class:`~edelweissfe.models.modelchangeobserver.ModelChangeType` of the mutation.
        change
            The structured :class:`ModelChange` describing what changed. If omitted, an empty one
            (bare ``changeType`` marker only, e.g. for a modifier that hasn't adopted the changeset
            yet) is recorded instead.
        """
        self.topologyVersion += 1
        if change is None:
            change = ModelChange(kind=changeType)
        change.version = self.topologyVersion
        self._changeLog.append(change)
        for observer in list(self._modelChangeObservers):
            observer.onModelChanged(self, changeType, change)

    def changesSince(self, version: int) -> ModelChange:
        """The :class:`ModelChange` coalesced across every mutation recorded after ``version``, or
        ``None`` if the model hasn't changed since. A pull-based consumer compares its own
        last-seen version against :attr:`topologyVersion` and, on a mismatch, reconciles from this,
        then adopts the new :attr:`topologyVersion` as its own last-seen version."""
        if version >= self.topologyVersion:
            return None
        return coalesce([c for c in self._changeLog if c.version > version])

    def _populateNodeFieldVariablesFromElements(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        elements.
        """
        for element in self.elements.values():
            for node, elementNodeFields in zip(element.nodes, element.fields):
                for field in elementNodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromConstraints(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        constraints.
        """

        for constraint in self.constraints.values():
            for node, nodeFields in zip(constraint.nodes, constraint.fieldsOnNodes):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _createNodeFieldsFromNodes(self, nodes: list, nodeSets: list) -> dict[str, NodeField]:
        """Bundle nodal FieldVariables together in contiguous NodeFields.

        Parameters
        ----------
        nodes
            The list of Nodes from which the NodeFields should be created.
        nodeSets
            The list of NodeSets, which should be considered in the index map of the NodeFields.

        Returns
        -------
        dict[str,NodeField]
            The dictionary containing the NodeField instances for every active field."""

        domainSize = self.domainSize

        theNodeFields = dict()
        for field in phenomena.keys():
            theNodeField = NodeField(field, getFieldSize(field, domainSize), nodes)

            if theNodeField.nodes:
                theNodeFields[field] = theNodeField

        return theNodeFields

    def _linkFieldVariableObjects(self, nodes):
        """Link NodeFields to individual FieldVariable objects.

        Parameters
        ----------
        nodes
            Nodes to be linked
        """

        for node in nodes:
            for field, fieldVariable in node.fields.items():
                nodeField = self.nodeFields[field]
                idx = nodeField._indicesOfNodesInArray[node]
                fieldVariable.values = self.nodeFields[field]["U"][idx, :]

        return

    def _requestAdditionalScalarVariable(self, name: str):
        """Create a new scalar variables

        Parameters
        ----------
        name
            The name of the variable.

        Returns
        -------
        ScalarVariable
            The instance of the variable.
        """
        if name in self.scalarVariables:
            raise Exception("ScalarVariable with name {:} already exists!".format(name))

        self.scalarVariables[name] = ScalarVariable()
        return self.scalarVariables[name]

    def _createAndAssignScalarVariableForConstraints(self, journal: Journal):
        """Create ScalarVariables for constraints.

        Parameters
        ----------
        journal
            The journal intance.
        """
        # we may have additional scalar degrees of freedom, not associated with any node (e.g, lagrangian multipliers of constraints)

        for constraintName, constraint in self.constraints.items():
            nAdditionalScalarVariables = constraint.getNumberOfAdditionalNeededScalarVariables()
            if nAdditionalScalarVariables > 0:
                journal.message(
                    "Constraint {:} requests {:} additional ScalarVariables".format(
                        constraintName, nAdditionalScalarVariables
                    ),
                    self.identification,
                    2,
                )

                scalarVariables = [
                    self._requestAdditionalScalarVariable("{:}_{:}".format(constraintName, i))
                    for i in range(nAdditionalScalarVariables)
                ]

                constraint.assignAdditionalScalarVariables(scalarVariables)

    def _prepareVariablesAndFields(self, journal):
        """Prepare all variables and fields for a simulation.

        Parameters
        ----------
        journal
            The journal instance.
        """
        journal.message(
            "Activating fields on nodes from Elements and Constraints",
            self.identification,
        )
        self._populateNodeFieldVariablesFromElements()
        self._populateNodeFieldVariablesFromConstraints()

        journal.message("Bundling fields on nodes to NodeFields", self.identification)
        self.nodeFields = self._createNodeFieldsFromNodes(self.nodeSets["all"], self.nodeFields.values())

        journal.message("Assembling ScalarVariables", self.identification)
        self.scalarVariables = dict()
        self._createAndAssignScalarVariableForConstraints(journal)

    def _resizeNodeFieldsForNodes(self, journal: Journal):
        """Resize the existing NodeFields in place for the current ``self.nodeSets["all"]``,
        instead of rebuilding :attr:`nodeFields` from scratch as :meth:`_prepareVariablesAndFields`
        does. Used by mesh mutators (e.g. AMR's ``hadaptivity._materialize``) so that NodeField
        identity -- and hence any :class:`~edelweissfe.fields.nodefield.NodeFieldSubset` or
        reference a consumer cached -- survives a topology change. ScalarVariables (e.g. Lagrange
        multipliers of constraints) are rebuilt, but values are preserved by name for constraints
        that still exist, so their converged state survives the refinement too.

        Parameters
        ----------
        journal
            The journal instance.
        """
        journal.message(
            "Activating fields on nodes from Elements and Constraints",
            self.identification,
        )
        self._populateNodeFieldVariablesFromElements()
        self._populateNodeFieldVariablesFromConstraints()

        journal.message("Resizing NodeFields", self.identification)
        nodes = self.nodeSets["all"]
        for nodeField in self.nodeFields.values():
            nodeField.resize(nodes)

        # a phenomenon activated for the first time (e.g. by a newly materialized constraint) has
        # no NodeField yet -- create it exactly as _prepareVariablesAndFields would
        for field in phenomena.keys():
            if field not in self.nodeFields:
                newNodeField = NodeField(field, getFieldSize(field, self.domainSize), nodes)
                if newNodeField.nodes:
                    self.nodeFields[field] = newNodeField

        journal.message("Assembling ScalarVariables", self.identification)
        # rebuilding scalarVariables cold-starts every value (ScalarVariable() defaults to 0.0), which
        # would silently discard converged Lagrange-multiplier values on every AMR refinement, even
        # though node fields are warm-started. Snapshot by name and restore the overlapping subset so
        # unchanged constraints keep their converged multiplier and only genuinely new ones cold-start.
        previousScalarVariableValues = {name: v.value for name, v in self.scalarVariables.items()}
        self.scalarVariables = dict()
        self._createAndAssignScalarVariableForConstraints(journal)
        for name, v in self.scalarVariables.items():
            if name in previousScalarVariableValues:
                v.value = previousScalarVariableValues[name]

    def _prepareElements(self, journal: Journal):
        """Prepare elements for a simulation.
        In detail, sections are assigned.


        Parameters
        ----------
        journal
            The journal instance.
        """
        for section in self.sections.values():
            section.assignSectionPropertiesToModel(self)

        # check if all elements are assigned a material
        materialAssigned = np.fromiter(map(attrgetter("hasMaterial"), self.elements.values()), dtype=bool)
        if not materialAssigned.all():
            elementIds = np.array([str(elId) for elId in self.elements.keys()])[np.logical_not(materialAssigned)]
            raise Exception(f"No material was assigned to element(s) with id(s) {', '.join(elementIds)}.")

    def prepareYourself(self, journal: Journal):
        """Prepare the model for a simulation.
        Creates the variables, bundles the fields,
        and initializes elements.


        Parameters
        ----------
        journal
            The journal instance.
        """

        self._prepareVariablesAndFields(journal)
        self._prepareElements(journal)

    def advanceToTime(self, time: float):
        """Accept the current state of the model and sub instances, and
        set the new time.

        Parameters
        ----------
        time
            The new time.
        """

        self.time = time

        for el in self.elements.values():
            el.acceptLastState()

        for constraint in self.constraints.values():
            constraint.acceptLastState()

    def writeRestart(self, restartFile: h5py.File):
        """Write the current (converged) state of the model to a restart checkpoint.

        Does not serialize model topology (nodes/elements/sets/sections/materials) -- only state
        that a rebuild from the original ``.inp`` file cannot reproduce: node field values, element
        quadrature-point history, scalar variables (e.g. Lagrange multipliers), and stateful
        constraints' internal history (e.g. frictional contact).

        Parameters
        ----------
        restartFile
            An open, writable :class:`h5py.File` (or group) to write the checkpoint into.
        """

        f = restartFile

        f.attrs["time"] = self.time

        nodeFieldsGroup = f.create_group("nodeFields")
        for nf in self.nodeFields.values():
            nodeFieldGroup = nodeFieldsGroup.create_group(nf.name)
            for entryName, entryValues in nf._values.items():
                nodeFieldGroup.create_dataset(entryName, data=entryValues)

        scalarVariablesGroup = f.create_group("scalarVariables")
        for name, scalarVariable in self.scalarVariables.items():
            scalarVariablesGroup.attrs[name] = scalarVariable.value

        elementsGroup = f.create_group("elements")
        for elNumber, element in self.elements.items():
            try:
                stateVars = element.getStateVars()
            except NotImplementedError:
                continue
            elementsGroup.create_dataset(str(elNumber), data=stateVars)

        constraintsGroup = f.create_group("constraints")
        for name, constraint in self.constraints.items():
            restartData = constraint.getRestartData()
            if restartData is None:
                continue
            constraintGroup = constraintsGroup.create_group(name)
            for entryName, entryValues in restartData.items():
                constraintGroup.create_dataset(entryName, data=entryValues)

        modelModifiersGroup = f.create_group("modelModifiers")
        for name, modifier in self.modelModifiers.items():
            restartData = modifier.getRestartData()
            if restartData is None:
                continue
            modifierGroup = modelModifiersGroup.create_group(name)
            for entryName, entryValues in restartData.items():
                modifierGroup.create_dataset(entryName, data=entryValues)

    def readRestart(self, restartFile: h5py.File):
        """Read the state of the model from a restart checkpoint written by :meth:`writeRestart`.

        The model must already have been rebuilt from the original ``.inp`` file (same topology)
        and :meth:`prepareYourself` called, before this is called.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        """

        f = restartFile

        self.time = f.attrs["time"]

        # Must run before every restore below: unlike a constraint's setRestartData (passive), a
        # model modifier's setRestartData (e.g. AMR replaying past refinements) can materialize
        # elements/nodes a plain rebuild from the .inp file cannot reproduce -- the node-field and
        # element-statevar restores that follow address them by label and would silently skip
        # anything not already in self.elements/self.nodeFields by this point.
        for name, modifier in self.modelModifiers.items():
            if name not in f["modelModifiers"]:
                continue
            restartData = {entryName: values[:] for entryName, values in f["modelModifiers"][name].items()}
            modifier.setRestartData(self, restartData)

        for nf in self.nodeFields.values():
            for entryName, entryValues in nf._values.items():
                nf[entryName][:] = f["nodeFields"][nf.name][entryName]

        for name, scalarVariable in self.scalarVariables.items():
            scalarVariable.value = f["scalarVariables"].attrs[name]

        # Elements a model modifier already restored itself (by a stable key -- see
        # ModelModifierBase.restoredElementLabels). Their numbers are reassigned by the replay, so
        # restoring them again by number would hand them another element's history, or hit a
        # stateless facet element that happens to hold the number now.
        alreadyRestored = set()
        for modifier in self.modelModifiers.values():
            alreadyRestored |= set(getattr(modifier, "restoredElementLabels", frozenset()))

        for elNumber, element in self.elements.items():
            elementKey = str(elNumber)
            if elementKey not in f["elements"] or elNumber in alreadyRestored:
                continue
            try:
                element.setStateVars(f["elements"][elementKey][:])
            except NotImplementedError:
                # This number belonged to a stateful element when the checkpoint was written and to
                # a stateless one now -- nothing sensible to restore.
                continue

        for name, constraint in self.constraints.items():
            if name not in f["constraints"]:
                continue
            restartData = {entryName: values[:] for entryName, values in f["constraints"][name].items()}
            constraint.setRestartData(restartData)


def printPrettyModelSummary(model: FEModel, journal: Journal):
    identification = "PrettyModelSummary"

    def wrapList(theList):
        for line in textwrap.wrap("[" + ", ".join(theList) + "]"):
            journal.message(
                "  {:<20} ".format(line),
                identification,
                0,
            )

    journal.message(
        "Finite element model with spatial dimension {:} has".format(model.domainSize),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15} ".format("nodes:", len(model.nodes)),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15} ".format("node sets:", len(model.nodeSets)),
        identification,
        0,
    )
    wrapList(model.nodeSets.keys())
    journal.message(
        " {:<20}{:<15} ".format("node fields:", len(model.nodeFields)),
        identification,
        0,
    )
    wrapList(["{:} ({:})".format(k, len(v.nodes)) for k, v in model.nodeFields.items()])
    journal.message(
        " {:<20}{:<15}".format("elements: ", len(model.elements)),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15}".format("element sets: ", len(model.elementSets)),
        identification,
        0,
    )
    wrapList(model.elementSets.keys())
    if model.constraints:
        journal.message(
            " {:<20}{:<15}".format("constraints: ", len(model.constraints)),
            identification,
            0,
        )
    if model.scalarVariables:
        journal.message(
            " {:<20}{:<15}".format("scalar variables: ", len(model.scalarVariables)),
            identification,
            0,
        )
    journal.message(
        " {:<20}{:<15}".format("materials: ", len(model.materials)),
        identification,
        0,
    )
    wrapList(model.materials.keys())

    if model.analyticalFields:
        journal.message(
            " {:<20}{:<15}".format("analytical fields: ", len(model.analyticalFields)),
            identification,
            0,
        )
        wrapList(model.analyticalFields.keys())
