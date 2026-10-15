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


from edelweissfe.config import registry
from edelweissfe.config.generators import getGeneratorClass
from edelweissfe.config.solvers import getSolverByName
from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.steps.stepmanager import (
    StepActionDefinition,
    StepDefinition,
    StepManager,
)
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import (
    keywordIdentifier,
    moduleLevelKeywordIdentifier,
)
from edelweissfe.utils.math import createMathExpression, createModelAccessibleFunction
from edelweissfe.utils.misc import (
    convertLinesToStringDictionary,
    convertLineToStringDictionary,
    isInteger,
    parseDatalinesToArgsAndKwargs,
    strCaseCmp,
    strToRange,
    withoutParserBookkeeping,
)
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import DatalineAggregatingSchema, buildSchemaFromOptions


def flattenDefinitions(ll):
    flat = []
    for item in ll:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def createFieldOutputFromInputFile(inputfile: dict, model: FEModel, journal: Journal) -> FieldOutputController:
    """Convenience helper function
    to create the FieldOutputController instance using the *fieldOutput keyword.

    A ``perNode`` output covers the nodes of its ``nSet`` or ``elSet`` -- for an element set, the nodes
    of the whole set, also where a domain-decomposed process created none of its elements -- and the
    whole field only if it names no set. A set without nodes gives an empty output, never the whole
    field.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    model
        The FEModel tree instance.
    journal
        The Journal for logging purposes.

    Returns
    -------
    FieldOutputController
        The configured FieldOutputController instance.
    """
    fieldOutputController = FieldOutputController(model, journal)
    for definition in inputfile["fieldOutput"]:
        moduleOptions = definition["moduleoptions"]
        perNodeDefs = moduleOptions.get("perNode", [])
        perElementDefs = moduleOptions.get("perElement", [])
        fromExpressionDefs = moduleOptions.get("fromExpression", [])

        for definition in perNodeDefs:
            field = definition["field"]

            if sum(bool(definition[key]) for key in ("nSet", "elSet", "rigidBody")) > 1:
                raise Exception(
                    f"During parsing of keyword {keywordIdentifier}fieldOutput ({moduleLevelKeywordIdentifier}perNode): Specify at most one of nSet, elSet, or rigidBody."
                )

            f_of_x = definition["f(x)"]
            if f_of_x:
                f_of_x = createMathExpression(f_of_x)

            f_export_of_x = definition["f_export(x)"]
            if f_export_of_x:
                f_export_of_x = createMathExpression(f_export_of_x)

            if definition["rigidBody"]:
                # A rigid body's visualization nodes are not independent degrees of
                # freedom -- they are fully determined by its reference point -- so
                # this bypasses NodeFields entirely rather than treating them as a
                # subset of one. There is consequently no "P"/"dU"-like distinction
                # to select via `result` -- RigidBodyFieldOutput always returns the
                # field's current value -- so reject anything but the "U" (current
                # value) convention rather than silently ignoring the user's intent.
                if definition["result"] not in ("U", "u"):
                    raise Exception(
                        f"During parsing of keyword {keywordIdentifier}fieldOutput "
                        f"({moduleLevelKeywordIdentifier}perNode) '{definition['name']}': rigidBody field outputs "
                        f"only support result=U (the rigid body's current field value), got "
                        f"result={definition['result']!r}."
                    )

                if definition["rigidBody"] not in model.rigidBodies:
                    raise Exception(
                        f"During parsing of keyword {keywordIdentifier}fieldOutput "
                        f"({moduleLevelKeywordIdentifier}perNode) '{definition['name']}': rigidBody "
                        f"'{definition['rigidBody']}' is not defined. Available rigid bodies: "
                        f"{list(model.rigidBodies.keys())}."
                    )

                fieldOutputController.addRigidBodyFieldOutput(
                    name=definition["name"],
                    rigidBody=model.rigidBodies[definition["rigidBody"]],
                    field=field,
                    saveHistory=definition["saveHistory"],
                    f_x=f_of_x,
                    export=definition["export"],
                    fExport_x=f_export_of_x,
                )
                continue

            nodeField = model.nodeFields[field]

            subset = None
            if definition["nSet"]:
                subset = model.nodeSets[definition["nSet"]]
            elif definition["elSet"]:
                # The nodes of the whole set, through the set, so that they follow a refinement of the
                # set; from the mesh, also where a domain-decomposed process created only part of it.
                subset = model.elementSets[definition["elSet"]]

            # A named set selects its nodes, even if it has none: an empty set gives an empty output, not
            # the whole field. (An element set may also hold none of its elements in this process, and is
            # then empty -- but its nodes are still those of the whole set.)
            if subset is not None:
                nodeField = nodeField.subset(subset)

            fieldOutputController.addPerNodeFieldOutput(
                name=definition["name"],
                nodeField=nodeField,
                result=definition["result"],
                saveHistory=definition["saveHistory"],
                f_x=f_of_x,
                export=definition["export"],
                fExport_x=f_export_of_x,
            )

        for definition in perElementDefs:
            elSet = model.elementSets[definition["elSet"]]

            qp = definition["quadraturePoint"]
            quadraturePoints = strToRange(qp) if not isInteger(qp) else [int(qp)]

            f_of_x = definition["f(x)"]
            if f_of_x:
                f_of_x = createMathExpression(f_of_x)

            f_export_of_x = definition["f_export(x)"]
            if f_export_of_x:
                f_export_of_x = createMathExpression(f_export_of_x)

            fieldOutputController.addPerElementFieldOutput(
                name=definition["name"],
                elSet=elSet,
                result=definition["result"],
                quadraturePoints=quadraturePoints,
                saveHistory=definition["saveHistory"],
                f_x=f_of_x,
                export=definition["export"],
                fExport_x=f_export_of_x,
            )

        for definition in fromExpressionDefs:
            if definition["nSet"]:
                associatedSet = model.nodeSets[definition["nSet"]]
            elif definition["elSet"]:
                # An expression over an element set reads the whole set (ExpressionFieldOutput takes its
                # len()), which raises if only part of the set is local to this process.
                associatedSet = model.elementSets[definition["elSet"]]
            else:
                raise Exception(
                    f"During parsing of keyword {keywordIdentifier}fieldOutput ({moduleLevelKeywordIdentifier}fromExpression): All fieldOuputs must be associated with a set!"
                )

            theExpression = createModelAccessibleFunction(definition["expression"], model)

            f_of_x = definition["f(x)"]
            if f_of_x:
                f_of_x = createMathExpression(f_of_x)

            f_export_of_x = definition["f_export(x)"]
            if f_export_of_x:
                f_export_of_x = createMathExpression(f_export_of_x)

            fieldOutputController.addExpressionFieldOutput(
                associatedSet=associatedSet,
                theExpression=theExpression,
                name=definition["name"],
                saveHistory=definition["saveHistory"],
                f_x=f_of_x,
                export=definition["export"],
                fExport_x=f_export_of_x,
            )

    return fieldOutputController


def fillFEModelFromInputFile(model: FEModel, inputfile: dict, journal: Journal) -> FEModel:
    """Convenience helper function
    to fill an existing (possibly empty) FEModel using the input file and generators.

    First the mesh generators and the ``*node``/``*nSet``/``*element``/``*elset``/``*surface`` keywords
    describe the mesh (``model.mesh``); then the nodes and the elements are made from it
    (:meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh`,
    :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`) -- the elements those of them the model's
    :attr:`~edelweissfe.models.femodel.FEModel.elementDistribution` creates in this process: every
    element, unless a domain-decomposed run distributes them. Everything that refers to
    element objects (sections, element properties, contact facets, constraints, model modifiers)
    comes after. A generator that needs element objects while the mesh is still being described
    (``executePythonCode``, which runs arbitrary code on the model) makes the elements described so
    far itself.

    Parameters
    ----------
    FEModel
        The model tree to be filled.
    input
        The processed inputfile in dictionary form.
    journal
        The Journal for logging purposes.

    Returns
    -------
    FEModel
        The updated, filled model tree.
    """

    # Model setup is a topology change: it is the one phase besides a model modifier's own update in
    # which elements may be created (see TopologyPipeline.changes). Everything below runs inside it.
    with model.topology.changes():
        return _fillFEModelFromInputFile(model, inputfile, journal)


#: The keywords whose data lines describe the mesh: read once, into the model, by
#: :func:`fillFEModelFromInputFile`, and released afterwards by :func:`releaseMeshDataLines`.
MESH_KEYWORDS = ("node", "element", "nSet", "elSet")


class ReleasedDataLines:
    """The data lines of a mesh keyword after :func:`releaseMeshDataLines`: reading them raises, so
    that a parsed input file used twice fails loudly instead of building a model without a mesh.

    Parameters
    ----------
    keyword
        The keyword whose data lines were released.
    """

    def __init__(self, keyword: str):
        self.keyword = keyword

    def __iter__(self):
        """Raise: the data lines are gone.

        Raises
        ------
        RuntimeError
            Always.
        """

        self._raiseReleased()

    def __len__(self):
        """Raise: the data lines are gone (see :meth:`__iter__`)."""

        self._raiseReleased()

    def __getitem__(self, index):
        """Raise: the data lines are gone (see :meth:`__iter__`)."""

        self._raiseReleased()

    def __bool__(self):
        """Raise: the data lines are gone (see :meth:`__iter__`)."""

        self._raiseReleased()

    def _raiseReleased(self):
        """Raise the error of reading the released data lines.

        Raises
        ------
        RuntimeError
            Always.
        """

        raise RuntimeError(
            "the data lines of *{:} were released after the model was built from them (releaseMeshDataLines); "
            "parse the input file again to build another model".format(self.keyword)
        )


def releaseMeshDataLines(inputfile: dict):
    """Release the data lines describing the mesh (``*node``, ``*element``, ``*nSet``, ``*elSet``),
    once the model holds the mesh they describe.

    They are the bulk of a parsed input file -- about 50 MB for a mesh of 280,000 nodes -- and every
    process of a domain-decomposed run parsed the whole file. Nothing reads them after
    :func:`fillFEModelFromInputFile`: a resumed run rebuilds its model from the file again. Reading
    them afterwards raises (:class:`ReleasedDataLines`).

    Parameters
    ----------
    inputfile
        The processed input file in dictionary form; its mesh keywords are changed in place.
    """

    for keyword in MESH_KEYWORDS:
        for definition in inputfile[keyword]:
            definition["datalines"] = ReleasedDataLines(keyword)


def _fillFEModelFromInputFile(model: FEModel, inputfile: dict, journal: Journal) -> FEModel:
    """The body of :func:`fillFEModelFromInputFile`, run inside an open topology window."""

    # call individual optional model generators with executeAfterManualGeneration == True
    for definition in inputfile["modelGenerator"]:
        if definition.get("executeAfterManualGeneration", False):
            continue
        generatorDefinition = CaseInsensitiveDict(definition.copy())

        generatorType = generatorDefinition.pop("generator")
        data = generatorDefinition.pop("datalines")
        args, kwargs = parseDatalinesToArgsAndKwargs(data)

        # raw code lines must not be comma-split -- same special case as in the
        # executeAfterManualGeneration loop below
        if strCaseCmp(generatorType, "executePythoncode"):
            args = data

        generatorClass = getGeneratorClass(generatorType)
        model = generatorClass.fromGeneratorDefinition(generatorDefinition["name"], model, journal, args, kwargs)

    # the standard 'Abaqus like' model generator is invoked unconditionally, and it has direct access to the inputfile
    abqModelConstructor = AbqModelConstructor(journal)
    model = abqModelConstructor.createGeometryFromInputFile(model, inputfile)

    # The mesh is described: decide which of its elements this process creates -- every one, unless
    # a domain-decomposed run distributes them over its processes -- and make them, and the nodes
    # (every process holds every node, for now).
    model.elementDistribution.decideLocalElements(model.mesh, model.domainSize)
    model.createNodesOfMesh()
    model.createElementsOfMesh()

    # The base mesh is complete here, and it numbers its nodes and elements from the input file
    # rather than from the allocators (see TopologyPipeline.adoptSetupElementNumbers). Raise both allocators
    # above it now, so that everything created from this point on -- contact facets, rigid-body
    # point masses, and later every model modifier -- draws numbers that cannot collide with it.
    model.topology.adoptSetupElementNumbers()
    model.topology.adoptSetupNodeNumbers()

    model = abqModelConstructor.createMaterialsFromInputFile(model, inputfile)
    model = abqModelConstructor.createAdvancedMaterialsFromInputFile(model, inputfile)
    model = abqModelConstructor.createAnalyticalFieldsFromInputFile(model, inputfile)
    model = abqModelConstructor.createSectionsFromInputFile(model, inputfile)
    model = abqModelConstructor.createElementPropertiesFromInputFile(model, inputfile)

    # call individual optional model generators with executeAfterManualGeneration == False
    for definition in inputfile["modelGenerator"]:
        if not definition.get("executeAfterManualGeneration", False):
            continue
        generatorDefinition = CaseInsensitiveDict(definition.copy())

        generatorType = generatorDefinition.pop("generator")
        data = generatorDefinition.pop("datalines")
        args, kwargs = parseDatalinesToArgsAndKwargs(data)

        if strCaseCmp(generatorType, "executePythoncode"):
            args = data

        generatorClass = getGeneratorClass(generatorType)
        model = generatorClass.fromGeneratorDefinition(generatorDefinition["name"], model, journal, args, kwargs)

    # A late generator may describe further nodes and elements (e.g. a boxgen with
    # executeAfterManualGeneration): make them too, before anything refers to their objects again.
    model.createNodesOfMesh()
    model.createElementsOfMesh()

    model = abqModelConstructor.createConstraintsFromInputFile(model, inputfile)
    model = abqModelConstructor.createModelModifiersFromInputFile(model, inputfile)
    return model


def createStepManagerFromInputFile(inputfile: dict):
    """Convenience helper function
    to create and fill a StepManager using *step definitions from the input file.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.

    Returns
    -------
    StepManager
        The StepManager with enqueued StepDefinitions.
    """
    stepManager = StepManager()

    autoNameCounter = 0

    for stepOptions in inputfile["step"]:
        stepType = stepOptions.pop("type")
        stepActionLines = stepOptions.pop("moduleOptions")

        stepOptions.pop("inputfile")
        data = stepOptions.pop("datalines")

        stepActionDefinitions = []

        args, kwargs = parseDatalinesToArgsAndKwargs(data)

        if args:
            raise ValueError(
                f"Unexpected positional options {args} in the datalines of *step; only key=value options are allowed."
            )

        stepOptions.update(kwargs)

        for actionModule, definitions in stepActionLines.items():
            for definition in definitions:
                # metadata added by the parser, of no relevance for the step actions:
                definition.pop("inputfile", None)

                # unnamed actions are identified by their category (e.g., 'options'
                # step actions), or get an auto-generated unique name:
                name = definition.pop("name", None) or definition.get("category")
                if name is None:
                    name = f"{actionModule}-{autoNameCounter}"
                    autoNameCounter += 1

                stepActionDefinitions.append(StepActionDefinition(name, actionModule, definition))

        stepDefinition = StepDefinition(stepType, stepOptions, stepActionDefinitions)

        stepManager.enqueueStepDefinition(stepDefinition)

    return stepManager


def createSolversFromInputFile(inputfile: dict, jobInfo: dict, journal: Journal) -> dict:
    """Convenience helper function
    to create instances of nonlinear solvers using the *solver keyword.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    jobInfo
        Additional informations about the job
    journal
        The Journal instance for logging purposes.

    Returns
    -------
    dict
        The dictionary containing the solver instances.
    """
    solvers = {}
    for solverDefinition in inputfile["solver"]:
        try:
            solverName = solverDefinition["name"]
        except KeyError:
            raise KeyError("Solver definition missing name")
        try:
            solverType = solverDefinition["solver"]
        except KeyError:
            raise KeyError(f"Missing type definition for solver {solverName}. Specify solver type with solver=...")

        solverData = solverDefinition["datalines"]

        Solver = getSolverByName(solverType)

        solverData = convertLinesToStringDictionary(solverData)

        solvers[solverName] = Solver(jobInfo, journal, **solverData)

    return solvers


def createOutputManagersFromInputFile(
    inputfile: dict,
    defaultName: str,
    model: FEModel,
    fieldOutputController: FieldOutputController,
    journal: Journal,
    plotter: Plotter,
) -> list:
    """Convenience helper function
    to create output managers using the *output keyword.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    defaultName
        The default name prefix for output managers without specified name.
    fieldOutputController
        The FieldOutputController instance.
    journal
        The Journal instance for logging purposes.
    plotter
        The plotter instance.

    Returns
    -------
    list
        The list containing the OutputManager instances.
    """
    outputManagers = []

    for definition in inputfile["output"]:
        outputManagerKwargs = CaseInsensitiveDict(definition.copy())

        outputManagerType = outputManagerKwargs.pop("type")

        if outputManagerKwargs["name"] is not None:
            outputManagerName = outputManagerKwargs.pop("name")
        elif strCaseCmp(outputManagerType, "ensight"):
            outputManagerName = "esExport"
        else:
            outputManagerName = f"OutputManager-{len(outputManagers)}"

        datalines = outputManagerKwargs.pop("datalines")

        outputManagerKwargs.pop("inputfile")

        moduleOptions = outputManagerKwargs.pop("moduleOptions")

        outputManagerClass, outputManagerSchema = registry.lookup("outputmanager", outputManagerType)

        if outputManagerSchema is not None and issubclass(outputManagerSchema, DatalineAggregatingSchema):
            # One instance aggregating a heterogeneous list of jobs, one or more per dataline,
            # instead of one instance per dataline -- see `DatalineAggregatingSchema` for why the
            # dataline interpretation belongs to the schema rather than to this adapter.
            try:
                configuration = outputManagerSchema.fromDatalines(
                    [convertLineToStringDictionary(line) for line in datalines]
                )
            except ValueError as e:
                raise ValueError(
                    f"Error during parsing of keyword {keywordIdentifier}output "
                    f"(type={outputManagerType}): " + e.args[0]
                ) from e

            outputManagers.append(
                outputManagerClass(
                    outputManagerName,
                    model,
                    fieldOutputController,
                    journal,
                    plotter,
                    configuration=configuration,
                )
            )
            continue

        if outputManagerSchema is None:
            raise ValueError(
                f"Output manager '{outputManagerType}' declares no option schema. Declare one as a "
                "`schema` class attribute (see `edelweissfe.utils.schema.OptionSchemaProvider`). The "
                "pre-schema `outputManagerFactory` protocol was removed once the last built-in output "
                "manager was ported, so there is no fallback path any more."
            )

        # Look up the output manager's class and schema in the registry, then parse -> validate/coerce
        # into the module's own option schema -> call its constructor. `moduleOptions` carries the
        # nested `>>` sub-keyword blocks (ensight's `>>perNode`, `>>perElement`, `>>configuration`)
        # and is forwarded as the schema's sub-keyword source; for a schema that declares no
        # sub-keyword fields it is `{}`, and a stray block under such a keyword is rejected by
        # `buildSchemaFromOptions` rather than silently dropped.
        for optionsForOneManager in [datalines] if len(datalines) == 0 else datalines:
            args, kwargs = parseDatalinesToArgsAndKwargs(optionsForOneManager)

            try:
                configuration = buildSchemaFromOptions(
                    outputManagerSchema,
                    kwargs,
                    {name: withoutParserBookkeeping(blocks) for name, blocks in moduleOptions.items()},
                )
            except ValueError as e:
                raise ValueError(
                    f"Error during parsing of keyword {keywordIdentifier}output "
                    f"(type={outputManagerType}): " + e.args[0]
                ) from e

            outputManagers.append(
                outputManagerClass(
                    outputManagerName,
                    model,
                    fieldOutputController,
                    journal,
                    plotter,
                    configuration=configuration,
                )
            )

    return outputManagers


def createPlotterFromInputFile(inputfile: dict, journal: Journal) -> Plotter:
    """Convenience helper function
    to create and configure the Plotter instance using the *configurePlots and *exportPlots keywords

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    journal
        The Journal instance for logging purposes.

    Returns
    -------
    Plotter
        The resulting plotter instance
    """
    plotConfigurations = [
        convertLineToStringDictionary(c)
        for configEntry in inputfile["configurePlots"]
        for c in configEntry["datalines"]
    ]

    exportJobs = [
        convertLineToStringDictionary(c) for configEntry in inputfile["exportPlots"] for c in configEntry["datalines"]
    ]

    plotter = Plotter(journal, plotConfigurations, exportJobs)

    return plotter
