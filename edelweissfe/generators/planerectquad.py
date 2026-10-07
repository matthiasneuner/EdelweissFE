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
# Created on Wed Apr 12 15:41:51 2017

# @author: Matthias Neuner
"""

A mesh generator, for rectangular geometries and structured quad meshes:


.. code-block:: console

        <-----l----->
         nX elements
         __ __ __ __
        |__|__|__|__|  A
        |__|__|__|__|  |
        |__|__|__|__|  | h
        |__|__|__|__|  | nY elements
      | |__|__|__|__|  |
      | |__|__|__|__|  V
    x0|_____
      y0

nSets, elSets, surface : 'name'_top, _bottom, _left, _right, ...
are automatically generated

Datalines:
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.config.elementlibrary import createPrototypeElement
from edelweissfe.generators.base.generatorbase import GeneratorBase, isNodeOfElements
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class PlaneRectQuadSchema:
    """The options this generator accepts, owned by this module and never mutated from outside
    it.

    ``length`` is spelled ``l`` in the input file -- a single-letter option name flake8 flags as
    ambiguous if used directly as a field/variable name, hence the ``optionName`` indirection.
    ``elType`` is declared ``required=True`` explicitly, but is still given a ``default=None`` so
    the schema remains constructible for the constructor's default argument.
    """

    x0: float = schemaField(description="Origin along the x axis.", dtype=float, default=0.0)
    y0: float = schemaField(description="Origin along the y axis.", dtype=float, default=0.0)
    z0: float = schemaField(description="Origin along the z axis.", dtype=float, default=0.0)
    length: float = schemaField(description="Height of the body.", dtype=float, default=1.0, optionName="l")
    h: float = schemaField(description="Length of the body.", dtype=float, default=1.0)
    nX: int = schemaField(description="Number of elements along the x axis.", dtype=int, default=1)
    nY: int = schemaField(description="Number of elements along the y axis.", dtype=int, default=1)
    nZ: int = schemaField(description="Number of elements along the z axis.", dtype=int, default=1)
    elType: str | None = schemaField(description="Element type.", dtype=str, default=None, required=True)
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


class Generator(GeneratorBase):
    """A mesh generator for rectangular geometries and structured quad meshes."""

    #: It only describes the mesh; see GeneratorBase.replicatedElementsReason.
    replicatedElementsReason = None

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = PlaneRectQuadSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: PlaneRectQuadSchema = PlaneRectQuadSchema(),
    ):
        """Constructible standalone, with no parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            The name of this generator instance, used as the prefix for the generated sets.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal object for logging.
        configuration
            The options this generator accepts; ``elType`` is still required, see
            :class:`PlaneRectQuadSchema`.
        """
        x0 = configuration.x0
        y0 = configuration.y0

        l = configuration.length  # noqa: E741
        h = configuration.h

        nX = configuration.nX
        nY = configuration.nY

        elTypeName = configuration.elType
        elProvider = configuration.elProvider

        # the number of nodes of the element type decides the node grid
        prototype = createPrototypeElement(elTypeName, elProvider)
        if prototype.nNodes == 4:
            nNodesX = nX + 1
            nNodesY = nY + 1

        if prototype.nNodes == 8:
            nNodesX = 2 * nX + 1
            nNodesY = 2 * nY + 1

        grid = np.mgrid[
            x0 : x0 + l : nNodesX * 1j,
            y0 : y0 + h : nNodesY * 1j,
        ]

        # Node labels come from the model's monotonic allocator (TopologyPipeline.reserveNodeNumbers), not
        # from max(model.nodes). Every grid position becomes a model node here, so the whole grid is
        # reserved as one batch.
        nodes = []
        reservedNodeLabels = iter(model.topology.reserveNodeNumbers(nNodesX * nNodesY))

        for x in range(nNodesX):
            for y in range(nNodesY):
                node = Node(next(reservedNodeLabels), grid[:, x, y])
                model.createNode(node)
                nodes.append(node)

        nG = np.asarray(nodes).reshape(nNodesX, nNodesY)

        # Element numbers come from the model's monotonic allocator (TopologyPipeline.reserveElementNumbers),
        # not from max(model.elements). Reserved one at a time so the count need not be predicted;
        # nothing else mints during this loop, so the numbers are consecutive exactly as before.

        elements = []
        connectivity = []
        for x in range(nX):
            for y in range(nY):
                (currentElementLabel,) = model.topology.reserveElementNumbers(1)
                if prototype.nNodes == 4:
                    elNodes = [nG[x, y], nG[x + 1, y], nG[x + 1, y + 1], nG[x, y + 1]]

                elif prototype.nNodes == 8:
                    elNodes = [
                        nG[2 * x, 2 * y],
                        nG[2 * x + 2, 2 * y],
                        nG[2 * x + 2, 2 * y + 2],
                        nG[2 * x, 2 * y + 2],
                        nG[2 * x + 1, 2 * y],
                        nG[2 * x + 2, 2 * y + 1],
                        nG[2 * x + 1, 2 * y + 2],
                        nG[2 * x, 2 * y + 1],
                    ]
                nodeLabels = [node.label for node in elNodes]
                model.mesh.addElement(currentElementLabel, elTypeName, elProvider, nodeLabels)
                elements.append(currentElementLabel)
                connectivity.append(nodeLabels)

        # nodesets:
        model.nodeSets["{:}_all".format(name)] = NodeSet(
            "{:}_all".format(name), np.ravel(nG)[np.ravel(isNodeOfElements(nG, connectivity))]
        )

        model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), np.ravel(nG[0, :]))
        model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), np.ravel(nG[-1, :]))
        model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), np.ravel(nG[:, -1]))
        model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), np.ravel(nG[:, 0]))

        model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), nG[0, 0])
        model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), nG[0, -1])
        model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), nG[-1, 0])
        model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), nG[-1, -1])

        # element sets
        elGrid = np.asarray(elements).reshape(nX, nY)
        model.mesh.setElementSet("{:}_bottom".format(name), np.ravel(elGrid[:, 0]))
        model.mesh.setElementSet("{:}_top".format(name), np.ravel(elGrid[:, -1]))
        model.mesh.setElementSet("{:}_central".format(name), [elGrid[int(nX / 2), int(nY / 2)]])
        model.mesh.setElementSet("{:}_right".format(name), np.ravel(elGrid[-1, :]))
        model.mesh.setElementSet("{:}_left".format(name), np.ravel(elGrid[0, :]))

        nShearBand = min(nX, nY)
        if nShearBand > 3:
            shearBand = [
                elGrid[int(nX / 2 + i - nShearBand / 2), int(nY / 2 + i - nShearBand / 2)] for i in range(nShearBand)
            ]
            model.mesh.setElementSet("{:}_shearBand".format(name), [e for e in shearBand])
            model.mesh.setElementSet(
                "{:}_shearBandCenter".format(name),
                [e for e in shearBand[int(nShearBand / 2) - 1 : int(nShearBand / 2) + 2]],
            )

        model.mesh.setElementSet("{:}_sandwichHorizontal".format(name), np.ravel(elGrid[1:-1, :]))

        model.mesh.setElementSet("{:}_sandwichVertical".format(name), np.ravel(elGrid[:, 1:-1]))

        model.mesh.setElementSet("{:}_core".format(name), np.ravel(elGrid[1:-1, 1:-1]))

        model.mesh.setElementSet("{:}_all".format(name), np.ravel(elGrid))
        # surfaces
        surfaceName = "{:}_bottom".format(name)
        model.mesh.addSurface(surfaceName, {1: "{:}_bottom".format(name)})
        surfaceName = "{:}_top".format(name)
        model.mesh.addSurface(surfaceName, {3: "{:}_top".format(name)})
        surfaceName = "{:}_right".format(name)
        model.mesh.addSurface(surfaceName, {2: "{:}_right".format(name)})
        surfaceName = "{:}_left".format(name)
        model.mesh.addSurface(surfaceName, {4: "{:}_left".format(name)})
