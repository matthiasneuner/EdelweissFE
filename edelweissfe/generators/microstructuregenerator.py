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
"""
A mesh generator for generating a structure from a single unit cell mesh:
The unit cell mesh must be readable by meshio (e.g., in .inp format).

.. code-block:: edelweiss
    :caption: Generate meshes on the fly. Example:

    *job, name=job, domain=2d, solver=NIST

    *modelGenerator, generator=microstructuregenerator, name=gen
        unitCellMeshFile = myUnitCellMesh.inp
        nX      =4
        nY      =8
        nZ      =2
"""

import time
from dataclasses import dataclass

import meshio
import numpy as np

from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.schema import schemaField

identification = "microgen"


@dataclass(frozen=True)
class MicrostructureGeneratorSchema:
    """The options this generator accepts, owned by this module and never mutated from outside
    it.

    ``elType`` is declared ``required=True`` explicitly, but is still given a ``default=None`` so
    the schema remains constructible for the constructor's default argument.
    """

    unitCellMeshFile: str | None = schemaField(description="Path to the unit cell mesh file.", dtype=str, default=None)
    nX: int = schemaField(description="Number of cells along the x axis.", dtype=int, default=1)
    nY: int = schemaField(description="Number of cells along the y axis.", dtype=int, default=1)
    nZ: int = schemaField(description="Number of cells along the z axis.", dtype=int, default=1)
    elType: str | None = schemaField(description="Element type.", dtype=str, default=None, required=True)
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


class Generator(GeneratorBase):
    """A mesh generator for generating a structure from a single unit cell mesh."""

    #: It only describes the mesh; see GeneratorBase.replicatedElementsReason.
    replicatedElementsReason = None

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = MicrostructureGeneratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: MicrostructureGeneratorSchema = MicrostructureGeneratorSchema(),
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
            :class:`MicrostructureGeneratorSchema`.
        """
        journal.message("Generating microstructure mesh from unit cell mesh...", identification)

        unitCellMeshFile = configuration.unitCellMeshFile

        nX = configuration.nX
        nY = configuration.nY
        nZ = configuration.nZ

        unitCellMesh = meshio.read(unitCellMeshFile)

        nodes = np.array(unitCellMesh.points)
        elements = unitCellMesh.cells

        all_nodes = nodes.copy()
        all_elements = np.array([], dtype=int).reshape(0, elements[0].data.shape[1])
        block_elements_assignments = {}
        for i, block in enumerate(elements):
            # stacke all elements together
            all_elements = np.vstack((all_elements, block.data))
            # store element ids for each block
            block_elements_assignments[i] = list(range(len(all_elements) - len(block.data), len(all_elements)))
            # create an empty element set for each block
            model.mesh.setElementSet(f"{name}_block-{i + 1}", [])

        # print information about the unit cell mesh
        journal.message(f"Unit cell mesh has {len(all_nodes)} nodes and {len(all_elements)} elements.", identification)
        journal.message(
            f"Element blocks in unit cell mesh: {len(block_elements_assignments)} with assignments:", identification
        )
        for block_id, el_ids in block_elements_assignments.items():
            journal.message(f"block-{block_id + 1}: {len(el_ids)} elements", identification)

        # get unit cell dimensions
        x_min = np.min(nodes[:, 0])
        x_max = np.max(nodes[:, 0])
        y_min = np.min(nodes[:, 1])
        y_max = np.max(nodes[:, 1])

        lX = x_max - x_min
        lY = y_max - y_min

        # create the nodes and describe the elements of the unit cell
        _nodes = []
        for label, node in zip(model.topology.reserveNodeNumbers(len(all_nodes)), all_nodes):
            _node = Node(label, np.array(node))
            _nodes.append(_node)
            model.createNode(_node)

        idx = 0
        for block_id, el_ids in block_elements_assignments.items():
            elements_per_block = []
            for local_el_id in el_ids:
                (elNumber,) = model.topology.reserveElementNumbers(1)
                nodeLabels = [_nodes[nid].label for nid in all_elements[local_el_id]]
                model.mesh.addElement(elNumber, configuration.elType, configuration.elProvider, nodeLabels)
                # add element to corresponding element set
                elements_per_block.append(elNumber)
                idx += 1

            # in the order of the mesh -- that order reaches facet/element numbering downstream (see
            # hadaptivity.py)
            model.mesh.setElementSet(f"{name}_block-{block_id + 1}", elements_per_block)

        # replicate the mesh of the unit cell in x direction
        model = replicateMesh(
            model,
            direction=0,
            nReplications=nX,
            elTypeName=configuration.elType,
            elProvider=configuration.elProvider,
            journal=journal,
        )

        # replicate the already replicated mesh in y direction
        model = replicateMesh(
            model,
            direction=1,
            nReplications=nY,
            elTypeName=configuration.elType,
            elProvider=configuration.elProvider,
            journal=journal,
        )

        if model.domainSize == 3:
            # replicate the already replicated mesh in z direction
            model = replicateMesh(
                model,
                direction=2,
                nReplications=nZ,
                elTypeName=configuration.elType,
                elProvider=configuration.elProvider,
                journal=journal,
            )

        # create node sets for boundary conditions
        nSet_left = set()
        nSet_right = set()
        nSet_bottom = set()
        nSet_top = set()
        # add sets for corners as well
        nSet_top_left = set()
        nSet_top_right = set()
        nSet_bottom_left = set()
        nSet_bottom_right = set()

        # create node sets for left and bottom boundaries
        for nodeID, node in model.nodes.items():
            if np.isclose(node.coordinates[1], y_min, atol=1e-8):
                nSet_bottom.add(node)
                if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                    nSet_bottom_left.add(node)
                elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                    nSet_bottom_right.add(node)
            if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                nSet_left.add(node)
            if np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                nSet_right.add(node)
            if np.isclose(node.coordinates[1], y_max + (nY - 1) * lY, atol=1e-8):
                nSet_top.add(node)
                if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                    nSet_top_left.add(node)
                elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                    nSet_top_right.add(node)

        model.nodeSets[f"{name}_left"] = NodeSet(f"{name}_left", nSet_left)
        model.nodeSets[f"{name}_right"] = NodeSet(f"{name}_right", nSet_right)
        model.nodeSets[f"{name}_bottom"] = NodeSet(f"{name}_bottom", nSet_bottom)
        model.nodeSets[f"{name}_top"] = NodeSet(f"{name}_top", nSet_top)
        model.nodeSets[f"{name}_bottom_left"] = NodeSet(f"{name}_bottom_left", nSet_bottom_left)
        model.nodeSets[f"{name}_bottom_right"] = NodeSet(f"{name}_bottom_right", nSet_bottom_right)
        model.nodeSets[f"{name}_top_left"] = NodeSet(f"{name}_top_left", nSet_top_left)
        model.nodeSets[f"{name}_top_right"] = NodeSet(f"{name}_top_right", nSet_top_right)


def findInterfaceNodes(nodes, coordIndex, coordValue, idx_offset=0):
    interfaceNodes = set()
    ids = np.where(np.isclose(nodes[:, coordIndex], coordValue, atol=1e-5))[0] + idx_offset
    interfaceNodes.update(ids.tolist())
    return interfaceNodes


def replicateMesh(
    model: FEModel, direction: int, nReplications: int, elTypeName: str, elProvider: str, journal: Journal = None
) -> FEModel:
    """Replicate the mesh described so far ``nReplications`` times along ``direction``, merging the
    coincident nodes of neighbouring copies. Each element set of the mesh gains the copies of its
    elements.

    Assumes that the mesh holds only the elements (numbered 1..N) and nodes (labelled 1..M) of the
    generator calling it.
    """

    mesh = model.mesh
    all_elements_to_copy = [[label - 1 for label in record.nodeLabels] for record in mesh.elements.values()]
    all_nodes_to_copy = [model.nodes[i + 1].coordinates for i in range(len(model.nodes))]

    elements_in_block = {}
    # separate elements according to their blocks
    for elset_name, numbers in mesh.elementSets.items():
        elements_in_block[elset_name] = [number - 1 for number in numbers]

    all_nodes = np.array(all_nodes_to_copy)

    direction_min = np.min([node[direction] for node in all_nodes])
    direction_max = np.max([node[direction] for node in all_nodes])
    length_in_direction = direction_max - direction_min

    shift = np.zeros(len(all_nodes[0]))

    minNodes = np.array(
        [k for k, node in enumerate(all_nodes_to_copy) if np.isclose(node[direction], direction_min, atol=1e-8)]
    )

    nodes_to_shift = [(k, node) for k, node in enumerate(all_nodes_to_copy) if k not in minNodes]

    for j in range(1, nReplications):
        tic_total = time.time()
        # shift nodes
        shift[direction] = j * length_in_direction
        new_nodes = []
        associated_nodes = []

        for k, node in nodes_to_shift:
            new_nodes.append(node + shift)
            # len(model.nodes) + 1 was a positional guess, not an allocator, and would silently
            # overwrite a live node the moment the label range had a gap. The rest of this function
            # addresses nodes as `label - 1` into all_nodes, so keep the association in that form.
            (label,) = model.topology.reserveNodeNumbers(1)
            associated_nodes.append([k, label - 1])
            _node = Node(label, np.array(new_nodes[-1]))
            model.createNode(_node)

        new_nodes = np.array(new_nodes)

        all_nodes = np.vstack((all_nodes, new_nodes))

        # create (smaller) array to search in
        idx_offset = (j - 1) * (len(all_nodes_to_copy) - len(minNodes))
        search_array = all_nodes[idx_offset:, :]

        for i_old in minNodes:
            i_new_ = (
                np.where(np.all(np.abs(search_array - all_nodes_to_copy[i_old] - shift) < 1e-5, axis=1))[0][0]
                + idx_offset
            )
            associated_nodes.append([i_old, i_new_])
        associated_nodes_array = np.array(associated_nodes)

        # create elements per block
        for elset_name, el_ids in elements_in_block.items():
            newElements = []
            for local_el_id in el_ids:
                el = all_elements_to_copy[local_el_id]
                new_el = []
                for nid in el:
                    k = np.where(associated_nodes_array[:, 0] == nid)[0][0]
                    new_el.append(int(associated_nodes_array[k, 1]))
                # len(model.elements) + 1 was not merely unidiomatic: element numbers are never
                # recycled, so the dict has gaps, and len()+1 could land on a live element and
                # silently overwrite it.
                (elNumber,) = model.topology.reserveElementNumbers(1)
                mesh.addElement(elNumber, elTypeName, elProvider, [nid + 1 for nid in new_el])
                # Keep the number itself, not a positional guess: element numbers come from the
                # allocator and are never recycled, so len(mesh.elements) says nothing about which
                # number this element got (the same reason the creation above no longer uses it).
                newElements.append(elNumber)

            mesh.setElementSet(elset_name, mesh.elementSets[elset_name] + newElements)

        # remove nodes that are now internal
        toc_total = time.time()

        if journal:
            journal.message(f"Replication step {j}/{nReplications - 1} in direction {direction} done.", identification)
            journal.message(f" Total nodes so far: {len(model.nodes)}", identification)
            journal.message(f" Total elements so far: {len(mesh.elements)}", identification)
            journal.message(
                f" Total time for replication step: {round(toc_total - tic_total, 2)} seconds", identification
            )

    return model
