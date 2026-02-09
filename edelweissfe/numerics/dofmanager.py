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
This module contains important classes for describing the global equation system by means of a sparse system.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np

from edelweissfe.config.phenomena import phenomena


class VIJSystemMatrix(np.ndarray):
    """
    This class represents the V Vector of VIJ triple (sparse matrix in COO format),
    which

      * also contains the I and J vectors as class members,
      * allows to directly access (contiguous read and write) access of each entity via the [] operator

    Parameters
    ----------
    nDof
        The size of the system.
    I
        The I vector for the VIJ triple.
    J
        The J vector for the VIJ triple.
    entitiesInVIJ
        A dictionary containing the indices of an entitiy in the value vector.
    """

    def __new__(cls, nDof: int, I: np.ndarray, J: np.ndarray, entitiesInVIJ: dict):  # noqa: E741
        obj = np.zeros_like(I, dtype=float).view(cls)

        obj.nDof = nDof
        obj.I = I  # noqa: E741
        obj.J = J
        obj.entitiesInVIJ = entitiesInVIJ

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.nDof = getattr(obj, "nDof", None)
        self.I = getattr(obj, "I", None)  # noqa: E741
        self.J = getattr(obj, "J", None)  # noqa: E741
        self.entitiesInVIJ = getattr(obj, "entitiesInVIJ", None)

    def __getitem__(self, key):
        if isinstance(key, (int, slice, np.ndarray, list)):
            return super().__getitem__(key)

        try:
            # Entity Lookup
            idxInVIJ = self.entitiesInVIJ[key]
            # Use local entity size (key.nDof) for the slice, not global nDof
            size = key.nDof**2
            return super().__getitem__(slice(idxInVIJ, idxInVIJ + size))
        except (KeyError, TypeError, AttributeError):
            # Fallback for any other weird key types
            return super().__getitem__(key)


class ScatterDofVector(np.ndarray):
    """
    A Scatter Vector that stores data for entities contiguously.
    Includes a fast lookup map to support random access by Entity.

    Parameters
    ----------
    entitiesInDofVector
        The dictionary mapping entities to their indices in the DofVector.
    nDof
        The total number of degrees of freedom.
    """

    def __new__(cls, entitiesInDofVector: dict, nDof: int):
        entities = list(entitiesInDofVector.keys())

        sizes = np.array([len(v) for v in entitiesInDofVector.values()], dtype=np.intc)
        total_size = np.sum(sizes)

        obj = np.zeros(total_size, dtype=float).view(cls)

        offsets = np.zeros(len(entities) + 1, dtype=np.intc)
        np.cumsum(sizes, out=offsets[1:])

        obj._offset_map = dict(zip(entities, offsets))

        obj._global_indices = np.empty(total_size, dtype=np.int32)

        current_offset = 0
        for entity, indices in entitiesInDofVector.items():
            n = len(indices)
            obj._global_indices[current_offset : current_offset + n] = indices
            current_offset += n

        obj._entitiesInDofVector = entitiesInDofVector
        obj._nDof = nDof

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._offset_map = getattr(obj, "_offset_map", None)
        self._entitiesInDofVector = getattr(obj, "_entitiesInDofVector", None)
        self._nDof = getattr(obj, "_nDof", None)
        self._global_indices = getattr(obj, "_global_indices", None)

    def __getitem__(self, key):
        """
        Returns a VIEW into the expanded buffer.

        Parameters
        ----------
        key
            The key for indexing, either an entity or an integer index.
        """
        if isinstance(key, (int, slice, np.ndarray, list)):
            return super().__getitem__(key)

        try:
            val = self._offset_map[key]
            size = len(self._entitiesInDofVector[key])
            return super().__getitem__(slice(val, val + size))
        except (KeyError, TypeError):
            return super().__getitem__(key)

    def assembleInto(self, targetDofVector, absolute=False):
        """Scatter-Add into the global vector.

        Parameters
        ----------
        targetDofVector
            The target DofVector to assemble into.
        absolute
            If True, assemble the absolute values.
        """
        data = np.abs(self) if absolute else self
        np.add.at(targetDofVector, self._global_indices, data)

    def toDofVector(self, absolute=False) -> "DofVector":
        """Create a new DofVector from this scatter vector.

        Parameters
        ----------
        absolute
            If True, use absolute values.

        Returns
        -------
        DofVector
            The new DofVector.
        """
        new_dof_vector = DofVector(self._nDof, self._entitiesInDofVector)
        self.assembleInto(new_dof_vector, absolute=absolute)
        return new_dof_vector


class DofVector(np.ndarray):
    """
    Represents a Dof Vector with entity-aware indexing.

    Parameters
    ----------
    nDof
        The total number of degrees of freedom.
    entitiesInDofVector
        A dictionary mapping entities to their indices in the DofVector.
    """

    def __new__(cls, nDof: int, entitiesInDofVector: dict):
        obj = np.zeros(nDof, dtype=float).view(cls)
        obj.entitiesInDofVector = entitiesInDofVector
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.entitiesInDofVector = getattr(obj, "entitiesInDofVector", None)

    def __getitem__(self, key):
        if isinstance(key, (int, slice, np.ndarray, list)):
            return super().__getitem__(key)

        try:
            return super().__getitem__(self.entitiesInDofVector[key])
        except (KeyError, TypeError):
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice, np.ndarray, list)):
            super().__setitem__(key, value)
            return

        try:
            super().__setitem__(self.entitiesInDofVector[key], value)
        except (KeyError, TypeError):
            super().__setitem__(key, value)

    def copy(self, order="C"):
        """
        Create a copy of this DofVector.

        Parameters
        ----------
        order
            The memory layout order.
        Returns
        -------
        DofVector
            The copied DofVector.
        """
        newDofVector = super().copy(order).view(DofVector)
        if self.entitiesInDofVector is not None:
            newDofVector.entitiesInDofVector = self.entitiesInDofVector.copy()
        return newDofVector

    def createScatterVector(self) -> ScatterDofVector:
        """
        Create a scatter vector for ALL entities in this DofVector.

        Returns
        -------
        ScatterDofVector
            The ScatterDofVector.
        """
        return ScatterDofVector(self.entitiesInDofVector, self.size)


class DofManager:
    """
    The DofManager.

    * analyzes the domain (nodes and constraints),
    * collects information about the necessary structure of the degrees of freedom
    * handles the active fields on each node
    * counts the accumulated number of associated elements on each dof
    * supplies the framework with DofVectors and VIJSystemMatrices

    Parameters
    ----------
    nodeFields : list[NodeField]
        The list of NodeFields which should be represented in the DofVector structure.
    scalarVariables : list[ScalarVariable], optional
        The list of ScalarVariables which should be represented in the DofVector structure.
    elements : list[BaseNodeCouplingEntity], optional
        The list of Elements for which a map to the respective indices should be created.
    constraints : list[BaseNodeCouplingEntity], optional
        The list of Constraints for which a map to the respective indices should be created.
    nodeSets : list[NodeSet], optional
        The list of NodeSets for which a map to the respective indices should be created.
    initializeVIJPattern : bool, default=True
        Whether to initialize the I and J vectors for sparse matrix construction.
    """

    def __init__(
        self,
        nodeFields,
        scalarVariables=None,
        elements=None,
        constraints=None,
        nodeSets=None,
        initializeVIJPattern=True,
    ):
        if scalarVariables is None:
            scalarVariables = []
        if elements is None:
            elements = []
        if constraints is None:
            constraints = []
        if nodeSets is None:
            nodeSets = []

        self.nDof = 0
        self.fields = []
        self.idcsOfFieldVariablesInDofVector = {}
        self.idcsOfNodeFieldsInDofVector = {}
        self.idcsOfScalarVariablesInDofVector = {}
        self.indexToHostObjectMapping = {}

        self.accumulatedElementNDof = 0
        self.largestNumberOfElNDof = 0
        self._accumulatedElementVIJSize = 0

        self.accumulatedConstraintNDof = 0
        self.largestNumberOfConstraintNDof = 0
        self._accumulatedConstraintVIJSize = 0

        self.idcsOfFieldsOnNodeSetsInDofVector = {}
        self.idcsOfElementsInDofVector = {}
        self.idcsOfConstraintsInDofVector = {}

        # 1. Determine DOFs (Sequental, usually fast)
        self._determineDofsAndTheirIndices(nodeFields, scalarVariables)

        # 2. Parallel Information Gathering
        self._gatherInformationParallel(elements, constraints, nodeSets)

        self.idcsOfFieldsInDofVector = self.idcsOfNodeFieldsInDofVector
        self.fields = self.idcsOfFieldsInDofVector.keys()
        self.nAccumulatedNodalFluxesFieldwise = self._computeAccumulatedNodalFluxesFieldWise(self.fields)

        self.idcsOfBasicVariablesInDofVector = (
            self.idcsOfFieldVariablesInDofVector | self.idcsOfScalarVariablesInDofVector
        )
        self.idcsOfHigherOrderEntitiesInDofVector = self.idcsOfElementsInDofVector | self.idcsOfConstraintsInDofVector

        self._sizeVIJ = self._accumulatedElementVIJSize + self._accumulatedConstraintVIJSize

        if initializeVIJPattern:
            (self.I, self.J, self.idcsOfHigherOrderEntitiesInVIJ) = self._initializeVIJPattern()

    def _gatherInformationParallel(self, elements, constraints, nodeSets):
        """Internal helper to orchestrate parallel setup using ThreadPoolExecutor."""

        def get_chunks(lst, n):
            if not lst:
                return []
            # Convert dict_values or other views to a list for subscripting
            items = list(lst)
            n = max(1, n)
            return [items[i::n] for i in range(n)]

        workers = os.cpu_count() or 4

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 1. Chunking (Safe for lists, tuples, or dict_values)
            el_chunks = get_chunks(elements, workers)
            con_chunks = get_chunks(constraints, workers)

            # 2. Map metadata gathering
            el_meta = list(executor.map(self._gatherElementsInformation, el_chunks))
            con_meta = list(executor.map(self._gatherElementsInformation, con_chunks))

            # 3. Parallelize dictionary building
            self.idcsOfElementsInDofVector = self._parallel_map_entities(el_chunks, executor, "element")
            self.idcsOfConstraintsInDofVector = self._parallel_map_entities(con_chunks, executor, "constraint")

        # Aggregate scalar results
        (
            self.accumulatedElementNDof,
            self._accumulatedElementVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromElements,
            self.largestNumberOfElNDof,
        ) = self._aggregate_meta(el_meta)

        (
            self.accumulatedConstraintNDof,
            self._accumulatedConstraintVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromConstraints,
            self.largestNumberOfConstraintNDof,
        ) = self._aggregate_meta(con_meta)

        self.idcsOfFieldsOnNodeSetsInDofVector = self._locateFieldsOnNodeSetsInDofVector(nodeSets)

    def _parallel_map_entities(self, chunks, executor, mode):
        """Worker to parallelize dictionary creation using pre-split chunks."""
        if not chunks:
            return {}

        def worker(chunk):
            fvm = self.idcsOfFieldVariablesInDofVector
            svm = self.idcsOfScalarVariablesInDofVector
            local_res = {}
            for ent in chunk:
                if mode == "element":
                    # Logic for elements
                    indices = [
                        idx
                        for iNode, node in enumerate(ent.nodes)
                        for nodeField in ent.fields[iNode]
                        for idx in fvm[node.fields[nodeField]]
                    ]
                    arr = np.array(indices, dtype=int)
                    local_res[ent] = arr[ent.dofIndicesPermutation] if ent.dofIndicesPermutation is not None else arr
                else:
                    # Logic for constraints
                    node_indices = [
                        idx
                        for iNode, node in enumerate(ent.nodes)
                        for nodeField in ent.fieldsOnNodes[iNode]
                        for idx in fvm[node.fields[nodeField]]
                    ]
                    scalar_indices = [svm[v] for v in ent.scalarVariables]
                    local_res[ent] = np.array(node_indices + scalar_indices, dtype=int)
            return local_res

        results = executor.map(worker, chunks)
        merged = {}
        for r in results:
            merged.update(r)
        return merged

    def _aggregate_meta(self, results):
        """Aggregates the tuples returned by the parallel info-gathering workers."""
        if not results:
            return 0, 0, {k: 0 for k in phenomena.keys()}, 0
        sum_ndof = sum(r[0] for r in results)
        sum_vij = sum(r[1] for r in results)
        max_dof = max(r[3] for r in results)
        merged_flux = {k: 0 for k in phenomena.keys()}
        for r in results:
            for field, val in r[2].items():
                merged_flux[field] += val
        return sum_ndof, sum_vij, merged_flux, max_dof

    def _determineDofsAndTheirIndices(self, nodeFields, scalarVariables):
        """Loop over all nodes to generate the global field-dof indices."""
        currentIdx = 0

        # Reserve Space for Node Fields
        for nodeField in nodeFields:
            num_nodes = len(nodeField.nodes)
            dim = nodeField.dimension
            total_field_dofs = dim * num_nodes

            nextIdx = currentIdx + total_field_dofs
            self.idcsOfNodeFieldsInDofVector[nodeField.name] = slice(currentIdx, nextIdx)

            all_indices = np.arange(currentIdx, nextIdx, dtype=int).reshape(num_nodes, dim)
            for i, n in enumerate(nodeField.nodes):
                fv = n.fields[nodeField.name]
                self.idcsOfFieldVariablesInDofVector[fv] = all_indices[i, :]
                # Map back to node
                for idx in all_indices[i, :]:
                    self.indexToHostObjectMapping[idx] = n

            currentIdx = nextIdx

        # Reserve Space for Scalar Variables
        for scalarVar in scalarVariables:
            self.idcsOfScalarVariablesInDofVector[scalarVar] = currentIdx
            currentIdx += 1

        self.nDof = currentIdx

    def _gatherElementsInformation(self, entities):
        """Scalar information gathering for a chunk of entities."""
        accNDof = 0
        accVIJ = 0
        maxDof = 0
        fluxes = {k: 0 for k in phenomena.keys()}
        fvm = self.idcsOfFieldVariablesInDofVector

        for e in entities:
            accNDof += e.nDof
            accVIJ += e.nDof**2
            maxDof = max(e.nDof, maxDof)
            for node in e.nodes:
                for field, fv in node.fields.items():
                    indices = fvm.get(fv)
                    if indices is not None:
                        fluxes[field] += len(indices)
        return accNDof, accVIJ, fluxes, maxDof

    def _computeAccumulatedNodalFluxesFieldWise(self, fields):
        """Combines fieldwise fluxes from elements and constraints."""
        res = {}
        for f in fields:
            res[f] = self._nAccumulatedNodalFluxesFieldwiseFromElements.get(
                f, 0
            ) + self._nAccumulatedNodalFluxesFieldwiseFromConstraints.get(f, 0)
        return res

    def _locateFieldsOnNodeSetsInDofVector(self, nodeSets):
        """Locates indices for NodeSets field-wise."""
        res = {}
        fvm = self.idcsOfFieldVariablesInDofVector
        for field in self.idcsOfNodeFieldsInDofVector:
            res[field] = {}
            for nSet in nodeSets:
                indices_gen = chain.from_iterable(fvm[node.fields[field]] for node in nSet if field in node.fields)
                res[field][nSet] = np.fromiter(indices_gen, dtype=int)
        return res

    def _initializeVIJPattern(self):
        """Generate the COO pattern for system matrices."""
        entitiesInVIJ = {}
        I_ = np.zeros(self._sizeVIJ, dtype=np.intc)
        J_ = np.zeros(self._sizeVIJ, dtype=np.intc)
        idxInVIJ = 0

        for entity, idcs in self.idcsOfHigherOrderEntitiesInDofVector.items():
            entitiesInVIJ[entity] = idxInVIJ
            nDofE = len(idcs)
            size = nDofE**2
            # Use broadcasting for efficiency
            I_[idxInVIJ : idxInVIJ + size] = np.repeat(idcs, nDofE)
            J_[idxInVIJ : idxInVIJ + size] = np.tile(idcs, nDofE)
            idxInVIJ += size

        return I_, J_, entitiesInVIJ

    def constructVIJSystemMatrix(self):
        """Constructs a VIJSystemMatrix instance."""
        return VIJSystemMatrix(self.nDof, self.I, self.J, self.idcsOfHigherOrderEntitiesInVIJ)

    def constructDofVector(self):
        """Constructs a DofVector instance."""
        return DofVector(self.nDof, self.idcsOfHigherOrderEntitiesInDofVector)

    def writeDofVectorToNodeField(self, dofVector, nodeField, resultName):
        """
        Write the current values of an entire NodeField from the respective
        locations in a given DofVector.

        Parameters
        ----------
        dofVector : DofVector
            The source DofVector.
        nodeField : NodeField
            The NodeField to get the updated values.
        resultName : str
            The name of the value entries held by the NodeField.

        Returns
        -------
        NodeField
            The updated NodeField.
        """
        if resultName not in nodeField:
            nodeField.createFieldValueEntry(resultName)

        indices = self.idcsOfNodeFieldsInDofVector[nodeField.name]
        data = dofVector[indices]
        nodeField[resultName][:] = data.reshape((-1, nodeField.dimension))

        return nodeField

    def writeNodeFieldToDofVector(self, dofVector, nodeField, resultName, nodeSet=None):
        """
        Write the current values of an entire NodeField to the respective
        locations in a given DofVector.

        Parameters
        ----------
        dofVector : DofVector
            The result DofVector.
        nodeField : NodeField
            The NodeField holding the values.
        resultName : str
            The name of the value entries held by the NodeField.
        nodeSet : NodeSet, optional
            The NodeSet to consider. If None, all nodes of the NodeField
            are considered.

        Returns
        -------
        DofVector
            The DofVector.
        """
        if nodeSet is not None:
            # Check if we have the indices for this specific nodeSet already
            if nodeSet not in self.idcsOfFieldsOnNodeSetsInDofVector.get(nodeField.name, {}):
                new_map = self._locateFieldsOnNodeSetsInDofVector([nodeSet])
                if nodeField.name not in self.idcsOfFieldsOnNodeSetsInDofVector:
                    self.idcsOfFieldsOnNodeSetsInDofVector[nodeField.name] = {}
                self.idcsOfFieldsOnNodeSetsInDofVector[nodeField.name].update(new_map[nodeField.name])

            indices = self.idcsOfFieldsOnNodeSetsInDofVector[nodeField.name][nodeSet]
            dofVector[indices] = nodeField.subset(nodeSet)[resultName].flatten()

        else:
            indices = self.idcsOfNodeFieldsInDofVector[nodeField.name]
            dofVector[indices] = nodeField[resultName].flatten()

        return dofVector

    def getNodeForIndexInDofVector(self, index: int):
        """
        Find the node for a given index in the equation system.

        Parameters
        ----------
        index : int
            The index in the DofVector.

        Returns
        -------
        Node
            The attached Node object.
        """
        return self.indexToHostObjectMapping[index]
