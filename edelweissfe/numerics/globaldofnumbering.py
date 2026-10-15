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
"""The global numbering of the degrees of freedom of a model, as data, from its mesh.

The :class:`~edelweissfe.numerics.dofmanager.DofManager` numbers the degrees of freedom of the node
objects it is given:

* field by field, each field taking a contiguous range of the global vector;
* within a field, its nodes in the order of the model's nodes -- which is mesh order;
* at each node, the components of the field;
* the scalar variables (e.g. Lagrange multipliers of constraints) last.

Which nodes carry which field follows from the mesh alone -- from the element types
(:meth:`~edelweissfe.models.mesh.Mesh.typeOf`) -- and from the fields the constraints couple. So the
same numbering can be written down without a single node object, as a few arrays per field:

``globalDof(field, node, component) = firstDof[field] + rankInField[field][node] * dimension[field] + component``

where ``rankInField[field]`` counts the nodes carrying the field, in mesh order. That is
:class:`GlobalDofNumbering`: about 4 bytes per node and field, the same in every process of a
domain-decomposed run, also in a process that holds only some of the nodes. A domain-decomposed run
checks it against the DofManager's numbering whenever the equation system is built
(:meth:`GlobalDofNumbering.requireAgreementWith`).
"""

import hashlib

import numpy as np

from edelweissfe.config.phenomena import getFieldSize, phenomena


class GlobalDofNumbering:
    """The global numbering of the degrees of freedom of a model, as data (see the module
    documentation).

    Parameters
    ----------
    nodeLabels
        The labels of the nodes of the mesh, in mesh order.
    nodesCarryingField
        Per field, in the order the fields are numbered, which nodes of the mesh carry it: a boolean
        array in mesh order.
    dimensionOfField
        Per field, the number of its components at a node.
    nScalarVariables
        The number of scalar variables, numbered after the fields.
    """

    def __init__(self, nodeLabels: np.ndarray, nodesCarryingField: dict, dimensionOfField: dict, nScalarVariables: int):
        #: The labels of the nodes of the mesh, in mesh order.
        self.nodeLabels = nodeLabels
        #: The fields, in the order they are numbered.
        self.fields = list(nodesCarryingField)
        #: Per field, the number of its components at a node.
        self.dimensionOfField = {field: dimensionOfField[field] for field in self.fields}
        #: Per field, which nodes of the mesh carry it (boolean, in mesh order).
        self.nodesCarryingField = nodesCarryingField
        #: Per field, the place of each node among the nodes carrying it, in mesh order; -1 where a node
        #: does not carry it.
        self.rankInField = {}
        #: Per field, the first of its degrees of freedom.
        self.firstDofOfField = {}

        firstDof = 0
        for field, carrying in nodesCarryingField.items():
            rank = np.full(carrying.shape[0], -1, dtype=np.int32)
            nCarrying = int(np.count_nonzero(carrying))
            rank[carrying] = np.arange(nCarrying, dtype=np.int32)
            self.rankInField[field] = rank
            self.firstDofOfField[field] = firstDof
            firstDof += nCarrying * self.dimensionOfField[field]

        #: The first degree of freedom of the scalar variables, after those of every field.
        self.firstDofOfScalarVariables = firstDof
        #: The number of scalar variables.
        self.nScalarVariables = nScalarVariables
        #: The number of degrees of freedom.
        self.nDof = firstDof + nScalarVariables

    @classmethod
    def ofModel(cls, model) -> "GlobalDofNumbering":
        """The numbering of a model, from its mesh, the fields its constraints couple and the number of
        its scalar variables.

        The fields are numbered in the order of :data:`~edelweissfe.config.phenomena.phenomena`, as the
        model creates its node fields.

        Parameters
        ----------
        model
            The model.

        Returns
        -------
        GlobalDofNumbering
            The numbering.
        """

        mesh = model.mesh
        carrying = {}

        def carryingOf(field: str) -> np.ndarray:
            if field not in carrying:
                carrying[field] = np.zeros(mesh.nNodes, dtype=bool)
            return carrying[field]

        # the elements, a group of the same type at a time, each one array operation per node and field
        for group in mesh.connectivity().groups:
            for i, fields in enumerate(group.typeInfo.fields):
                for field in fields:
                    carryingOf(field)[group.nodeRows[:, i]] = True

        for constraint in model.constraints.values():
            rows = mesh.rowsOfNodes([node.label for node in constraint.nodes])
            for row, fields in zip(rows, constraint.fieldsOnNodes):
                for field in fields:
                    carryingOf(field)[row] = True

        fields = [field for field in phenomena if field in carrying]
        return cls(
            mesh.nodeLabels.copy(),
            {field: carrying[field] for field in fields},
            {field: getFieldSize(field, model.domainSize) for field in fields},
            len(model.scalarVariables),
        )

    def digest(self) -> str:
        """A digest of the numbering: the node labels in mesh order, and per field its name, its
        dimension and the nodes carrying it; and the number of scalar variables.

        Returns
        -------
        str
            A hex digest.
        """

        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(self.nodeLabels, dtype=np.int64).tobytes())
        for field in self.fields:
            digest.update(field.encode() + b"|%d|" % self.dimensionOfField[field])
            digest.update(np.packbits(self.nodesCarryingField[field]).tobytes())
        digest.update(b"scalar variables|%d" % self.nScalarVariables)
        return digest.hexdigest()

    def requireAgreementWith(self, dofManager, nodeFields: dict, scalarVariables: dict):
        """Raise unless a DofManager numbers the degrees of freedom exactly as this numbering does,
        degree of freedom by degree of freedom.

        Parameters
        ----------
        dofManager
            The DofManager, built on ``nodeFields`` and ``scalarVariables``.
        nodeFields
            The node fields of the model, by name.
        scalarVariables
            The scalar variables of the model, by name.

        Raises
        ------
        RuntimeError
            Naming the first difference.
        """

        def differ(what: str):
            raise RuntimeError(
                "the DofManager's numbering of the degrees of freedom differs from the global numbering of the "
                "mesh (GlobalDofNumbering): {:}".format(what)
            )

        if list(nodeFields) != self.fields:
            differ("fields {:} instead of {:}".format(list(nodeFields), self.fields))

        dofsOfFieldVariables = dofManager.idcsOfFieldVariablesInDofVector
        for field in self.fields:
            nodes = nodeFields[field].nodes
            labels = np.fromiter((node.label for node in nodes), dtype=np.int64, count=len(nodes))
            if not np.array_equal(labels, self.nodeLabels[self.nodesCarryingField[field]]):
                differ("the nodes of field {:} are not the nodes of the mesh carrying it, in mesh order".format(field))

            first, dimension = self.firstDofOfField[field], self.dimensionOfField[field]
            dofs = np.concatenate(
                [dofsOfFieldVariables[node.fields[field]] for node in nodes] or [np.empty(0, dtype=np.int64)]
            )
            if not np.array_equal(dofs, np.arange(first, first + labels.shape[0] * dimension)):
                differ("the degrees of freedom of field {:}".format(field))
            if dofManager.idcsOfFieldsInDofVector[field] != slice(first, first + labels.shape[0] * dimension):
                differ("the range of field {:}".format(field))

        scalarDofs = np.asarray(
            [dofManager.idcsOfScalarVariablesInDofVector[variable] for variable in scalarVariables.values()],
            dtype=np.int64,
        )
        if not np.array_equal(
            scalarDofs, self.firstDofOfScalarVariables + np.arange(self.nScalarVariables, dtype=np.int64)
        ):
            differ("the degrees of freedom of the scalar variables")
        if dofManager.nDof != self.nDof:
            differ("{:} degrees of freedom instead of {:}".format(dofManager.nDof, self.nDof))

    def dofsAtNodes(self, field: str, rows: np.ndarray) -> np.ndarray:
        """The degrees of freedom of a field at nodes of the mesh.

        Parameters
        ----------
        field
            The field.
        rows
            The nodes, by their row in the mesh; each must carry the field.

        Returns
        -------
        np.ndarray
            One row per node, one column per component of the field.

        Raises
        ------
        KeyError
            If a node does not carry the field.
        """

        ranks = self.rankInField[field][np.asarray(rows, dtype=np.int64)].astype(np.int64)
        if np.any(ranks < 0):
            raise KeyError("a node does not carry field {:}".format(field))
        dimension = self.dimensionOfField[field]
        return self.firstDofOfField[field] + ranks[:, None] * dimension + np.arange(dimension, dtype=np.int64)

    def dofsOfFieldVariables(self, mesh) -> "GlobalDofsOfFieldVariables":
        """The global degrees of freedom of the field variables of nodes of the mesh: what a model entity
        reads from a DofManager (``idcsOfFieldVariablesInDofVector``), answered from this numbering
        instead -- e.g. for the records of the multi-point constraints by global degree of freedom, in a
        process whose DofManager numbers only its subdomain.

        Parameters
        ----------
        mesh
            The mesh this numbering was made from.

        Returns
        -------
        GlobalDofsOfFieldVariables
            The lookup.
        """

        return GlobalDofsOfFieldVariables(self, mesh)

    def nodesOfFieldsAgreeWith(self, nodeFields: dict) -> bool:
        """Whether the node fields of a model hold, per field, the nodes of the mesh carrying it, in mesh
        order -- the order of this numbering.

        Parameters
        ----------
        nodeFields
            The node fields of the model, by name.

        Returns
        -------
        bool
            Whether they agree.
        """

        if list(nodeFields) != self.fields:
            return False
        for field in self.fields:
            nodes = nodeFields[field].nodes
            labels = np.fromiter((node.label for node in nodes), dtype=np.int64, count=len(nodes))
            if not np.array_equal(labels, self.nodeLabels[self.nodesCarryingField[field]]):
                return False
        return True

    def requireAgreementOfSubdomain(self, dofManager, rowsOfFields: dict, nodeFields: dict, subdomainDofs: np.ndarray):
        """Raise unless a DofManager of the node fields of a subdomain numbers its degrees of freedom as
        the order-preserving compression of this numbering -- its degree of freedom ``i`` is the global
        ``subdomainDofs[i]`` --, and unless the node fields of the model, which hold every node, are
        ordered as this numbering.

        Parameters
        ----------
        dofManager
            The DofManager of the subdomain's node fields.
        rowsOfFields
            Per field, in the order the DofManager numbers them, the nodes of the subdomain's node field
            by their row in the mesh, in its order.
        nodeFields
            The node fields of the model, by name.
        subdomainDofs
            The global degrees of freedom of the subdomain, sorted.

        Raises
        ------
        RuntimeError
            Naming the first difference.
        """

        def differ(what: str):
            raise RuntimeError(
                "the DofManager of the subdomain does not number its degrees of freedom as the global "
                "numbering of the mesh (GlobalDofNumbering), compressed: {:}".format(what)
            )

        if not self.nodesOfFieldsAgreeWith(nodeFields):
            differ("the node fields of the model are not the nodes of the mesh carrying them, in mesh order")
        if list(dofManager.idcsOfFieldsInDofVector) != self.fields or list(rowsOfFields) != self.fields:
            differ("its fields are not those of the numbering, in its order")
        first = 0
        globalDofs = []
        for field, rows in rowsOfFields.items():
            dofs = self.dofsAtNodes(field, rows).ravel()
            if dofManager.idcsOfFieldsInDofVector[field] != slice(first, first + dofs.shape[0]):
                differ("the range of field {:}".format(field))
            first += dofs.shape[0]
            globalDofs.append(dofs)
        if dofManager.nDof != first or not np.array_equal(np.concatenate(globalDofs), subdomainDofs):
            differ("its degrees of freedom are not the subdomain degrees of freedom, in global order")


class GlobalDofsOfFieldVariables:
    """The global degrees of freedom of field variables, by field variable, read from a
    :class:`GlobalDofNumbering`; see :meth:`GlobalDofNumbering.dofsOfFieldVariables`.

    It answers what a model entity asks a :class:`~edelweissfe.numerics.dofmanager.DofManager` for the
    degrees of freedom of a field variable -- through the same attribute,
    :attr:`idcsOfFieldVariablesInDofVector` --, so that the entity can be asked for its degrees of
    freedom in the global numbering wherever no DofManager numbers the whole model.

    Parameters
    ----------
    numbering
        The numbering.
    mesh
        The mesh it was made from.
    """

    def __init__(self, numbering: GlobalDofNumbering, mesh):
        self._numbering = numbering
        self._mesh = mesh

    @property
    def idcsOfFieldVariablesInDofVector(self) -> "GlobalDofsOfFieldVariables":
        """The lookup itself, under the name of the DofManager's mapping."""

        return self

    def __contains__(self, fieldVariable) -> bool:
        """Whether the node of a field variable carries its field in the numbering.

        Parameters
        ----------
        fieldVariable
            The field variable.

        Returns
        -------
        bool
            Whether it has degrees of freedom.
        """

        field = fieldVariable.field
        if field not in self._numbering.rankInField or not self._mesh.hasNodes([fieldVariable.node.label])[0]:
            return False
        row = self._mesh.rowsOfNodes([fieldVariable.node.label])[0]
        return bool(self._numbering.rankInField[field][row] >= 0)

    def __getitem__(self, fieldVariable) -> np.ndarray:
        """The global degrees of freedom of a field variable, one per component.

        Parameters
        ----------
        fieldVariable
            The field variable.

        Returns
        -------
        np.ndarray
            The degrees of freedom.

        Raises
        ------
        KeyError
            If its node does not carry its field.
        """

        rows = self._mesh.rowsOfNodes([fieldVariable.node.label])
        return self._numbering.dofsAtNodes(fieldVariable.field, rows)[0]
