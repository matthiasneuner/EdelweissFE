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
Linearized rigid body enforced by master-slave DOF elimination (multi-point constraint), like Abaqus'
``*RIGID BODY, TIE NSET=...`` -- the DOF-elimination twin of
:mod:`~edelweissfe.constraints.linearizedrigidbody`, which enforces the same kinematics by Lagrange multipliers.

Every node :math:`s` of the node set follows the reference point (RP) rigidly, for infinitesimal rotations:

.. math::
    \\boldsymbol{u}_s = \\boldsymbol{u}_{RP}
    + \\boldsymbol{\\theta}_{RP} \\times ( \\boldsymbol{X}_s - \\boldsymbol{X}_{RP} )
    \\qquad \\text{(2D: } \\boldsymbol{u}_s = \\boldsymbol{u}_{RP} + \\theta_{RP} \\, \\boldsymbol{e}_z \\times
    ( \\boldsymbol{X}_s - \\boldsymbol{X}_{RP} ) \\text{)}.

The slave displacements are condensed out of the equation system; the RP carries the ``displacement`` and the
``rotation`` field (created by this constraint if no element provides them), and loads and boundary conditions
applied to the RP act on the whole rigid node set. Unlike the Lagrange-multiplier version, the condensed system has
no zero diagonal (no saddle point), so it is solvable by iterative/AMG linear solvers such as ``blockamg``. The
reaction forces of the tied nodes are condensed onto the RP (read them there, field ``displacement`` result ``P``
and field ``rotation`` result ``P`` for the moments).

Like every multi-point constraint, a node may be the slave of only one of them, and the slaves must not carry
Dirichlet conditions of their own (constrain the RP instead).

.. code-block:: edelweiss
    :caption: Example:

    *constraint, type=rigidBodyMPC, name=endFace
    nSet=endFaceNodes
    referencePoint=RP
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.exceptions import WrongDomain
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class RigidBodyMPCSchema:
    """The options this constraint accepts, owned by this module and never mutated from outside it."""

    nSet: str | None = schemaField(
        description="The node set tied rigidly to the reference point.", dtype=str, default=None, required=True
    )
    referencePoint: str | None = schemaField(
        description="The node set containing only the reference point.", dtype=str, default=None, required=True
    )


def rigidBodyCoefficients(distance: np.ndarray) -> list[list[tuple[str, int, float]]]:
    """The linear dependency of the displacement of a node at ``distance = X_s - X_RP`` on the RP's displacement and
    rotation (infinitesimal), ``u_s = u_RP + theta x distance`` (2D: ``theta e_z x distance``).

    Returns
    -------
    list
        Per displacement component of the slave, the ``(field, component, coefficient)`` of its masters.
    """
    if distance.shape[0] == 2:
        dx, dy = distance
        return [
            [("displacement", 0, 1.0), ("rotation", 0, -dy)],
            [("displacement", 1, 1.0), ("rotation", 0, dx)],
        ]
    dx, dy, dz = distance
    return [
        [("displacement", 0, 1.0), ("rotation", 1, dz), ("rotation", 2, -dy)],
        [("displacement", 1, 1.0), ("rotation", 2, dx), ("rotation", 0, -dz)],
        [("displacement", 2, 1.0), ("rotation", 0, dy), ("rotation", 1, -dx)],
    ]


class Constraint(MultiPointConstraintBase):
    """Linearized rigid body multi-point constraint enforced by DOF elimination.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    journal
        The journal.
    configuration
        The options this constraint accepts.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = RigidBodyMPCSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal = None,
        *,
        configuration: RigidBodyMPCSchema = RigidBodyMPCSchema(),
    ):
        if model.domainSize not in (2, 3):
            raise WrongDomain(f"{name}: rigidBodyMPC needs a 2D or 3D domain")

        self._name = name
        self._model = model

        referencePoint = model.nodeSets[configuration.referencePoint]
        if len(referencePoint) != 1:
            raise ValueError(
                f"{name}: the reference point set '{configuration.referencePoint}' must contain exactly one node"
            )
        self.rp = referencePoint[0]

        xRP = np.asarray(self.rp.coordinates, dtype=float)[: model.domainSize]
        # (slaveNode, [[(field, component, coefficient), ...] per displacement component]); slave node first
        self._records = [
            (node, rigidBodyCoefficients(np.asarray(node.coordinates, dtype=float)[: model.domainSize] - xRP))
            for node in dict.fromkeys(model.nodeSets[configuration.nSet])
            if node is not self.rp
        ]

        if journal:
            journal.message(f"{len(self._records)} node(s) tied rigidly to reference point {self.rp.label}", name)

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, journal, configuration=configuration)

    def fieldsOnMasterNodes(self) -> list:
        return [(self.rp, ["displacement", "rotation"])]

    def getMultiPointConstraints(self, dofManager) -> list:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector
        rpDofs = {field: fieldVariableIndices[self.rp.fields[field]] for field in ("displacement", "rotation")}
        records = []
        for slaveNode, coefficients in self._records:
            if "displacement" not in slaveNode.fields:
                continue
            slaveDofs = fieldVariableIndices[slaveNode.fields["displacement"]]
            for component, masters in enumerate(coefficients):
                records.append((slaveDofs[component], [(rpDofs[field][i], c) for field, i, c in masters if c != 0.0]))
        return records
