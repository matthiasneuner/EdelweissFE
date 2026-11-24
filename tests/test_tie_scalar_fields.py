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
"""The tie of a scalar field across an interface (``surfaceToDeformableSurfacePenalty``, ``type=tied``, ``tiedFields``):
the nonlocal field of gradient-enhanced elements held equal on both sides by a penalty per unit area
:math:`\\kappa = \\beta l^2 / h` (see :class:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty.Constraint`,
*Tied scalar fields*).

The interface patch test: a linear field crosses the tie without a force, on matching and non-matching meshes, and a
uniform jump of the field is distributed as the consistent nodal flux of a uniform flux on both faces. Then the
tangent by finite differences, the bound of the stiffness for the critical time step against the tangent and against
the exact eigenvalue, the block layout, and what is refused.
"""

import unittest

import numpy as np
from _hexa20cube import _SIDE, twoBlockModel  # noqa: E402  (tests/ is on sys.path)

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)

#: The penalty of the displacement per unit area, the nonlocal radius, and the factor of the penalty of the field.
PENALTY, RADIUS, FACTOR = 1.0e4, 0.5, 10.0


def _quad8(corners: np.ndarray) -> list:
    """The eight node coordinates of a quadrilateral, corners counterclockwise, then the midsides."""

    return list(corners) + [0.5 * (corners[i] + corners[(i + 1) % 4]) for i in range(4)]


def twoPlaneBlocksModel(journal, nMasterElements: int) -> tuple:
    """Two plane strain CPE8 blocks meeting at y = 0: one element of the side 2 below (its top face the slave), and
    ``nMasterElements`` elements side by side above (their bottom faces the master), matching for one, not for two; a
    plane section of unit thickness.

    Returns
    -------
    tuple
        ``(model, slaveSurface, masterSurface)``.
    """

    model = FEModel(2)
    blocks = {
        "lower": [(0.0, -2.0, 2.0)],
        "upper": [(2.0 * i / nMasterElements, 0.0, 2.0 / nMasterElements) for i in range(nMasterElements)],
    }
    nodeAt, elements = {}, {}
    with model.topology.changes():
        for name, rectangles in blocks.items():
            elements[name] = []
            for x0, y0, width in rectangles:
                corners = np.array([[x0, y0], [x0 + width, y0], [x0 + width, y0 + 2.0], [x0, y0 + 2.0]])
                nodes = []
                for x in _quad8(corners):
                    key = (name, round(x[0], 12), round(x[1], 12))
                    if key not in nodeAt:
                        label = len(nodeAt) + 1
                        nodeAt[key] = model.nodes[label] = Node(label, np.array(x))
                    nodes.append(nodeAt[key])
                (elNumber,) = model.topology.reserveElementNumbers(1)
                element = DisplacementElement("CPE8", elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                elements[name].append(element)
        model.sections["blocks"] = PlaneSection(
            "blocks",
            model,
            {},
            [ElementSet("blocks", elements["lower"] + elements["upper"])],
            configuration=PlaneSectionSchema(thickness=1.0),
        )
        model.surfaces["slaveFace"] = {3: ElementSet("s", elements["lower"])}
        model.surfaces["masterFace"] = {1: ElementSet("m", elements["upper"])}
        masterSetName, _ = buildContactFacets(model, "masterFace", "mst", "midside", "facetConsistent", journal)
        slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", "facetConsistent", journal)
    return model, model.elementSets[slaveSetName], model.elementSets[masterSetName]


class TestTiedScalarField(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _tie(self, model, slaveSurface, masterSurface, **options) -> SurfaceContact:
        """The tie of the displacement and the nonlocal field (unless ``tiedFields`` says otherwise), its points the
        quadrature points of the slave facets, paired."""

        configuration = SurfaceToDeformableSurfacePenaltySchema(
            penalty=PENALTY,
            **(
                {"contactType": "tied", "tiedFields": "displacement; nonlocal damage", "nonlocalRadius": RADIUS}
                | options
            ),
        )
        tie = SurfaceContact(
            "tie",
            model,
            ContactPointsOnSlaveSurface(slaveSurface, configuration.nQuadraturePoints, "tie"),
            masterSurface,
            self.journal,
            configuration=configuration,
        )
        tie.updateConnectivity(model)
        return tie

    def _models(self) -> list:
        """The interfaces: 2D matching and non-matching, 3D matching (hexa20)."""

        return [
            ("2D matching", *twoPlaneBlocksModel(self.journal, 1)),
            ("2D non-matching", *twoPlaneBlocksModel(self.journal, 2)),
            ("3D", *twoBlockModel(self.journal, 0.0)),
        ]

    @staticmethod
    def _fieldPart(tie) -> slice:
        """The degrees of freedom of the nonlocal field in the block of the tie."""

        nNodes = len(tie._nodes)
        return slice(tie.nDim * nNodes, (tie.nDim + 1) * nNodes)

    def test_the_default_ties_the_displacement_alone(self):
        """Without ``tiedFields``, the block holds the displacement alone, as before the tie could tie other fields."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, 0.0)
        tie = self._tie(model, slaveSurface, masterSurface, tiedFields="displacement", nonlocalRadius=None)
        self.assertEqual(tie.tiedScalarFields, [])
        self.assertEqual(tie.nDof, 3 * len(tie.nodes))
        self.assertEqual(tie.fields, [["displacement"]] * len(tie.nodes))

        tied = self._tie(model, slaveSurface, masterSurface)
        nNodes = len(tied._nodes)
        self.assertEqual(tied.nodes, tied._nodes + tied._nodes)
        self.assertEqual(tied.fields, [["displacement"]] * nNodes + [["nonlocal damage"]] * nNodes)
        self.assertEqual(tied.nDof, 4 * nNodes)

    def test_a_linear_field_crosses_the_tie_without_a_force(self):
        """A linear field, continuous across the interface, has no jump at any tied point: both sides interpolate it
        exactly (the quadratic parent faces of the slave and the master), on matching and non-matching meshes alike,
        and the tie exerts no force on it."""

        for name, model, slaveSurface, masterSurface in self._models():
            tie = self._tie(model, slaveSurface, masterSurface)
            X = np.array([node.coordinates for node in tie._nodes])
            U = np.zeros(tie.nDof)
            U[self._fieldPart(tie)] = 0.3 + X @ np.array([0.7, -1.1, 0.4][: tie.nDim])
            P = np.zeros(tie.nDof)
            tie.computeKernels(U, np.zeros_like(U), P, None, None)
            scale = FACTOR * RADIUS**2 * np.max(np.abs(U))
            with self.subTest(name):
                np.testing.assert_allclose(P, 0.0, atol=1e-13 * scale)
                # the same field displaced by a constant: a uniform jump, which the tie does see
                U[self._fieldPart(tie)][: tie._nSlaveNodes] += 1.0
                tie.computeKernels(U, np.zeros_like(U), P, None, None)
                self.assertGreater(np.max(np.abs(P)), 1e6 * 1e-13 * scale)

    def test_a_uniform_jump_is_the_consistent_flux_on_both_faces(self):
        """A uniform jump :math:`j` of the field, the slave side above the master: the force on the slave is the
        consistent nodal flux of the uniform flux :math:`\\kappa j` over the face, :math:`-\\kappa j \\int N_a\\,
        dA` -- for a quadratic line face of the length :math:`L` :math:`L/6` at the ends and :math:`2L/3` at the
        midside, for a quad8 face of the area :math:`A` :math:`-A/12` at the corners and :math:`A/3` at the midsides
        --, the force on the master its opposite, node by node, and the penalty is :math:`\\beta l^2/h` with the size
        :math:`h` of the master face."""

        for name, model, slaveSurface, masterSurface in (self._models()[0], self._models()[2]):
            tie = self._tie(model, slaveSurface, masterSurface)
            size = _SIDE
            kappa = FACTOR * RADIUS**2 / size
            np.testing.assert_allclose(tie._scalarPenaltyTimesAreaOfLocated, kappa * tie.slave.areas, rtol=1e-14)

            jump = 0.25
            U = np.zeros(tie.nDof)
            U[self._fieldPart(tie).start : self._fieldPart(tie).start + tie._nSlaveNodes] = jump
            P = np.zeros(tie.nDof)
            tie.computeKernels(U, np.zeros_like(U), P, None, None)
            forces = -P[self._fieldPart(tie)]

            # the force on each node, summed over the copies of the node in the block
            onNode = {}
            for node, force in zip(tie._nodes, forces):
                onNode[node] = onNode.get(node, 0.0) + force
            slaveNodes, masterNodes = tie._nodes[: tie._nSlaveNodes], tie._nodes[tie._nSlaveNodes :]
            nCorners = 2 if tie.nDim == 2 else 4
            weights = [1 / 6, 1 / 6, 2 / 3] if tie.nDim == 2 else [-1 / 12] * 4 + [1 / 3] * 4
            area = size ** (tie.nDim - 1)
            faceOrder = lambda nodes: list(dict.fromkeys(nodes))  # noqa: E731
            with self.subTest(name):
                self.assertEqual(len(faceOrder(slaveNodes)), len(weights))
                for side, sign in ((slaveNodes, -1.0), (masterNodes, 1.0)):
                    nodes = faceOrder(side)
                    flux = np.array([onNode[node] for node in nodes])
                    corners = flux[:nCorners] if tie.nDim == 3 else flux[[0, 1]]
                    np.testing.assert_allclose(flux.sum(), sign * kappa * jump * area, rtol=1e-13)
                    np.testing.assert_allclose(
                        np.sort(flux), np.sort(sign * kappa * jump * area * np.array(weights)), rtol=1e-12
                    )
                    self.assertTrue(np.all(np.sign(corners) == np.sign(sign * weights[0])))

    def test_the_tangent_is_the_derivative_of_the_residual_contribution(self):
        """``K == dP/dU`` by central differences, for the displacement and the field together, at a random state."""

        for name, model, slaveSurface, masterSurface in self._models():
            tie = self._tie(model, slaveSurface, masterSurface)
            U = 1e-2 * np.random.default_rng(5).standard_normal(tie.nDof)
            P, K = residualContributionAndTangent(tie, U, np.zeros_like(U))
            with self.subTest(name):
                np.testing.assert_allclose(
                    K, finiteDifferenceTangent(tie, U, np.zeros_like(U)), atol=1e-6 * np.max(np.abs(K))
                )
                np.testing.assert_allclose(K, K.T, atol=1e-12 * np.max(np.abs(K)))
                # the field and the displacement do not couple
                part = self._fieldPart(tie)
                np.testing.assert_array_equal(K[part, : part.start], 0.0)

    def test_the_field_springs_add_nothing_to_the_reported_forces_and_energy(self):
        """The springs of the field are not mechanical: the energy, the tractions and the resultants the tie reports
        are those of the displacement springs, with or without the field tied."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, 0.0)
        displacementOnly = self._tie(model, slaveSurface, masterSurface, tiedFields="displacement")
        tied = self._tie(model, slaveSurface, masterSurface)
        U = 1e-2 * np.random.default_rng(6).standard_normal(tied.nDof)
        nDisplacement = displacementOnly.nDof
        for tie, Utie in ((displacementOnly, U[:nDisplacement]), (tied, U)):
            tie.computeKernels(Utie, np.zeros_like(Utie), np.zeros(tie.nDof), None, None)
        self.assertGreater(displacementOnly.penaltyEnergy, 0.0)
        self.assertEqual(tied.penaltyEnergy, displacementOnly.penaltyEnergy)
        np.testing.assert_array_equal(tied.getTractions(), displacementOnly.getTractions())
        np.testing.assert_array_equal(tied.contactForceOnMaster, displacementOnly.contactForceOnMaster)

        P, Ptied = np.zeros(nDisplacement), np.zeros(tied.nDof)
        displacementOnly.computeKernels(U[:nDisplacement], np.zeros(nDisplacement), P, None, None)
        tied.computeKernels(U, np.zeros_like(U), Ptied, None, None)
        np.testing.assert_array_equal(Ptied[:nDisplacement], P)

    def test_the_bound_of_the_stiffness_is_the_one_of_the_tangent_and_bounds_the_exact_eigenvalue(self):
        """The bound of the stiffness for the critical time step (from the springs) equals the row sums of the
        absolute value of the tangent, weighted by random columns; and with the weights :math:`1/\\sqrt{c}` of random
        lumped coefficients (a mass of a second-order field, a damping of a first-order one), the largest weighted row
        sum of the field's block bounds the largest eigenvalue of its scaled stiffness, so the critical time step it
        gives (:math:`2/\\sqrt{\\rho}` or :math:`2/\\rho`) is below the exact one."""

        for name, model, slaveSurface, masterSurface in self._models():
            tie = self._tie(model, slaveSurface, masterSurface)
            U = 1e-3 * np.random.default_rng(8).standard_normal(tie.nDof)
            rng = np.random.default_rng(3)
            weights = rng.uniform(0.5, 2.0, tie.nDof)
            bound, boundOfTheTangent = np.zeros(tie.nDof), np.zeros(tie.nDof)
            tie.addBoundOfAbsoluteStiffnessTimes(U, np.zeros_like(U), weights, bound, None)
            ConstraintBase.addBoundOfAbsoluteStiffnessTimes(tie, U, np.zeros_like(U), weights, boundOfTheTangent, None)
            part = self._fieldPart(tie)
            with self.subTest(name):
                self.assertGreater(boundOfTheTangent[part].max(), 0.0)
                np.testing.assert_allclose(bound, boundOfTheTangent, rtol=1e-12, atol=1e-12 * bound.max())

                # the nodes of the block, each once, with a lumped coefficient: the stiffness assembled onto them
                nodes = list(dict.fromkeys(tie._nodes))
                indexOfNode = {node: i for i, node in enumerate(nodes)}
                assemble = np.zeros((len(tie._nodes), len(nodes)))
                assemble[np.arange(len(tie._nodes)), [indexOfNode[node] for node in tie._nodes]] = 1.0
                K = residualContributionAndTangent(tie, U, np.zeros_like(U))[1][part, part]
                coefficients = rng.uniform(0.5, 3.0, len(nodes))
                s = 1.0 / np.sqrt(coefficients)
                scaled = s[:, None] * (assemble.T @ K @ assemble) * s[None, :]
                weightsOfTheBlock = assemble @ s
                bound = np.zeros(tie.nDof)
                tie.addBoundOfAbsoluteStiffnessTimes(
                    U, np.zeros_like(U), np.r_[np.zeros(part.start), weightsOfTheBlock], bound, None
                )
                rowSums = s * (assemble.T @ bound[part])
                largestEigenvalue = np.max(np.linalg.eigvalsh(scaled))
                self.assertGreaterEqual(np.max(rowSums), largestEigenvalue * (1 - 1e-12))
                self.assertLess(np.max(rowSums), 10.0 * largestEigenvalue)

    def test_what_is_refused(self):
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, 0.0)
        for options in (
            {"tiedFields": "nonlocal damage"},
            {"tiedFields": "displacement; displacement"},
            {"tiedFields": "displacement; rotation"},
            {"tiedFields": "displacement; nonlocal damage", "nonlocalRadius": None},
            {"tiedFields": "displacement; nonlocal damage", "nonlocalPenaltyFactor": 0.0},
            {"tiedFields": "displacement; nonlocal damage", "contactType": "linear"},
        ):
            with self.subTest(options), self.assertRaises(ValueError):
                self._tie(model, slaveSurface, masterSurface, **options)


if __name__ == "__main__":
    unittest.main()
