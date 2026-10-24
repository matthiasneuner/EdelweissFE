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
"""The rigid-body coarse level (``fieldPreconds`` key ``rigidBodyCoarseLevel``,
:mod:`edelweissfe.linsolve.blockamg.rigidbodycoarselevel`): the defining properties of its construction -- the
tentative prolongation reproduces the rigid-body modes, the coarse operator is the Galerkin product, Dirichlet rows stay
out of the coarse space, linearly dependent modes are dropped -- its control (with the translations alone it is AMGCL's
own first coarse level), its purpose (fewer iterations on a bending-dominated structure), and its configuration in
``blockamg``."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from _blockamgsystems import NODAL_COUPLING, gridLaplacian, hierarchyLevels
from _elasticbeam import cantilever, equilibrated
from scipy.sparse.linalg import LinearOperator, gmres

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg import createSolver  # noqa: E402
from edelweissfe.linsolve.blockamg.blockamg import (  # noqa: E402
    BlockAMGSolver,
    _amgclTree,
    _defaultFieldPrecond,
)
from edelweissfe.linsolve.blockamg.rigidbodycoarselevel import (  # noqa: E402
    RigidBodyCoarseLevelPreconditioner,
)
from edelweissfe.linsolve.blockamg.schurcomplement import (  # noqa: E402
    SchurComplementCondensation,
)
from edelweissfe.linsolve.nullspace import (  # noqa: E402
    rigidBodyNullspace,
    translationNullspace,
)
from edelweissfe.linsolve.superlu import SuperLUSolver  # noqa: E402


@functools.lru_cache
def _beam(dimension: int):
    """A propped cantilever, 16:1 slender: the stiffness, the node coordinates and the constrained dofs."""
    if dimension == 3:
        return cantilever((32, 2, 2), (16.0, 1.0, 1.0))
    return cantilever((64, 4), (16.0, 1.0))


def _equilibratedBeam(dimension: int):
    """The equilibrated propped cantilever: ``D K D``, ``D``, the node coordinates and the constrained dofs."""
    K, coordinates, constrained = _beam(dimension)
    As, scaling = equilibrated(K)
    return As, scaling, coordinates, constrained


def _preconditioner(dimension: int, backendPrecision: str = "double", **options) -> RigidBodyCoarseLevelPreconditioner:
    """The rigid-body coarse level of the default tree of a field of this dimension, stored in backendPrecision."""
    return RigidBodyCoarseLevelPreconditioner(
        _amgclTree(_defaultFieldPrecond(dimension)), dimension, backendPrecision, **options
    )


def _relativeDifference(x, y) -> float:
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def _rigidBodyModes(As, scaling, coordinates, dimension, aggregates):
    """The field's rigid-body modes about its centroid, divided by the equilibration, zero on the Dirichlet rows and on
    the nodes in no aggregate: what P_tent reproduces from the coarse near null-space."""
    modes = rigidBodyNullspace(FieldBlock("displacement", 0, As.shape[0], dimension), coordinates, scaling)
    modes[np.diff(As.indptr) == 1] = 0.0
    modes[np.repeat(aggregates < 0, dimension)] = 0.0
    return modes


#: The iteration budget of :func:`_gmresIterations`.
_GMRES_CAP = 300


def _gmresIterations(A, preconditioner) -> int:
    """The iterations of (left-preconditioned, unrestarted) GMRES to a preconditioned relative residual of 1e-8, or
    ``_GMRES_CAP + 1`` if it does not reach a true relative residual of 1e-5 within ``_GMRES_CAP``."""
    b = np.random.default_rng(7).standard_normal(A.shape[0])
    M = LinearOperator(A.shape, matvec=preconditioner.applyPreconditioner)
    history = []
    x, _ = gmres(A, b, M=M, rtol=1e-8, restart=_GMRES_CAP, maxiter=1, callback=history.append, callback_type="pr_norm")
    converged = np.linalg.norm(b - A @ x) <= 1e-5 * np.linalg.norm(b)
    return len(history) if converged and len(history) <= _GMRES_CAP else _GMRES_CAP + 1


@pytest.mark.parametrize("dimension", [3, 2], ids=["3D, 6 modes", "2D, 3 modes"])
def test_the_tentative_prolongation_reproduces_the_rigid_body_modes(dimension):
    # P_tent B_c = B: the coarse near null-space are the coefficients of the field's rigid-body modes in the coarse
    # basis, so the hierarchy below inherits exactly the modes of the fine level. And the columns of P_tent are
    # orthonormal, the QR factors of every aggregate's modes.
    As, scaling, coordinates, _ = _equilibratedBeam(dimension)
    preconditioner = _preconditioner(dimension, prolongation="tentative")
    preconditioner.build(As, scaling, coordinates)
    modesPerAggregate = dimension * (dimension + 1) // 2
    assert preconditioner.nCoarse == modesPerAggregate * preconditioner.nAggregates > 0

    P, coarseNullspace = preconditioner.prolongationMatrix(), preconditioner.coarseNullspace()
    assert coarseNullspace.shape == (preconditioner.nCoarse, modesPerAggregate)
    modes = _rigidBodyModes(As, scaling, coordinates, dimension, preconditioner.aggregates)
    assert _relativeDifference(P @ coarseNullspace, modes) < 1e-12

    PtP = (P.T @ P).toarray()
    kept = np.diag(PtP) > 0.5
    assert np.abs(PtP - np.diag(kept.astype(float))).max() < 1e-12


@pytest.mark.parametrize("prolongation", ["smoothed", "tentative"])
def test_the_coarse_operator_is_the_galerkin_product(prolongation):
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    preconditioner = _preconditioner(3, prolongation=prolongation)
    preconditioner.build(As, scaling, coordinates)
    P, coarseOperator = preconditioner.prolongationMatrix(), preconditioner.coarseOperator()
    expected = (P.T @ As @ P).toarray()
    # a dropped mode (none expected here) has a zero column of P and the padding as its diagonal entry
    dropped = np.flatnonzero(abs(P).sum(axis=0).A1 == 0.0)
    difference = coarseOperator.toarray() - expected
    difference[dropped, dropped] = 0.0
    assert np.linalg.norm(difference) <= 1e-12 * np.linalg.norm(expected)
    if prolongation == "smoothed":
        # one damped Jacobi step widens every column of P_tent by a layer of nodes
        tentative = _preconditioner(3, prolongation="tentative")
        tentative.build(As, scaling, coordinates)
        assert P.nnz > tentative.prolongationMatrix().nnz
        assert 0.0 < preconditioner.prolongationSmoothingWeight < 2.0
    else:
        assert preconditioner.prolongationSmoothingWeight == 0.0


def test_dirichlet_rows_stay_out_of_the_coarse_space_and_are_solved_exactly():
    # the beam is clamped (all components) and propped (one component): fully and partially constrained nodes
    As, scaling, coordinates, constrained = _equilibratedBeam(3)
    for backendPrecision in ("double", "float"):
        preconditioner = _preconditioner(3, backendPrecision)
        preconditioner.build(As, scaling, coordinates)
        P = preconditioner.prolongationMatrix()
        assert abs(P[constrained]).max() == 0.0
        r = np.random.default_rng(3).standard_normal(As.shape[0])
        x = preconditioner.applyPreconditioner(r)
        assert np.array_equal(x[constrained], r[constrained] / As.diagonal()[constrained])
    # a fully constrained node has no strong connection and is in no aggregate
    clampedNodes = np.flatnonzero(coordinates[:, 0] == 0.0)
    assert np.all(preconditioner.aggregates[clampedNodes] < 0)


@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (1.0, 2.0, 3.0)], ids=["along x", "oblique"])
def test_a_collinear_aggregate_drops_the_rotation_about_its_line(direction):
    # A chain of nodes: every aggregate is collinear, and the rotation about its line is no displacement at all. Along
    # an axis that rotation is one of the three modes, exactly zero; along an oblique line it is a combination of the
    # three, which the Gram-Schmidt sweeps reduce to rounding. Either is dropped, and the remaining modes still
    # reproduce all six rigid-body modes of the chain.
    nNodes = 40
    chain = sp.diags([-1.0, 2.01, -1.0], [-1, 0, 1], shape=(nNodes, nNodes))
    As, scaling = equilibrated(sp.kron(chain, NODAL_COUPLING).tocsr())
    coordinates = np.outer(np.arange(nNodes, dtype=float), np.asarray(direction) / np.linalg.norm(direction))
    preconditioner = _preconditioner(3, prolongation="tentative")
    preconditioner.build(As, scaling, coordinates)

    aggregates = preconditioner.aggregates
    sizes = np.bincount(aggregates[aggregates >= 0])
    # one rotation of an aggregate of two or more nodes, all three of a single node
    assert preconditioner.droppedModes == np.sum(np.where(sizes > 1, 1, 3)) > 0
    P, coarseNullspace = preconditioner.prolongationMatrix(), preconditioner.coarseNullspace()
    modes = _rigidBodyModes(As, scaling, coordinates, 3, aggregates)
    assert _relativeDifference(P @ coarseNullspace, modes) < 1e-12
    # the dropped modes are decoupled coarse dofs: their diagonal entry is the padding, their right-hand side zero
    r = np.random.default_rng(4).standard_normal(As.shape[0])
    assert np.all(np.isfinite(preconditioner.applyPreconditioner(r)))


def test_the_translations_alone_are_amgcls_own_first_coarse_level():
    # The control of the construction: with the rotations left out, the coarse space of every aggregate is spanned by
    # its translations D^-1 t, as in AMGCL's hierarchy on the block backend with blockTranslationNullspace -- the same
    # aggregates, the same tentative prolongation (per component, normalized), the same smoothing and the same
    # Galerkin product. With both coarse levels solved directly, the two preconditioners then differ only by the
    # spectral-radius estimate of the fine Chebyshev smoother (AMGCL's from per-thread random vectors, the two-grid
    # cycle's from a deterministic one), which enough power iterations make negligible. Without Dirichlet rows:
    # AMGCL's translations are not zero there, those of the rigid-body coarse level are.
    n = 12
    stiffness = sp.kron(gridLaplacian(n) + 0.01 * sp.identity(n**3), NODAL_COUPLING)
    rowScaling = sp.diags(10.0 ** np.random.default_rng(3).uniform(0.0, 2.0, 3 * n**3))
    As, scaling = equilibrated((rowScaling @ stiffness @ rowScaling).tocsr())
    axis = np.arange(n, dtype=float)
    # in the node order of gridLaplacian, the last direction fastest
    coordinates = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    tree = _defaultFieldPrecond(3)
    tree["relax"]["power_iters"] = 3000

    amgcl = PyAMGCLSolver(
        {
            "precond": _amgclTree(tree),
            "solver": {"type": "preonly"},
            "backendBlockSize": 3,
            "backendPrecision": "double",
        }
    )
    amgcl.set_nullspace(translationNullspace(FieldBlock("displacement", 0, As.shape[0], 3), scaling))
    amgcl.build(As)
    assert hierarchyLevels(amgcl) == 2

    translations = RigidBodyCoarseLevelPreconditioner(_amgclTree(tree), 3, "double", rotations=False)
    translations.build(As, scaling, coordinates)
    rigidBody = RigidBodyCoarseLevelPreconditioner(_amgclTree(tree), 3, "double")
    rigidBody.build(As, scaling, coordinates)
    assert translations.coarseNullspace().shape[1] == 3 and translations.droppedModes == 0

    r = np.random.default_rng(5).standard_normal(As.shape[0])
    expected = amgcl.applyPreconditioner(r)
    print(
        "relative difference to AMGCL's first coarse level: translations alone {:.1e}, rigid-body modes {:.1e}".format(
            _relativeDifference(translations.applyPreconditioner(r), expected),
            _relativeDifference(rigidBody.applyPreconditioner(r), expected),
        )
    )
    assert _relativeDifference(translations.applyPreconditioner(r), expected) < 1e-6
    # and the test can tell: the rotations are another coarse space
    assert _relativeDifference(rigidBody.applyPreconditioner(r), expected) > 1e-3


@pytest.mark.parametrize("dimension", [3, 2], ids=["3D hexahedra", "2D quadrilaterals"])
def test_the_rotations_cut_the_iterations_of_a_bending_dominated_beam(dimension):
    # What the rigid-body coarse level is for: the smooth error of a slender beam is bending, which rotates every
    # aggregate. The translations alone cannot represent it, the rotations can.
    As, scaling, coordinates, _ = _equilibratedBeam(dimension)
    iterations = {}
    for rotations in (False, True):
        preconditioner = _preconditioner(dimension, rotations=rotations)
        preconditioner.build(As, scaling, coordinates)
        iterations[rotations] = _gmresIterations(As, preconditioner)
    print("GMRES iterations, translations alone / rigid-body modes:", iterations[False], iterations[True])
    assert iterations[True] <= _GMRES_CAP and iterations[True] < 0.5 * iterations[False]


def test_float_storage_rounds_only_the_stored_entries():
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    double, single = _preconditioner(3, "double"), _preconditioner(3, "float")
    double.build(As, scaling, coordinates)
    single.build(As, scaling, coordinates)
    # the spectral radius comes from the double operator in both, and the setup is in double
    assert single.fineSpectralRadius == double.fineSpectralRadius
    r = np.random.default_rng(6).standard_normal(As.shape[0])
    assert _relativeDifference(single.applyPreconditioner(r), double.applyPreconditioner(r)) < 1e-5
    assert _relativeDifference(single.prolongationMatrix().data, double.prolongationMatrix().data) < 1e-7


def test_a_supplied_spectral_radius_is_rechecked_and_kept_on_the_same_operator():
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    first = _preconditioner(3)
    first.build(As, scaling, coordinates)
    assert first.fineSpectralRadiusSource == "estimated"
    again = _preconditioner(3, fineSpectralRadius=first.fineSpectralRadius, fineDominantVector=first.fineDominantVector)
    again.build(As, scaling, coordinates)
    assert again.fineSpectralRadiusSource == "supplied" and again.fineSpectralRadius == first.fineSpectralRadius
    # the same smoother; the coarse level is rebuilt, and AMGCL's power iterations in it (the prolongation smoothing
    # weight, the coarse smoother's spectral radius) sum their per-thread parts in arrival order, which differs from
    # run to run in the last bits
    r = np.random.default_rng(8).standard_normal(As.shape[0])
    assert _relativeDifference(again.applyPreconditioner(r), first.applyPreconditioner(r)) < 1e-9


def test_without_a_strong_connection_there_is_no_coarse_level():
    # nodes that are not coupled at all: no aggregate, and the cycle is the smoother alone (as AMGCL ends its own
    # hierarchy at an empty level)
    nNodes = 50
    As, scaling = equilibrated(sp.kron(sp.identity(nNodes), NODAL_COUPLING).tocsr())
    preconditioner = _preconditioner(3)
    preconditioner.build(As, scaling, np.random.default_rng(9).uniform(size=(nNodes, 3)))
    assert preconditioner.nAggregates == 0 and preconditioner.nCoarse == 0
    assert "the smoother alone" in preconditioner.report()
    r = np.random.default_rng(10).standard_normal(As.shape[0])
    # one Chebyshev sweep of degree 16 on the nodal blocks reduces the residual
    assert _relativeDifference(As @ preconditioner.applyPreconditioner(r), r) < 0.5


def test_a_fully_constrained_field_is_the_identity():
    nNodes = 20
    preconditioner = _preconditioner(3)
    preconditioner.build(sp.identity(3 * nNodes, format="csr"), np.ones(3 * nNodes), np.zeros((nNodes, 3)))
    r = np.random.default_rng(11).standard_normal(3 * nNodes)
    assert np.array_equal(preconditioner.applyPreconditioner(r), r)
    assert "identity" in preconditioner.report()


def test_coordinates_of_other_nodes_are_refused():
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    preconditioner = _preconditioner(3)
    with pytest.raises(RuntimeError, match="not the field's nodes"):
        preconditioner.build(As, scaling, coordinates[:-1])
    with pytest.raises(RuntimeError, match="no preconditioner built"):
        preconditioner.applyPreconditioner(np.ones(As.shape[0]))


@pytest.mark.parametrize(
    "tree, blockSize, backendPrecision, options, reason",
    [
        ({"relax": {"type": "spai0"}}, 3, "double", {}, "relax.type must be chebyshev"),
        ({"relax": {"type": "chebyshev", "scale": True, "power_iters": 10}}, 3, "double", {}, "relax.scale"),
        ({"relax": {"type": "chebyshev"}}, 3, "double", {}, "relax.power_iters must be at least 1"),
        ({"coarsening": {"type": "ruge_stuben"}}, 3, "double", {}, "coarsening.type must be smoothed_aggregation"),
        ({}, 1, "double", {}, "backendBlockSize 2 or 3"),
        ({}, 2, "float", {}, "2x2 blocks are stored in double only"),
        ({}, 3, "double", {"prolongation": "galerkin"}, "prolongation must be smoothed or tentative"),
        ({}, 3, "double", {"coarseHierarchyPrecision": "half"}, "coarseHierarchyPrecision must be double or float"),
    ],
    ids=["relax", "scale", "power_iters", "coarsening", "scalar", "2D float", "prolongation", "precision"],
)
def test_settings_it_cannot_use_raise_before_any_build(tree, blockSize, backendPrecision, options, reason):
    fullTree = {key: value for key, value in _amgclTree(_defaultFieldPrecond(3)).items() if key not in tree} | tree
    with pytest.raises(ValueError, match=reason):
        RigidBodyCoarseLevelPreconditioner(fullTree, blockSize, backendPrecision, **options)


# ----- in blockamg -----------------------------------------------------------------------------------------------


def _rigidBodyTree(value=True) -> dict:
    """The default 3D tree with the rigid-body coarse level."""
    return {**_defaultFieldPrecond(3), "rigidBodyCoarseLevel": value}


def _coarsenedTree(tree: dict) -> dict:
    """A tree whose AMGCL hierarchy coarsens also the small beam (AMGCL solves directly below 3000 unknowns): so that
    the default tree is a two-level method too, not a direct solve."""
    return {**tree, "coarse_enough": 100}


def _beamSystem():
    """The propped cantilever (unequilibrated: blockamg equilibrates itself), a right-hand side zero on its
    constrained dofs, its field block and node coordinates."""
    K, coordinates, constrained = _beam(3)
    b = np.random.default_rng(12).standard_normal(K.shape[0])
    b[constrained] = 0.0
    return K, b, FieldBlock("displacement", 0, K.shape[0], 3), coordinates


def _solve(fieldTree, **options):
    """Solve the beam with blockamg and the displacement field's tree; returns the solver and the relative residual."""
    K, b, block, coordinates = _beamSystem()
    settings = {"fieldPreconds": {"displacement": fieldTree}, "outerTol": 1e-8, "outerRestart": 300}
    settings.update({"verbosity": "silent", "nodeCoordinates": {"displacement": coordinates}} | options)
    solver = BlockAMGSolver(**settings)
    solver.setFieldStructure([block])
    x = solver(K, b)
    return solver, np.linalg.norm(b - K @ x) / np.linalg.norm(b)


def test_blockamg_solves_a_bending_dominated_beam_in_fewer_outer_iterations():
    default, defaultResidual = _solve(_coarsenedTree(_defaultFieldPrecond(3)))
    rigidBody, rigidBodyResidual = _solve(_coarsenedTree(_rigidBodyTree()))
    assert type(rigidBody._preconditioners[0]).__name__ == "RigidBodyCoarseLevelPreconditioner"
    assert defaultResidual <= 1.0001e-8 and rigidBodyResidual <= 1.0001e-8
    print("outer iterations, default / rigid-body coarse level:", default._lastOuterIters, rigidBody._lastOuterIters)
    assert rigidBody._lastOuterIters < 0.5 * default._lastOuterIters


def test_the_tentative_prolongation_and_a_float_hierarchy_below_are_options():
    solver, residual = _solve(_rigidBodyTree({"prolongation": "tentative", "coarseHierarchyPrecision": "float"}))
    assert residual <= 1.0001e-8
    preconditioner = solver._preconditioners[0]
    assert preconditioner.prolongationSmoothingWeight == 0.0
    assert "tentative prolongation" in preconditioner.report() and "unknowns" in preconditioner.report()


def test_without_node_coordinates_the_field_gets_its_tree_without_the_key_and_a_warning(capsys):
    K, b, block, _ = _beamSystem()
    solver = BlockAMGSolver(fieldPreconds={"displacement": _rigidBodyTree()}, outerTol=1e-8, outerRestart=300)
    solver.setFieldStructure([block])
    x = solver(K, b)
    assert "asks for rigidBodyCoarseLevel, but its node coordinates are unavailable" in capsys.readouterr().out
    assert isinstance(solver._preconditioners[0], PyAMGCLSolver)
    assert np.linalg.norm(b - K @ x) <= 1.0001e-8 * np.linalg.norm(b)


def _stubModel(coordinates):
    """A model and DOF manager with exactly what ``setModel`` reads: one 3D field with its node coordinates."""
    nodes = [SimpleNamespace(coordinates=c) for c in coordinates]
    model = SimpleNamespace(nodeFields={"displacement": SimpleNamespace(dimension=3, nodes=nodes)})
    dofManager = SimpleNamespace(idcsOfFieldsInDofVector={"displacement": slice(0, 3 * len(coordinates))})
    return model, dofManager


def test_the_coordinates_come_from_the_model_and_are_read_again_when_it_changes():
    K, b, block, coordinates = _beamSystem()
    solver = BlockAMGSolver(fieldPreconds={"displacement": _rigidBodyTree()}, outerTol=1e-8, verbosity="silent")
    solver.setModel(*_stubModel(coordinates))
    solver(K, b)
    assert type(solver._preconditioners[0]).__name__ == "RigidBodyCoarseLevelPreconditioner"
    assert np.array_equal(solver._nodeCoordinateCache["displacement"], coordinates)
    # a new model (e.g. after adaptive refinement) drops the coordinates of the old one
    solver.setModel(*_stubModel(coordinates))
    assert solver._nodeCoordinateCache == {}


def test_a_reused_fine_spectral_radius_is_rechecked_on_the_next_build():
    solver, _ = _solve(_rigidBodyTree({"fineSpectralRadiusReuse": True}), reuseHierarchy=False)
    first = solver._preconditioners[0]
    assert first.fineSpectralRadiusSource == "estimated"
    assert solver._fineSpectralRadiusCache["displacement"][0] == first.fineSpectralRadius
    K, b, _, _ = _beamSystem()
    solver(K, b)
    assert solver._preconditioners[0] is not first
    assert solver._preconditioners[0].fineSpectralRadiusSource == "supplied"


class _CondensedNodes(SchurComplementCondensation):
    """A Schur complement condensation of all dofs of given nodes, without a model."""

    def __init__(self, nodes, dimension):
        super().__init__([], SuperLUSolver)
        self.nodes, self.dimension = np.asarray(nodes), dimension

    def _selectCondensedDofs(self):
        dofs = (self.dimension * self.nodes[:, None] + np.arange(self.dimension)).ravel()
        return dofs, np.repeat(self.nodes, self.dimension)


def test_the_remaining_nodes_of_a_schur_complement_condensation_get_their_coordinates():
    # the condensed field of schurComplement is a field like any other: its remaining nodes, with their coordinates
    K, b, block, coordinates = _beamSystem()
    condensedNodes = np.flatnonzero(coordinates[:, 0] > 14.0)
    solver = BlockAMGSolver(
        fieldPreconds={"displacement": _rigidBodyTree()},
        outerTol=1e-8,
        verbosity="silent",
        nodeCoordinates={"displacement": coordinates},
        condensation=_CondensedNodes(condensedNodes, 3),
    )
    solver.setFieldStructure([block])
    x = solver(K, b)
    assert np.linalg.norm(b - K @ x) <= 1.0001e-8 * np.linalg.norm(b)
    preconditioner = solver._preconditioners[0]
    assert type(preconditioner).__name__ == "RigidBodyCoarseLevelPreconditioner"
    assert len(preconditioner.aggregates) == len(coordinates) - len(condensedNodes)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"rigidBodyCoarseLevel": True, "backendBlockSize": 1, "backendPrecision": "double"}, "needs a block backend"),
        ({"rigidBodyCoarseLevel": "yes"}, "must be true, false or an object"),
        ({"rigidBodyCoarseLevel": {"smoothing": "jacobi"}}, "unknown option"),
        ({"rigidBodyCoarseLevel": {"fineSpectralRadiusReuse": 1}}, "fineSpectralRadiusReuse must be true or false"),
        ({"rigidBodyCoarseLevel": {"prolongation": "galerkin"}}, "prolongation must be smoothed or tentative"),
        ({"rigidBodyCoarseLevel": True, "relax": {"type": "spai0"}}, "relax.type must be chebyshev"),
        (
            {"rigidBodyCoarseLevel": True, "relax": {"type": "chebyshev", "degree": 0, "power_iters": 10}},
            "relax.degree must be at least 1",
        ),
    ],
    ids=["scalar backend", "not a bool", "unknown option", "reuse not a bool", "prolongation", "relax", "degree"],
)
def test_blockamg_refuses_a_rigid_body_coarse_level_it_cannot_build(change, reason):
    with pytest.raises(ValueError, match=reason):
        createSolver({"fieldPreconds": {"displacement": {**_defaultFieldPrecond(3), **change}}})


def test_a_field_has_either_the_p_two_grid_or_the_rigid_body_coarse_level():
    with pytest.raises(ValueError, match="ask for both the p-two-grid and the rigid-body coarse level"):
        createSolver({"p1FieldNames": ["displacement"], "fieldPreconds": {"displacement": _rigidBodyTree()}})
    # switched off, the key is just a key
    createSolver({"p1FieldNames": ["displacement"], "fieldPreconds": {"displacement": _rigidBodyTree(False)}})
