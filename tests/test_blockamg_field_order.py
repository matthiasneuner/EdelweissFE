#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver: the order in which a block Gauss-Seidel sweep visits the fields (``fieldOrder``).

A gradient-enhanced damage system is (nearly) block upper-triangular: the displacement rows depend on the
nonlocal field (K_ud != 0), the nonlocal rows barely on the displacement (K_du ~ 0). Visiting the nonlocal
field first makes one sweep the block back-substitution of that structure; the DOF order (displacement first)
sweeps with the nonlocal correction still zero and ignores K_ud.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver


def _oneWayCoupledSystem(n=10, coupling=0.5):
    """A displacement-like block (3 dofs per node of an n^3 grid Laplacian) driven by a scalar block through
    K_ud only; the scalar rows do not see the displacement."""
    lap1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    lap3 = sp.kron(sp.kron(lap1, eye), eye) + sp.kron(sp.kron(eye, lap1), eye) + sp.kron(sp.kron(eye, eye), lap1)
    nodes = n**3
    Kuu = sp.kron(lap3 + 0.01 * sp.identity(nodes), sp.identity(3)).tocsr()
    Kdd = (lap3 + sp.identity(nodes)).tocsr()
    Kud = coupling * sp.kron(sp.identity(nodes), np.ones((3, 1))).tocsr()
    A = sp.bmat([[Kuu, Kud], [None, Kdd]], format="csr")
    blocks = [FieldBlock("displacement", 0, 3 * nodes, 3), FieldBlock("nonlocal damage", 3 * nodes, 4 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, blocks


def _outerIterations(A, b, blocks, **options):
    config = {"outerTol": 1e-8, "verbosity": "silent", "symmetric": False, "trueResidualMaxContinuations": 0}
    config.update(options)
    solver = createSolver(config)
    solver.setFieldStructure(blocks)
    solver(A, b)
    return solver._lastOuterIters, solver


def test_the_default_sweep_order_is_the_dof_order():
    _, _, blocks = _oneWayCoupledSystem(n=4)
    solver = createSolver({"verbosity": "silent"})
    assert solver._sweepOrder(blocks) == [0, 1]


def test_listed_fields_come_first_the_rest_follow_in_dof_order():
    _, _, blocks = _oneWayCoupledSystem(n=4)
    solver = createSolver({"verbosity": "silent", "fieldOrder": "nonlocal damage"})
    assert solver._sweepOrder(blocks) == [1, 0]


def test_an_unknown_field_name_raises():
    _, _, blocks = _oneWayCoupledSystem(n=4)
    solver = createSolver({"verbosity": "silent", "fieldOrder": ["damage"]})
    with pytest.raises(ValueError, match="fieldOrder"):
        solver._sweepOrder(blocks)


def test_visiting_the_driving_field_first_needs_fewer_outer_iterations():
    A, b, blocks = _oneWayCoupledSystem()
    dofOrder, _ = _outerIterations(A, b, blocks)
    drivingFirst, solver = _outerIterations(A, b, blocks, fieldOrder=["nonlocal damage"])

    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) < 1e-7
    assert drivingFirst < dofOrder
