#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver with ``krylovOnScaledSystem=False``: the outer LGMRES runs on the unscaled system A,
right-preconditioned by x = D M^-1 (D r), so the residual it converges is the true one the solve is accepted
on -- no gap, no continuations.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver


def _system(n=12):
    """Two weakly coupled fields, rows scaled over two orders of magnitude so the equilibration matters."""
    lap1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    lap3 = sp.kron(sp.kron(lap1, eye), eye) + sp.kron(sp.kron(eye, lap1), eye) + sp.kron(sp.kron(eye, eye), lap1)
    nodes = n**3
    Kuu = sp.kron(lap3 + 0.01 * sp.identity(nodes), sp.identity(3))
    rowScale = sp.diags(np.logspace(0, 2, 3 * nodes))
    Kuu = (rowScale @ Kuu @ rowScale).tocsr()
    Kdd = (lap3 + sp.identity(nodes)).tocsr()
    Kud = sp.random(3 * nodes, nodes, density=1e-3, random_state=1) * 1e-2
    A = sp.bmat([[Kuu, Kud], [None, Kdd]], format="csr")
    blocks = [FieldBlock("displacement", 0, 3 * nodes, 3), FieldBlock("nonlocal damage", 3 * nodes, 4 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, blocks


@pytest.mark.parametrize("outerSolver", ["amgcl_lgmres", "scipy"])
def test_the_unscaled_outer_solve_converges_the_true_residual(outerSolver):
    A, b, blocks = _system()
    solver = createSolver(
        {
            "verbosity": "silent",
            "outerTol": 1e-8,
            "outerRestart": 200,
            "outerSolver": outerSolver,
            "krylovOnScaledSystem": False,
        }
    )
    solver.setFieldStructure(blocks)
    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) <= 1.0001e-8 * np.linalg.norm(b)
    assert solver._lastContinuations == 0


def test_both_modes_solve_the_same_system():
    A, b, blocks = _system()
    solutions = []
    for scaled in (True, False):
        solver = createSolver({"verbosity": "silent", "outerTol": 1e-10, "krylovOnScaledSystem": scaled})
        solver.setFieldStructure(blocks)
        solutions.append(solver(A, b))
    assert np.linalg.norm(solutions[0] - solutions[1]) < 1e-7 * np.linalg.norm(solutions[1])
