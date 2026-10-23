#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver's preconditioner on its own: preparePreconditioner builds it once for a matrix, applyPreconditioner
applies it (as within the Krylov solve of another solver, e.g. the outer solve of barschur)."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from _blockamgsystems import gridLaplacian

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver


def _system(n=16):
    """Two scalar fields, coupled, on an n^3 grid Laplacian each (above AMGCL's coarse_enough, so that the fields get
    real hierarchies)."""
    nodes = n**3
    K = (gridLaplacian(n) + 0.05 * sp.identity(nodes)).tocsr()
    A = sp.bmat([[K, 0.5 * sp.identity(nodes)], [0.5 * sp.identity(nodes), 2.0 * K]], format="csr")
    blocks = [FieldBlock("first", 0, nodes, 1), FieldBlock("second", nodes, 2 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, blocks


def test_prepared_preconditioner_accelerates_a_krylov_solve():
    A, b, blocks = _system()
    solver = createSolver({"verbosity": "silent"})
    solver.setFieldStructure(blocks)
    assert solver.preparePreconditioner(A)

    iterations = {"plain": 0, "preconditioned": 0}

    def counter(key):
        def count(_):
            iterations[key] += 1

        return count

    M = spla.LinearOperator(A.shape, matvec=solver.applyPreconditioner)
    x, info = spla.gmres(
        A, b, M=M, rtol=1e-8, restart=50, maxiter=20, callback=counter("preconditioned"), callback_type="pr_norm"
    )
    assert info == 0
    assert np.linalg.norm(b - A @ x) <= 1e-7 * np.linalg.norm(b)
    spla.gmres(A, b, rtol=1e-8, restart=50, maxiter=20, callback=counter("plain"), callback_type="pr_norm")
    assert iterations["preconditioned"] < iterations["plain"] / 3


def test_prepared_preconditioner_is_a_fixed_linear_operator():
    A, b, blocks = _system()
    solver = createSolver({"verbosity": "silent"})
    solver.setFieldStructure(blocks)
    solver.preparePreconditioner(A)
    r = np.random.default_rng(1).standard_normal(A.shape[0])
    first = solver.applyPreconditioner(b)
    assert np.array_equal(solver.applyPreconditioner(b), first)  # nothing is updated by an application
    assert np.allclose(solver.applyPreconditioner(b + 2.0 * r), first + 2.0 * solver.applyPreconditioner(r))


def test_preparing_a_non_finite_matrix_is_declined():
    A, b, blocks = _system()
    A = A.tolil()
    A[3, 3] = np.inf
    solver = createSolver({"verbosity": "silent"})
    solver.setFieldStructure(blocks)
    assert not solver.preparePreconditioner(A.tocsr())
