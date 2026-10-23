import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from edelweissfe.linsolve.barschur import createSolver
from edelweissfe.linsolve.base import FieldBlock


def _bondSlipSystem(nx: int = 24, ny: int = 12, nBars: int = 3, bond: float = 50.0):
    """A small bond-slip-like system: a 2D grid of 'concrete' nodes (2 dofs each, a vector Laplacian, fixed at x = 0),
    and bars along x (2 dofs per node, stiff axial springs along the bar), each bar node tied by bond springs to the
    concrete node it lies on. The bar dofs follow the concrete dofs in one 'displacement' field."""
    nC = nx * ny
    rows, cols, vals = [], [], []

    def spring(i, j, k):
        for d in range(2):
            a, b = 2 * i + d, 2 * j + d
            rows.extend([a, b, a, b]), cols.extend([a, b, b, a]), vals.extend([k, k, -k, -k])

    node = lambda ix, iy: iy * nx + ix  # noqa: E731
    for iy in range(ny):
        for ix in range(nx):
            if ix + 1 < nx:
                spring(node(ix, iy), node(ix + 1, iy), 1.0)
            if iy + 1 < ny:
                spring(node(ix, iy), node(ix, iy + 1), 1.0)
    barNodes = []
    for bar in range(nBars):
        iy = (bar + 1) * ny // (nBars + 1)
        first = nC + bar * nx
        for ix in range(nx):
            barNodes.append(first + ix)
            spring(first + ix, node(ix, iy), bond)
            if ix + 1 < nx:
                spring(first + ix, first + ix + 1, 1000.0)
    n = 2 * (nC + nBars * nx)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # fix the concrete and the bars at x = 0 (identity rows)
    fixed = np.concatenate([[2 * node(0, iy), 2 * node(0, iy) + 1] for iy in range(ny)])
    fixed = np.concatenate([fixed] + [[2 * (nC + bar * nx), 2 * (nC + bar * nx) + 1] for bar in range(nBars)])
    keep = np.ones(n)
    keep[fixed] = 0.0
    D = sp.diags(keep)
    A = (D @ A @ D + sp.diags(1.0 - keep)).tocsr()
    b = np.zeros(n)
    b[2 * node(nx - 1, ny // 2)] = 1.0  # pull at the free end
    dofsPerNode = [np.array([2 * k, 2 * k + 1]) for k in barNodes]
    return A, b, [FieldBlock("displacement", 0, n, 2)], dofsPerNode


@pytest.mark.parametrize(
    "options", [{"schurApproximation": "local"}, {"schurApproximation": "nodal"}, {"innerCycles": None}]
)
def test_bar_schur_complement_preconditioner_converges(options):
    A, b, fieldStructure, dofsPerNode = _bondSlipSystem()
    solver = createSolver(options | {"schurWindow": 8, "blockamg": {"verbosity": "silent"}})
    solver.setBarDofs(fieldStructure, dofsPerNode, A.shape[0])
    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) <= 1e-3 * np.linalg.norm(b)
    assert solver.lastSolveSummary.residualMet
    reference = spsolve(A.tocsc(), b)
    assert np.linalg.norm(x - reference) <= 1e-2 * np.linalg.norm(reference)


def test_bar_schur_complement_is_reused_for_the_same_pattern():
    A, b, fieldStructure, dofsPerNode = _bondSlipSystem()
    solver = createSolver({"schurWindow": 8, "blockamg": {"verbosity": "silent"}})
    solver.setBarDofs(fieldStructure, dofsPerNode, A.shape[0])
    solver(A, b)
    assert "rebuilt" in solver.lastSolveSummary.detailLines[0]
    x = solver(1.01 * A, b)  # a new matrix with the same sparsity pattern
    assert "reused" in solver.lastSolveSummary.detailLines[0]
    assert np.linalg.norm(b - 1.01 * A @ x) <= 1e-3 * np.linalg.norm(b)


def test_blocks_are_those_of_the_matrix_for_the_same_pattern():
    A, b, fieldStructure, dofsPerNode = _bondSlipSystem()
    solver = createSolver({})
    solver.setBarDofs(fieldStructure, dofsPerNode, A.shape[0])
    bars, others = solver._barDofs, solver._otherDofs
    for M in (A, 1.01 * A):  # the second with the structures found for the first
        expected = (M[bars][:, bars], M[bars][:, others], M[others][:, bars], M[others][:, others])
        for block, reference in zip(solver._splitBlocks(M), expected):
            assert (block != reference).nnz == 0


@pytest.mark.parametrize("bandwidth", [3, None])
def test_transposed_solve_of_banded_and_dense_matrices(bandwidth):
    from edelweissfe.linsolve.barschur.barschur import _solveTransposed

    rng = np.random.default_rng(0)
    n = 60
    dense = rng.standard_normal((n, n)) + n * np.eye(n)
    if bandwidth is not None:  # a non-symmetric band: 3 below, 2 above the diagonal
        i, j = np.indices((n, n))
        dense[(i - j > bandwidth) | (j - i > bandwidth - 1)] = 0.0
    B = rng.standard_normal((n, 5))
    assert np.allclose(_solveTransposed(sp.coo_matrix(dense), B), np.linalg.solve(dense.T, B), rtol=1e-12, atol=1e-14)


def test_inner_vector_fields_get_fewer_chebyshev_power_iterations():
    from edelweissfe.linsolve.barschur.barschur import INNER_CHEBYSHEV_POWER_ITERATIONS
    from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond

    fields = [FieldBlock("displacement", 0, 30, 3), FieldBlock("damage", 30, 40, 1)]
    dofsPerNode = [np.arange(3), np.arange(3, 6)]
    solver = createSolver({})
    solver.setBarDofs(fields, dofsPerNode, 40)
    trees = solver._inner._fieldPreconds
    assert trees["displacement"]["relax"]["power_iters"] == INNER_CHEBYSHEV_POWER_ITERATIONS
    assert "damage" not in trees  # scalar fields keep blockamg's default
    own = _defaultFieldPrecond(3)
    solver = createSolver({"blockamg": {"fieldPreconds": {"displacement": own}}})
    solver.setBarDofs(fields, dofsPerNode, 40)
    assert solver._inner._fieldPreconds["displacement"] == own


def test_zero_right_hand_side_needs_no_setup():
    A, b, fieldStructure, dofsPerNode = _bondSlipSystem()
    solver = createSolver({})
    solver.setBarDofs(fieldStructure, dofsPerNode, A.shape[0])
    assert not np.any(solver(A, np.zeros_like(b)))
    assert solver.lastSolveSummary.iters == 0


def test_slow_solves_are_dumped_and_can_be_replayed(tmp_path):
    A, b, fieldStructure, dofsPerNode = _bondSlipSystem()
    solver = createSolver({"schurWindow": 8, "dumpSlowSolvesDir": str(tmp_path), "dumpSlowSolveSeconds": 0.0})
    solver.setBarDofs(fieldStructure, dofsPerNode, A.shape[0])
    x = solver(A, b)
    A2 = sp.load_npz(tmp_path / "slow_0_A.npz")
    sizes = np.load(tmp_path / "slow_0_barNodeSizes.npy")
    dofs = np.split(np.load(tmp_path / "slow_0_barDofs.npy"), np.cumsum(sizes)[:-1])
    replay = createSolver({"schurWindow": 8})
    replay.setBarDofs(fieldStructure, dofs, A2.shape[0])
    assert np.allclose(replay(A2, np.load(tmp_path / "slow_0_b.npy")), x)
