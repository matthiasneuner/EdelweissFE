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

"""Schur complement preconditioner for embedded bars with their own dofs (bond-slip), around ``blockamg``.

Bars coupled to a host continuum by bond elements (:mod:`edelweissfe.generators.embeddedbond`) have their own
displacement dofs (and, for beams, rotation dofs). The rows of these dofs are dominated by the bond springs to the
nodes of the host element, i.e., by a relation :math:`u_{bar} \\approx \\sum_i N_i u_i` that an aggregation AMG coarse
space cannot represent: ``blockamg`` alone stalls on such systems. With the dofs ordered as bar dofs :math:`b` and all
other dofs :math:`k`,

.. math::

    \\begin{bmatrix} A_{bb} & A_{bk} \\\\ A_{kb} & A_{kk} \\end{bmatrix}
    \\begin{bmatrix} x_b \\\\ x_k \\end{bmatrix} =
    \\begin{bmatrix} r_b \\\\ r_k \\end{bmatrix},

the bar block :math:`A_{bb}` consists of independent chains (one per bar) and is factorized exactly (sparse LU). The
other dofs see the Schur complement :math:`S = A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}`, which is dense along each bar. It
is approximated by

- ``schurApproximation = "local"`` (the default): the correction :math:`A_{kb} A_{bb}^{-1} A_{bk}` computed in
  overlapping windows along each bar (``schurWindow``) by dense solves, with the entries
  :math:`|\\tilde{S}_{ij}| < \\tau \\sqrt{|\\tilde{S}_{ii} \\tilde{S}_{jj}|}` dropped (``schurDropTolerance``
  :math:`\\tau`). The response of a bar on its bond decays within a few host elements, so this is sparse and
  accurate, for trusses and for beams, at a cost linear in the number of bar dofs,
- ``schurApproximation = "nodal"``: :math:`\\tilde{S} = A_{kk} - A_{kb} D^{-1} A_{bk}` with :math:`D` the diagonal
  block of :math:`A_{bb}` of each bar node, which couples only the nodes of the host element of a bar node; good for
  trusses, too crude for beams, whose bar rows are dominated by the bending stiffness along the bar,
- ``schurApproximation = "none"``: :math:`\\tilde{S} = A_{kk}`.

The preconditioner is the block LU factorization with :math:`\\tilde{S}` solved by an inner ``blockamg`` solve to a
loose tolerance:

.. math::

    y_b = A_{bb}^{-1} r_b, \\quad z_k = \\tilde{S}^{-1} (r_k - A_{kb} y_b), \\quad
    z_b = A_{bb}^{-1} (r_b - A_{bk} z_k).

The inner solves make the preconditioner vary slightly from one application to the next, so the outer solve is a
flexible GMRES (FGMRES) on the full system. With ``innerCycles``, :math:`\\tilde{S}^{-1}` is instead a fixed number of
applications of the inner ``blockamg`` preconditioner, built once per :math:`\\tilde{S}`
(:meth:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver.preparePreconditioner`).

The bar dofs are all dofs of the nodes that belong only to 1D elements (``bar2``/``bar3``: the bars and the bond
elements, but not to the host elements). With perfect bond (``embeddedRegion``)
these dofs are eliminated as multi-point constraint slaves: use ``blockamg`` directly there.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee
from scipy.sparse.linalg import splu

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import FieldBlock, LinearSolver, LinearSolveSummary
from edelweissfe.linsolve.blockamg import createSolver as createBlockAMGSolver
from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond

#: The power iterations of the Chebyshev smoother's spectral radius estimate in the inner hierarchies of 3D vector
#: fields, instead of blockamg's 300: an inner hierarchy is rebuilt with every approximate Schur complement, and on a
#: hex20 tunnel segment with bond-slip (2.07 M dofs) 50 gave the same outer iterations, elastic and through cracking,
#: at a third of the build time (3.5 instead of 10.4 s).
INNER_CHEBYSHEV_POWER_ITERATIONS = 50


class BarSchurSolver(LinearSolver):
    """Flexible GMRES, preconditioned by the block LU factorization of the bar dofs and an approximate Schur complement
    of all other dofs, see the module documentation.

    Parameters
    ----------
    blockamgOptions
        The options of the inner ``blockamg`` solver of the approximate Schur complement. Its tolerance is
        ``innerTolerance``, and its hierarchies are built once per approximate Schur complement. A 3D vector field
        without its own ``fieldPreconds`` tree gets ``blockamg``'s default one with
        :data:`INNER_CHEBYSHEV_POWER_ITERATIONS` power iterations for the Chebyshev smoother's spectral radius.
    outerTolerance
        The relative residual :math:`\\|b - A x\\| / \\|b\\|` of the outer FGMRES solve.
    innerTolerance
        The relative tolerance of each inner ``blockamg`` solve (``innerCycles`` ``None`` only).
    innerCycles
        Instead of an inner ``blockamg`` solve to ``innerTolerance``, apply its preconditioner (one block
        Gauss-Seidel sweep of AMG cycles) this many times as a stationary iteration on :math:`\\tilde{S}`; the outer
        FGMRES does the acceleration. ``None``: inner solves.
    restart
        The FGMRES restart length.
    maxIterations
        The maximum number of FGMRES iterations of a solve.
    schurApproximation
        ``"local"``, ``"nodal"`` or ``"none"``, see the module documentation.
    schurDropTolerance
        ``"local"`` only: the relative size below which entries of the approximate Schur complement are dropped.
    schurWindow
        ``"local"`` only: the bar dofs per window core, see :meth:`_barWindows`.
    schurHalo
        ``"local"`` only: the bar dofs of the halo on either side of a window core (default: ``schurWindow``).
    reuseSchurComplement
        Keep the approximate Schur complement and the inner hierarchies for the following solves (Newton iterations)
        as long as the sparsity pattern does not change and the outer iterations do not drift, see
        ``schurStalenessFactor``. A solve with a reused one that is not converged after twice (and 20 more than) the
        outer iterations of the first solve after the last rebuild is repeated with a rebuilt one, and the next three
        solves rebuild directly.
    dumpSlowSolvesDir
        Write the systems of solves slower than ``dumpSlowSolveSeconds`` to this directory, for offline diagnosis
        (``slow_<n>_A.npz``, ``_b.npy``, ``_barDofs.npy``, ``_barNodeSizes.npy``, ``_fields.json``); ``None`` (the
        default) writes nothing.
    dumpSlowSolveSeconds
        The wall time above which a solve is dumped.
    dumpSlowSolvesMax
        The number of slow solves dumped at most.
    schurStalenessFactor
        A rebuild is due when a solve needs more than this factor times (and more than 10 more than) the outer
        iterations of the first solve after the last rebuild.
    """

    identification = "BarSchurSolver"
    reportsSolveSummary = True
    #: The approximate Schur complement and the inner hierarchies are reused while the sparsity pattern stays the
    #: same, see :attr:`~edelweissfe.linsolve.base.LinearSolver.needsStableSparsityPattern`.
    needsStableSparsityPattern = True

    def __init__(
        self,
        blockamgOptions: dict,
        outerTolerance: float = 1e-3,
        innerTolerance: float = 1e-1,
        innerCycles: int = 2,
        restart: int = 50,
        maxIterations: int = 1000,
        schurApproximation: str = "local",
        schurDropTolerance: float = 1e-4,
        schurWindow: int = 600,
        schurHalo: int = None,
        reuseSchurComplement: bool = True,
        schurStalenessFactor: float = 1.5,
        dumpSlowSolvesDir: str = None,
        dumpSlowSolveSeconds: float = 300.0,
        dumpSlowSolvesMax: int = 5,
    ):
        if schurApproximation not in ("local", "nodal", "none"):
            raise ValueError(
                f"barschur: schurApproximation must be 'local', 'nodal' or 'none', not '{schurApproximation}'"
            )
        if innerCycles is not None and innerCycles < 1:
            raise ValueError(f"barschur: innerCycles must be at least 1, not {innerCycles}")
        self._innerCycles = innerCycles
        self._schurDropTolerance = schurDropTolerance
        self._schurWindow = schurWindow
        self._schurHalo = schurWindow if schurHalo is None else schurHalo
        self._dumpSlowSolvesDir = dumpSlowSolvesDir
        self._dumpSlowSolveSeconds = dumpSlowSolveSeconds
        self._dumpSlowSolvesMax = dumpSlowSolvesMax
        self._slowSolvesDumped = 0
        self._windows = None  # (sparsity key of the bar block, windows), see _barWindows
        self._split = None  # (sparsity key of A, the structure of each block), see _splitBlocks
        self._reuseSchurComplement = reuseSchurComplement
        self._schurStalenessFactor = schurStalenessFactor
        self._schur = None  # (sparsity key of A, S~, S~ as a threaded product) of the last rebuild
        self._referenceIterations = None  # the outer iterations of the first solve after the last rebuild
        self._rebuildNext = False
        self._rebuildsWithoutReuse = 0  # after a reused approximation failed, the next solves rebuild directly
        self._nThreads = max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
        self._threadPool = ThreadPoolExecutor(self._nThreads) if self._nThreads > 1 else None
        # one matrix per outer solve, many inner solves with it: the inner hierarchies are refreshed explicitly
        self._innerOptions = {"verbosity": "silent"} | dict(blockamgOptions)
        self._innerOptions |= {
            "outerTol": innerTolerance,
            "reuseHierarchy": True,
            "refreshOnNewIncrement": False,
            "refreshOnCapHit": False,
        }
        self._inner = None  # created for the field blocks of the other dofs, see setBarDofs
        self._outerTolerance = outerTolerance
        self._restart = restart
        self._maxIterations = maxIterations
        self._schurApproximation = schurApproximation
        self._barDofs = None
        self._otherDofs = None
        self._barNodeBlocks = None
        self.lastSolveSummary = None

    def setModel(self, model, dofManager) -> None:
        """Find the bar dofs and the field blocks of all other dofs, see
        :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel`."""
        super().setModel(model, dofManager)
        # bar nodes carry only 1D elements (bars and the bond elements, whose other nodes belong to the host elements)
        isBarElement = {el: el.ensightType in ("bar2", "bar3") for el in model.elements.values()}
        continuumNodes = {node for el, isBar in isBarElement.items() if not isBar for node in el.nodes}
        barNodes = sorted(
            {node for el, isBar in isBarElement.items() if isBar for node in el.nodes} - continuumNodes,
            key=lambda node: node.label,
        )
        indices = dofManager.idcsOfFieldVariablesInDofVector
        dofsPerNode = [
            np.concatenate([np.atleast_1d(indices[fv]) for fv in node.fields.values() if fv in indices])
            for node in barNodes
        ]
        self.setBarDofs(self._fieldStructure, dofsPerNode, dofManager.nDof)
        if self._journal is not None:
            self._journal.message(
                f"{len(self._barDofs)} bar dofs of {len(barNodes)} bar nodes condensed, {len(self._otherDofs)} other "
                f"dofs by blockamg ({self._schurApproximation} Schur complement approximation)",
                self.identification,
                1,
            )

    def setBarDofs(self, fieldStructure: list, dofsPerNode: list, nDof: int) -> None:
        """Set the bar dofs and the field blocks of all other dofs directly, without a model.

        Parameters
        ----------
        fieldStructure
            The :class:`~edelweissfe.linsolve.base.FieldBlock` of every field of the system.
        dofsPerNode
            The dofs of each bar node (3 for a bar, 6 for a beam with rotations).
        nDof
            The size of the system.
        """
        self._fieldStructure = list(fieldStructure)
        self._barDofs = np.concatenate(dofsPerNode)
        isBar = np.zeros(nDof, dtype=bool)
        isBar[self._barDofs] = True
        self._otherDofs = np.flatnonzero(~isBar)
        # the diagonal blocks of the bar nodes: (first position in the bar dofs, number of dofs)
        sizes = np.array([len(d) for d in dofsPerNode])
        self._barNodeBlocks = (np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes)

        # the field blocks of the other dofs, which keep their order (fields are contiguous and field-major)
        otherBlocks, start = [], 0
        for block in fieldStructure:
            n = int((~isBar[block.start : block.stop]).sum())
            if n:
                otherBlocks.append(FieldBlock(block.name, start, start + n, block.dimension))
                start += n
        fieldPreconds = dict(self._innerOptions.get("fieldPreconds", {}))
        for block in otherBlocks:
            if block.dimension == 3 and block.name not in fieldPreconds:
                tree = _defaultFieldPrecond(3)
                tree["relax"] = tree["relax"] | {"power_iters": INNER_CHEBYSHEV_POWER_ITERATIONS}
                fieldPreconds[block.name] = tree
        self._inner = createBlockAMGSolver(self._innerOptions | {"fieldPreconds": fieldPreconds})
        self._inner.setFieldStructure(otherBlocks)

    def _inverseNodalBlocks(self, Abb: sp.csr_matrix) -> sp.csr_matrix:
        """The block diagonal matrix of the inverted diagonal blocks of the bar nodes (3 or 6 dofs each)."""
        starts, sizes = self._barNodeBlocks
        nodeOfDof = np.repeat(np.arange(len(sizes)), sizes)
        localOfDof = np.arange(Abb.shape[0]) - np.repeat(starts, sizes)
        m = int(sizes.max())
        # padded to m x m, with the identity on the padding
        blocks = np.zeros((len(sizes), m, m))
        blocks[:, np.arange(m), np.arange(m)] = (np.arange(m)[None, :] >= sizes[:, None]).astype(float)
        coo = Abb.tocoo()
        inBlock = nodeOfDof[coo.row] == nodeOfDof[coo.col]
        rows, cols = coo.row[inBlock], coo.col[inBlock]
        blocks[nodeOfDof[rows], localOfDof[rows], localOfDof[cols]] = coo.data[inBlock]
        inverses = np.linalg.inv(blocks)
        # all (row, col) pairs of each node's block, as positions in the bar dofs
        pairRow = np.repeat(np.arange(Abb.shape[0]), sizes[nodeOfDof])
        pairCol = np.repeat(starts[nodeOfDof], sizes[nodeOfDof]) + np.concatenate(
            [np.arange(sizes[k]) for k in nodeOfDof]
        )
        values = inverses[nodeOfDof[pairRow], localOfDof[pairRow], localOfDof[pairCol]]
        return sp.csr_matrix((values, (pairRow, pairCol)), shape=Abb.shape)

    def _approximateSchurComplement(self, Abb, Abk, Akb, Akk) -> sp.csr_matrix:
        """The approximate Schur complement of the other dofs that the inner ``blockamg`` solves, see the module
        documentation."""
        if self._schurApproximation == "local":
            return self._localSchurComplement(Abb, Abk, Akb, Akk)
        if self._schurApproximation == "nodal":
            return (Akk - Akb @ self._inverseNodalBlocks(Abb.tocsr()) @ Abk).tocsr()
        return Akk

    def _barWindows(self, Abb: sp.csr_matrix) -> list:
        """Overlapping windows along the bars, cached for the sparsity pattern of the bar block.

        The dofs of each bar (a connected component of the bar block) are ordered along the bar by the reverse
        Cuthill-McKee ordering and cut into cores of ``schurWindow`` dofs; each window is a core with a halo of
        ``schurHalo`` dofs on either side. A bar no longer than a core with both halos is one window.

        Returns
        -------
        list
            ``(dofs of the window, positions of its core in the window)`` for every window; the cores partition the
            bar dofs.
        """
        key = (Abb.shape[0], Abb.nnz)
        if self._windows is not None and self._windows[0] == key:
            return self._windows[1]
        order = reverse_cuthill_mckee(Abb, symmetric_mode=True)
        nBars, barOfDof = connected_components(Abb, directed=False)
        alongBars = order[np.argsort(barOfDof[order], kind="stable")]
        w, h = self._schurWindow, self._schurHalo
        windows = []
        for bar in np.split(alongBars, np.cumsum(np.bincount(barOfDof))[:-1]):
            n = len(bar)
            if n <= w + 2 * h:
                windows.append((bar, np.arange(n)))
                continue
            for start in range(0, n, w):
                stop = min(start + w, n)
                low, high = max(0, start - h), min(n, stop + h)
                windows.append((bar[low:high], np.arange(start - low, stop - low)))
        self._windows = (key, windows)
        return windows

    def _localSchurComplement(self, Abb, Abk, Akb, Akk) -> sp.csr_matrix:
        """The Schur complement with the correction :math:`A_{kb} A_{bb}^{-1} A_{bk}` computed window by window along
        the bars (:meth:`_barWindows`), and its small entries dropped.

        For the dofs of a window core, :math:`A_{bb}^{-1}` is replaced by the inverse of the window's block: the
        response of a bar on its bond decays within a few host elements, far within the halo. Each window's
        contribution is thinned out at a tenth of the drop tolerance (relative to :math:`A_{kk}`) before the windows
        are combined, which bounds the memory."""
        tau = self._schurDropTolerance
        Abb, Abk, Akb = Abb.tocsr(), Abk.tocsr(), Akb.tocsc()
        akk = np.sqrt(np.abs(Akk.diagonal()))

        def windowCorrection(window):
            dofs, core = window
            Bk = Abk[dofs]
            hostDofs = np.unique(Bk.indices)  # the other dofs coupled to the window
            Kb = Akb[:, dofs[core]]
            coreHostDofs = np.unique(Kb.indices)  # the other dofs coupled to the core
            if len(hostDofs) == 0 or len(coreHostDofs) == 0:
                return None
            # the core rows of the window's inverse, G = I[core] A_ww^-1 (from A_ww^T G^T = I[:, core]): far fewer
            # right-hand sides than host dofs, and the products with the sparse coupling blocks stay sparse
            unitColumns = np.zeros((len(dofs), len(core)))
            unitColumns[core, np.arange(len(core))] = 1.0
            G = _solveTransposed(Abb[dofs][:, dofs].tocoo(), unitColumns).T
            correction = (Bk[:, hostDofs].T @ (Kb[coreHostDofs] @ G).T).T
            significant = np.abs(correction) >= 0.1 * tau * np.outer(akk[coreHostDofs], akk[hostDofs])
            i, j = np.nonzero(significant)
            return coreHostDofs[i], hostDofs[j], correction[i, j]

        with performancetiming.timeit("S~ window solves"):
            windows = self._barWindows(Abb)
            parts = [p for p in self._map(windowCorrection, windows) if p is not None]
        with performancetiming.timeit("S~ assembly"):
            with performancetiming.timeit("correction matrix"):
                rows, cols, values = (np.concatenate(v) for v in zip(*parts))
                correction = sp.csr_matrix((values, (rows, cols)), shape=Akk.shape)
            with performancetiming.timeit("subtraction"):
                S = Akk - correction
            with performancetiming.timeit("drop"):
                s = np.sqrt(np.abs(S.diagonal()))

                def dropSmallEntries(rowRange):
                    low, high = rowRange
                    entries = slice(S.indptr[low], S.indptr[high])
                    rowOfEntry = np.repeat(np.arange(low, high), np.diff(S.indptr[low : high + 1]))
                    keep = np.abs(S.data[entries]) >= tau * s[rowOfEntry] * s[S.indices[entries]]
                    keptPerRow = np.bincount(rowOfEntry[keep] - low, minlength=high - low)
                    return S.data[entries][keep], S.indices[entries][keep], keptPerRow

                bounds = np.linspace(0, S.shape[0], self._nThreads + 1).astype(int)
                parts = self._map(dropSmallEntries, list(zip(bounds[:-1], bounds[1:])))
                data, indices, keptPerRow = (np.concatenate(v) for v in zip(*parts))
                indptr = np.concatenate([[0], np.cumsum(keptPerRow)])
                return sp.csr_matrix((data, indices, indptr), shape=S.shape)

    def _splitBlocks(self, A: sp.csr_matrix) -> list:
        """The blocks :math:`A_{bb}` (CSC), :math:`A_{bk}`, :math:`A_{kb}` and :math:`A_{kk}` (CSR) of ``A``.

        The sparsity pattern of ``A`` stays the same from one solve to the next (see
        :attr:`needsStableSparsityPattern`): the blocks are cut out once per pattern from a matrix of the positions of
        ``A``'s entries, and each solve gathers the values at these positions into the stored structures."""
        key = (A.shape[0], A.nnz)
        if self._split is None or self._split[0] != key:
            bars, others = self._barDofs, self._otherDofs
            # positions + 1, so that no position is a zero that could be dropped as such
            positions = sp.csr_matrix((np.arange(1, A.nnz + 1, dtype=float), A.indices, A.indptr), shape=A.shape)
            blocks = (
                positions[bars][:, bars].tocsc(),
                positions[bars][:, others].tocsr(),
                positions[others][:, bars].tocsr(),
                positions[others][:, others].tocsr(),
            )
            self._split = (key, [(B.data.astype(np.int64) - 1, B.indices, B.indptr, B.shape, type(B)) for B in blocks])
        return [kind((A.data[at], indices, indptr), shape=shape) for at, indices, indptr, shape, kind in self._split[1]]

    def _map(self, function, items) -> list:
        """``function`` applied to every item, in the thread pool (the dense solves release the GIL)."""
        if self._threadPool is None:
            return [function(item) for item in items]
        return list(self._threadPool.map(function, items))

    def __call__(self, A, b):
        tic = time.time()
        b = np.asarray(b, dtype=float)
        if not np.any(b):
            # nothing to solve (e.g. the first iteration of a step without load change): no setup either
            self.lastSolveSummary = LinearSolveSummary(iters=0, residual=0.0, residualMet=True, retries=0)
            return np.zeros_like(b)
        A = A.tocsr()
        bars, others = self._barDofs, self._otherDofs
        with performancetiming.timeit("barschur split"):
            Abb, Abk, Akb, Akk = self._splitBlocks(A)
        with performancetiming.timeit("bar LU"):
            barFactorization = splu(Abb)
        product = _RowBlockProduct(A, self._nThreads, self._threadPool)
        timing = {"split+LU": time.time() - tic, "S~": 0.0, "inner": 0.0, "nnz": 0}
        innerIterations = []

        key = (A.shape[0], A.nnz)
        rebuild = (
            not self._reuseSchurComplement
            or self._schur is None
            or self._schur[0] != key
            or self._rebuildNext
            or self._rebuildsWithoutReuse > 0
        )
        self._rebuildsWithoutReuse = max(0, self._rebuildsWithoutReuse - 1)

        def solve(rebuild: bool, maxIterations: int) -> tuple:
            if rebuild:
                tSchur = time.time()
                S = self._approximateSchurComplement(Abb, Abk, Akb, Akk)
                self._schur = (key, S, _RowBlockProduct(S, self._nThreads, self._threadPool))
                timing["S~"] += time.time() - tSchur
                if self._innerCycles is None:
                    self._inner.requestHierarchyRefresh()
                else:
                    tInner = time.time()
                    prepared = self._inner.preparePreconditioner(S)
                    timing["inner"] += time.time() - tInner
                    if not prepared:
                        # S~ is not finite: no solution (a diverged one for the nonlinear solver), rebuild next time
                        self._schur = None
                        return np.full_like(b, np.nan), 0, np.inf
            S, Sproduct = self._schur[1], self._schur[2]
            timing["nnz"] = S.nnz

            def innerSolve(rk: np.ndarray) -> np.ndarray:
                if self._innerCycles is None:
                    zk = np.asarray(self._inner(S, rk))
                    innerIterations.append(self._inner.lastSolveSummary.iters)
                    return zk
                zk = self._inner.applyPreconditioner(rk)
                for _ in range(self._innerCycles - 1):
                    zk += self._inner.applyPreconditioner(rk - Sproduct @ zk)
                innerIterations.append(self._innerCycles)
                return zk

            def precondition(r: np.ndarray) -> np.ndarray:
                rb = r[bars]
                yb = barFactorization.solve(rb)
                tInner = time.time()
                zk = innerSolve(r[others] - Akb @ yb)
                timing["inner"] += time.time() - tInner
                z = np.empty_like(r)
                z[others] = zk
                z[bars] = barFactorization.solve(rb - Abk @ zk)
                return z

            return _flexibleGMRES(product, precondition, b, self._outerTolerance, self._restart, maxIterations)

        # a reused approximation gets a short leash: when it no longer does, rebuilding is cheaper than iterating on
        reference = self._referenceIterations
        leash = self._maxIterations if rebuild else min(self._maxIterations, max(2 * reference, reference + 20))
        x, iterations, relativeResidual = solve(rebuild, leash)
        retried = 0
        if relativeResidual > self._outerTolerance and not rebuild:
            rebuild, retried = True, 1
            # the matrix changes too fast for reuse (e.g. at a damage onset): rebuild directly for the next solves
            self._rebuildsWithoutReuse = 3
            x, retryIterations, relativeResidual = solve(True, self._maxIterations)
            iterations += retryIterations
            finalIterations = retryIterations
        else:
            finalIterations = iterations
        converged = relativeResidual <= self._outerTolerance

        if rebuild:
            self._referenceIterations = finalIterations
            reference = finalIterations
        self._rebuildNext = not converged or finalIterations > max(
            self._schurStalenessFactor * reference, reference + 10
        )

        self.lastSolveSummary = LinearSolveSummary(
            iters=iterations,
            residual=relativeResidual,
            residualMet=converged,
            retries=retried,
            detailLines=(
                f"inner {sum(innerIterations)} its · {time.time() - tic:.1f}s · "
                f"S~ {'rebuilt' if rebuild else 'reused'}",
                f"split+LU {timing['split+LU']:.1f}s · S~ {timing['S~']:.1f}s (nnz {timing['nnz']}) · "
                f"inner {timing['inner']:.1f}s",
            ),
        )
        if not converged and self._journal is not None:
            self._journal.message(
                f"⚠ FGMRES did not converge: ‖r‖/‖b‖ = {relativeResidual:.1e} after {iterations} iterations",
                self.identification,
                1,
            )
        if time.time() - tic > self._dumpSlowSolveSeconds:
            self._dumpSlowSolve(A, b)
        return x

    def _dumpSlowSolve(self, A: sp.csr_matrix, b: np.ndarray) -> None:
        """Write the system of a slow solve for offline diagnosis, see ``dumpSlowSolvesDir``."""
        if self._dumpSlowSolvesDir is None or self._slowSolvesDumped >= self._dumpSlowSolvesMax:
            return
        os.makedirs(self._dumpSlowSolvesDir, exist_ok=True)
        stem = os.path.join(self._dumpSlowSolvesDir, f"slow_{self._slowSolvesDumped}")
        sp.save_npz(stem + "_A.npz", A)
        np.save(stem + "_b.npy", b)
        np.save(stem + "_barDofs.npy", self._barDofs)
        np.save(stem + "_barNodeSizes.npy", self._barNodeBlocks[1])
        with open(stem + "_fields.json", "w") as f:
            json.dump([[fb.name, fb.start, fb.stop, fb.dimension] for fb in self._fieldStructure], f)
        self._slowSolvesDumped += 1
        if self._journal is not None:
            self._journal.message(f"slow solve dumped to {stem}_*", self.identification, 1)


def _solveTransposed(A: sp.coo_matrix, B: np.ndarray) -> np.ndarray:
    """:math:`A^{-T} B` for a square sparse ``A``, by a banded LU factorization if ``A`` is banded
    (a window of a bar in the order along the bar), else by a dense one.

    The bandwidth of a window is a few dofs, so the banded solve costs a small fraction of the dense one; it runs in the
    threads of the window loop, where a threaded dense solve in every thread oversubscribes the cores."""
    n = A.shape[0]
    lower = int(max(0, (A.row - A.col).max(initial=0)))  # the bandwidths of A
    upper = int(max(0, (A.col - A.row).max(initial=0)))
    if lower + upper + 1 > n // 4:
        return np.linalg.solve(A.toarray().T, B)
    # A^T in LAPACK band storage: (A^T)_ij = A_ji at [upper(A^T) + i - j, j], and upper(A^T) = lower(A)
    banded = np.zeros((lower + upper + 1, n))
    banded[lower + A.col - A.row, A.row] = A.data
    return solve_banded((upper, lower), banded, B, check_finite=False)


class _RowBlockProduct:
    """The product ``A @ x`` with the rows of ``A`` split into blocks that are multiplied in parallel threads."""

    def __init__(self, A: sp.csr_matrix, nBlocks: int, threadPool):
        bounds = np.linspace(0, A.shape[0], nBlocks + 1).astype(int)
        self._blocks = [(low, high, A[low:high]) for low, high in zip(bounds[:-1], bounds[1:]) if high > low]
        self._threadPool = threadPool
        self._n = A.shape[0]

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        y = np.empty(self._n)

        def multiply(block):
            low, high, rows = block
            y[low:high] = rows @ x

        if self._threadPool is None:
            for block in self._blocks:
                multiply(block)
        else:
            list(self._threadPool.map(multiply, self._blocks))
        return y


def _flexibleGMRES(A, precondition, b: np.ndarray, tolerance: float, restart: int, maxIterations: int) -> tuple:
    """Flexible GMRES with right preconditioning (Saad 1993): the preconditioned vectors are kept, so the
    preconditioner may change from one application to the next.

    Returns
    -------
    tuple
        The solution, the number of iterations and the relative residual :math:`\\|b - A x\\| / \\|b\\|`.
    """
    x = np.zeros_like(b)
    bNorm = np.linalg.norm(b)
    if bNorm == 0.0:
        return x, 0, 0.0
    iterations = 0
    # the Krylov basis and the preconditioned vectors as rows, so that the orthogonalization is matrix-vector
    # products (threaded BLAS) instead of a loop over the vectors
    V = np.empty((restart + 1, len(b)))
    Z = np.empty((restart, len(b)))
    while True:
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta <= tolerance * bNorm or iterations >= maxIterations:
            return x, iterations, beta / bNorm
        V[0] = r / beta
        H = np.zeros((restart + 1, restart))
        cs, sn = np.zeros(restart), np.zeros(restart)
        g = np.zeros(restart + 1)
        g[0] = beta
        for j in range(restart):
            Z[j] = precondition(V[j])
            w = A @ Z[j]
            for _ in range(2):  # classical Gram-Schmidt, twice (as accurate as modified Gram-Schmidt)
                h = V[: j + 1] @ w
                w -= h @ V[: j + 1]
                H[: j + 1, j] += h
            H[j + 1, j] = np.linalg.norm(w)
            V[j + 1] = w / H[j + 1, j] if H[j + 1, j] > 0.0 else w
            for i in range(j):  # the previous Givens rotations, applied to the new column
                H[i, j], H[i + 1, j] = cs[i] * H[i, j] + sn[i] * H[i + 1, j], -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            denominator = np.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = H[j, j] / denominator, H[j + 1, j] / denominator
            H[j, j], H[j + 1, j] = denominator, 0.0
            g[j], g[j + 1] = cs[j] * g[j], -sn[j] * g[j]
            iterations += 1
            if abs(g[j + 1]) <= tolerance * bNorm or iterations >= maxIterations:
                break
        k = j + 1
        x = x + np.linalg.solve(np.triu(H[:k, :k]), g[:k]) @ Z[:k]
