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

"""Schur complement condensation of selected dofs, an option of ``blockamg`` (``schurComplement``).

Some dofs make a system hard for AMG: the dofs of embedded bars tied to their host elements by bond springs
(:math:`u_{bar} \\approx \\sum_i N_i u_i`, which an aggregation coarse space cannot represent), or the dofs of a
softening zone. With these dofs as :math:`b` and all others as :math:`k`,

.. math::

    \\begin{bmatrix} A_{bb} & A_{bk} \\\\ A_{kb} & A_{kk} \\end{bmatrix}
    \\begin{bmatrix} x_b \\\\ x_k \\end{bmatrix} =
    \\begin{bmatrix} r_b \\\\ r_k \\end{bmatrix},

the block :math:`A_{bb}` is factorized exactly (a direct solver), and the other dofs see the Schur complement
:math:`S = A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}`, on which ``blockamg`` builds its field-split AMG as usual. The
preconditioner of the outer Krylov solve is the block LU factorization

.. math::

    y_b = A_{bb}^{-1} r_b, \\quad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\quad
    z_b = A_{bb}^{-1} (r_b - A_{bk} z_k),

with :math:`M_{\\tilde S}^{-1}` the field-split block Gauss-Seidel sweep of ``blockamg`` on an approximation
:math:`\\tilde S` of :math:`S`. With :math:`\\tilde S = S` and an exact sweep it is :math:`A^{-1}`.

:math:`S` is dense wherever :math:`A_{bb}^{-1}` is, e.g. along a whole bar. Its entries decay quickly away from
the diagonal, though (a bar on its bond is a beam on an elastic foundation: a load is felt only a few host elements
away), so :math:`\\tilde S` keeps only the significant ones:

- a long, banded component of :math:`A_{bb}` (a bar, ordered along the bar) is cut into overlapping windows;
  for the dofs of a window's core, :math:`A_{bb}^{-1}` is replaced by the inverse of the window's block
  (restricted additive Schwarz); a short component is a single, exact window,
- a long component that is not banded (a softening zone) gets its exact correction, from the factorization of
  :math:`A_{bb}`, in batches of right-hand sides,
- entries :math:`|\\tilde S_{ij}| < \\tau \\sqrt{|\\tilde S_{ii} \\tilde S_{jj}|}` are dropped (``dropTolerance``).

Which dofs are condensed is decided by markers, the same ones that mark elements for adaptive refinement
(:mod:`edelweissfe.adaptivity.marking`): :math:`b` holds all dofs of the *interior nodes* of the marked elements,
i.e., of the nodes all of whose elements are marked. For embedded bars, mark the bar and the bond elements: the
bar nodes belong only to them, the host nodes also to continuum elements. For a softening zone, mark the elements
of a ``perElement`` fieldOutput above a threshold, with a ``halo`` so that the zone's own nodes become interior.
"""

from collections import Counter
from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.adaptivity.marking import RefineableElements
from edelweissfe.linsolve.base import FieldBlock, LinearSolver
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
)

#: The configuration keys of ``schurComplement`` and their defaults (``markers`` is required).
DEFAULT_OPTIONS = {
    "markers": None,
    "approximation": "local",
    "dropTolerance": 1e-4,
    "windowSize": 300,
    "haloSize": None,
    "directSolver": {"linsolver": "superlu"},
}

#: The number of right-hand sides per batch of the exact correction of a component (:meth:`_exactCorrection`) ...
EXACT_CORRECTION_BATCH = 256
#: ... at most, and fewer when there are so many condensed dofs that a batch would exceed this many entries.
EXACT_CORRECTION_MAX_ENTRIES = 2**25


def optionsRejection(options) -> "str | None":
    """Why ``schurComplement`` options cannot be used, or ``None``. Checks what can be checked without the model.

    Parameters
    ----------
    options
        The ``schurComplement`` section of the ``blockamg`` configuration.
    """
    if not isinstance(options, Mapping):
        return "must be a JSON object"
    unknown = sorted(set(options) - set(DEFAULT_OPTIONS))
    if unknown:
        return "unknown key(s) {:}; known are {:}".format(unknown, sorted(DEFAULT_OPTIONS))
    markers = options.get("markers")
    if not isinstance(markers, list) or not markers or not all(isinstance(m, Mapping) for m in markers):
        return "'markers' must be a non-empty list of marker objects, each with a 'type'"
    if options.get("approximation", "local") not in ("local", "none"):
        return "'approximation' must be 'local' or 'none'"
    if not 0.0 <= options.get("dropTolerance", 1e-4) < 1.0:
        return "'dropTolerance' must be at least 0 and less than 1 (else the diagonal would be dropped)"
    if not options.get("windowSize", 300) >= 1 or not (options.get("haloSize") or 0) >= 0:
        return "'windowSize' must be at least 1 and 'haloSize' not negative"
    directSolver = options.get("directSolver", DEFAULT_OPTIONS["directSolver"])
    if not isinstance(directSolver, Mapping) or "linsolver" not in directSolver:
        return "'directSolver' must be an object with a 'linsolver' name (and optional 'options')"
    return None


def createCondensation(options) -> "SchurComplementCondensation":
    """The condensation of the ``schurComplement`` options (see :func:`optionsRejection`).

    The markers are created through the marker registry (:mod:`edelweissfe.config.markerlibrary`) from exactly the
    options of a ``>>marker`` line, the direct solver through the linear solver registry (SuperLU by default: its
    back substitution, twice per outer iteration, is cheap for the chains of bars).
    """
    from edelweissfe.config.linsolve import getLinSolverByName
    from edelweissfe.config.markerlibrary import getMarkerClass

    reason = optionsRejection(options)
    if reason is not None:
        raise ValueError("blockamg: schurComplement: " + reason)
    options = DEFAULT_OPTIONS | dict(options)

    markers = []
    for markerOptions in options["markers"]:
        markerOptions = dict(markerOptions)
        markerType = markerOptions.pop("type", None)
        if markerType is None:
            raise ValueError("blockamg: schurComplement: every marker needs a 'type'")
        markers.append(getMarkerClass(markerType).fromOptions(markerOptions))

    directSolver = getLinSolverByName(
        options["directSolver"]["linsolver"], dict(options["directSolver"].get("options", {}))
    )

    windowSize = int(options["windowSize"])
    return SchurComplementCondensation(
        markers,
        directSolver,
        approximation=options["approximation"],
        dropTolerance=float(options["dropTolerance"]),
        windowSize=windowSize,
        haloSize=windowSize if options["haloSize"] is None else int(options["haloSize"]),
    )


class SchurComplementCondensation:
    """The condensed dofs :math:`b`, the factorization of :math:`A_{bb}`, the approximate Schur complement
    :math:`\\tilde S` of all other dofs :math:`k`, and the block LU preconditioner; see the module documentation.

    ``blockamg`` uses it in three steps: :meth:`update` on every solve (select :math:`b`, split :math:`A`,
    factorize :math:`A_{bb}`), :meth:`approximateSchurComplement` whenever it rebuilds its hierarchies (they are
    built on :math:`\\tilde S` instead of :math:`A`), and :meth:`precondition` in every outer iteration.

    Parameters
    ----------
    markers
        The markers whose marked elements' interior nodes are condensed.
    directSolver
        The linear solver of :math:`A_{bb}`, used through :meth:`~edelweissfe.linsolve.base.LinearSolver.factorize`
        and :meth:`~edelweissfe.linsolve.base.LinearSolver.solveFactorized`.
    approximation
        ``"local"``: :math:`\\tilde S` as described in the module documentation; ``"none"``:
        :math:`\\tilde S = A_{kk}` (the coupling through :math:`b` is ignored).
    dropTolerance
        The relative size :math:`\\tau` below which entries of :math:`\\tilde S` are dropped.
    windowSize, haloSize
        The dofs of a window's core along a banded component, and of its halo on either side.
    """

    def __init__(
        self,
        markers: list,
        directSolver: LinearSolver,
        approximation: str = "local",
        dropTolerance: float = 1e-4,
        windowSize: int = 300,
        haloSize: int = 300,
    ):
        self._markers = markers
        self._directSolver = directSolver
        self._approximation = approximation
        self._dropTolerance = dropTolerance
        self._windowSize = windowSize
        self._haloSize = haloSize

        # the model, its elements as the candidates of the markers, and per node its number of elements and its dofs,
        # see setModel
        self._model = None
        self._candidateElements = None
        self._elementsAtNode = None
        self._dofsOfNode = None
        #: The condensed dofs b and all other dofs k, sorted.
        self.condensedDofs = np.empty(0, dtype=np.int64)
        self.remainingDofs = None
        #: Whether the markers selected every dof at the last :meth:`update`, which leaves nothing for the field
        #: split: then nothing is condensed (a direct solver is the better choice for such a system).
        self.selectedAllDofs = False
        # the blocks of A: cut out once per sparsity pattern of A and choice of b, see _split
        self._pattern = None
        self._blockPositions = None
        self.A_bb = self.A_bk = self.A_kb = None
        # A_kb by columns, for cutting out the columns of the windows, see approximateSchurComplement
        self._A_kb_byColumns = None
        # per field: which of its nodes remain (are not condensed), see remainingFieldBlocks
        self._remainingNodes = {}

    # ----- the condensed dofs -----

    def setModel(self, model, dofManager) -> None:
        """Take the model and its dofs: check that what the markers read exists, and count each node's elements
        for the interior-node rule."""
        for marker in self._markers:
            marker.checkReferences(model)
        self._model = model
        self._candidateElements = RefineableElements(list(model.elements.values()))
        self._elementsAtNode = Counter(node for element in model.elements.values() for node in element.nodes)
        indices = dofManager.idcsOfFieldVariablesInDofVector
        self._dofsOfNode = {}
        for node in self._elementsAtNode:
            dofs = [np.atleast_1d(indices[fv]) for fv in node.fields.values() if fv in indices]
            if dofs:
                self._dofsOfNode[node] = np.concatenate(dofs)
        self.condensedDofs = np.empty(0, dtype=np.int64)
        self._pattern = None

    def _selectCondensedDofs(self) -> np.ndarray:
        """All dofs of the interior nodes of the marked elements, sorted: a node is interior if it has at least one
        element and all of its elements are marked. A marker that has nothing to decide on yet (a fieldOutput
        before the first converged increment) marks nothing."""
        if self._model is None:
            raise RuntimeError(
                "schurComplement: the condensed dofs are chosen on the model, but setModel was not called"
            )
        marked = set()
        for marker in self._markers:
            if marker.isReady(self._model):
                marked |= marker.mark(self._model, self._candidateElements, None)
        markedElementsAtNode = Counter(node for element in marked for node in element.nodes)
        interiorNodes = [node for node, count in markedElementsAtNode.items() if count == self._elementsAtNode[node]]
        if not interiorNodes:
            return np.empty(0, dtype=np.int64)
        dofs = [self._dofsOfNode[node] for node in interiorNodes if node in self._dofsOfNode]
        return np.sort(np.concatenate(dofs)).astype(np.int64) if dofs else np.empty(0, dtype=np.int64)

    def hasCondensedDofs(self) -> bool:
        """Whether anything is condensed (after the last :meth:`update`)."""
        return self.condensedDofs.size > 0

    # ----- every solve -----

    def update(self, A: sp.csr_matrix) -> None:
        """Select the condensed dofs, cut the blocks :math:`A_{bb}`, :math:`A_{bk}`, :math:`A_{kb}` out of ``A`` and
        factorize :math:`A_{bb}`.

        Parameters
        ----------
        A
            The system matrix, CSR in canonical format.
        """
        with performancetiming.timeit("schur dof selection"):
            dofs = self._selectCondensedDofs()
        self.selectedAllDofs = dofs.size == A.shape[0]
        if self.selectedAllDofs:
            dofs = np.empty(0, dtype=np.int64)
        if not np.array_equal(dofs, self.condensedDofs):
            self.condensedDofs = dofs
            isCondensed = np.zeros(A.shape[0], dtype=bool)
            isCondensed[dofs] = True
            self.remainingDofs = np.flatnonzero(~isCondensed)
            self._pattern = None
        if not self.hasCondensedDofs():
            return
        with performancetiming.timeit("schur split"):
            self.A_bb, self.A_bk, self.A_kb = self._split(A, ("bb", "bk", "kb"))
        with performancetiming.timeit("schur factorization"):
            self._directSolver.factorize(self.A_bb)

    def _split(self, A: sp.csr_matrix, which: tuple) -> list:
        """The blocks ``which`` (of ``"bb"``, ``"bk"``, ``"kb"``, ``"kk"``) of ``A``, as CSR matrices.

        Cutting blocks out of a large matrix by fancy indexing is slow, but the sparsity pattern of ``A`` stays the
        same over many solves: the positions of every block's entries in ``A.data`` are found once per pattern
        (and choice of b), and each solve only gathers the values at these positions."""
        if self._pattern is None or not (
            np.array_equal(self._pattern[0], A.indptr) and np.array_equal(self._pattern[1], A.indices)
        ):
            self._pattern = (A.indptr.copy(), A.indices.copy())
            self._blockPositions = {}
        dofsOf = {"b": self.condensedDofs, "k": self.remainingDofs}
        blocks = []
        for name in which:
            if name not in self._blockPositions:
                # positions + 1, so that no position is a zero that the slicing could drop
                positions = sp.csr_matrix((np.arange(1, A.nnz + 1, dtype=np.float64), A.indices, A.indptr), A.shape)
                block = positions[dofsOf[name[0]]][:, dofsOf[name[1]]].tocsr()
                block.sort_indices()
                self._blockPositions[name] = (block.data.astype(np.int64) - 1, block.indices, block.indptr, block.shape)
            at, indices, indptr, shape = self._blockPositions[name]
            blocks.append(sp.csr_matrix((A.data[at], indices, indptr), shape=shape))
        return blocks

    def precondition(self, r: np.ndarray, preconditionRemaining) -> np.ndarray:
        """One application of the block LU preconditioner, see the module documentation.

        Parameters
        ----------
        r
            The vector to precondition (all dofs).
        preconditionRemaining
            :math:`M_{\\tilde S}^{-1}`, applied to a vector of the remaining dofs k.

        Returns
        -------
        np.ndarray
            :math:`z \\approx A^{-1} r`.
        """
        b, k = self.condensedDofs, self.remainingDofs
        r_b = r[b]
        y_b = self._directSolver.solveFactorized(r_b)
        z = np.empty_like(r)
        z[k] = preconditionRemaining(r[k] - self.A_kb @ y_b)
        z[b] = self._directSolver.solveFactorized(r_b - self.A_bk @ z[k])
        return z

    # ----- the remaining dofs, on which blockamg builds its field split -----

    def remainingFieldBlocks(self, blocks: list) -> list:
        """The field blocks of the remaining dofs: every field without its condensed dofs, in the same order (and
        contiguous, as the remaining dofs keep their order). A field whose dofs are all condensed (e.g. the rotations
        of embedded beams) has no block. Also records which nodes of every field remain, see
        :meth:`remainingNodes`."""
        remainingBlocks, start = [], 0
        self._remainingNodes = {}
        for block in blocks:
            inBlock = self.condensedDofs[(self.condensedDofs >= block.start) & (self.condensedDofs < block.stop)]
            n = (block.stop - block.start) - inBlock.size
            if n == 0:
                continue
            remainingBlocks.append(FieldBlock(block.name, start, start + n, block.dimension))
            start += n
            # condensation takes whole nodes, and a field's dofs are node-major
            remains = np.ones((block.stop - block.start) // block.dimension, dtype=bool)
            remains[(inBlock - block.start) // block.dimension] = False
            self._remainingNodes[block.name] = remains
        return remainingBlocks

    def remainingNodes(self, fieldName: str) -> "np.ndarray | None":
        """Which nodes of the field ``fieldName`` remain, as a boolean mask over the field's nodes (in the field's
        dof order), or ``None`` if no block of that name remains."""
        return self._remainingNodes.get(fieldName)

    # ----- the approximate Schur complement -----

    def approximateSchurComplement(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """:math:`\\tilde S`, see the module documentation.

        Parameters
        ----------
        A
            The system matrix of the last :meth:`update`.
        """
        (A_kk,) = self._split(A, ("kk",))
        if self._approximation == "none":
            return A_kk
        tau = self._dropTolerance
        akk = np.sqrt(np.abs(A_kk.diagonal()))
        # each part of the correction is thinned at a tenth of the drop tolerance before the parts are summed,
        # which bounds the memory of the correction
        thinning = 0.1 * tau

        windows, exactDofs = self._windowsAndExactDofs()
        # the windows cut columns out of A_kb
        self._A_kb_byColumns = self.A_kb.tocsc()
        parts = []
        with performancetiming.timeit("schur windows"):
            pool = getThreadPool(getNumberOfThreads())
            for part in pool.map(lambda window: self._windowCorrection(window, akk, thinning), windows):
                if part is not None:
                    parts.append(part)
        with performancetiming.timeit("schur exact correction"):
            parts += self._exactCorrection(exactDofs, akk, thinning)

        with performancetiming.timeit("schur assembly"):
            if parts:
                rows, cols, values = (np.concatenate(v) for v in zip(*parts))
                S = (A_kk - sp.csr_matrix((values, (rows, cols)), shape=A_kk.shape)).tocsr()
            else:
                S = A_kk.copy()
            return _dropSmallEntries(S, tau)

    def _windowsAndExactDofs(self) -> tuple:
        """The windows along the long banded components of :math:`A_{bb}`, and the dofs of all other components.

        Each component is ordered by the reverse Cuthill-McKee ordering (along the bar, for a bar). A banded component
        (bandwidth at most a quarter of a core: a bar) is cut into cores of ``windowSize`` dofs, each with a halo of
        ``haloSize`` dofs on either side; one no longer than a core with both halos is a single, exact window. A
        component that is not banded (a zone of a 2D or 3D mesh) is corrected exactly with the factorization of
        :math:`A_{bb}`, see :meth:`_exactCorrection`.

        Returns
        -------
        tuple
            ``(windows, exactDofs)``: ``(dofs of the window, positions of its core in the window)`` for every window
            (the cores partition the windowed dofs), and the dofs of the other components (positions in b).
        """
        A_bb = self.A_bb
        order = reverse_cuthill_mckee(A_bb, symmetric_mode=True)
        nComponents, componentOf = connected_components(A_bb, directed=False)
        alongComponents = order[np.argsort(componentOf[order], kind="stable")]
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        coo = A_bb.tocoo()
        bandwidth = np.zeros(nComponents, dtype=np.int64)
        np.maximum.at(bandwidth, componentOf[coo.row], np.abs(rank[coo.row] - rank[coo.col]))

        w, h = self._windowSize, self._haloSize
        windows, exactDofs = [], []
        for component, dofs in enumerate(np.split(alongComponents, np.cumsum(np.bincount(componentOf))[:-1])):
            n = len(dofs)
            if bandwidth[component] > w // 4:
                exactDofs.append(dofs)
                continue
            if n <= w + 2 * h:
                windows.append((dofs, np.arange(n)))
                continue
            for start in range(0, n, w):
                stop = min(start + w, n)
                low, high = max(0, start - h), min(n, stop + h)
                windows.append((dofs[low:high], np.arange(start - low, stop - low)))
        exactDofs = np.sort(np.concatenate(exactDofs)) if exactDofs else np.empty(0, dtype=np.int64)
        return windows, exactDofs

    def _windowCorrection(self, window: tuple, akk: np.ndarray, thinning: float) -> "tuple | None":
        """The correction :math:`A_{kb} A_{bb}^{-1} A_{bk}` of one window's core: :math:`A_{bb}^{-1}` is replaced by
        the inverse of the window's block, of which only the core rows :math:`G = I_{core} A_{ww}^{-1}` are needed
        (far fewer right-hand sides than the remaining dofs coupled to the window).

        Returns
        -------
        tuple | None
            ``(rows, cols, values)`` of the significant entries, as positions in k, or ``None``.
        """
        dofs, core = window
        A_bk_window = self.A_bk[dofs]
        coupledToWindow = np.unique(A_bk_window.indices)
        A_kb_core = self._A_kb_byColumns[:, dofs[core]]
        coupledToCore = np.unique(A_kb_core.indices)
        if len(coupledToWindow) == 0 or len(coupledToCore) == 0:
            return None
        # G^T = A_ww^-T I[:, core]
        unitColumns = np.zeros((len(dofs), len(core)))
        unitColumns[core, np.arange(len(core))] = 1.0
        G = _solveTransposed(self.A_bb[dofs][:, dofs].tocoo(), unitColumns).T
        # (A_kb[:, core] G) A_bk[window, :], written so that the sparse factor multiplies from the left
        correction = (A_bk_window[:, coupledToWindow].T @ (A_kb_core[coupledToCore] @ G).T).T
        return _significantEntries(correction, coupledToCore, coupledToWindow, akk, thinning)

    def _exactCorrection(self, exactDofs: np.ndarray, akk: np.ndarray, thinning: float) -> list:
        """The exact correction :math:`A_{kb} A_{bb}^{-1} A_{bk}` of the components ``exactDofs``, with the
        factorization of :math:`A_{bb}` (block diagonal over the components, so right-hand sides on these components
        give solutions on them), for the remaining dofs coupled to them in batches of
        :data:`EXACT_CORRECTION_BATCH`.

        Returns
        -------
        list
            ``(rows, cols, values)`` of the significant entries of every batch, as positions in k.
        """
        if exactDofs.size == 0:
            return []
        A_bk_zone = self.A_bk[exactDofs].tocsc()
        coupledColumns = np.flatnonzero(np.diff(A_bk_zone.indptr))
        A_kb_zone = self.A_kb[:, exactDofs].tocsr()
        coupledRows = np.flatnonzero(np.diff(A_kb_zone.indptr))
        A_kb_zone = A_kb_zone[coupledRows]
        nCondensed = self.A_bb.shape[0]
        batch = max(1, min(EXACT_CORRECTION_BATCH, EXACT_CORRECTION_MAX_ENTRIES // nCondensed))
        parts = []
        for start in range(0, len(coupledColumns), batch):
            columns = coupledColumns[start : start + batch]
            rightHandSides = np.zeros((nCondensed, len(columns)))
            rightHandSides[exactDofs] = A_bk_zone[:, columns].toarray()
            X = np.asarray(self._directSolver.solveFactorized(rightHandSides)).reshape(rightHandSides.shape)
            correction = A_kb_zone @ X[exactDofs]
            parts.append(_significantEntries(correction, coupledRows, columns, akk, thinning))
        return parts


def _significantEntries(
    correction: np.ndarray, rows: np.ndarray, cols: np.ndarray, akk: np.ndarray, thinning: float
) -> tuple:
    """The entries of a dense correction block (rows ``rows``, columns ``cols`` of k) at least ``thinning`` times
    :math:`\\sqrt{|A_{ii} A_{jj}|}`, as ``(rows, cols, values)``."""
    significant = np.abs(correction) >= thinning * np.outer(akk[rows], akk[cols])
    i, j = np.nonzero(significant)
    return rows[i], cols[j], correction[i, j]


def _dropSmallEntries(S: sp.csr_matrix, tau: float) -> sp.csr_matrix:
    """``S`` without the entries :math:`|S_{ij}| < \\tau \\sqrt{|S_{ii} S_{jj}|}` (and without stored zeros), the
    rows in parallel."""
    s = np.sqrt(np.abs(S.diagonal()))

    def keptEntries(rowRange):
        low, high = rowRange
        entries = slice(S.indptr[low], S.indptr[high])
        rowOfEntry = np.repeat(np.arange(low, high), np.diff(S.indptr[low : high + 1]))
        keep = np.abs(S.data[entries]) >= tau * s[rowOfEntry] * s[S.indices[entries]]
        keep &= S.data[entries] != 0.0
        return (
            S.data[entries][keep],
            S.indices[entries][keep],
            np.bincount(rowOfEntry[keep] - low, minlength=high - low),
        )

    nThreads = getNumberOfThreads()
    bounds = np.linspace(0, S.shape[0], nThreads + 1).astype(np.int64)
    parts = list(getThreadPool(nThreads).map(keptEntries, zip(bounds[:-1], bounds[1:])))
    data, indices, keptPerRow = (np.concatenate(v) for v in zip(*parts))
    return sp.csr_matrix((data, indices, np.concatenate([[0], np.cumsum(keptPerRow)])), shape=S.shape)


def _solveTransposed(A: sp.coo_matrix, B: np.ndarray) -> np.ndarray:
    """:math:`A^{-T} B` for a square sparse ``A``: by a banded LU factorization if ``A`` is banded (a window along
    a bar), else by a dense one (a short window, for which the banded storage would not pay).

    The bandwidth of a window is a few dofs, so the banded solve costs a small fraction of the dense one; it runs in
    the threads of the window loop."""
    n = A.shape[0]
    lower = int(max(0, (A.row - A.col).max(initial=0)))
    upper = int(max(0, (A.col - A.row).max(initial=0)))
    if lower + upper + 1 > n // 4:
        return np.linalg.solve(A.toarray().T, B)
    # A^T in LAPACK band storage: (A^T)_ij = A_ji at [upper(A^T) + i - j, j], and upper(A^T) = lower(A)
    banded = np.zeros((lower + upper + 1, n))
    banded[lower + A.col - A.row, A.row] = A.data
    return solve_banded((upper, lower), banded, B, check_finite=False)


def restrictP1Map(isCorner: np.ndarray, edgeEndpoints: np.ndarray, remains: np.ndarray) -> tuple:
    """A field's P1 map (see :func:`~edelweissfe.numerics.p1topology.buildP1Map`) restricted to its remaining
    nodes ``remains``: the rows are renumbered, and a midside node whose edge endpoint is condensed becomes a corner
    (always structurally safe for the P1 restriction, which takes a corner as is).

    Returns
    -------
    tuple
        ``(isCorner, edgeEndpoints)`` of the remaining nodes.
    """
    newRow = np.cumsum(remains) - 1
    isCorner, edgeEndpoints = isCorner[remains].copy(), edgeEndpoints[remains].copy()
    isMidside = ~isCorner
    endpointRemains = np.ones_like(isMidside)
    endpointRemains[isMidside] = remains[edgeEndpoints[isMidside]].all(axis=1)
    becomesCorner = isMidside & ~endpointRemains
    isCorner[becomesCorner] = True
    edgeEndpoints[becomesCorner] = -1
    stillMidside = ~isCorner
    edgeEndpoints[stillMidside] = newRow[edgeEndpoints[stillMidside]]
    return isCorner, edgeEndpoints
