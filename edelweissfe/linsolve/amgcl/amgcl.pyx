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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

import json

import numpy as np
import scipy.sparse

cimport numpy as np


cdef class PyAMGCLSolver:
    cdef LinearSolver* solver
    cdef LinearSolverFloat* solverFloat
    cdef LinearSolverBlock2* solverBlock2
    cdef LinearSolverBlock3* solverBlock3
    cdef LinearSolverBlock3Float* solverBlock3Float
    cdef LinearSolverScalarSetupBlock3Float* solverScalarSetup
    cdef readonly bint isFloat
    """True if this instance runs a mixed-precision hierarchy: builtin<float> for the scalar backend, or
    3x3 blocks set up in double but stored in float (``LinearSolverBlock3Float``) for block size 3."""
    cdef readonly int blockSize
    """1 (scalar, default), 2, or 3 -- the AMGCL backend's block size."""
    cdef readonly int lastIterations
    """The iteration count AMGCL reported for the most recent solve, -1 before the first."""
    cdef readonly double lastError
    """The relative residual AMGCL reported for the most recent solve, NaN before the first."""

    def __cinit__(self, dict params=None):
        """
        Initialize with parameters only.

        The AMG smoother key is ``relax``, not ``relaxation``; AMGCL ignores unknown keys with only a
        warning on stderr, so a misspelling here silently runs the default smoother instead of the
        requested one.

        ``backendPrecision``: ``"double"`` (default) or ``"float"``. Selects the AMGCL backend's value
        type -- ``"float"`` halves the memory traffic of the hierarchy build and, more importantly, of
        every :meth:`applyPreconditioner` call (the dominant cost on large coupled solves). Not an
        AMGCL parameter itself, so it is popped out before the rest of ``params`` is forwarded as JSON.
        The near null-space (:meth:`set_nullspace`) always stays double regardless -- AMGCL's own
        ``coarsening::nullspace_params::B`` is hardcoded double, independent of the backend.

        ``backendBlockSize``: ``1`` (default, scalar), ``2``, or ``3``. Selects a block-valued backend
        operating on node-major B x B nodal blocks instead of scalar entries -- shrinks the
        CSR index traffic by ~B² and lets block-aware smoothers (block-ILU0, block-GS) invert each
        node's coupling exactly. The matrix still arrives as a plain scalar CSR; the wrapper adapts it
        internally. With ``backendPrecision == "float"``, block size 3 sets the hierarchy up in double
        -- coarsening, Galerkin products and every Chebyshev spectral-radius estimate -- and stores
        only the level matrices the cycle reads in float, with double vectors and double accumulation
        (``LinearSolverBlock3Float``); block size 2 with float raises. :meth:`set_nullspace` raises on a
        block backend -- AMGCL's own near-null-space path is unimplemented for block value types -- unless
        ``backendScalarSetup`` is set (block size 3 and float only): the hierarchy is then set up on the
        scalar entries, with the full near null-space, and only its level matrices are stored and applied
        as float 3x3 blocks (``LinearSolverScalarSetupBlock3Float``).

        Example params:
        {
            "solver": {"type": "bicgstab", "tol": 1e-6},
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "ilu0"}}
        }
        """
        if params is None:
            # Default to Robust Blackbox (BiCGStab + ILU0)
            params = {
                "solver": {"type": "bicgstab", "tol": 1e-6},
                "precond": {
                    "coarsening": {"type": "smoothed_aggregation"},
                    "relax": {"type": "ilu0"}
                }
            }
        else:
            params = dict(params)

        backendPrecision = params.pop("backendPrecision", "double")
        if backendPrecision not in ("double", "float"):
            raise ValueError(f"backendPrecision must be 'double' or 'float', got {backendPrecision!r}")
        self.isFloat = backendPrecision == "float"

        scalarSetup = params.pop("backendScalarSetup", False)
        blockSize = params.pop("backendBlockSize", 1)
        if blockSize not in (1, 2, 3):
            raise ValueError(f"backendBlockSize must be 1, 2, or 3, got {blockSize!r}")
        self.blockSize = blockSize
        if scalarSetup and not (self.blockSize == 3 and self.isFloat):
            raise NotImplementedError(
                "backendScalarSetup is implemented for backendBlockSize 3 with backendPrecision 'float' only."
            )
        if self.blockSize == 2 and self.isFloat:
            raise NotImplementedError(
                "backendBlockSize 2 combined with backendPrecision='float' is not implemented."
            )

        self.lastIterations = -1
        self.lastError = float("nan")

        # Convert dict to JSON string for C++
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.solver = NULL
        self.solverFloat = NULL
        self.solverBlock2 = NULL
        self.solverBlock3 = NULL
        self.solverBlock3Float = NULL
        self.solverScalarSetup = NULL
        if self.blockSize == 2:
            self.solverBlock2 = new LinearSolverBlock2(c_json)
        elif scalarSetup:
            self.solverScalarSetup = new LinearSolverScalarSetupBlock3Float(c_json)
        elif self.blockSize == 3 and self.isFloat:
            self.solverBlock3Float = new LinearSolverBlock3Float(c_json)
        elif self.blockSize == 3:
            self.solverBlock3 = new LinearSolverBlock3(c_json)
        elif self.isFloat:
            self.solverFloat = new LinearSolverFloat(c_json)
        else:
            self.solver = new LinearSolver(c_json)

    def __dealloc__(self):
        if self.solver != NULL:
            del self.solver
        if self.solverFloat != NULL:
            del self.solverFloat
        if self.solverBlock2 != NULL:
            del self.solverBlock2
        if self.solverBlock3 != NULL:
            del self.solverBlock3
        if self.solverBlock3Float != NULL:
            del self.solverBlock3Float
        if self.solverScalarSetup != NULL:
            del self.solverScalarSetup

    def set_nullspace(self, object B):
        """
        Supply near null-space vectors for smoothed-aggregation coarsening.

        AMGCL's smoothed aggregation defaults to a single constant near null-space vector, which is
        wrong for an elasticity operator -- there the near null-space is the rigid body modes. AMGCL
        takes these vectors as a raw pointer in its property tree, so they cannot be passed through the
        JSON parameter string like every other option; this is the separate entry point for them.

        B: an (n, cols) array-like; column j is the j-th near null-space vector (float64). Copied into
        the C++ solver, so the array need not be kept alive. Must be called before the first solve().
        Passing an (n, 0) array clears any previously set null-space. Always double, regardless of
        ``backendPrecision`` -- see the class docstring. Raises on a block backend (``blockSize > 1``)
        -- AMGCL's own near-null-space path is unimplemented for block value types.
        """
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] B_arr = np.ascontiguousarray(B, dtype=np.float64)
        cdef int rows = B_arr.shape[0]
        cdef int cols = B_arr.shape[1]
        cdef double[:, ::1] B_view
        if cols == 0:
            if self.blockSize == 2:
                self.solverBlock2.set_nullspace(<const double*> 0, rows, 0)
            elif self.blockSize == 3:
                if self.solverScalarSetup != NULL:
                    self.solverScalarSetup.set_nullspace(<const double*> 0, rows, 0)
                elif self.solverBlock3Float != NULL:
                    self.solverBlock3Float.set_nullspace(<const double*> 0, rows, 0)
                else:
                    self.solverBlock3.set_nullspace(<const double*> 0, rows, 0)
            elif self.isFloat:
                self.solverFloat.set_nullspace(<const double*> 0, rows, 0)
            else:
                self.solver.set_nullspace(<const double*> 0, rows, 0)
            return
        B_view = B_arr
        if self.blockSize == 2:
            self.solverBlock2.set_nullspace(&B_view[0, 0], rows, cols)
        elif self.blockSize == 3:
            if self.solverScalarSetup != NULL:
                self.solverScalarSetup.set_nullspace(&B_view[0, 0], rows, cols)
            elif self.solverBlock3Float != NULL:
                self.solverBlock3Float.set_nullspace(&B_view[0, 0], rows, cols)
            else:
                self.solverBlock3.set_nullspace(&B_view[0, 0], rows, cols)
        elif self.isFloat:
            self.solverFloat.set_nullspace(&B_view[0, 0], rows, cols)
        else:
            self.solver.set_nullspace(&B_view[0, 0], rows, cols)

    def build(self, object A):
        """
        Build the AMG hierarchy for A once, for repeated preconditioner application via
        :meth:`applyPreconditioner` -- the build-once / apply-many split :meth:`solve` fuses.

        A: scipy.sparse.csr_matrix. Its values narrow to float32 first if ``backendPrecision ==
        "float"`` -- cheap here (build() runs once per solve, not once per outer Krylov iteration). A
        block backend (``blockSize > 1``) instead takes the plain scalar CSR unchanged and adapts it
        to node-major blocks inside the C++ wrapper.
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()

        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices

        cdef np.ndarray[np.float32_t, ndim=1, mode="c"] dataFloat
        cdef float[::1] dataFloat_
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data
        cdef double[::1] data_

        if self.blockSize == 2 or self.blockSize == 3:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            if self.blockSize == 2:
                self.solverBlock2.build(n, &indptr_[0], &indices_[0], &data_[0])
            else:
                if self.solverScalarSetup != NULL:
                    self.solverScalarSetup.build(n, &indptr_[0], &indices_[0], &data_[0])
                elif self.solverBlock3Float != NULL:
                    self.solverBlock3Float.build(n, &indptr_[0], &indices_[0], &data_[0])
                else:
                    self.solverBlock3.build(n, &indptr_[0], &indices_[0], &data_[0])
        elif self.isFloat:
            dataFloat = np.ascontiguousarray(A.data, dtype=np.float32)
            dataFloat_ = dataFloat
            self.solverFloat.build(n, &indptr_[0], &indices_[0], &dataFloat_[0])
        else:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            self.solver.build(n, &indptr_[0], &indices_[0], &data_[0])

    def applyPreconditioner(self, object rhs):
        """
        Apply one AMG cycle of the hierarchy built by :meth:`build` to rhs: returns M^-1 rhs.

        rhs: array-like, converted to 1D float64 (C-contiguous). Always double at this boundary --
        any backend-specific narrowing/widening or block reinterpretation happens inside the C++
        wrapper, not here, since this is called once per outer Krylov iteration.
        """
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)
        cdef int n = rhs_arr.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x
        if self.blockSize == 2:
            self.solverBlock2.applyPreconditioner(n, &rhs_[0], &x_[0])
        elif self.blockSize == 3:
            if self.solverScalarSetup != NULL:
                self.solverScalarSetup.applyPreconditioner(n, &rhs_[0], &x_[0])
            elif self.solverBlock3Float != NULL:
                self.solverBlock3Float.applyPreconditioner(n, &rhs_[0], &x_[0])
            else:
                self.solverBlock3.applyPreconditioner(n, &rhs_[0], &x_[0])
        elif self.isFloat:
            self.solverFloat.applyPreconditioner(n, &rhs_[0], &x_[0])
        else:
            self.solver.applyPreconditioner(n, &rhs_[0], &x_[0])
        return x

    def report(self):
        """
        Human-readable AMGCL hierarchy report (levels, operator complexity, coarse size) for the most
        recently built() or solve()'d hierarchy.

        Raises RuntimeError if nothing has been built yet.
        """
        if self.blockSize == 2:
            return self.solverBlock2.report().decode("utf-8")
        if self.solverScalarSetup != NULL:
            return self.solverScalarSetup.report().decode("utf-8")
        if self.blockSize == 3 and self.isFloat:
            return self.solverBlock3Float.report().decode("utf-8")
        if self.blockSize == 3:
            return self.solverBlock3.report().decode("utf-8")
        if self.isFloat:
            return self.solverFloat.report().decode("utf-8")
        return self.solver.report().decode("utf-8")

    def solve(self, object A, object rhs):
        """
        A: scipy.sparse.csr_matrix
        rhs: array-like, will be converted to 1D float64 (C-contiguous)
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.asarray(rhs, dtype=np.float64, order="C")
        if rhs_arr.ndim != 1:
            raise ValueError("rhs must be a 1D array-like")

        cdef int n = A.shape[0]
        if rhs_arr.shape[0] != n:
            raise ValueError(f"Dimension mismatch: Matrix {n}x{n}, RHS {rhs_arr.shape[0]}")
        if n > np.iinfo(np.int32).max:
            raise OverflowError(f"Matrix dimension {n} exceeds supported int32 range for AMGCL wrapper")

        cdef long long int32_max = np.iinfo(np.int32).max
        cdef long long nnz = A.nnz
        if nnz > int32_max:
            raise OverflowError(f"CSR nnz value {nnz} exceeds supported int32 range for AMGCL wrapper")
        if int(A.indptr.min()) < 0:
            raise OverflowError("CSR indptr contains negative values, which are unsupported")
        if A.indices.size > 0:
            if int(A.indices.max()) > int32_max:
                raise OverflowError("CSR indices contain values that exceed int32 range for AMGCL wrapper")
            if int(A.indices.min()) < 0:
                raise OverflowError("CSR indices contain negative values, which are unsupported")

        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = A.indptr.astype(np.int32, copy=False)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = A.indices.astype(np.int32, copy=False)

        # Ensure contiguous (astype might return non-contiguous if copy=False was possible)
        if not indptr.flags["C_CONTIGUOUS"]:
            indptr = np.ascontiguousarray(indptr)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = np.ascontiguousarray(indices)

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

        cdef int iters = 0
        cdef double error = 0.0

        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x

        cdef np.ndarray[np.float32_t, ndim=1, mode="c"] dataFloat
        cdef float[::1] dataFloat_
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data
        cdef double[::1] data_

        if self.blockSize == 2 or self.blockSize == 3:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            if self.blockSize == 2:
                self.solverBlock2.solve(
                        n, &indptr_[0], &indices_[0], &data_[0], &rhs_[0], &x_[0], iters, error
                    )
            else:
                if self.solverScalarSetup != NULL:
                    self.solverScalarSetup.solve(
                            n, &indptr_[0], &indices_[0], &data_[0], &rhs_[0], &x_[0], iters, error
                        )
                elif self.solverBlock3Float != NULL:
                    self.solverBlock3Float.solve(
                            n, &indptr_[0], &indices_[0], &data_[0], &rhs_[0], &x_[0], iters, error
                        )
                else:
                    self.solverBlock3.solve(
                            n, &indptr_[0], &indices_[0], &data_[0], &rhs_[0], &x_[0], iters, error
                        )
        elif self.isFloat:
            dataFloat = np.ascontiguousarray(A.data, dtype=np.float32)
            dataFloat_ = dataFloat
            self.solverFloat.solve(
                    n,
                    &indptr_[0],
                    &indices_[0],
                    &dataFloat_[0],
                    &rhs_[0],
                    &x_[0],
                    iters,
                    error
                )
        else:
            data = np.ascontiguousarray(A.data, dtype=np.float64)
            data_ = data
            self.solver.solve(
                    n,
                    &indptr_[0],
                    &indices_[0],
                    &data_[0],
                    &rhs_[0],
                    &x_[0],
                    iters,
                    error
                )

        # AMGCL reports these by reference and they were being dropped on the floor, which left an
        # unconverged solve indistinguishable from a converged one: an iterative solver that hits
        # maxiter still returns a finite x, so the nonlinear solvers' NaN check does not catch it and
        # the Newton loop silently consumes a wrong correction. Surfaced here so callers can check.
        self.lastIterations = iters
        self.lastError = error

        return x


cdef class PyAMGCLEquilibratedSystem:
    """blockamg's equilibrated, field-split system ``As = D A D`` (``D = diag(dinv)``), built in one
    OpenMP-threaded pass over ``A``: the outer Krylov operator (:meth:`matvec`, :meth:`residual`, and
    the operator :meth:`PyAMGCLLGMRESSolver.solve` iterates on), the coupling blocks between fields
    (:meth:`couplingMatvec`) and, on demand, a field's diagonal block (:meth:`diagonalBlock`).

    Replaces a chain of serial SciPy operations -- a full copy of ``A`` scaled in place, a threaded
    copy of that for the operator, row and column slices for every coupling and diagonal block --
    measured at about 4 s of every solve on a 0.8-1.1M DOF system. The operator is bitwise the former
    one (see ``EquilibratedSystemT`` in ``amgcl-wrapper.hpp``).
    """
    cdef EquilibratedSystem* system
    cdef readonly int n
    """The system size."""

    def __cinit__(self):
        self.system = new EquilibratedSystem()
        self.n = 0

    def __dealloc__(self):
        if self.system != NULL:
            del self.system

    @staticmethod
    def absoluteDiagonal(object A):
        """``abs(A.diagonal())`` of a scipy.sparse CSR matrix, threaded."""
        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        cdef double[::1] out_ = out
        EquilibratedSystem.absoluteDiagonal(n, &indptr_[0], &indices_[0], &data_[0], &out_[0])
        return out

    def build(self, object A, object dinv, object fieldBounds, bint keepUnscaled=False):
        """
        A: scipy.sparse csr_matrix (square); dinv: the equilibration, length n; fieldBounds: a
        sequence of ``(start, stop)`` DOF ranges, one per field, disjoint. keepUnscaled: also keep A
        itself, for :meth:`unscaledMatvec`/:meth:`unscaledResidual` and an outer solve on it.
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] dinv_arr = np.ascontiguousarray(dinv, dtype=np.float64)
        cdef np.ndarray[np.int32_t, ndim=2, mode="c"] bounds = np.ascontiguousarray(
            np.asarray(fieldBounds, dtype=np.int32).reshape(-1, 2)
        )
        if dinv_arr.shape[0] != n:
            raise ValueError("dinv has length {:}, the system has {:} rows".format(dinv_arr.shape[0], n))
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        cdef double[::1] dinv_ = dinv_arr
        cdef int[:, ::1] bounds_ = bounds
        cdef int nFields = bounds.shape[0]
        self.system.build(n, &indptr_[0], &indices_[0], &data_[0], &dinv_[0], nFields,
                          &bounds_[0, 0] if nFields > 0 else NULL, keepUnscaled)
        self.n = n

    def diagonalBlock(self, int field):
        """``As[field, field]`` as a new scipy.sparse csr_matrix (sorted columns if ``A``'s were)."""
        cdef int size = self.system.diagonalBlockSize(field)
        cdef int nnz = self.system.diagonalBlockNnz(field)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.empty(size + 1, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.empty(max(nnz, 1), dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.empty(max(nnz, 1), dtype=np.float64)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        self.system.fillDiagonalBlock(field, &indptr_[0], &indices_[0], &data_[0])
        return scipy.sparse.csr_matrix((data[:nnz], indices[:nnz], indptr), shape=(size, size), copy=False)

    def matvec(self, object x):
        """Returns ``As @ x`` as a new float64 array."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(self.n, dtype=np.float64)
        cdef double[::1] x_ = x_arr
        cdef double[::1] y_ = y
        self.system.matvec(&x_[0], &y_[0])
        return y

    def unscaledMatvec(self, object x):
        """Returns ``A @ x`` on the unscaled operator (``build(..., keepUnscaled=True)``)."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(self.n, dtype=np.float64)
        cdef double[::1] x_ = x_arr
        cdef double[::1] y_ = y
        self.system.unscaledMatvec(&x_[0], &y_[0])
        return y

    def unscaledResidual(self, object rhs, object x):
        """Returns ``rhs - A @ x`` on the unscaled operator."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] r = np.empty(self.n, dtype=np.float64)
        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x_arr
        cdef double[::1] r_ = r
        self.system.unscaledResidual(&rhs_[0], &x_[0], &r_[0])
        return r

    def residual(self, object rhs, object x):
        """Returns ``rhs - As @ x`` as a new float64 array."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] r = np.empty(self.n, dtype=np.float64)
        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x_arr
        cdef double[::1] r_ = r
        self.system.residual(&rhs_[0], &x_[0], &r_[0])
        return r

    def diagonalBlockMatvec(self, int field, object x):
        """Returns ``As[field, field] @ x`` as a new float64 array of this field's size."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.shape[0] != self.system.diagonalBlockSize(field):
            raise ValueError("x has the wrong length for field {:}".format(field))
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(x_arr.shape[0], dtype=np.float64)
        if x_arr.shape[0] == 0:
            return y
        cdef double[::1] x_ = x_arr
        cdef double[::1] y_ = y
        self.system.diagonalBlockMatvec(field, &x_[0], &y_[0])
        return y

    def couplingMatvec(self, int field, int otherField, object x):
        """Returns ``As[field, otherField] @ x`` (``x`` of the other field's size) as a new float64
        array of this field's size."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.shape[0] != self.system.diagonalBlockSize(otherField):
            raise ValueError("x has the wrong length for field {:}".format(otherField))
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(
            self.system.diagonalBlockSize(field), dtype=np.float64
        )
        cdef double[::1] x_ = x_arr
        cdef double[::1] y_ = y
        if x_arr.shape[0] == 0 or y.shape[0] == 0:
            y[:] = 0.0
            return y
        self.system.couplingMatvec(field, otherField, &x_[0], &y_[0])
        return y


cdef void _lgmresPrecondApplyTrampoline(void* ctx, int n, const double* rhs, double* x) noexcept:
    """The C-callable trampoline bridging amgcl::solver::lgmres's per-``apply()`` Precond callback
    back into Python -- passed to :class:`LGMRESOuterSolver` as a bare function pointer (see
    ``amgcl-wrapper.hpp``'s ``PyPrecondApplyFn``/``PyLGMRESPrecondT``).

    ``ctx`` is the calling :class:`PyAMGCLLGMRESSolver` instance itself, cast to ``void*`` and back --
    Cython's own documented pattern for passing a Python object through an opaque C callback context.
    No manual ``PyObject*`` reference counting is needed: ``ctx`` only ever points at ``self`` from the
    *same*, still-on-stack :meth:`PyAMGCLLGMRESSolver.solve` call that constructed it (see there), so
    its lifetime is strictly nested inside this function's own call and never outlives it.

    Declared ``noexcept`` and *not* ``nogil``: every call into this function happens nested inside
    :meth:`PyAMGCLLGMRESSolver.solve`'s own C++ call, made from a plain ``def`` method that never
    releases the GIL (matching every other AMGCL wrapper method in this module -- the GIL only blocks
    *other* Python threads from running during the solve; AMGCL's own OpenMP worker threads never touch
    a Python object, so they are unaffected either way). Not releasing the GIL around the C++ call was
    a deliberate choice, not an oversight: doing so would require re-acquiring it here (``with gil:``)
    on every single preconditioner application -- the hottest inner loop this whole class exists to
    speed up -- for a benefit (letting unrelated Python threads run during one linear solve) this
    codebase's actual threading model does not use.

    A Python exception raised by the preconditioner callable cannot propagate through this C
    function-pointer boundary (there is no C++ exception handler on the AMGCL side expecting one), so
    it is caught here and stashed on the calling instance instead; :meth:`PyAMGCLLGMRESSolver.solve`
    re-raises it once the (now-abandoned) AMGCL call returns. ``x`` is zeroed on that path so the
    (about-to-be-discarded) AMGCL iteration continues on defined, if meaningless, numbers rather than
    whatever uninitialized/stale data happened to be there, and every further call returns at once.
    """
    cdef PyAMGCLLGMRESSolver self = <PyAMGCLLGMRESSolver> ctx
    cdef Py_ssize_t i
    # rhs arrives as a `const double*` (it is AMGCL's own internal Arnoldi-vector storage, never meant
    # to be written through) -- copied element-by-element into a fresh, ordinary (non-const) numpy
    # array rather than wrapped in a zero-copy memoryview, which would need casting away the pointer's
    # constness. n is at most the outer Krylov restart length, so this copy is not the cost driver
    # here -- the Python-level preconditioner call below is.
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhsArr
    cdef double[::1] xView
    if self._pendingException is not None:
        # an earlier application already failed: the AMGCL call is abandoned anyway, so do not keep
        # calling into Python (and raising) for every remaining Arnoldi vector
        for i in range(n):
            x[i] = 0.0
        return
    rhsArr = np.empty(n, dtype=np.float64)
    try:
        for i in range(n):
            rhsArr[i] = rhs[i]
        result = self._precondCallable(rhsArr)
        xView = np.ascontiguousarray(result, dtype=np.float64)
        if xView.shape[0] != n:
            raise ValueError(
                "PyAMGCLLGMRESSolver: preconditioner callable returned a vector of length {:} for an "
                "input of length {:}".format(xView.shape[0], n)
            )
        for i in range(n):
            x[i] = xView[i]
    except BaseException as exc:
        self._pendingException = exc
        for i in range(n):
            x[i] = 0.0


cdef class PyAMGCLLGMRESSolver:
    """AMGCL's own native ``amgcl::solver::lgmres`` as blockamg.py's outer Krylov solve, in
    place of ``scipy.sparse.linalg.gmres`` -- see ``amgcl-wrapper.hpp``'s ``LGMRESOuterSolverT`` for
    the full motivation and design reasoning (this class is a thin Cython shell around it).

    Bridges the field-split block Gauss-Seidel preconditioner
    (:mod:`edelweissfe.linsolve.blockamg.blockamg`'s ``blockGaussSeidel`` closure, dispatching to
    per-field :class:`PyAMGCLSolver`/:class:`~edelweissfe.linsolve.blockamg.ptwogrid.
    PTwoGridPreconditioner` objects) to AMGCL's native ``Precond`` interface via
    :func:`_lgmresPrecondApplyTrampoline`, called once per Arnoldi vector.

    One instance per system size (AMGCL preallocates every scratch vector for it), reused across solves
    with per-call tolerances. Every call starts from an empty augmentation subspace (``always_reset`` is
    forced): carrying the augmentation vectors across calls was measured to buy nothing and to turn a
    struggling solve's subspace into much more expensive or NaN subsequent ones, so it is not offered.
    """
    cdef LGMRESOuterSolver* solver
    cdef readonly int n
    """The fixed problem size this instance was constructed for."""
    cdef readonly int lastIterations
    """The iteration count AMGCL's lgmres reported for the most recent solve, -1 before the first."""
    cdef readonly double lastError
    """The relative residual AMGCL's lgmres reported for the most recent solve -- of the *scaled* system,
    right-preconditioned, so not the true residual -- NaN before the first."""
    cdef object _precondCallable
    cdef object _pendingException

    def __cinit__(self, int n, dict params=None):
        """
        n
            The fixed problem size this instance is built for.
        params
            AMGCL's own ``lgmres::params`` field names, forwarded verbatim as JSON (``M``, ``K``,
            ``pside``, ``maxiter``, ``tol``, ``abstol``, ``ns_search``, ``verbose``; ``always_reset`` is
            forced to true)
            -- unlike :class:`PyAMGCLSolver`, there is no ``precond``/``solver`` nesting here, since
            this wraps a single ``amgcl::solver::lgmres`` object directly, not ``amgcl::make_solver``.
            ``maxiter``/``tol`` given here only set the *initial* defaults -- :meth:`solve` overrides
            both on every call (see ``amgcl-wrapper.hpp``'s ``LGMRESOuterSolverT`` for why).
        """
        if params is None:
            params = {}
        self.n = n
        self.lastIterations = -1
        self.lastError = float("nan")
        self._precondCallable = None
        self._pendingException = None

        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.solver = new LGMRESOuterSolver(n, c_json)

    def __dealloc__(self):
        if self.solver != NULL:
            del self.solver

    def solve(self, PyAMGCLEquilibratedSystem system, object rhs, object applyPreconditioner, double tol,
              int maxiter, object x0=None, bint onUnscaledOperator=False):
        """
        system
            This solve's equilibrated full coupled operator -- or, with ``onUnscaledOperator``, the
            unscaled one it kept (``build(..., keepUnscaled=True)``).
        rhs
            array-like, converted to 1D float64 (C-contiguous).
        applyPreconditioner
            A Python callable, ``residual (1D float64 array) -> correction (1D float64 array)`` --
            exactly :mod:`~edelweissfe.linsolve.blockamg.blockamg`'s ``blockGaussSeidel`` closure's own
            shape. Called once per Arnoldi vector via :func:`_lgmresPrecondApplyTrampoline`; any
            exception it raises is re-raised here once the (now-abandoned) AMGCL call returns.
        tol, maxiter
            This call's relative residual tolerance and maximum total Arnoldi iterations (across every
            internal restart -- AMGCL's own ``maxiter`` semantics differ from scipy's restart x maxiter,
            see blockamg.py's call site), set on the underlying ``amgcl::solver::lgmres`` object's own
            ``prm`` for this call.
        x0
            Optional initial guess (warm start), e.g. blockamg.py's true-residual continuation retry.
            Defaults to zero.
        """
        cdef int n = system.n
        if n != self.n:
            raise ValueError(
                "PyAMGCLLGMRESSolver: matrix size {:} does not match the size {:} this instance was "
                "constructed for -- construct a fresh instance on any field-structure/size "
                "change.".format(n, self.n)
            )

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x
        if x0 is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.ascontiguousarray(x0, dtype=np.float64).copy()

        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x

        cdef int iters = 0
        cdef double error = 0.0

        # self is passed through as the callback's opaque context, cast to void* and back in
        # _lgmresPrecondApplyTrampoline -- see that function's own docstring for why this needs no
        # manual PyObject* reference counting (self is kept alive by this very call's own stack frame
        # for as long as the cast pointer is in use).
        self._precondCallable = applyPreconditioner
        self._pendingException = None
        cdef void* ctx = <void*> self
        try:
            self.solver.solve(
                system.system[0], &rhs_[0], &x_[0],
                tol, maxiter, _lgmresPrecondApplyTrampoline, ctx, iters, error, onUnscaledOperator
            )
        finally:
            self._precondCallable = None

        if self._pendingException is not None:
            exc = self._pendingException
            self._pendingException = None
            raise exc

        self.lastIterations = iters
        self.lastError = error
        return x


cdef class PyAMGCLSpGEMM:
    """OpenMP-threaded sparse-matrix product/sum, wrapping AMGCL's own
    ``amgcl::backend::builtin::product()``/``sum()`` -- verified directly (not assumed) against the
    installed headers that both are OpenMP-parallel (``spgemm_saad``/``spgemm_rmerge`` and ``sum()``
    itself each carry ``#pragma omp parallel``/``#pragma omp for``).

    Used by :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.
    _transformSystemMatrixAmgcl` as an opt-in alternative (``useAmgclSpgemm``/
    ``useAmgclMPCCondensation``) to the plain ``T^T @ K @ T + C`` expression's ``scipy.sparse`` CSR
    products/addition, all single-threaded (SciPy's own sparse routines carry no OpenMP parallelism
    at all). This class exposes the two primitives (``product``, ``sum``) generically, one call at a
    time, matching ``SpGEMMHelperT``'s own design (square matrices only; compose calls in Python).
    """
    cdef SpGEMMHelper* helper

    def __cinit__(self):
        self.helper = new SpGEMMHelper()

    def __dealloc__(self):
        if self.helper != NULL:
            del self.helper

    def _copyOut(self):
        cdef int nRows = self.helper.resultNRows()
        cdef int nCols = self.helper.resultNCols()
        if nRows != nCols:
            raise AssertionError(
                "PyAMGCLSpGEMM: result is {:}x{:}, not square -- product()/sum() are only defined "
                "for square, equal-size inputs (enforced in SpGEMMHelperT::product/sum), so this "
                "should be unreachable.".format(nRows, nCols)
            )
        cdef int nnz = self.helper.resultNnz()
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] ptrOut = np.empty(nRows + 1, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] colOut = np.empty(nnz, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] valOut = np.empty(nnz, dtype=np.float64)
        cdef int[::1] ptrOut_ = ptrOut
        cdef int[::1] colOut_ = colOut
        cdef double[::1] valOut_ = valOut
        self.helper.copyResult(&ptrOut_[0], &colOut_[0], &valOut_[0])
        return scipy.sparse.csr_matrix((valOut, colOut, ptrOut), shape=(nRows, nRows))

    def product(self, object A, object B):
        """Returns A @ B (both square scipy.sparse CSR, same size) as a new csr_matrix."""
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        if not scipy.sparse.isspmatrix_csr(B):
            B = B.tocsr()

        cdef int aN = A.shape[0]
        cdef int bN = B.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] aPtr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] aCol = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] aVal = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] bPtr = np.ascontiguousarray(B.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] bCol = np.ascontiguousarray(B.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] bVal = np.ascontiguousarray(B.data, dtype=np.float64)
        cdef int[::1] aPtr_ = aPtr
        cdef int[::1] aCol_ = aCol
        cdef double[::1] aVal_ = aVal
        cdef int[::1] bPtr_ = bPtr
        cdef int[::1] bCol_ = bCol
        cdef double[::1] bVal_ = bVal
        self.helper.product(aN, &aPtr_[0], &aCol_[0], &aVal_[0], bN, &bPtr_[0], &bCol_[0], &bVal_[0])
        return self._copyOut()

    def sum(self, double alpha, object A, double beta, object B):
        """Returns alpha*A + beta*B (both square scipy.sparse CSR, same size) as a new csr_matrix."""
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        if not scipy.sparse.isspmatrix_csr(B):
            B = B.tocsr()

        cdef int aN = A.shape[0]
        cdef int bN = B.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] aPtr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] aCol = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] aVal = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] bPtr = np.ascontiguousarray(B.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] bCol = np.ascontiguousarray(B.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] bVal = np.ascontiguousarray(B.data, dtype=np.float64)
        cdef int[::1] aPtr_ = aPtr
        cdef int[::1] aCol_ = aCol
        cdef double[::1] aVal_ = aVal
        cdef int[::1] bPtr_ = bPtr
        cdef int[::1] bCol_ = bCol
        cdef double[::1] bVal_ = bVal
        self.helper.sum(alpha, aN, &aPtr_[0], &aCol_[0], &aVal_[0], beta, bN, &bPtr_[0], &bCol_[0], &bVal_[0])
        return self._copyOut()


cdef class PyAMGCLPTwoGrid:
    """The p-two-grid (P1) preconditioner of :mod:`edelweissfe.linsolve.blockamg.ptwogrid`, built and
    applied entirely in C++: Dirichlet detection, the restricted prolongation ``P``, the Galerkin
    coarse operator ``P^T A P``, the coarse AMGCL hierarchy, the fine Chebyshev smoother and the
    whole two-grid cycle, all OpenMP-threaded -- no Python runs inside :meth:`applyPreconditioner`.
    See ``PTwoGridPreconditionerT`` in ``amgcl-wrapper.hpp`` for the Dirichlet scheme and the
    fine spectral-radius estimate.
    """
    cdef PTwoGridPreconditioner* preconditioner

    def __cinit__(self, dict params):
        """
        params: the flat parameter tree, e.g.
        {"nu": 1, "fineDegree": 5, "fineLower": 0.01, "fineHigher": 1.1, "finePowerIters": 300,
        "fineScale": false, "fineBlockSize": 1, "coarse": {<AMGCL preconditioner tree>}}
        """
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.preconditioner = new PTwoGridPreconditioner(c_json)

    def __dealloc__(self):
        if self.preconditioner != NULL:
            del self.preconditioner

    @staticmethod
    def dirichletRows(object A):
        """The rows of a scipy.sparse CSR matrix this preconditioner treats as Dirichlet rows: a nonzero
        diagonal and no nonzero off-diagonal entry (explicitly stored zeros, which the Dirichlet
        elimination keeps, do not count). Returns a bool array."""
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef np.ndarray[np.int8_t, ndim=1, mode="c"] mask = np.zeros(n, dtype=np.int8)
        if n == 0 or A.nnz == 0:
            return mask.astype(bool)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        cdef signed char[::1] mask_ = mask
        PTwoGridPreconditioner.dirichletRowMask(n, &indptr_[0], &indices_[0], &data_[0], <char*> &mask_[0])
        return mask.astype(bool)

    def build(self, object A, object isCorner, object edgeEndpoints, object nullspace=None,
              double spectralRadius=0.0, object dominantVector=None):
        """
        Build the preconditioner for A (scipy.sparse csr_matrix, the field's equilibrated diagonal
        block with its Dirichlet identity rows).

        isCorner: (nNodes,) bool/int8; edgeEndpoints: (nNodes, 2) int, read for midside nodes only.
        nullspace: (nFreeCoarseDofs, k) float64 coarse near null-space, or None.
        spectralRadius: > 0 to start from this fine spectral radius instead of estimating it in full.
        dominantVector: with spectralRadius, the previous build's :attr:`fineDominantVector`; the
        radius is then re-checked by a few power iterations from it (``fineRecheckIters``) and
        raised, or re-estimated in full, if the operator has stiffened.
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()
        if not A.has_sorted_indices:
            A = A.sorted_indices()

        cdef int n = A.shape[0]
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.ascontiguousarray(A.data, dtype=np.float64)
        cdef np.ndarray[np.int8_t, ndim=1, mode="c"] corner = np.ascontiguousarray(isCorner, dtype=np.int8)
        cdef np.ndarray[np.int32_t, ndim=2, mode="c"] endpoints = np.ascontiguousarray(edgeEndpoints, dtype=np.int32)
        cdef int nNodes = corner.shape[0]
        if endpoints.shape[0] != nNodes or endpoints.shape[1] != 2:
            raise ValueError("edgeEndpoints must have shape (nNodes, 2)")

        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] B
        cdef int nullspaceRows = 0
        cdef int nullspaceCols = 0
        cdef const double* nullspacePtr = NULL
        if nullspace is not None:
            B = np.ascontiguousarray(nullspace, dtype=np.float64)
            nullspaceRows = B.shape[0]
            nullspaceCols = B.shape[1]
            if nullspaceRows > 0 and nullspaceCols > 0:
                nullspacePtr = &B[0, 0]

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] dominant
        cdef const double* dominantPtr = NULL
        cdef int dominantSize = 0
        if dominantVector is not None:
            dominant = np.ascontiguousarray(dominantVector, dtype=np.float64)
            dominantSize = dominant.shape[0]
            if dominantSize > 0:
                dominantPtr = &dominant[0]

        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        cdef signed char[::1] corner_ = corner
        cdef int[:, ::1] endpoints_ = endpoints
        self.preconditioner.build(n, &indptr_[0], &indices_[0], &data_[0], nNodes, &corner_[0], &endpoints_[0, 0],
                                  nullspacePtr, nullspaceRows, nullspaceCols, spectralRadius, dominantPtr, dominantSize)

    def applyPreconditioner(self, object r):
        """One two-grid cycle: returns M^-1 r as a new float64 array (Dirichlet entries pass through)."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] r_arr = np.ascontiguousarray(r, dtype=np.float64)
        cdef int n = r_arr.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.empty(n, dtype=np.float64)
        cdef double[::1] r_ = r_arr
        cdef double[::1] x_ = x
        self.preconditioner.apply(n, &r_[0], &x_[0])
        return x

    @property
    def fineSpectralRadius(self):
        """The fine operator's spectral radius the Chebyshev smoother was built with -- estimated, or
        the one handed to :meth:`build`."""
        return self.preconditioner.fineSpectralRadius()

    @property
    def fineDominantVector(self):
        """The power iteration's last iterate, the estimate of the fine operator's dominant
        eigenvector -- hand it back to the next :meth:`build` with :attr:`fineSpectralRadius`. Empty
        if the radius was used as supplied, without a re-check."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] v = np.empty(
            self.preconditioner.fineDominantVectorSize(), dtype=np.float64
        )
        cdef double[::1] v_ = v
        if v.shape[0] > 0:
            self.preconditioner.copyFineDominantVector(&v_[0])
        return v

    @property
    def fineSpectralRadiusSource(self):
        """How the last build arrived at its fine spectral radius: ``"estimated"`` in full,
        ``"supplied"`` and kept, ``"raised"`` by the re-check, or ``"re-estimated"`` in full because
        the re-check found the operator stiffened beyond the smoother's margin."""
        return ("estimated", "supplied", "raised", "re-estimated")[self.preconditioner.fineSpectralRadiusSource()]

    @property
    def nCoarseFree(self):
        """The number of free coarse DOFs (rows of the Galerkin operator)."""
        return self.preconditioner.nCoarseFree()

    def timings(self):
        """Cumulative wall time [s] of the build stages (of the last build) and of the apply (fine
        smoothing incl. the residual, coarse correction incl. restriction and prolongation)."""
        return {
            "setup: dirichlet + P": self.preconditioner.setupDirichletAndPSeconds_,
            "setup: galerkin P^T A P": self.preconditioner.setupGalerkinSeconds_,
            "setup: coarse hierarchy": self.preconditioner.setupCoarseSeconds_,
            "setup: fine operator": self.preconditioner.setupFineOperatorSeconds_,
            "setup: fine spectral radius": self.preconditioner.setupSpectralRadiusSeconds_,
            "apply: fine": self.preconditioner.applyFineSeconds_,
            "apply: coarse": self.preconditioner.applyCoarseSeconds_,
        }

    def report(self):
        """The coarse level's AMGCL hierarchy report."""
        return self.preconditioner.report().decode("utf-8")
