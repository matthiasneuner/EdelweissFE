# Comprehensive Code Review & Strategic Recommendations Report

**Target Branch**: `perf/linsolve-investigation`
**Subproject**: EdelweissFE
**Reference Benchmark**: Anchor Pry-out simulation (280,155 DOFs, 14,036 active elements, 1,208 hanging nodes, 16,556 MPC slave DOFs)
**Primary Reference Files**:
- `EdelweissFE/PERF_LINSOLVE_INVESTIGATION.md` (Phases 1–24)
- `EdelweissFE/edelweissfe/linsolve/blockamg/blockamg.py`
- `EdelweissFE/edelweissfe/linsolve/blockamg/ptwogrid.py`
- `EdelweissFE/edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp`
- `EdelweissFE/edelweissfe/linsolve/amgcl/amgcl.pyx`
- `EdelweissFE/edelweissfe/numerics/mpctransformation.py`

---

## 1. Executive Summary & Investigation Plan Status

The investigation in `PERF_LINSOLVE_INVESTIGATION.md` chronically documents the transition from direct factorizations (`MKL PARDISO`) to field-split block algebraic multigrid (`blockamg`) to eliminate the $O(N^{1.5-2})$ memory/scaling wall for coupled multi-field simulations (>500k DOFs).

### Key Shipped Achievements
1. **Threaded Outer SpMV** (`ThreadedMatrixT` in `amgcl-wrapper.hpp`): Replaced single-threaded SciPy CSR matrix-vector products with OpenMP-threaded C++ SpMV. Shipped default yielding **1.15× live speedup**.
2. **Native C++ LGMRES as Shipped Default** (`LGMRESOuterSolverT` / `amgcl::solver::lgmres`): Replaced SciPy outer GMRES orchestration with AMGCL's native C++ solver. With `lgmresAlwaysReset=True` (§23.10), LGMRES scales smoothly across NUMA nodes (**2.18× faster than SciPy at 32 threads**).
3. **Eisenstat–Walker Forcing & True Residual Continuations**: Adaptive outer tolerance ($\eta_{\max} = 3\times 10^{-4}$) reduces outer GMRES iterations per Newton step while preserving exact Newton convergence paths by enforcing true relative residual checks. The forcing formula $\eta_k = \gamma \left( \frac{\|r_k\|}{\|r_{k-1}\|} \right)^\alpha$ successfully prevents linear oversolving.
4. **Threaded MPC Condensation (`useAmgclMPCCondensation`)**: Accelerates master-slave condensation $K_c = T^T K T + C$ using AMGCL's OpenMP `product()` / `sum()` primitives (`PyAMGCLSpGEMM`), giving **~2.5× offline / 7.3% live job speedup**.

---

## 2. Granular Reference Catalog of Investigation Phases 1–24

| Phase / Section | Core Focus / Hypothesis | Key Outcome / Headline Number | Status |
|---|---|---|---|
| **Phase 1 (§2)** | Profiling Baseline | Direct solve = 82% of step time (Factorization = 77%, MPC = 14%) | Baseline established |
| **Phase 2 (§3.1–3.3)** | Sparsity Churn & Threading | Pattern churns every iterate ($\pm 200\text{k}$ NNZ); unified pattern gives **1.78×** speedup | Pattern reuse limited by contact/MPC |
| **Phase 3, Lead 1 (§3.4, §10)** | Lagged Direct LU + Inexact Newton | Reused LU factor converges in 4–9 GMRES iters (**2–3× speedup**) | Implemented (`inexactnewton`) |
| **Phase 3, Lead 3, Step 1 (§11)** | Monolithic AMG Near-Nullspace | Monolithic AMG fails (stalls at 0.2–0.65 residual) even with translation vectors | Falsified monolithic AMG |
| **Phase 3, Lead 3, Step 2 (§12)** | Block AMG Feasibility | Field-split Block AMG converges (93–117 outer iters) where monolithic fails | Feasibility proven |
| **Phase 3, Lead 3, Step 3 (§13–14)** | Initial `blockamg` Delivery | Delivered `linsolver=blockamg`; 68 outer iters on coupled system | Shipped feasibility grade |
| **Phases A/B (§16–17)** | Retracting "Needs MueLu" | Symmetrizing free submatrix & tuning `aggr.eps_strong=0.01` + Chebyshev (d=8) $\to$ **29 iters** | Re-tuned default |
| **Phase 18 (§18)** | Iteration Count vs. Wall Clock | Tuning Chebyshev (d=5, npre=npost=1) cut wall-clock by **~23%** | Re-tuned default |
| **Phase 4 (§19)** | Wall-Clock Parity & Mixed Precision | EW forcing gave 1.56× offline win; float32 backend gave only 1.03× win | EW rescue needed |
| **Phase 5 (§20)** | Block-Valued B3 & EW Rescue | B3 backend regressed 23%; EW forcing + true residual check shipped (**etaMax=3e-4**) | Shipped EW default |
| **Phase 6 (§21)** | Cached-Pattern MPC Condensation | Value scatter was 1.7× slower due to SciPy SpGEMM row scanning | Replaced by Phase 24 |
| **Phase 7 (§22.1–22.2)** | p-Multigrid Corner Topology Map | Corner/midside map built; $A_1 = P^T A_2 P$ coarse solve converged in **26 iters** | Enabler validated |
| **Phase 7 (§22.3–22.4)** | Threaded Relaxation Smoother | `RelaxationSmootherT` replaced serial Chebyshev (2.15× smoother win; parity overall) | Opt-in (`p1Maps`) |
| **Phase 8 (§23.1–23.5)** | Threaded Outer SpMV | `ThreadedMatrixT` OpenMP SpMV delivered **1.22–1.27× offline / 1.15× live speedup** | **Shipped default** |
| **Phase 8 (§23.7–23.10)** | Native C++ `lgmres` Outer Solver | Fixed stale Krylov NaN (`lgmresAlwaysReset=True`); **2.18× speedup at 32 threads** | **Shipped default** |
| **Phase 24 (§24, §24.1)** | Threaded MPC Condensation | AMGCL SpGEMM `product()`/`sum()` delivered **2.5× offline / 7.3% live speedup** | Shipped opt-in |

---

## 3. Detailed Code Review & Technical Flaws Identified

### Flaw 1: Sparsity Pattern Change Detection Heuristic (`A.nnz`)
- **Location**: `edelweissfe/linsolve/blockamg/blockamg.py:L497–L514`
- **Code**: `patternChanged = self._lastNnz is not None and A.nnz != self._lastNnz`
- **Mechanism**: `A.nnz` (total non-zero count) is used as an $O(1)$ proxy for pattern changes. Two sparse matrices $A_1$ and $A_2$ can have identical non-zero counts while having different sparsity structures (e.g., contact opening at node A while closing at node B).
- **Consequence**: `mustRefresh` evaluates to `False`, causing `blockamg` to reuse an AMG hierarchy built for a different sparsity structure. As documented in §3.1/§19.2, reusing a hierarchy built for a different pattern causes outer GMRES iterations to explode (**494 iterations vs. 94 iterations**).
- **Fix**: Compare `indptr` directly or hash `indptr`. This provides an exact structural equality check for CSR matrices in negligible time:
  ```python
  patternChanged = self._lastIndptr is not None and not np.array_equal(A.indptr, self._lastIndptr)
  ```

### Flaw 2: Heap Memory Allocations Per Krylov Step in C++ Wrapper
- **Location**: `edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp:L109–L117`
- **Code**:
  ```cpp
  void applyPreconditioner( int n, const double* rhs, double* x ) {
      std::vector< ValueType > rhs_v( rhs, rhs + n );
      std::vector< ValueType > x_v( n, ValueType( 0 ) );
      // ...
  }
  ```
- **Mechanism**: On every outer GMRES iteration inside Block Gauss–Seidel, two `std::vector` instances of size $N$ are allocated and freed on the heap ($\approx 3.4\text{ MB}$ per Krylov step at 214k DOFs).
- **Consequence**: Triggers tens of thousands of dynamic heap allocations during a simulation, causing cache pollution, heap fragmentation, and unnecessary allocation latency.
- **Fix**: For `double` (`ValueType == double`), utilize `amgcl::make_iterator_range` to pass `rhs` and `x` zero-copy directly to the preconditioner, leveraging the contiguous memory layout:
  ```cpp
  auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
  auto x_rng   = amgcl::make_iterator_range( x, x + n );
  ```

### Flaw 3: Dynamic NumPy Array Allocation in Outer Loop Trampoline
- **Location**: `edelweissfe/linsolve/amgcl/amgcl.pyx:L515–L525`
- **Code**: `cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhsArr = np.empty(n, dtype=np.float64)`
- **Mechanism**: `_lgmresPrecondApplyTrampoline` allocates a new 1D NumPy array (`np.empty`) on **every single Arnoldi iteration call** from C++ LGMRES.
- **Consequence**: Incurs CPython C-API allocation and Garbage Collection tracking overhead inside the performance-critical inner C++ Arnoldi loop.
- **Fix**: Pre-allocate `_rhsScratch` and `_xScratch` memoryview buffers on the `PyAMGCLLGMRESSolver` instance during initialization and reuse them per iteration, ensuring zero dynamic allocations inside the Krylov subspace generation.

### Flaw 4: OpenMP vs. OpenBLAS Thread Oversubscription
- **Location**: Execution environment & driver runbooks (`run_bench.sh`).
- **Mechanism**: Scripts set `OMP_NUM_THREADS=16` (or `32`), but NumPy/SciPy linked against OpenBLAS spawn their own pthreads pool controlled by `OPENBLAS_NUM_THREADS`.
- **Consequence**: SciPy GMRES / BLAS operations run OpenBLAS threads simultaneously alongside OpenMP threads, producing massive 32–64 thread contention on 36 physical cores (causing SciPy GMRES to be 75% slower at 32 threads).
- **Fix**: Explicitly export `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in all run scripts to ensure OpenMP maintains exclusive ownership of the thread pool.

### Flaw 5: Serendipity P-Multigrid Coarse Grid Stencil Over-Density
- **Location**: `edelweissfe/linsolve/blockamg/ptwogrid.py:L225`
- **Code**: `A1_free = (P_free.T @ A_free @ P_free).tocsr()`
- **Mechanism**: Galerkin projection for 20-node serendipity hexahedral elements ($Q_2$) produces a coarse $P_1$ matrix $A_1$ with $\approx 71$ non-zeros per row (vs $\approx 27$ for trilinear hexes), dramatically increasing the density of the coarse grid operator.
- **Consequence**: Coarse AMG V-cycles on this densified $A_1$ (`coarseSeconds`) consume **55–61% of total p-two-grid preconditioner time**, neutralizing outer iteration reductions ($66 \to 50$) and holding p-multigrid at wall-clock parity.
- **Fix**: Apply a numerical threshold drop-tolerance filtering to $A_1$ post-projection to prune weak fill-in entries, preserving the spectral properties of the coarse operator while drastically reducing SpMV latency during the V-cycle.

### Flaw 6: Unhandled Preconditioner Non-Convergence Propagation
- **Location**: `edelweissfe/linsolve/blockamg/blockamg.py:L781–L787`
- **Mechanism**: When GMRES/LGMRES fails to converge within `maxiter` (`info != 0`), `__call__` logs a warning and returns `x` to the nonlinear solver anyway.
- **Consequence**: The Newton solver receives an inaccurate correction vector $\delta U$. A non-converged finite vector causes Newton iterations to missteer or take unnecessary step cutbacks instead of immediately triggering a linear refactorization or matrix pattern refresh.
- **Fix**: Raise a dedicated `LinearSolverFailedException` or set an explicit error flag. This allows the nonlinear driver to handle non-converged iterations deterministically (e.g., by cutting the time step or refreshing the preconditioner).

---

## 4. Unexploited AMGCL Capabilities Catalog

| Feature | AMGCL Header / Class | Unexploited Power & Application in EdelweissFE |
|---|---|---|
| **Native Block Schur Preconditioner** | `amgcl::preconditioner::schur_pressure_correction` | Executes full physics-based block factorizations (**SIMPLE**, **SIMPLEC**, **LDU**) **entirely inside C++**, eliminating all Python callback overhead and GIL retention during Krylov iterations. Crucial for scaling beyond 1M DOFs. |
| **Energy Minimization Coarsening** | `amgcl::coarsening::smoothed_aggr_emin` | Minimizes prolongation energy $\|P^T A P\|$; adapts AMG hierarchy to localized damage/phase-field crack bands where standard smoothed aggregation creates overly stiff coarse modes. |
| **Flexible GMRES** | `amgcl::solver::fgmres` | Allows preconditioners to vary per Krylov iteration; enables loose adaptive inner block solves inside `BlockAMGSolver`, reducing over-solving on inner fields. |
| **Mixed-Precision Block Backend** | `builtin<static_matrix<float, 3, 3>>` | Combines 9× reduction in CSR index traffic ($3\times 3$ block) with 50% memory traffic reduction (`float32`), yielding **>60% memory bandwidth savings** for the 3D displacement block. |
| **Sparse Approximate Inverse Level 1** | `amgcl::relaxation::spai1` | Incorporates nearest-neighbor sparsity couplings; provides robust non-symmetric smoothing for penalty contact and heavily condensed MPC matrices ($T^T K T + C$) where Chebyshev struggles. |
| **Threshold ILU ($ILUT$)** | `amgcl::relaxation::ilu<N>` | Exposes higher fill-in $ILU(k)$ or threshold $ILUT$ for severely ill-conditioned field blocks (like non-local damage) where $ILU(0)$ or Chebyshev stall. |
| **GPU Acceleration Backends** | `amgcl::backend::cuda` / `vexcl` | Offloads entire AMG V-cycle and LGMRES Krylov solves to GPUs by swapping the C++ backend template parameter without changing solver code. |

---

## 5. Strategic Architectural Recommendations

### Recommendation A: Native C++ Block Gauss-Seidel / Schur Complement
The current Python-orchestrated block Gauss-Seidel preconditioner (`blockamg.py`) forces constant C++/Python boundary crossings during the inner Krylov iterations. Moving this field-split logic into a native C++ `BlockGSPrecondT` class inside `amgcl-wrapper.hpp`—or leveraging AMGCL's native `amgcl::preconditioner::schur_pressure_correction`—will eliminate Python overhead entirely. This ensures maximum cache locality and thread efficiency for the LGMRES outer solver.

### Recommendation B: 6-DOF Rigid Body Modes for 3D Elasticity
Extend the near-nullspace definition (`_translationNullspace`) to include the 3 rotational rigid-body modes ($x \times e_i$) alongside the 3 translations:
$$B = \begin{bmatrix} 1 & 0 & 0 & 0 & z & -y \\ 0 & 1 & 0 & -z & 0 & x \\ 0 & 0 & 1 & y & -x & 0 \end{bmatrix}$$
Smoothed Aggregation AMG depends critically on exactly representing the near-nullspace of the operator on all coarse grids. For bending-dominated domains (like pry-out anchors or cantilever structures), lacking rotational modes severely degrades the spectral radius of the iteration matrix, causing convergence stalls. Supplying all 6 modes will drastically cut outer iterations.

### Recommendation C: Native C++ P-Multigrid ($P^T A P$) Projection
Port the serendipity Galerkin projection $A_1 = P^T A_2 P$ and the fine Chebyshev smoother from `ptwogrid.py` entirely into C++ using `SpGEMMHelperT` and `RelaxationSmootherT`. Furthermore, implement a rigorous weak-connection dropping strategy on $A_1$ (e.g., pruning entries where $|A_{ij}| < \epsilon \sqrt{A_{ii} A_{jj}}$) to convert the established outer iteration count reductions into true wall-clock speedups, bypassing the memory bandwidth bottleneck of the densified coarse grid.
