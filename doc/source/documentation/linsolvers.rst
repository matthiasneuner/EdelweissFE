Linear solvers
==============

Linear solvers are defined in EdelweissFE after the ``*solver`` keyword using ``linsolver`` and an optional configuration file ``linsolverConfigFile`` as a data line.
The ``linsolverConfigFile`` needs to be in ``.json`` format.

Choose a linsolver after the ``*solver`` keyword:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=gmres
    linsolverConfigFile=opt.json

.. list-table:: Currently available linear solvers
    :width: 100%
    :widths: 15 1 25
    :header-rows: 1

    * - Name
      - Direct solver
      - Relevant module
    * - ``superlu``
      - ✓
      - ``scipy.sparse.linalg.spsolve``
    * - ``umfpack``
      - ✓
      - ``scipy.sparse.linalg.spsolve``
    * - ``pardiso``
      - ✓
      - ``edelweissfe.linsolve.pardiso.pardiso``
    * - ``panuapardiso``
      - ✓
      - ``edelweissfe.linsolve.panuapardiso.panuapardiso``
    * - ``klu``
      - ✓
      - ``edelweissfe.linsolve.klu.klu``
    * - ``petsclu``
      - ✓
      - ``edelweissfe.linsolve.petsclu.petsclu``
    * - ``mumps``
      - ✓
      - ``edelweissfe.linsolve.mumps.mumps``
    * - ``gmres``
      - ✗
      - ``edelweissfe.linsolve.gmres.gmres``
    * - ``amgcl``
      - ✗
      - ``edelweissfe.linsolve.amgcl.amgcl``
    * - ``inexactnewton``
      - ✗
      - ``edelweissfe.linsolve.inexactnewton.inexactnewton``
    * - ``blockamg``
      - ✗
      - ``edelweissfe.linsolve.blockamg.blockamg``
    * - ``matrixdump``
      - —
      - ``edelweissfe.linsolve.matrixdump.matrixdump``

Several linsolvers accept an optional configuration file ``linsolverConfigFile`` (a ``.json`` file), among them ``gmres``, ``amgcl``, ``inexactnewton`` and ``matrixdump``; the direct solvers ignore it (``pardiso`` additionally reads a single ``reuseSymbolicFactorization`` flag).

Choose the options for the linsolver (in this case ``gmres``) in an extra file:

.. code-block:: json

    	{
	"precondopts":
	{
	"presmoother": ["block_gauss_seidel", {"iterations": 15}],
	"postsmoother": ["block_gauss_seidel", {"iterations": 15}],
	},
	"linsolveopts": {"maxiter": 1, "restart": 1500}
	}


The ``inexactnewton`` solver
----------------------------

``inexactnewton`` is not a solver in its own right but a *modified-Newton–Krylov* scheme intended for large, coupled nonlinear models (for example penalty contact combined with adaptive mesh refinement and gradient-enhanced damage) where a direct factorization dominates the run time while the Jacobian changes only slightly from one Newton iteration to the next.

Instead of factorizing the system matrix on every Newton iteration, it keeps an **exact LU factorization of one iterate** (computed by a *delegate* direct solver, ``pardiso`` by default) and reuses it as a **preconditioner for GMRES** on the next few iterates. The linear tolerance follows an **Eisenstat–Walker forcing sequence** rather than being solved tightly: a Newton correction does not need the linear system solved to machine precision, so the reuse solves converge in only a handful of GMRES iterations. When the factorization goes stale it is refreshed automatically; the first solve of an increment and its large first correction — the iterates that precondition worst — are kept direct.

Because it exposes the ordinary ``(A, b) -> x`` interface, selecting it requires no other change to the analysis:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=inexactnewton
    linsolverConfigFile=inexactnewton.json

All configuration keys are optional; the defaults are a turnkey configuration (the PARDISO delegate with the measured sweet-spot policy). The recognised keys:

.. list-table:: ``inexactnewton`` configuration keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``delegate``
      - ``"pardiso"``
      - Factorizing backend supplying the lagged LU. ``"pardiso"`` in production, or ``"superlu"`` (SciPy, dependency-free) for testing or installs without the PARDISO extension.
    * - ``maxReuse``
      - ``8``
      - How many consecutive reuse solves one factorization may serve before it is refreshed.
    * - ``residualGrowthFactor``
      - ``4.0``
      - A solve whose ``||b||`` exceeds this multiple of the previous one is treated as a new increment (or a cutback) and refactorized.
    * - ``etaMin`` / ``etaMax``
      - ``1e-6`` / ``1e-3``
      - Clamp on the Eisenstat–Walker forcing tolerance (tightest / loosest a reuse solve may use).
    * - ``ewGamma`` / ``ewAlpha``
      - ``0.9`` / ``1.618…``
      - Eisenstat–Walker "choice 2" parameters, ``eta_k = ewGamma * (||b_k|| / ||b_{k-1}||) ** ewAlpha``.
    * - ``gmresRestart`` / ``gmresMaxOuter``
      - ``25`` / ``1``
      - GMRES Krylov dimension between restarts and maximum restart cycles; their product caps the iterations before a reuse falls back to a direct solve (default cap 25, just above the ~22-iteration break-even of the reference condensed system).
    * - ``staleIterationThreshold``
      - ``20``
      - A reuse converging in more iterations than this marks the region as hardening: refresh next iterate and grow the probe backoff. Set a little below the break-even.
    * - ``cheapIterationThreshold``
      - ``10``
      - A reuse converging within this many iterations marks the region as easy and resets the probe backoff.
    * - ``maxProbeBackoff``
      - ``8``
      - Ceiling on the direct-solve run inserted between reuse probes in a persistently hard region.
    * - ``verbose``
      - ``false``
      - Print one line per solve (refactorize?, forcing tolerance, iteration count).

Example configuration selecting the SuperLU delegate for a dependency-free run:

.. code-block:: json

    {
        "delegate": "superlu",
        "etaMax": 1e-3,
        "maxReuse": 8
    }


The ``blockamg`` solver
-----------------------

``blockamg`` is a field-split block-AMG solver for **large coupled multi-field systems** (e.g. displacement + gradient-enhanced damage). It is the O(n)-memory route to problem sizes a direct factorization cannot reach — past roughly a million DOFs its fill-in exceeds memory, whereas algebraic multigrid stays linear.

Applied *monolithically*, AMG is ineffective on such a coupled system (a single hierarchy cannot represent the disparate fields at once — their physical scales and near-null-spaces differ). ``blockamg`` instead builds **one AMGCL algebraic-multigrid hierarchy per field** and combines them with a **block Gauss–Seidel** sweep to precondition an outer Krylov solve, following Alkmim et al. (IJNME 2026). Per solve it equilibrates the system (symmetric diagonal scaling, to tame the large dynamic range coupled multi-field systems typically have), splits it into field blocks, and preconditions the outer solve with the block sweep.

The block structure — which DOFs belong to which field, and each field's dimension — is **discovered automatically** from the ``DofManager`` and the live ``FEModel``, both handed to the solver by the nonlinear solver via ``setModel()`` whenever the equation system is (re)built (the first solve, and again after any adaptive-mesh-refinement or connectivity change). Nothing about the block layout, node coordinates, or mesh topology needed below is specified by hand. A ``linsolverConfigFile`` is therefore optional and carries only solver knobs. Requires the optional ``amgcl`` extension.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "outerSolver": "amgcl_lgmres",
        "sweeps": 1,
        "symmetric": true
    }

Near-null-space and the Chebyshev smoother
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each field's AMG hierarchy uses **smoothed aggregation**, coarsened with a **Chebyshev polynomial smoother**. Both pieces need to be understood together, because they play deliberately complementary roles:

- The smoother damps error components whose eigenvalues lie in an estimated window ``[lower, higher] * rho`` of the operator's spectral radius ``rho`` (by default ``lower = 0.01``, i.e. eigenvalues below 1% of the spectral radius are, by design, **not** targeted by the smoother at all).
- Everything below that window — including the operator's *near-null-space*, i.e. directions in which the operator has almost no stiffness — is left for the coarse-grid correction to remove, via smoothed aggregation's own null-space-aware construction of the prolongation operator.

For 3D linear elasticity, the physical near-null-space is the 6-dimensional space of **rigid-body motions**: 3 translations plus 3 infinitesimal rotations, each of which costs zero elastic strain energy and therefore corresponds to a genuinely (near-)zero eigenvalue of the discretized stiffness operator. Standard AMG practice for elasticity is to hand smoothed aggregation the *full* 6-mode basis (translations **and** rotations) as the near-null-space, computed from each vector field's node coordinates about its own coordinate centroid (translations: unit displacement per component; rotations: the classic infinitesimal-rigid-rotation fields, e.g. about the z-axis, ``(-y, x, 0)``). A scalar field (e.g. ``nonlocal damage``) has no rotational rigid-body mode; its near-null-space is just the constant field.

Giving smoothed aggregation only the translations (a common simplification, since building the rotational modes needs node coordinates rather than just DOF-block structure) leaves the coarse grid unable to represent rotational rigid-body error exactly — and because the Chebyshev smoother is, by the construction above, *not* targeting that error class either, it has nowhere efficient to go, and outer-iteration counts suffer measurably (on the order of 30% more outer iterations on real coupled systems, isolated per field). The two AMG components genuinely need to agree on what "smooth error" means; a near-null-space that is missing part of the operator's true kernel is a gap neither component covers.

The Chebyshev smoother's own configuration has a second, independent subtlety: it estimates the operator's spectral radius via a fixed number of power-iteration steps rather than an exact eigenvalue computation. A short power-iteration budget can converge to a materially different (and sometimes badly under-converged) estimate depending on how many parallel worker threads are used to build the hierarchy — because a parallel implementation typically partitions the power-iteration's random starting vector across worker threads and seeds each thread's random-number generator independently, so the number of independent streams (and hence the effective starting vector) changes with thread count even though the true operator, and its true spectral radius, does not. A badly under-converged estimate degrades the smoother's own effectiveness and can defeat an otherwise-healthy near-null-space just as thoroughly as an incomplete one. Running the power iteration to convergence (i.e. increasing its iteration budget well past AMGCL's own low default) removes the thread-count sensitivity and is measurably worth the modest extra one-time hierarchy-build cost.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

Recognised keys, all optional:

.. list-table:: ``blockamg`` configuration keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``outerSolver``
      - ``"amgcl_lgmres"``
      - The outer Krylov solve. ``"amgcl_lgmres"`` uses AMGCL's own native, OpenMP-threaded Loose GMRES implementation, which scales better across NUMA nodes than orchestrating the same iteration from Python; ``"scipy"`` uses SciPy's ``gmres`` instead, kept as a fallback.
    * - ``sweeps`` / ``symmetric``
      - ``1`` / ``true``
      - Number of block Gauss–Seidel sweeps per preconditioner application, and whether the sweep pattern is made symmetric (forward then backward), which keeps the preconditioner compatible with a symmetric outer Krylov method.
    * - ``useRigidBodyNullspace``
      - ``true``
      - Use the full 6-mode rigid-body near-null-space (translations and rotations, see above) for every vector field whose node coordinates are available; automatically falls back to translations-only for a field whose coordinates cannot be determined (e.g. a solver driven directly through the low-level ``setFieldStructure`` API instead of ``setModel``). Set to ``false`` to force translations-only unconditionally.
    * - ``fieldPreconds``
      - ``{}``
      - A mapping of field name (e.g. ``"displacement"``) to an AMGCL parameter tree overriding the dimension-based default for that field, including its own ``relax.power_iters``/``relax.lower``/``relax.higher`` Chebyshev settings.
    * - ``p1FieldNames``
      - ``[]``
      - **Experimental, opt-in only, not recommended as a default** — see "p-multigrid" below.
    * - ``dumpOnDegradationDir`` / ``dumpOnDegradationThreshold`` / ``dumpOnDegradationMaxDumps`` / ``dumpOnDegradationContextSolves``
      - ``None`` (off)
      - Diagnostic capture: when set, write the raw ``(A, b)`` system, its field-block layout, and (optionally) a rolling window of the preceding solves' own state, to disk for any solve whose outer-iteration count exceeds ``dumpOnDegradationThreshold`` — up to a process-wide cap of ``dumpOnDegradationMaxDumps`` — so a pathological live-run system can be captured for offline diagnosis without knowing in advance which solve will misbehave. Off by default; negligible cost when unused.
    * - ``etaMin`` / ``etaMax`` / ``ewGamma`` / ``ewAlpha``
      - see the ``inexactnewton`` table above
      - The same Eisenstat–Walker forcing-tolerance scheme, applied to the outer Krylov solve's own stopping tolerance.
    * - ``verbosity``
      - ``"warning"``
      - Log level (``"debug"``/``"info"``/``"warning"``/``"error"``) for this solver's own diagnostic output.

.. note::
   ``blockamg`` is the O(n)-memory route to the 1M+-DOF regime a direct factorization cannot reach, not necessarily the fastest solver at moderate problem sizes — a direct factorization (e.g. ``pardiso``) can still be competitive, or faster, on systems that comfortably fit its fill-in. Which is faster depends on problem size, conditioning, and how severely damage/contact nonlinearity degrades the per-field AMG hierarchies' convergence on a given increment.

p-multigrid (experimental, opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a field discretized with quadratic (serendipity) elements, ``blockamg`` can optionally replace a field's single-level AMG hierarchy with a genuine **two-grid V-cycle**: a coarse level built purely from the mesh's corner nodes (a topological ``P1`` restriction/prolongation, requiring no re-discretization — corner values pass through unchanged, midside values are the average of their two edge-endpoint corners), with its own small AMGCL solve, sandwiched between Chebyshev pre- and post-smoothing sweeps on the full quadratic mesh.

The underlying two-grid algorithm is sound and does reduce the outer-iteration count on real coupled systems. However, it has two structural costs the single-level default does not pay, which can outweigh the iteration-count win depending on how AMG-friendly the operator otherwise is: a fixed per-solve setup cost (projecting the operator onto the coarse corner-node space via a sparse matrix triple product, plus building the coarse hierarchy itself), and a near-null-space handling gap — the coarse solve can be given the same rigid-body near-null-space treatment described above (restricted to the corner-node subset), but the fine-level Chebyshev smoother, unlike a full recursive AMG hierarchy, has no coarsening step of its own and is not given any near-null-space information at all. On at least one real reference model this made the two-grid variant measurably slower overall than the single-level default, on the range of operators tested (well short of severe convergence degradation). Whether it is worthwhile depends heavily on how badly a given field's single-level AMG hierarchy is already struggling — it is not recommended as a default, and is offered as an opt-in tool for a user who has independently confirmed it helps their own model.

Enable it per field via ``p1FieldNames`` (a list of field names to enable it for; the field's mesh must consist entirely of quadratic serendipity elements, or the solver falls back to the single-level default for that field with a warning).


The ``amgcl`` solver
--------------------

``amgcl`` is an iterative solver (Krylov method plus algebraic-multigrid or single-level preconditioner) built on the `AMGCL <https://github.com/ddemidov/amgcl>`_ library. Its ``linsolverConfigFile`` is forwarded as an AMGCL parameter tree; note that AMGCL silently ignores unknown parameter keys (warning only on stderr), so check its stderr if a configuration behaves unexpectedly.


The ``matrixdump`` diagnostic solver
------------------------------------

``matrixdump`` is not a solver but a diagnostic wrapper: it writes the equation systems it is handed to disk and then delegates the actual solve to a real linear solver, so a sequence of authentic ``(A, b)`` pairs can be replayed offline instead of by rerunning the simulation. Its ``linsolverConfigFile`` selects the ``delegate`` solver, the dump ``directory``, and which solves to capture (``dumpAt`` / ``skipFirst`` / ``maxDumps`` / ``instances``).

Its instance/dump-ordinal counters are process-wide and not part of a ``*restart`` checkpoint, so a resumed run starts them back at zero -- point a resumed analysis using ``matrixdump`` at a fresh ``directory`` if you need both runs' dumps kept, rather than the resumed run silently overwriting the interrupted one's.
