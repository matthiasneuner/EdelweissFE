Linear solvers
==============

Linear solvers are defined in EdelweissFE after the ``*solver`` keyword using ``linsolver`` and an optional configuration file ``linsolverConfigFile`` as a data line.
The ``linsolverConfigFile`` needs to be in ``.json`` format.

Choose a linsolver after the ``*solver`` keyword:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=pardiso

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
    * - ``amgcl``
      - ✗
      - ``edelweissfe.linsolve.amgcl.amgcl``
    * - ``blockamg``
      - ✗
      - ``edelweissfe.linsolve.blockamg.blockamg``
    * - ``barschur``
      - ✗
      - ``edelweissfe.linsolve.barschur.barschur``
    * - ``matrixdump``
      - —
      - ``edelweissfe.linsolve.matrixdump.matrixdump``

How the available solvers relate
--------------------------------

Every entry in the table above is selected the same way (``linsolver=<name>``) and presents the same
``(A, b) -> x`` interface, but they fall into three groups that differ in what they actually do:

**Direct solvers** factorize the matrix and back-substitute: ``superlu`` and ``umfpack`` (SciPy,
always available), ``pardiso`` (Intel MKL, the production default), ``panuapardiso``, ``klu``,
``petsclu`` and ``mumps``. They ignore ``linsolverConfigFile`` (except ``pardiso``, which reads a
single ``reuseSymbolicFactorization`` flag). Pick one of these unless the factorization has become
the run's bottleneck or exceeds available memory.

**Iterative solvers** never factorize the full matrix, trading exactness for O(n) memory:

* ``amgcl`` -- the AMGCL library's own solver/preconditioner combinations, configured through its
  JSON parameter tree.
* ``blockamg`` -- a *field-split* variant for coupled multi-field models: one AMG hierarchy per
  physical field (e.g. displacement, nonlocal damage), combined by a block Gauss--Seidel sweep to
  precondition an outer GMRES/LGMRES. This is the route to problem sizes a direct factorization
  cannot reach at all.
* ``barschur`` -- for embedded bars with their own dofs (bond-slip, ``embeddedBond``), on which
  ``blockamg`` alone stalls: the bar dofs are condensed exactly, the other dofs are solved by an inner
  ``blockamg`` on an approximate Schur complement, within an outer flexible GMRES.

**A diagnostic wrapper**, which does not solve anything itself:

* ``matrixdump`` -- writes every :math:`(A, b)` pair it is handed to disk, then hands the solve to
  a real solver chosen with its ``delegate`` option, so the simulation proceeds normally. The point
  is to lift solver comparisons out of the finite element run: rerunning a simulation per solver
  variant is slow and not like-for-like (a variant that changes the Newton iterates changes the
  *sequence* of matrices being compared). Dump one authentic sequence once, then replay it offline
  with ``scripts/benchmark_linsolve.py`` so every variant sees byte-identical input. Use it to
  investigate solver performance, never in a production run.

Several linsolvers accept an optional configuration file ``linsolverConfigFile`` (a ``.json`` file), among them ``amgcl``, ``blockamg`` and ``matrixdump``; the direct solvers ignore it (``pardiso`` additionally reads a single ``reuseSymbolicFactorization`` flag). See each solver's own section below for its configuration format.


The ``blockamg`` solver
-----------------------

``blockamg`` is a field-split block-AMG solver for **large coupled multi-field systems** (e.g. displacement + gradient-enhanced damage). It is the O(n)-memory route to problem sizes a direct factorization cannot reach -- past roughly a million DOFs its fill-in exceeds memory, whereas algebraic multigrid stays linear.

Applied *monolithically*, AMG is ineffective on such a coupled system (a single hierarchy cannot represent the disparate fields at once -- their physical scales and near-null-spaces differ). ``blockamg`` instead builds **one AMGCL algebraic-multigrid hierarchy per field** and combines them with a **block Gauss--Seidel** sweep to precondition an outer Krylov solve. Per solve it equilibrates the system, splits it into field blocks, builds or reuses the per-field hierarchies, and preconditions the outer solve with the block sweep.

The block structure -- which DOFs belong to which field, and each field's dimension -- is **discovered automatically** from the ``DofManager`` and the live ``FEModel``, both handed to the solver via ``setModel()`` whenever the equation system is (re)built. A ``linsolverConfigFile`` is therefore optional and carries only solver knobs. Requires the optional ``amgcl`` extension.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "outerSolver": "amgcl_lgmres",
        "sweeps": 1,
        "symmetric": false
    }

The method, the near-null-space construction, the adaptive outer tolerance, hierarchy reuse, every configuration key with its default, and the experimental p-multigrid variant are documented in detail on their own page:

.. toctree::
   :maxdepth: 1

   blockamgtheory


The ``barschur`` solver
-----------------------

``barschur`` solves systems with **embedded bars that carry their own dofs**, i.e. bond-slip between reinforcement
and concrete (the ``embeddedBond`` generator). The rows of the bar dofs are dominated by the bond springs to the nodes
of their host elements, :math:`u_{bar} \approx \sum_i N_i u_i`, a relation that an aggregation AMG coarse space cannot
represent: ``blockamg`` alone stalls on such systems from the first load step. With perfect bond (``embeddedRegion``)
the same relation is a multi-point constraint, eliminated before the solve, and ``blockamg`` is the right choice.

``barschur`` treats the bar dofs :math:`b` (all dofs of the nodes that carry only 1D elements: the bars and the bond
elements) and all other dofs :math:`k` as the two blocks of a block LU factorization:

.. math::

    y_b = A_{bb}^{-1} r_b, \quad z_k = \tilde{S}^{-1} (r_k - A_{kb} y_b), \quad z_b = A_{bb}^{-1} (r_b - A_{bk} z_k).

The bar block :math:`A_{bb}` decouples into one chain per bar and is factorized exactly. The Schur complement of the
other dofs, :math:`S = A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}`, is dense along each bar, but its entries decay within a
few host elements (a bar on an elastic foundation): ``schurApproximation = "local"`` (the default) computes it in
overlapping windows along the bars and drops the small entries. :math:`\tilde{S}^{-1}` is ``innerCycles`` (default 2)
applications of the preconditioner of an inner ``blockamg`` (one block Gauss-Seidel sweep of AMG cycles over the
fields), as a stationary iteration on :math:`\tilde{S}`; the outer solve is a flexible GMRES on the full system and
does the acceleration. With ``"innerCycles": null``, :math:`\tilde{S}^{-1}` is instead an inner ``blockamg`` solve to
``innerTolerance``. On a hex20 tunnel segment with bond-slip (2.07 M dofs), two cycles took a third less time per solve
than inner solves to 0.3 (51 outer iterations instead of 32, but 102 AMG applications instead of 164, and no inner
Krylov iterations or per-solve setup of the inner solver); one cycle needed three times the outer iterations, three
cycles were slower than two. Through cracking, the run was about four times faster than with inner solves. The
inner hierarchies of 3D vector fields use 50 instead of 300 power iterations for the Chebyshev smoother's spectral
radius (unless the ``blockamg`` options give the field its own tree): they are rebuilt with every approximate Schur
complement, and 50 gave the same outer iterations at a third of the build time. The approximate Schur complement and
the inner hierarchies are reused for the following solves while the sparsity pattern stays the same and the outer
iterations do not drift; a solve that fails with a reused one is repeated with a rebuilt one.

.. code-block:: edelweiss

    *solver, solver=NISTParallel, name=theSolver
    linsolver=barschur
    linsolverConfigFile=barschur.json

.. code-block:: json

    {
        "schurApproximation": "local",
        "schurWindow": 300,
        "blockamg": {}
    }

Configuration keys (defaults in parentheses): ``outerTolerance`` (1e-3, relative residual of the outer solve),
``innerCycles`` (2; ``null``: inner solves to ``innerTolerance``), ``innerTolerance`` (0.1), ``restart`` (50; much
shorter restarts cost many more outer iterations), ``maxIterations`` (1000), ``schurApproximation`` (``"local"``; ``"nodal"``
uses the diagonal block of each bar node only, which suffices for trusses but not for beams; ``"none"`` uses
:math:`A_{kk}`), ``schurDropTolerance`` (1e-4), ``schurWindow`` (600 bar dofs per window core), ``schurHalo`` (bar dofs per halo on
either side of a core; default ``schurWindow``), ``reuseSchurComplement`` (true), ``schurStalenessFactor`` (1.5),
``dumpSlowSolvesDir`` / ``dumpSlowSolveSeconds`` / ``dumpSlowSolvesMax`` (write the systems of slow solves for offline
diagnosis; off / 300 / 5) and ``blockamg`` (the options of the inner solver; with inner solves, an inner iteration
cap, e.g. ``"outerRestart": 20``, bounds the cost of inner solves that converge poorly).
Requires the optional ``amgcl`` extension.

Very short bar elements make the equation system nearly singular for any solver (a beam of length :math:`L` adds the
bending stiffness :math:`12 EI / L^3`); with ``splitBars``, set the ``embeddedBond`` option ``minimumPartLength`` to
avoid them.


The ``amgcl`` solver
--------------------

``amgcl`` is an iterative solver (Krylov method plus algebraic-multigrid or single-level preconditioner) built on the `AMGCL <https://github.com/ddemidov/amgcl>`_ library. Its ``linsolverConfigFile`` is forwarded as an AMGCL parameter tree; note that AMGCL silently ignores unknown parameter keys (warning only on stderr), so check its stderr if a configuration behaves unexpectedly.


The ``matrixdump`` diagnostic solver
------------------------------------

``matrixdump`` is not a solver but a diagnostic wrapper: it writes the equation systems it is handed to disk and then delegates the actual solve to a real linear solver, so a sequence of authentic ``(A, b)`` pairs can be replayed offline instead of by rerunning the simulation. Its ``linsolverConfigFile`` selects the ``delegate`` solver, the dump ``directory``, and which solves to capture (``dumpAt`` / ``skipFirst`` / ``maxDumps`` / ``instances``).

Its instance/dump-ordinal counters are process-wide and not part of a ``*restart`` checkpoint, so a resumed run starts them back at zero -- point a resumed analysis using ``matrixdump`` at a fresh ``directory`` if you need both runs' dumps kept, rather than the resumed run silently overwriting the interrupted one's.
