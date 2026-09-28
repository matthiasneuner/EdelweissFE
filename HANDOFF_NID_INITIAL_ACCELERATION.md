# NID initial acceleration: reduced, converged mass solve (2026-09-27/28, xeon)

Branch `fix/nid-initial-acceleration` (off origin/next_v26.11 8e63e8e9), fix commit **ce0d8e89**, pushed to `mn`, no PR.
It is independent of `fix/nid-mass-assembly-memory` / PR #177, whose HANDOFF (Phase 5) found the bug.

## Bug
`_computeInitialAcceleration` solved the full system `M + I(static)`, with Dirichlet row replacement, using the
step's linear solver. That operator is:
- non-symmetric (asymmetry 0.75);
- badly scaled: unit rows next to mass entries of 1e-6;
- indefinite in its symmetric part.

blockamg, configured for the tangent, stopped at a relative residual of 1e-1 to 1e0 on the edge-breakout decks.

## Fix
- Solve `M_ff a_f = R_f` on the free dynamic dofs only:
  - `a_p = 0` on the Dirichlet dofs;
  - no equation for the quasi-static fields;
  - MPCs condensed with the Newton solves' own `transformSystemMatrix` / `transformResidual`; slaves are excluded and then get `applySlaveKinematics`.
- Solver: Jacobi-preconditioned CG to a relative residual of 1e-14, checked on the true residual; raises above 1e-13.
- Docs: a module-docstring paragraph plus method docstrings (Sphinx automodule).
- Tests: `tests/test_nid_initial_acceleration.py`.
  - A GC3D20R bar with a quasi-static AT2 field, lateral Dirichlet conditions and hanging nodes; a = b/ρ at every node, slaves included, and the residual at rounding level.
  - The non-convergence error fires.

## CG iterations (journal, level 2)

| case | free dynamic dofs | CG iterations | relative residual |
|---|---|---|---|
| c1_50, step start | 135,066 | 191 | 1.3e-14 |
| c1_50, after the live refinement | 135,990 | 199 | 9.6e-15 |
| NIDLiveAMR, after refinement | 487 | 55 | 4.8e-15 |

About 14 iterations per decade, and no growth with refinement.

## testfiles/marmot suite, branch vs origin/next_v26.11
Setup: same compiled extensions, OMP=8, MKL_CBWR=AUTO,STRICT, `--create` into copies.
- **All 144 cases that produce a U.ref are bitwise identical, including NID, NIDParallel and NIDLiveAMR. No U.ref was regenerated.**
- 11 cases produce no U.ref on either side. 2 fail on both sides on matplotlib usetex (known on xeon): IndirectDisplacementControl, OutputManagers.
- Correction to the ce0d8e89 commit message: its "NID/NIDParallel 1.4e-17, NIDLiveAMR 1.6e-13" came from a comparison against a base copy whose `csrgeneratorv2` extension was stale (built from the pre-8e63e8e9 source). With identical extensions the three NID testfiles are bitwise identical.

## A/B on c1_50: first live refinement at increment 11, then increments 11-13 (runs `~/nidmem/aba_*`)

| variant | KE, inc 11 | max\|A\|, inc 11 | RF at t = 0.055 / 0.060 / 0.065 | Newton iterations, incs 11 / 12 / 13 |
|---|---|---|---|---|
| consistent, converged (new default) | 1.1075 | 5.2e5 | 1464.0 / 1774.1 / 2123.4 | 8 / 9 / 10 |
| (a) keep interpolated | 1.1060 (-0.13 %) | 147 | 1474.9 (+0.74 %) / 1771.0 (-0.17 %) / 2124.3 (+0.04 %) | 7 / 9 / 8 |
| (b) lumped (HRZ) | 1.652 (+49 %) | 2.4e5 | 1246.5 (-15 %) / 1672.7 (-5.7 %) / 2208.1 (+4.0 %) | 7 / 7 / 7 |

Before the fix, the unconverged consistent solve gave: KE 1.014, max|A| 7.2e3, RF 1565.6 at t = 0.055 (+6.9 % off the converged value).

## Verdict
- **Lumped (b):** rejected.
- **Interpolated (a):** close. RF is within 0.74 %, the gap decays to 0.04 % two increments later, and it takes 1-2 fewer Newton iterations per increment.
  - But it is not identical, and its acceleration field is not in equilibrium: max|A| is 147 against 5.2e5 for the converged solve.
  - The large converged acceleration is real. It is the out-of-balance force of the refinement's state transfer acting on small masses, not a solver artefact.
- **Decision:** the default stays `computeInitialAcceleration=True`, the consistent solve. The difference is not "clearly equivalent" at the first increment. The cheaper option stays available (`computeInitialAcceleration=False`).
- **Open:**
  - The acceleration spike after refinement belongs to the AMR state transfer (see the state-transfer PRs), not to NID.
  - The +6.9 % RF error of the old unconverged solve means earlier NID+AMR results on blockamg decks carry that error for a few increments after each refinement.
