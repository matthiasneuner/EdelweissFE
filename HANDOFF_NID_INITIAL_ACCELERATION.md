# NID initial acceleration: reduced, converged mass solve (2026-09-27/28, xeon)

Branch `fix/nid-initial-acceleration` (off origin/next_v26.11 8e63e8e9), fix commit **e42849c9** (reworded from ce0d8e89), pushed to `mn`, no PR.
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
- The first version of the fix commit claimed "NID/NIDParallel 1.4e-17, NIDLiveAMR 1.6e-13". That came from a base copy whose `csrgeneratorv2` extension was stale (built from the pre-8e63e8e9 source); the message is corrected in e42849c9.

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

## Do the NID testfiles reach the new solve? (2026-09-28 check)
Probe: `~/nidmem/accprobe.py`. On each side it prints `edelweissfe.__file__` and the loaded `csrgeneratorv2` .so:
`~/nidmem/fe_accbase/...` for base, `EdelweissFE-nidacc/...` for the branch, with identical compiled extensions.

| testfile | calls of `_computeInitialAcceleration` | a0 | base solver | branch solver |
|---|---|---|---|---|
| NID, NIDParallel | 1, at t = 0 | max\|a0\| 0.3948 (nonzero) | PARDISO | CG, 4 free dofs, 1 iteration, relres 9.9e-17 |
| NIDLiveAMR | 2: at t = 0, and after the refinement at t = 0.1 | t = 0: exactly 0 (at rest, no load). t = 0.1: max\|a0\| 0.72 | PARDISO | CG, 487 free dofs, 55 iterations, relres 1.7e-15 |

- The new path is reached with a nonzero a0 in all three testfiles, and NIDLiveAMR's refinement solve does run.
- a0 differs from base at rounding level only: NID 1.7e-16 (1 ulp), NIDLiveAMR 4.1e-13 of 0.72. The old path was a direct PARDISO solve on these small systems, so it was accurate there.
- **Why U.ref is still bitwise identical:** the stored U is the final displacement, and in these decks inertia is tiny against stiffness. NIDLiveAMR is displacement-driven, with ρ/E = 1e-7. A 4e-13 change in a0 moves the equilibrium displacement by about (ρ/E)·δa ~ 1e-20, below one ulp of U (about 4e-19).
- **New testfile `NIDInitialAccelerationAMR`:** two C3D20R elements, the left one refined at the start (13 hanging-node slave dofs), a sudden end load, and ρ ~ E. Here a0 enters U visibly.
  - a0: max 259; branch vs base differs by 6.8e-13 (2.6e-15 relative). CG took 19 iterations on 59 free dofs.
  - U: branch vs base differs by 2.8e-16 of 0.126, in 69 of 279 entries. That is CG rounding: the old PARDISO solve was accurate here too.
  - Its U.ref was created with the branch.
- **Where the old solve was actually wrong:** only with an iterative tangent solver (blockamg on the edge-breakout decks), as tabulated above. No small testfile reproduces that without blockamg.
- The fix commit message was reworded (now **e42849c9**; the tree is unchanged, force-pushed): the earlier ΔU figures came from a stale-extension comparison.
