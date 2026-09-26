# NID mass/damping assembly memory (2026-09-26, xeon, 1 h budget)

Branch `fix/nid-mass-assembly-memory` (EdelweissFE, off origin/next_v26.11 f38b05de), pushed to `mn`.
Scratch data on xeon: `~/nidmem/` (prof.py instrumentation, base50/base50b/fix50 runs, tf_* NID refs, asym.py).
Caveat: xeon's libMarmot is a Sep-15 build of Marmot `feat/elements-bulk-viscosity` (eea4c784), NOT origin/next_v26.11
(c591c8dc); the Marmot checkout has staged WIP, so it was not switched/rebuilt. All comparisons are relative (same lib).

## Measured (c1_50 s2s deck, 3 increments, OMP=16, MKL_CBWR=AUTO,STRICT)
Sizes after initial AMR: nDof 193,486; VIJ length 71.09 M (int32 I/J); nnz(M) = nnz(C) = 40.9 M; C has 0 nonzeros
(GCDP damping sits on the non-local field, which NID zeroes). One float64 VIJ vector = 0.53 GiB, CSR(M) = 0.46 GiB.

| stage | baseline | fix |
|---|---|---|
| `_assembleMassAndDamping` wall | 10.2 s | 9.4 s |
| its RSS peak above entry | +3.19 GiB | +2.77 GiB |
| its tracemalloc peak | +4.09 GiB | +2.75 GiB (-1.34 GiB = M.T/M-M.T + VIJ isfinite temporaries) |
| retained after (Mvij, Cvij, M, C) | 2.65 GiB | 2.65 GiB |
| `initializeIncrement` 1st (incl. initial-acceleration solve) | 23.8 s, +4.77 GiB RSS | 24.4 s, +4.77 GiB |
| `initializeIncrement` later (dynamicStiffness) | +1.06 GiB peak, +0.53 retained | same (2 VIJ vectors alive at once: result + scaled Cvij temp) |

Scaled to c1_100 (failed allocation 199.8 M float64 = the M-M.T data array): the removed global symmetry check was
~ 3x that array transient (M.T view is free, M-M.T data+indices, abs temp) i.e. ~4 GiB at the crash point.

Not measured (budget): c1_100 deck on xeon to its first refinement; NIST comparison; blockamg's own copies;
per-array breakdown beyond the above; reuse-path frequency on c1_100 (reuse path is only taken for
constraint-connectivity-only rebuilds; every AMR refinement does a full reassembly by design).

## Symmetry check verdict
Claim 1 is half right. No solver relies on symmetry (blockamg `symmetric` is the GS sweep direction; K_eff is
non-symmetric anyway). But a non-symmetric mass is NOT harmless: injecting +/-5% antisymmetric off-diagonal mass
into NIDParallel (asym.py) converges (2 -> 8 Newton iterations) and silently gives a wrong answer
(tipU at end -1.1e-4 vs 1.36e-6). So the check is kept, moved to the element loop on each element's own
nDof x nDof block: same failure detected, zero global memory. (Per-element firing not yet verified by injection:
Marmot elements are cdef, needs a fake element in tests/test_nid_*.py -- OPEN.)

## Implemented (commit on branch)
1. Global `M - M.T` check -> per-element block check.
2. `isfinite` on summed CSR data instead of the two VIJ vectors.
3. `dynamicStiffness` formed with in-place multiply/add (bitwise identical ops).
Retained: finite, damping >= 0, massless-dynamic-dof, per-field total-mass journal.

## Ranked open fixes (expected savings at c1_50 sizes; x~2.8 for c1_100)
1. Damping as a dof-vector (diagonal by construction): drop Cvij (0.53 GiB) + C CSR (0.46 GiB, all explicit zeros here)
   + the dynamicStiffness temp (0.53 GiB transient). Put per-element Ce on its VIJ diagonal slots (few M entries).
   Touches `_NewmarkSystem`, `_reuseMassAndDamping`, `_warnAboutDiscardedInertia` (+ test_nid_gradient_enhanced).
2. `_computeInitialAcceleration` (+4.77 GiB, re-armed after every refinement): builds a full K VIJ + CSR MEff only
   to throw K away -- investigate.
3. Bool temporaries (isDynamic[I] & isDynamic[J], ~couplesDynamicOnly twice, discarded mask): ~5 x 71 MB, minor.

## Bitwise
- testfiles/marmot NID, NIDParallel, NIDLiveAMR: fix == baseline U bitwise (OMP=4, MKL_CBWR=AUTO,STRICT).
- c1_50, 3 increments: fix differs from baseline at 4e-14 rel in RF, BUT two baseline runs differ from each
  other equally (base50 vs base50b) -> the deck is not run-to-run reproducible at OMP=16 (threaded assembly /
  blockamg); no bitwise verdict possible there. U_loading, maxDamage identical.
- tests/test_nid_*.py: 28 pass, 1 fail (test_a_reduced_integration_element_starts_under_load) -- fails identically on
  baseline (stale libMarmot, likely needs #174's Marmot side).
