# Plan: correctness round R2 (2026-10-15)

Source: REVIEW_MPI_2026-10-15_architect.md (F3, F4, F7, F8, F11) and REVIEW_MPI_2026-10-15_teaching.md (setup hang,
hand-enumerated (U,V,P)); empty-part fix report (outlier timings). User decisions 2026-10-15: order R2 (correctness)
→ T2 (teaching cleanup incl. ModelPartition→ModelPart rename + standard vocabulary/glossary) → P6.3; own Cython
scatter-add instead of np.bincount: yes.
Base: `feat/mpi-explicit-dynamics-v2` at `fe6d6cf8` (pushed). Worktree `EdelweissFE-r2`, branch `fix/robustness-r2`.
Rabbit, env next_v2611_mpi (editable install points at EdelweissFE-mpi2: put this worktree first on PYTHONPATH, print
edelweissfe.__file__; copy .so files + built_extensions.log from EdelweissFE-mpi2, rebuild the extension you add).

## Items
1. **Unexpected exceptions exit non-zero (F3).** `allRanksFailTogether` must turn only the failures serial NED treats
   as a step failure/cutback into the agreed `StepFailedOnAllRanks`; any other exception (KeyError, MemoryError, …) on
   any rank must end the job with a non-zero exit on all ranks, as the serial run does (agreed raise of an error that
   is not a step failure, or abort). Test: a programming error on rank 1 of 3 inside an agreement → non-zero exit, no
   hang (timeout-guarded), negative control on the base.
2. **Setup agreement (F4 / teaching).** One agreement around model setup (input → model → solver construction), so a
   failure on one rank, even one a caller catches (test runner), does not hang the others. Remove/update the
   documented limitation. Test with timeout.
3. **Outlier-resistant element timing.** A descheduled process must not make a single element weigh 40–1e5× its
   peers (see the empty-part report: per-element wall time charges a whole stall to one element). Choose the simplest
   robust measure (e.g. per element the minimum or median over increments, or per element type/cost class); speed
   only, decisions identical on all ranks. Show a loaded-machine run no longer gives a lopsided partition.
4. **One list of the carried/published entries (teaching).** (U, V, P, … ) are enumerated by hand in ~9 places in
   NED/NEDMPI; make one shared definition (tuple) that every sync/publish/migration/checkpoint path iterates, so a new
   carried vector cannot go stale silently; test that adding an entry reaches every path (or fails loudly).
5. **Replicated copies consistent by assertion (F8).** Copies of constraints/rigid-body surfaces are kept consistent
   only by call order in acceptIncrement; add a cheap loud assertion (e.g. digest/increment stamp) where they are read.
6. **One message tag per exchange (F11).** Distinct tags per exchange kind, named in one place.
7. **Own scatter-add (F7).** Replace np.bincount left folds by a small Cython scatter-add with the order documented
   as the contract (left fold in contribution order); a test proving bitwise equality with the old bincount results
   on adversarial inputs; all gates bitwise unchanged.

## Gates
Bitwise everything (verify_refactor.sh quick after each code commit, full once at the end on an idle machine — check
ps; no leftover processes, no self-matching pgrep waiters); testfiles/mpi -n 2/3/4 `--tolerance 0` vs fresh serial
NED; checkpoints; mode check; full pytest (3 test_tie failures possible at base); suites (base: CDPPlane); sphinx;
lint; c1_150 8x4 + serial32 timing vs base (no regression). Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small
Conventional Commits with the footer lines; stage by name; no push; no helper agents. Time cap ~5 h; stop and report
rather than force a design decision.

## Decision (user, 2026-10-15)
Keep the setup agreement `allRanksFailTogetherWhileCommunicating` ("agree before each collective"), with the
review fix: no communication while an exception is being handled (sys.exception()), mpiCommunicator private.
Splitting setup into communication-free phases is not pursued.
