# Plan P3: each process creates only its own elements (2026-10-06)

Context: DESIGN_DISTRIBUTED_MEMORY.md (§3, §5 P3, §6 decisions), DESIGN_P2_MESH_AS_DATA.md (§3 guard,
§5b migration, §9 decisions, §10 P3 follow-ups), REVIEW_MPI_2026-10-06.md (architect: gather-to-root,
migration recipe, loud partial-set guard). Base: `feat/mpi-explicit-dynamics-v2` at `41eae721` (pushed to
mn; no PR). Machine rabbit, env next_v2611_mpi.

## Decision (user, 2026-10-06)

Models that still need the whole model on every process — **live AMR, contact, executePythonCode,
cubit** — fall back to creating every element on every process (today's behaviour), by **one visible
rule** decided in one place and logged once on rank 0 ("whole model on every process because: …").
Everything else runs **distributed**: each process creates only the elements it needs. P4 (AMR) and P5
(contact) remove the fallback piece by piece.

## What P3 does

1. **Partition before creation.** Under an MPI launcher, after the mesh is described and before
   `createElementsOfMesh`, the elements are partitioned on the mesh connectivity (METIS, as
   `partitioning.partitionElements` does today on objects — move it onto `model.mesh`; same result, so
   the partition is identical to today's). The predicate creates a process' own elements plus exactly
   those others it must evaluate (e.g. elements whose distributed load or body force reaches its
   subdomain — today `Subdomain` evaluates such loads on replicated neighbours to keep bit-identity;
   keep bit-identity, decide the minimal set and document it). Decide where this lives so the driver
   stays readable (e.g. a `domaindecomposition` function the driver calls when the job runs on several
   processes); the solver later adopts the partition instead of computing its own.
2. **Subdomain without the whole model** (distributed mode): no whole-model state synchronization;
   vectors shared from owners as today. Whole-model readers go through the gather path to rank 0
   (`gatherElementResults` or similar, by mesh element number in mesh order), and `requireComplete` /
   `requireCompleteMesh` stay loud for any reader not converted. Convert what the distributed decks
   need: element field outputs over sets, Ensight (geometry from `model.mesh` on rank 0 + gathered
   results), monitor/conditional stop (via field outputs), checkpoints (rank 0 writes the gathered
   element states in the existing format — checkpoint files must stay identical), restart (each process
   restores its own elements from the same file). The layout fingerprint and `_checkReplicatedLayout`
   key on `model.mesh` (P2 §10).
3. **Rebalancing = migration** (P2 §5b: `createElementOfMesh` + section + `setStateVars` on the
   receiver, drop on the sender). If it does not fit the time cap, disable rebalancing in distributed
   mode with a logged reason and report.
4. **Fallback rule** for live AMR / contact / executePythonCode / cubit, as decided above; also refuse
   (loud) anything else found to need the whole model that is neither converted nor in the rule.
5. **Measure memory**: per-process peak RSS in distributed mode on a large model without AMR/contact
   (e.g. the 192k GC3D8 GCDP block under `mpi_explicit_benchmarks/strong_scaling/`, or a boxgen model of
   similar size) at 1/8/32 processes vs. the fallback (whole model) — the point of the whole exercise.

## Gates

- **Bitwise** everything: `mpi_explicit_benchmarks/tools/verify_refactor.sh <wt> <label> quick|full`
  (c1_150 runs in the fallback and must stay bitwise and as fast; decks without AMR/contact run
  distributed — add a check that proves which mode each `testfiles/mpi` deck ran in, e.g. the logged
  mode or created-element counts, so distributed mode is verified on the failing path, not assumed).
- Checkpoint written in distributed mode == serial NED's checkpoint (h5cmp); resume in distributed mode
  bitwise to a straight run (use a distributed deck with restart, e.g. a Marmot NED deck with a restart
  writer).
- Suites, pytest, sphinx, lint unchanged vs base; testfiles/mpi under mpirun -n 2/3 (also with
  `run_tests_edelweissfe --tolerance 0` against fresh serial `--create`).
- Rules: as PLAN_MPI_REFACTOR.md "Rules for both" (teaching vocabulary, docstrings, no
  hasattr/getattr/setattr, no duplication, elements parallelization-agnostic, commits with footers,
  stage by name, no push). Time cap ~5 h; stop and report rather than force anything costing
  bit-identity or performance, or needing a design change.
