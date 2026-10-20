# MPI explicit dynamics (NEDMPI): status and roadmap — TEMPORARY

**Temporary development notes. Remove this directory (`devnotes/`) before the branch is merged upstream.**

Paused on 2026-10-20 at `feat/mpi-explicit-dynamics-v2` = `1e8fd57b` (pushed to remote `mn`). These notes are copies
of the planning documents kept outside the repository (in the `next_v2611/` workspace), so that the plan survives
the pause. The user-facing documentation is `doc/source/documentation/domaindecomposition.rst`.

## Where things stand

NEDMPI is a domain-decomposed explicit central-difference solver whose results are bitwise identical to serial
`NED` for any number of processes (hard gate D3). Fixed design decisions (see `designs/DESIGN_DISTRIBUTED_MEMORY.md`):
D1 replicated mesh description, distributed element objects; D2 output and checkpoints through rank 0, file format
identical to serial; D3 bitwise identity; D4 explicit only; elements know nothing of parallelization; whole contact
constraints evaluated by one process. Vocabulary: `Subdomain` = the piece one process computes (serially the whole
model), `DomainDecomposition` = NEDMPI's machinery, *partition* = the METIS division only.

Done and pushed (chronological; details in `HANDOFF_SESSION_2026-10-02.md` §5b–§5ab):

| Phase | Content |
|---|---|
| P2–P5 | mesh as data, owned elements, migration, distributed AMR, distributed constraints |
| gather-to-reader, hybrid exchange, C0 | outputs gathered only to readers; point-to-point constraint forces; load balancing counts constraint cost |
| R1, R2 | robustness: agreed failures, setup agreement, exact external work, own scatter-add, message tags, outlier-robust timing |
| T2 | teaching cleanup: Subdomain/DomainDecomposition naming, vocabulary + invariants in the docs, NED seams |
| exact sums | kinetic energy and momentum as exact sums (identical for any process count) |
| P6.1–P6.2 | mesh owns the nodes; global DOF numbering as replicated data |
| P6.3a | subdomain interface derived from the partition (no global gathers); constraints by node majority, re-decided at migrations |
| P6.3b | subdomain-sized DofManager, vectors, lumped operators |
| P6.3c | local Node objects/NodeFields; local MPC (tie) transformation; one rule for "whole model" |
| P6.4a | AMR octree as compact arrays (c1_150 mirror 277 → 54 MB); coordinates only in the mesh |
| P6.4b | local nodes under AMR (new-node values interpolated locally, completion exchange) |

Measured effect of P6.3–P6.4 (1 thread/process, peak RSS per process): 192k block 1909 → 1499 MB at 8 and
1143 → 685 MB at 32 processes; AMR block (24k → 69k elements) ~2580 → ~1760 MB at 8 and ~1910 → ~1057 MB at 32,
increment 0.065 → 0.042 s at 32. c1_150 (contact) still holds every node until P6.5.

Accepted incompatibilities: checkpoints with AMR level ≥ 2 on warped meshes written before P6.4a fail to resume
(loud fingerprint error); the sign of an exactly-zero interpolated AMR value may differ from pre-P6.4b runs.

## Next steps (when the work resumes)

1. **P6.5 — contact with local nodes**: the process evaluating a contact constraint holds that constraint's surface
   nodes, receiving copies of those it does not integrate; master positions gathered surface-sized; rigid-body
   reference node received. Contact splitting (C1–C3, `designs/DESIGN_CONTACT_BY_SLAVE.md`) stays deferred
   (revisit beyond ~64 processes). Needed for c1_150's memory to fall.
2. **Hanging-node records from labels + GlobalDofNumbering** (as ties in P6.3c), so only hanging groups touching
   local elements are held (today the whole front is held on every process).
3. **P6.6** — retire the whole-model rule; then NED loses `[dofs]` (user decision: not before).
4. Remaining O(N) transients at a topology check (whole-vector gather, recovery-marker gather); the ~20 s topology
   update on the AMR block deck (AMR itself ~5 s).
5. Later: distribution machinery out of the serial core (load balancer / receive plans out of
   `DomainDecomposition`/NEDMPI, design note first); ParMETIS/array connectivity beyond ~5M elements.
6. Known unrelated: EdelweissMeshfree broken against current EdelweissFE (user: not now); 3 `test_tie` failures on
   base; `testfiles/mpi/edelweiss-only/NED/U.ref` not bitwise on rabbit (machine provenance).

## Working method (how every round was run)

Plan file → implementing agent in its own worktree → independent review → fixes → own verification → merge →
push only after user approval, only to `mn`. Gates, every round: `harness/verify_refactor.sh <worktree> <label>
quick|full` (full with `BASELINE=…/refactor_reference_marmot_e992a3f4/c1_150_baseline.npy`, idle machine);
`testfiles/mpi` at `-n 2/3/4 --tolerance 0` against a *fresh* serial NED; checkpoints (`testfiles/mpi/
compare_checkpoints.py`); mode check (`testfiles/mpi/check_element_distribution.py`); full pytest over `tests` and
`edelweissfe`; suites; sphinx; lint. Rules: `plans/PLAN_MPI_REFACTOR.md` "Rules for both".

## Contents

- `HANDOFF_SESSION_2026-10-02.md` — master log of the whole session (§5b onwards: every round and decision).
- `HANDOFF_MPI_EXPLICIT.md` — the original prototype handoff.
- `designs/` — design notes (distributed memory, mesh as data, local DOFs (P6) with all user decisions, contact).
- `plans/` — one plan per round, each with its gates and the user's decisions.
- `reviews/` — the independent reviews (2026-10-06 … 2026-10-15 architect/teaching).
- `harness/` — verification and measurement scripts (normally in `next_v2611/mpi_explicit_benchmarks/tools/`; the
  Python scripts carry a `.py.txt` suffix here so the repository's linters skip them — drop it to use them;
  paths inside are absolute to that workspace). Reference data (`refactor_reference_marmot_e992a3f4/`, c1_150
  baseline) and benchmark decks stay in the workspace, not in the repository.
