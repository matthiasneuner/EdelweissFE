# Plan: teaching cleanup T2 (2026-10-15)
Base: feat/mpi-explicit-dynamics-v2 at 2adfeaee (pushed, R2 merged). Worktree EdelweissFE-t2, branch refactor/teaching-t2.
Rabbit, env next_v2611_mpi (editable install points at EdelweissFE-mpi2: this worktree first on PYTHONPATH, print
edelweissfe.__file__; copy .so files + built_extensions.log from EdelweissFE-mpi2). Rules: PLAN_MPI_REFACTOR.md "Rules
for both"; small Conventional Commits; stage by name; no push; no helper agents; time cap ~5 h; no leftover processes.

Source: REVIEW_MPI_2026-10-15_teaching.md (prioritized list) + naming feedback (two agents, 2026-10-15).
User decision 2026-10-15: option B naming.

## 1. Renames (first commit series; pure renames, bitwise unchanged)
- Big class `Subdomain` (domaindecomposition/subdomain.py) → `DomainDecomposition` (NEDMPI only: METIS division,
  exchanges, gathers, load balancing, migration). Do this FIRST to avoid a transient name clash.
- `ModelPartition` → `Subdomain` (the piece one process computes: elements, constraints, DOFs; serially
  `Subdomain.wholeModel`); `NED.partition` → `NED.subdomain`; hook `partitionModel` → `defineSubdomain(model,
  topologyChanged) -> Subdomain`. Serial NED never says "partition".
- "partition" only for the METIS division (partitionElementsOfMesh, repartition, element owners).
- Read every doc/code occurrence of "subdomain" (some paragraphs use both senses, e.g. domaindecomposition.rst
  ~l.222–228); keep the docstring rule: exchanges are the decomposition's business, not the subdomain's. Define
  "subdomain" once in the docs (no relation to libMesh/MOOSE material blocks).
## 2. Vocabulary + docs
- Standard terms (owned, replicated, subdomain DOFs, interface, collective) instead of "integrated here / computed
  here"; glossary + invariants block at the top of domaindecomposition.rst; measurements to an appendix; "Collective."
  as the first word of collective methods' docstrings; serial-first docstrings in femodel/sets/base classes.
## 3. Seams
- One "What NEDMPI overrides" section at the end of NED's docstring incl. the inline-reduction rule.
- kineticEnergy()/momentum as the only whole-vector reductions, one overridable hook.
- NED splits constraint handling into evaluate + add (no hidden _constraintForcesOfIncrement handshake).
Gates: as R2 (bitwise everything, harness quick/full, testfiles/mpi, pytest, suites, sphinx, lint).
