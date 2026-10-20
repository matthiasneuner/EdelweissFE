# Plan: loads assembled like element forces (fix of the stale-U load bug, 2026-10-07)

Context: REVIEW_MPI_2026-10-07.md (architect, risk 2a/2b, priority 1). Base: `feat/mpi-explicit-dynamics-v2`
after P5 is merged. Rabbit, env next_v2611_mpi.

## The bug (reproduced by the orchestrator, predates P3–P5)

NEDMPI does not exchange loads: every process integrating a DOF re-evaluates the distributed loads / body
forces of the elements touching its subdomain ("loaded neighbours"), with `U_np[el]`. U is advanced only at
the DOFs a process integrates; at a loaded neighbour's other nodes it is only as current as the last
synchronization. Configuration-dependent loads therefore read stale U: a 6x6x6 C3D8UL box with a follower
pressure (Marmot UL pressure uses coordinates + qU) differs from serial by 5.4 of max 9.1 (U/P/V) at n=3,
2.6e-3 with output every increment; whole-model fallback fails identically; small-strain decks hide it.
Reproduction deck: /tmp/claude-1000/-home-matthias-constitutive-modeling-next-v2611/9b4e6ca5-3207-4233-8964-0733bf6b0f74/scratchpad/c3/PressureUL3/test.inp.
Second, loud symptom: a body/surface load plus a distributed constraint whose DOFs span subdomains ->
`TopologyError: element … not created there` at increment 1 (subdomain.py, _requireLoadedElementsCreatedHere).

## The fix (by design)

Loads are contributions of elements, so assemble them like element forces: **the owner of an element
evaluates that element's distributed loads and body forces** (with its own, current U — the element's DOFs are
in its subdomain), and the contributions are exchanged through the same tagged interface assembly as the
internal forces, summed per DOF in a canonical order that reproduces serial NED's summation order bit for bit
(serial: internal forces of all elements in model order, negated, then node forces, then distributed loads in
deck order over their surfaces' elements, then body forces, then constraints — establish the exact serial
order from the code and reproduce it; if the internal-force sum and the load sum are separate sums in serial,
keep them separate). Concentrated node forces stay owner-of-DOF.
Consequences: the loaded-neighbour halo (DistributedElements' loaded elements, elementsLoadedByDefinition,
Subdomain._requireLoadedElementsCreatedHere, the load views handed to the shared assembly) disappears or
shrinks to what is still needed — remove dead code; elements stay parallelization-agnostic; the shared load
assembly in NonlinearSolverBase is reused, not copied.

## Gates

- New regression decks in testfiles/mpi: the C3D8UL follower-pressure box (distributed), a body force +
  a constraint spanning subdomains (the former TopologyError), both bitwise vs fresh serial NED at -n 2/3/4
  (`--tolerance 0`), also with output-frequency > 1 (the stale-U trigger), and in the whole-model fallback.
- Everything else bitwise: verify_refactor.sh quick|full with REF=…/refactor_reference_fsum (full once, idle
  machine, check ps, leave no background runs); testfiles/mpi -n 2/3 --tolerance 0; mode check; checkpoints
  of the restart decks == serial; full pytest, suites, sphinx, lint; no slowdown on c1_150 8x4 (distributed)
  and the 192k block (alternate base/head if close).
- Docs: domaindecomposition.rst — how loads are assembled, why it is bit-identical; remove the
  "loads are not exchanged" text.
- Rules: PLAN_MPI_REFACTOR.md "Rules for both"; small Conventional Commits with the two footer lines; stage
  by name; no push. Time cap ~4 h; stop and report rather than force anything costing bit-identity.
