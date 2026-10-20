# Opus reviews of the distributed-memory MPI work (snapshot f1e96b3f, 2026-10-07)

## Architect — keep and evolve; redesign load assembly and constraint ownership
- Sound: mesh as replicated data + owner-local elements; partition before creation; tagged (DOF, model
  position) interface sums; migration by state vector; fail-loud fingerprints.
- CONFIRMED BUG (reproduced by orchestrator, also on 373d1496 = pre-P5): configuration-dependent loads
  (C3D8UL follower pressure) evaluated on loaded neighbours with stale U outside the subdomain ->
  NOT bit-identical (n=3 maxdiff 5.4 of 9.1 in U/P/V; 2.6e-3 with output every increment). Fallback mode
  fails the same way. Small-strain load decks hide it.
- Loud: body/surface load + a distributed constraint spanning subdomains -> TopologyError at increment 1.
- Balancer blind to constraint cost; agreedOnByAllParts can hang if a rank raises before an allgather inside
  it; restart replay not parallel, checkpoint = O(N) gather to rank 0; O(nDof) collectives (U/V/P allgather
  per search, constraint-force allgather per increment, touched-DOF allgather per redefinition).
- Priorities: (1) assemble loads like element forces (owner evaluates, tagged exchange; drop loaded
  neighbours; C3D8UL regression deck); (2) constraints evaluated, not integrated, by owner (halo read,
  point-to-point forces), then split contact by slave entity + constraint cost in balancer; (3) gather only
  what is read; (4) restart I/O per rank + rebalance after replay; (5) guard collectives in agreedOnByAllParts.

## Teaching professor — acceptable with substantial changes
- Model building: two stages good, but Mesh lacks nodes; createElement (generic name) is the special path;
  `model.elements[n] = el` silently ignored by field activation (check runs one direction only — confirmed).
- NED harder than before: plan/bincount element loop, VPlain[dofs] with slice, ownedDofMask, IncrementPlan,
  NodeFieldSlot, releaseEquationSystem, deferred fsum queue — only ModelPartition defensible serially.
- MPI leaks into core: FEModel.elementDistribution null object (~10 hooks), ownership versions in field
  outputs, capability attributes on every base class, TimeStepperBase.incrementsLeftEstimate ABSTRACT
  (confirmed; breaks user steppers; only NEDMPI calls it).
- Partial ElementSetOfMesh iterates silently (opt-in requireComplete: 9 call sites vs 57 elementSets[...]
  reads — confirmed). Jargon ("made by its owner", "settle", "agreed on by all parts"). CRLF->LF noise in
  adaptivetimestepper.py (confirmed). `if subset is not None` changes serial behaviour for empty sets.
- Required: (1) NED readable as the textbook method (MPI concerns behind NEDMPI / one exchange object);
  (2) mesh single source of truth by construction (incl. nodes or rename), rename createElement;
  (3) partial sets fail safe (raise; ask localElements() explicitly); (4) MPI obligations off base classes,
  incrementsLeftEstimate not abstract, owned/ghost vocabulary; (5) tighten bit-identity claim (bincount
  dependence, adversarial ordering test, recipe incl. states/checkpoints).
