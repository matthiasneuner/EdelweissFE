# Plan: P6 round-1 review fixes (2026-10-15)
Worktree EdelweissFE-p6, branch feat/p6-local-nodes (on top of 1ec04436). Review findings: see the reviewer summary
below; user decisions: DESIGN_P6_LOCAL_DOFS.md "Decisions (user, 2026-10-15)".
1. Field order: `_resizeNodeFieldsForNodes` keeps `model.nodeFields` in phenomena order (NodeField identity kept);
   test activating fields across two topology changes (repro: scratchpad/field_order_repro.py) asserting agreement;
   negative tests for the "fields" and "scalar variables" branches of `requireAgreementWith`.
2. `FEModel.moveNodes(labels, coordinates)`: writes mesh + held Node objects (in place) in one place; tie
   (tie.py ~369/401) and surfacesnap (~455-468) use it. AMR registry part is deferred to P6.4 (do not do it now).
3. Auxiliary nodes docstring: "kept only so the float32 rigid-body arithmetic stays bitwise".
4. `ReleasedDataLines`: `__len__`/`__getitem__`/`__bool__` raise the same error; `finiteElementSimulation` docstring
   says the parsed mesh data is consumed.
5. Redefining a node set: move `hadaptivity._redefineNodeSet` onto the model (`FEModel.redefineNodeSet`) or make it
   polymorphic on the set; one-line comment on the mesh-identity caches shared by `__copy__`; replace the
   kindOfSet/localMembersAre/localPartHas message-fragment attributes by one overridable describing method if it
   reads better.
Gates as PLAN_P6_ROUND1.md (quick after each code commit, full once at the end idle, testfiles/mpi -n 2/3/4 tol 0,
checkpoints, mode check, full pytest, suites, sphinx, lint). No push. Time cap ~2.5 h.
