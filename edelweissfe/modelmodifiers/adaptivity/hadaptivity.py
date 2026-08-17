#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

"""Dynamic h-adaptivity model modifier for HEX20 hanging-node AMR."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh
from edelweissfe.adaptivity.statetransfer.perstatevar import PerStateVarStateTransfer
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.markerlibrary import getMarkerClass
from edelweissfe.config.registry import RegistryLookupError
from edelweissfe.config.statetransferstrategies import getStateTransferStrategyClass
from edelweissfe.constraints.hangingnode import Constraint as HangingNodeConstraint
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.base.modelmodifierbase import ModelModifierBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.points.node import Node
from edelweissfe.utils.performancetiming import timeit
from edelweissfe.utils.schema import (
    buildSchemaFromOptions,
    schemaField,
    subKeywordField,
)


@dataclass(frozen=True)
class HAdaptivityMarkerSchema:
    """L2: the grammar common to every ``>>marker`` block.

    A ``>>marker`` block is polymorphic on ``type``: the remaining options depend on which marker
    that selects, and are owned/validated by that marker's own schema (a
    :class:`~edelweissfe.adaptivity.marking.MarkerOptionsBase` subclass, e.g.
    :class:`~edelweissfe.adaptivity.marking.RecoveryErrorMarkerSchema`) rather than being flattened
    into one union here. This schema therefore declares only the two options every marker shares --
    ``type`` (the dispatch key) and ``initialOnly`` -- with the type-specific options documented on
    each marker in :mod:`edelweissfe.adaptivity.marking` and reachable through the ``marker`` L3
    registry category (:mod:`edelweissfe.config.markerlibrary`).
    """

    type: str | None = schemaField(
        description=(
            "Marker type, resolved through the 'marker' registry: fieldOutput, elementSet, nodeSet, "
            "surface, recoveryError. The type-specific options are defined by the selected marker's "
            "own schema."
        ),
        dtype=str,
        default=None,
        required=True,
    )
    initialOnly: bool = schemaField(description="Evaluate only once at simulation start", dtype=bool, default=False)


@dataclass(frozen=True)
class HAdaptivitySchema:
    """L2: the options this model modifier accepts, owned by this module and never mutated from
    outside it.

    ``marker`` is declared optional even though at least one is required in practice -- that
    invariant is enforced in :meth:`ModelModifier.__init__`, not by the grammar (a schema field
    cannot express "at least one of a repeatable sub-keyword").
    """

    moduleOptions: dict = schemaField(description="Internal", dtype=dict, default_factory=dict)
    elSet: str | None = schemaField(
        description=(
            "Fallback for 'refineElSet' if that is not given. Each '>>marker' scopes its own "
            "eligible elements (a fieldOutput's associated set, an elementSet/nodeSet/surface's "
            "members); this no longer restricts marking itself."
        ),
        dtype=str,
        default=None,
    )
    refineElSet: str | None = schemaField(
        description=(
            "Restrict the AMR octree mirror itself to this element set, e.g. the solid elements in "
            "a mesh that also contains contact-facet elements. Elements outside this set never "
            "become octree roots and are left untouched by refinement. Defaults to 'elSet' if given, "
            "otherwise to every 20-node (HEX20-family) element in the model."
        ),
        dtype=str,
        default=None,
    )
    maxLevel: int = schemaField(description="Maximum refinement level.", dtype=int, default=1)
    minMarkedElements: int = schemaField(
        description=(
            "Minimum number of eligible elements that must be marked before a refinement pass is "
            "triggered. Marked elements persist (accumulate) across increments -- across calls where "
            "fewer than this many are marked, no refinement happens and no equation system rebuild is "
            "triggered -- until the accumulated count reaches this threshold, at which point all of "
            "them are refined together in a single pass. Default 1 refines as soon as any element is "
            "marked (previous behavior)."
        ),
        dtype=int,
        default=1,
    )
    splitFactor: int = schemaField(
        description=(
            "Number of equal parts per axis a marked element is split into (2 = octree bisection "
            "into 8 children; 3 = 3x3x3 = 27 children, etc.). The hanging-node coupling stays exact "
            "for any factor."
        ),
        dtype=int,
        default=2,
    )
    elementType: str | None = schemaField(
        description="Element type to instantiate for children (default: like parents).", dtype=str, default=None
    )
    elementProvider: str = schemaField(description="Element provider.", dtype=str, default="marmot")
    stateTransfer: str = schemaField(
        description="Quadrature-point state-transfer strategy for the whole state block: nearestQp|projection|virgin.",
        dtype=str,
        default="nearestQp",
    )
    stateTransferOverrides: str | None = schemaField(
        description=(
            "Per-state-variable overrides routing named variables to a different strategy, e.g. "
            "'strain:projection, stress:virgin'. Comma-separated 'name:strategy' pairs."
        ),
        dtype=str,
        default=None,
    )
    marker: tuple = subKeywordField(
        description="AMR marker definition. At least one is required.", schema=HAdaptivityMarkerSchema
    )


def _buildStateTransferStrategy(defaultName, overridesSpec):
    """Construct the state-transfer strategy from the input arguments. With no per-variable
    overrides this is just the named default strategy; otherwise a
    :class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer` wrapping the
    default with the named overrides."""
    default = getStateTransferStrategyClass(defaultName)()
    if not overridesSpec:
        return default
    overrides = {}
    for entry in overridesSpec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, strategyName = entry.rsplit(":", 1)
        overrides[name.strip()] = getStateTransferStrategyClass(strategyName.strip())()
    return PerStateVarStateTransfer(default, overrides) if overrides else default


def _connectedComponents(elements: list) -> dict:
    """Partition elements into connected bodies via union-find over shared node labels.

    Two elements belong to the same body if they share at least one node label. The resulting
    component id namespaces the refinement node registry and confines hanging-node classification
    to one body, so two bodies meeting at a flush interface -- a tied surface pair (``adjust`` moves
    the slave nodes exactly onto the master surface), a zero-gap contact pair, a duplicated-node
    crack plane -- are neither collapsed onto shared node labels nor welded together by refinement.

    Parameters
    ----------
    elements
        The refineable elements, in the order in which they become octree roots.

    Returns
    -------
    dict
        element -> component id, densely numbered from 0 in order of first appearance.
    """
    parentOf = list(range(len(elements)))

    def find(i):
        while parentOf[i] != i:
            parentOf[i] = parentOf[parentOf[i]]  # path halving
            i = parentOf[i]
        return i

    def union(i, j):
        rootI, rootJ = find(i), find(j)
        if rootI != rootJ:
            parentOf[max(rootI, rootJ)] = min(rootI, rootJ)

    firstElementAtNode = {}
    for i, element in enumerate(elements):
        for node in element.nodes:
            union(i, firstElementAtNode.setdefault(node.label, i))

    componentOfElement = {}
    denseIds = {}
    for i, element in enumerate(elements):
        componentOfElement[element] = denseIds.setdefault(find(i), len(denseIds))
    return componentOfElement


class ModelModifier(ModelModifierBase):
    #: L2 schema declared for the L3 registry, per OptionSchemaProvider. Documentation-only for now
    #: (see HAdaptivitySchema's own docstring) -- construction still goes through the legacy
    #: Module-based mechanism below, unchanged.
    schema = HAdaptivitySchema

    def __init__(self, name: str, model: FEModel, journal: Journal, *args, **kwargs):
        super().__init__(name, model, journal, *args, **kwargs)
        options = buildSchemaFromOptions(HAdaptivitySchema, kwargs)

        self._name = name
        self._model = model
        self._journal = journal

        # Markers are resolved by 'type' through the L3 marker registry and each builds itself from
        # its own >>marker options via fromOptions (validated against that marker's own schema), so
        # this loop is marker-agnostic: adding a marker means registering it, not editing an if/elif
        # here. The 'type' key is the dispatch key, not a marker option, so it is stripped before the
        # marker validates the rest.
        self.markers = []
        for m_opt in options.moduleOptions.get("marker", []):
            m_type = m_opt.get("type")
            if not m_type:
                raise ValueError(
                    f"hAdaptivity modifier {name!r}: a '>>marker' block is missing its required "
                    "'type' (e.g. 'type=fieldOutput', 'type=recoveryError', 'type=nodeSet')."
                )
            try:
                markerClass = getMarkerClass(m_type)
            except RegistryLookupError as e:
                raise ValueError(f"hAdaptivity modifier {name!r}: {e}") from e
            # 'type' is the dispatch key (already consumed above); 'inputFile' is parser bookkeeping
            # stamped onto every module keyword's options. Everything else is a real marker option,
            # validated against the marker's own schema inside fromOptions.
            markerOptions = {key: value for key, value in m_opt.items() if key.casefold() not in ("type", "inputfile")}
            self.markers.append(markerClass.fromOptions(markerOptions))
        if not self.markers:
            raise ValueError(
                f"hAdaptivity modifier {name!r} defines no '>>marker' block. At least one is required, "
                "e.g. '>>marker, type=fieldOutput, fieldOutput=stress, expression=\"abs(x) > 0.1\"' "
                "(referencing an already-declared 'perElement' *fieldOutput)."
            )

        self.maxLevel = options.maxLevel
        self.minMarkedElements = max(1, options.minMarkedElements)
        self._pendingMarkedElements = set()  # elements marked but not yet refined (below minMarkedElements)
        # element labels that triggered each refinement that has actually materialized, in commit
        # order -- the record restart replays (see getRestartData/setRestartData) to reproduce this
        # instance's topology history without re-running (and trusting bit-identical) marker evaluation.
        self._committedOccasions = []
        self.splitFactor = options.splitFactor
        self._stateTransfer = _buildStateTransferStrategy(options.stateTransfer, options.stateTransferOverrides)
        self._provider = options.elementProvider
        # element -> its section, so children inherit the parent's material (multi-material meshes)
        self._sectionOf = {}
        for section in model.sections.values():
            for elementSet in section.elSets:
                for element in elementSet:
                    self._sectionOf[element] = section

        # restrict the octree mirror to the refineable solid elements: a model that also contains
        # e.g. contact-facet elements (2/3 nodes) must not have those become octree roots. Prefer an
        # explicit restriction; otherwise fall back to the 20-node (HEX20-family) elements, which is
        # the only family this modifier supports anyway.
        refineSetName = options.refineElSet or options.elSet
        if refineSetName is not None:
            refineElements = list(model.elementSets[refineSetName])
        else:
            refineElements = [el for el in model.elements.values() if len(el.nodes) == 20]
        if not refineElements:
            raise ValueError(
                "hAdaptivity found no refineable (20-node) elements in the model; specify "
                "'refineElSet' (or 'elSet') to select the solid element set explicitly."
            )

        # two hAdaptivity instances cannot independently own overlapping elements: each maintains
        # its own AdaptiveMesh mirror and materializes/deletes elements directly in the model, so a
        # second instance refining/removing an element the first still tracks leaves the first with
        # a stale reference (an Element object no longer in model.elements) -- which later corrupts
        # element-set membership (a "deleted" element gets carried back into e.g. 'fixed_all') and
        # can surface as a node simultaneously Dirichlet-prescribed and a hanging-node MPC slave.
        # Fail loud at construction time instead of silently corrupting state deep in the solve loop.
        refineElementNumbers = {el.elNumber for el in refineElements}
        for otherName, otherModifier in model.modelModifiers.items():
            if isinstance(otherModifier, ModelModifier):
                overlap = refineElementNumbers & otherModifier._refineElementNumbers
                if overlap:
                    raise ValueError(
                        f"hAdaptivity modifier {name!r} and existing modifier {otherName!r} both "
                        f"claim {len(overlap)} of the same element(s) (e.g. label "
                        f"{sorted(overlap)[0]}) as refineable roots via overlapping 'refineElSet'/"
                        "'elSet' (or no restriction at all). Combine all markers -- including "
                        "'initialOnly' ones -- into a single hAdaptivity block via multiple "
                        "'>>marker' lines instead of stacking separate modifiers over the same "
                        "elements."
                    )
        self._refineElementNumbers = refineElementNumbers

        # element type: infer from a refineable element if not given
        anyEl = refineElements[0]
        self._elementType = options.elementType or anyEl.elType
        self._elementClass = getElementClass(self._elementType, self._provider)

        # bodies of the refineable mesh: node labels are namespaced per body, so coincident nodes of
        # two bodies (a tied interface -- 'adjust' makes it flush by default --, a zero-gap contact
        # pair, a duplicated-node crack plane) are never deduplicated into one label
        componentOfElement = _connectedComponents(refineElements)

        # build the AdaptiveMesh mirror, sharing node labels with the live model. Only the nodes of
        # the refineable elements are seeded: a node the octree does not own must not be able to
        # claim a coordinate key, and only an octree-owned node can be seeded with a body.
        self._topology = Hex20Topology()
        self._mesh = AdaptiveMesh(splitFactor=self.splitFactor, topology=self._topology)
        self._eidToEl = {}  # mesh element id -> live element
        #: True only while setRestartData() replays committed occasions; lets _materialize skip
        #: per-occasion work whose result the checkpoint restore overwrites anyway.
        self._replayMode = False
        #: Octree eids of each committed occasion, parallel to _committedOccasions (which holds
        #: element numbers). Eids stay valid across a restart replay; element numbers do not.
        self._committedOccasionEids = []
        for el in refineElements:
            componentId = componentOfElement[el]
            for n in el.nodes:
                self._mesh.registry.seed(n.label, n.coordinates, componentId)
            coords = np.array([n.coordinates for n in el.nodes])
            eid = self._mesh.add_root(coords, componentId)
            self._eidToEl[eid] = el
        # nodes outside the refineable mesh are deliberately not seeded, but their labels are taken:
        # keep the registry's high-water mark above them so new nodes never collide with them
        self._mesh.registry.reserve_labels_up_to(max(model.nodes.keys(), default=0))

        # all-encompassing sets (contain every node, e.g. 'all', 'ALLNODES') are not boundary BCs --
        # they just gain every new node; rebuild them wholesale, don't guard/track them
        allLabels = set(model.nodes.keys())
        self._allLikeSets = {name for name, ns in model.nodeSets.items() if {n.label for n in ns.nodes} == allLabels}
        # track the remaining (boundary) node sets so real BCs gain new boundary nodes on refinement
        for setName, nodeSet in model.nodeSets.items():
            if setName not in self._allLikeSets:
                self._mesh.define_node_set(setName, [n.label for n in nodeSet])

        # track element sets so user element sets propagate child elements on refinement (Finding 1)
        elToEid = {el: eid for eid, el in self._eidToEl.items()}
        # passengers of a tracked set: members the octree mirror does not know (non-refineable
        # elements, e.g. HEX8, interface elements, contact facets). A mixed set would lose them on
        # the first refinement, since _materialize rebuilds the set from mesh element ids only
        self._untrackedOfElementSet = {}  # element set name -> list of non-mirrored members
        for setName, elementSet in model.elementSets.items():
            eids = [elToEid[el] for el in elementSet if el in elToEid]
            # a set with no refineable member (e.g. a contact-facet-only set) is left untracked, so
            # _materialize never overwrites it with an emptied-out ElementSet
            if eids:
                self._mesh.define_element_set(setName, eids)
                self._untrackedOfElementSet[setName] = [el for el in elementSet if el not in elToEid]

        # track element-based surfaces so surface loads stay consistent under refinement (Finding 2)
        for surfaceName, surface in model.surfaces.items():
            pairs = [
                (elToEid[el], faceID) for faceID, elementSet in surface.items() for el in elementSet if el in elToEid
            ]
            if pairs:
                self._mesh.define_surface(surfaceName, pairs)

        # companion hanging-node MPC (records set in memory), registered as a multi-point constraint
        self._hanging = HangingNodeConstraint(name + "_hanging", model)
        model.multiPointConstraints[name + "_hanging"] = self._hanging
        self._converged = False  # set True once an increment has converged
        self._lastRefinedTime = None  # model.time of the last refinement (guards re-refine on cutback)
        self._isFirstCall = True
        # parent-parametric coords of each child's nodes (used for warm-start interpolation)
        self._octantParams = self._topology.subdivision_children_param(self.splitFactor)

    @timeit("AMR")
    def updateModel(self, model: FEModel, step, timeStep: float) -> bool:
        # Do not re-refine if the solver is re-trying the exact same time state after a cutback
        if self._lastRefinedTime is not None and abs(model.time - self._lastRefinedTime) < 1e-12:
            return False

        elForEid = {v: k for k, v in self._eidToEl.items()}
        marked_elements = set()

        if self._isFirstCall:
            initial_markers = [m for m in self.markers if m.initialOnly]
            for m in initial_markers:
                elements = m.mark(model, self._eidToEl.values(), self._mesh)
                marked_elements.update(elements)

        # dynamic markers (not initialOnly) need a non-zero state to evaluate safely
        dynamic_markers = [m for m in self.markers if not m.initialOnly]

        stateMagnitude = max(
            (float(np.abs(np.asarray(nf["U"])).max()) for nf in model.nodeFields.values() if "U" in nf),
            default=0.0,
        )
        if stateMagnitude >= 1e-12:
            for m in dynamic_markers:
                marked_elements.update(m.mark(model, self._eidToEl.values(), self._mesh))

        self._isFirstCall = False

        # freshly marked elements accumulate onto any still-pending ones from earlier increments (WS-
        # minMarkedElements): a stale pending element that another path already refined/removed is
        # dropped by the elForEid/maxLevel filter below, same as a freshly marked one would be.
        self._pendingMarkedElements.update(marked_elements)

        if not self._pendingMarkedElements:
            return False

        # keep only active elements below maxLevel
        with timeit("marking filter"):
            eligible = [
                el
                for el in sorted(self._pendingMarkedElements, key=lambda e: e.elNumber)
                if el in elForEid and self._mesh.elements[elForEid[el]]["level"] < self.maxLevel
            ]
        self._pendingMarkedElements = set(eligible)

        if len(eligible) < self.minMarkedElements:
            if eligible:
                self._journal.message(
                    "AMR ModelModifier: {:} element(s) marked, deferring refinement until {:} accumulate".format(
                        len(eligible), self.minMarkedElements
                    ),
                    "hadaptivity",
                    1,
                )
            return False

        self._pendingMarkedElements = set()
        self._refineAndMaterialize(model, eligible)
        return True

    def _refineAndMaterialize(self, model: FEModel, eligible: list) -> None:
        """Refine exactly ``eligible`` and materialize the resulting children: the deterministic,
        marking-decision-independent half of a refinement pass (octree split, 2:1 balance,
        hanging-node MPCs, element/node/set bookkeeping, and the :class:`ModelChange` notification).

        Shared by the live marking path (:meth:`updateModel`, which has already decided *which*
        elements to refine before calling this) and restart's occasion replay
        (:meth:`setRestartData`, which recreates a past decision from the checkpoint instead of
        re-evaluating markers) -- both grow :attr:`_committedOccasions` identically and produce
        byte-identical topology given the same ``eligible`` input, since everything here is pure
        octree/topology mechanics with no dependence on solution history.

        Parameters
        ----------
        model
            The FEModel object.
        eligible
            The (already-decided) elements to refine, sorted by ``elNumber``.
        """

        elForEid = {v: k for k, v in self._eidToEl.items()}
        markedEids = [elForEid[el] for el in eligible]

        # (WS-B/C) refine + 2:1 balance in the mirror
        nBefore = len(self._mesh.active())
        with timeit("refine & balance"):
            for eid in markedEids:
                if self._mesh.elements[eid]["active"]:
                    self._mesh.refine(eid)
            self._mesh.balance_2to1()

        with timeit("hanging nodes"):
            records = self._mesh.hanging_mpc_records()  # computed once (expensive), reused below

        with timeit("materialize"):
            change = self._materialize(model, records)

        self._hanging.setRecords(records)
        # notify observers (e.g. Dirichlet BCs, Ensight output manager) so they re-index against the mutated mesh
        with timeit("notify observers"):
            model.notifyModelChanged(ModelChangeType.REFINEMENT, change)
        self._journal.message(
            "AMR ModelModifier: marked {:}, refined -> active elements {:} -> {:}, {:} hanging nodes".format(
                len(markedEids), nBefore, len(self._mesh.active()), len(records)
            ),
            "hadaptivity",
            0,
        )
        self._lastRefinedTime = float(model.time)
        self._committedOccasions.append([el.elNumber for el in eligible])
        self._committedOccasionEids.append(list(markedEids))

    def _materialize(self, model: FEModel, records: dict):
        mesh = self._mesh
        reg = mesh.registry

        # Element numbers come from the model's single monotonic allocator
        # (FEModel.reserveElementNumbers). This modifier deliberately keeps no counter of its own:
        # the one it used to keep had to be resynced against max(model.elements) on every call,
        # because a tied surface's facets -- rebuilt via the observer/MeshDependent escape hatches
        # fired at the end of THIS very call -- claim labels in between, and a private counter would
        # collide with (and silently overwrite) them, after which they were deleted as "stale",
        # orphaning the solid elements that had taken their labels.

        # snapshot the converged nodal values BEFORE the mesh mutates, for the warm start
        oldValues = {}
        # On the restart replay path the warm start is dead work: readRestart overwrites every node
        # field right afterwards. Leaving oldValues empty makes the interpolation below and the
        # fields-restore block no-ops, and skips one array copy per node per field per occasion.
        if not self._replayMode:
            for fieldName, nodeField in model.nodeFields.items():
                if "U" in nodeField:
                    U = np.asarray(nodeField["U"])
                    oldValues[fieldName] = {
                        node: U[nodeField._indicesOfNodesInArray[node]].copy() for node in nodeField.nodes
                    }

        # new nodes
        newNodes = {}
        for label, coord in reg.coordinates.items():
            if label not in model.nodes:
                node = Node(label, np.asarray(coord, dtype=float))
                model.nodes[label] = node
                newNodes[label] = node

        active = set(mesh.active())
        materialized = set(self._eidToEl.keys())
        newValues = {fieldName: {} for fieldName in oldValues}  # interpolated values for new nodes
        # Only children whose parent is already materialised: the batched restart replay can leave
        # several refinement levels pending at once, and this keeps each pass to one level.
        newChildEids = {eid for eid in (active - materialized) if mesh.elements[eid]["parent"] in self._eidToEl}

        # the changeset this call produces (Finding 1/2 above become its faceMap/*Sets entries)
        change = ModelChange(kind=ModelChangeType.REFINEMENT, addedNodes=set(newNodes.keys()))

        # new child elements (single level of new refinement per call -> parents are materialized)
        # Sorted, with the whole batch's numbers reserved up front: which octree child gets which
        # element number is then a pure function of this sorted list of eids -- not of the order an
        # unordered set happened to iterate in, and not of what else claimed a number partway
        # through the loop.
        newChildEidsInOrder = sorted(newChildEids)
        childNumbers = model.reserveElementNumbers(len(newChildEidsInOrder)) if newChildEidsInOrder else []
        with timeit("elements & state transfer"):
            for eid, elNumber in zip(newChildEidsInOrder, childNumbers):
                e = mesh.elements[eid]
                parentEid = e["parent"]
                parentEl = self._eidToEl[parentEid]
                child = self._elementClass(self._elementType, elNumber)
                child.setNodes([model.nodes[label] for label in e["conn"]])
                self._sectionOf[parentEl].assignSectionPropertiesToElement(child)
                if not self._replayMode:
                    # Replay restores each child's checkpointed state by eid afterwards, so the
                    # transferred values would be overwritten (see setRestartData).
                    self._stateTransfer.transferState(parentEl, [child], self._topology)  # WS-F (state)

                # warm start (WS-H): interpolate each NEW node's field values from the parent via the
                # HEX20 isoparametric map, so the increment restarts from a consistent state, not zero
                octant = mesh.elements[parentEid]["children"].index(eid)
                childParams = self._octantParams[octant]
                for i, label in enumerate(e["conn"]):
                    node = model.nodes[label]
                    if label in newNodes and any(node not in newValues[f] for f in oldValues):
                        N = self._topology.shape_functions(*childParams[i])
                        for fieldName, vals in oldValues.items():
                            if all(pn in vals for pn in parentEl.nodes):
                                parentVals = np.array([vals[pn] for pn in parentEl.nodes])
                                newValues[fieldName][node] = N @ parentVals

                model.createElement(child)
                self._eidToEl[eid] = child
                self._sectionOf[child] = self._sectionOf[parentEl]

                change.addedElements.add(child.elNumber)
                change.parentToChildren.setdefault(parentEl.elNumber, []).append(child.elNumber)

        # per-face parent -> child tiling (Finding 2's faceMap), while parents are still materialized
        newlyRefinedParentEids = {mesh.elements[eid]["parent"] for eid in newChildEids}
        for parentEid in newlyRefinedParentEids:
            parentLabel = self._eidToEl[parentEid].elNumber
            childEids = mesh.elements[parentEid]["children"]
            for faceID, faceIndex in self._topology.faceid_to_face.items():
                childLabels = [
                    self._eidToEl[childEids[j]].elNumber
                    for j in self._topology.face_child_indices(faceIndex, self.splitFactor)
                ]
                change.faceMap[(parentLabel, faceID)] = [(label, faceID) for label in childLabels]

        # remove refined parents (sorted, so the changeset is built in a reproducible order)
        for eid in sorted(materialized - active):
            el = self._eidToEl.pop(eid)
            model.removeElement(el.elNumber)
            change.removedElements.add(el.elNumber)

        # keep model.surfaces in sync (Finding 2): parent (eid,faceID) -> child faces
        for surfaceName, pairs in mesh.surfaces.items():
            if surfaceName in model.surfaces:
                if any(meid in newChildEids for meid, _ in pairs):
                    change.changedSurfaces.add(surfaceName)
                byFace = defaultdict(list)
                # Sorted: this fixes the member order of the rebuilt surface, and a contact/tie
                # facet generator hands out facet element labels in exactly that order.
                for meid, faceID in sorted(pairs):
                    if meid in self._eidToEl:
                        byFace[faceID].append(self._eidToEl[meid])
                model.surfaces[surfaceName].replaceData({f: els for f, els in byFace.items()})

        with timeit("sets & fields sync"):
            # Tracked (non-all) node sets that gain nodes are rebuilt with the new members (excluding
            # hanging slave nodes, whose motion is set by the MPC).
            slaves = set(records.keys())
            for setName, labels in mesh.nodeSets.items():
                present = {n.label for n in model.nodeSets[setName].nodes}
                if any(label not in present and label not in slaves for label in labels):
                    members = [model.nodes[label] for label in sorted(labels) if label not in slaves]
                    model.nodeSets[setName].replaceMembers(members)
                    change.changedNodeSets.add(setName)

            # sync all element sets (user sets like 'concrete' and all-encompassing sets) -- Finding 1
            allNodes = list(model.nodes.values())
            if newNodes:
                for setName in self._allLikeSets | {"all"}:
                    model.nodeSets[setName].replaceMembers(allNodes)
                    change.changedNodeSets.add(setName)
            for setName, eids in mesh.elementSets.items():
                if setName in model.elementSets:
                    if eids & newChildEids:
                        change.changedElementSets.add(setName)
                    # sorted, for the same reason as the surface sync above: this order becomes the
                    # element set's member order, which downstream generators number entities by
                    elements = [self._eidToEl[eid] for eid in sorted(eids) if eid in self._eidToEl]
                    # carry the non-mirrored members along: the octree only knows refineable elements,
                    # so a mixed set would silently drop them here. Members deleted from the model in
                    # the meantime are filtered out by their label
                    elements += [el for el in self._untrackedOfElementSet[setName] if el.elNumber in model.elements]
                    model.elementSets[setName].replaceMembers(elements)
            model.elementSets["all"].replaceMembers(list(model.elements.values()))
            change.changedElementSets.add("all")

        with timeit("fields resize & restore"):
            # resize node fields in place to include the new nodes, then restore the warm start:
            # converged values on the retained nodes and interpolated values on the new nodes (Finding 1).
            # Both U (current) and P (previous converged) get the same warm-start value, so the first
            # Newton iteration after refinement sees a normal residual rather than a spurious dU = U - P
            # = U - 0 cold-restart spike on every retained/new node (P-field warm-start fix).
            model._resizeNodeFieldsForNodes(self._journal)
            for fieldName, nodeField in model.nodeFields.items():
                if "U" not in nodeField:
                    nodeField.createFieldValueEntry("U")
                if "P" not in nodeField:
                    nodeField.createFieldValueEntry("P")
                U = nodeField["U"]
                P = nodeField["P"]
                old = oldValues.get(fieldName, {})
                new = newValues.get(fieldName, {})
                for node in nodeField.nodes:
                    idx = nodeField._indicesOfNodesInArray[node]
                    if node in old:
                        U[idx] = old[node]
                        P[idx] = old[node]
                    elif node in new:
                        U[idx] = new[node]
                        P[idx] = new[node]

            model._linkFieldVariableObjects(model.nodeSets["all"])
        return change

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """This instance's refinement history: every committed occasion's marked element labels
        (flattened CSR-style, since occasions have varying size), the not-yet-refined pending
        labels, and the cutback-reentry guard. ``None`` if nothing has ever happened (no
        refinement, nothing pending) -- see :meth:`ModelModifierBase.getRestartData`.

        Deliberately does not store node coordinates, connectivity, or hanging-node records --
        :meth:`setRestartData` rederives all of that deterministically by replaying each occasion
        through :meth:`_refineAndMaterialize`, the same mechanics a live run uses.
        """

        if not self._committedOccasions and not self._pendingMarkedElements:
            return None

        occasionSizes = [len(labels) for labels in self._committedOccasions]
        occasionLabels = [label for labels in self._committedOccasions for label in labels]
        occasionEids = [eid for eids in self._committedOccasionEids for eid in eids]
        pendingLabels = [el.elNumber for el in self._pendingMarkedElements]

        # Material state (quadrature-point history) of every currently-materialized leaf element,
        # keyed by the octree element id (eid) rather than the element number. AMR child element
        # numbers are NOT reproducible across a restart replay -- contact/tie facet elements claim
        # element labels between refinements, so the running label counter interleaves differently --
        # which would leave every refined child at a number FEModel.readRestart's number-keyed state
        # restore cannot match, i.e. restored virgin. The eid IS reproducible (topology is
        # byte-identical given the replayed occasions), so setRestartData restores by it instead.
        stateEids = []
        stateSizes = []
        stateChunks = []
        for eid, el in self._eidToEl.items():
            try:
                sv = np.asarray(el.getStateVars(), dtype=float).ravel()
            except NotImplementedError:
                continue
            stateEids.append(int(eid))
            stateSizes.append(sv.size)
            stateChunks.append(sv)
        stateData = np.concatenate(stateChunks) if stateChunks else np.zeros(0, dtype=float)

        return {
            "occasionSizes": np.array(occasionSizes, dtype=int),
            "occasionLabels": np.array(occasionLabels, dtype=int),
            "occasionEids": np.array(occasionEids, dtype=int),
            "pendingLabels": np.array(pendingLabels, dtype=int),
            "lastRefinedTime": np.array([self._lastRefinedTime if self._lastRefinedTime is not None else np.nan]),
            "stateEids": np.array(stateEids, dtype=int),
            "stateSizes": np.array(stateSizes, dtype=int),
            "stateData": stateData,
        }

    def _replayOccasionsByEid(self, model: FEModel, data: dict[str, np.ndarray]) -> None:
        """Reconstruct the refinement history from recorded octree eids.

        Refines the octree mirror for every occasion first -- pure topology, no model objects, so
        none of the per-occasion model-side cost is paid -- then materialises the model once per
        refinement *level*. The 2:1 balance stays per occasion because a later occasion's eids were
        chosen in a mesh that had already been balanced, so batching it could change the topology.
        """
        offset = 0
        occasions = []
        for size in data["occasionSizes"]:
            occasions.append([int(eid) for eid in data["occasionEids"][offset : offset + int(size)]])
            offset += int(size)

        for eids in occasions:
            for eid in eids:
                if self._mesh.elements[eid]["active"]:
                    self._mesh.refine(eid)
            self._mesh.balance_2to1()

        # Materialise level by level: _materialize only takes children whose parent is already
        # materialised, so one pass per refinement level is enough (maxLevel bounds the count).
        passes = 0
        while set(self._mesh.active()) - set(self._eidToEl.keys()):
            with timeit("hanging nodes"):
                records = self._mesh.hanging_mpc_records()
            with timeit("materialize"):
                change = self._materialize(model, records)
            self._hanging.setRecords(records)
            with timeit("notify observers"):
                model.notifyModelChanged(ModelChangeType.REFINEMENT, change)
            passes += 1
            if not change.addedElements or passes > 64:
                break

        self._committedOccasions = [list(labels) for labels in self._splitByOccasion(data, "occasionLabels")]
        self._committedOccasionEids = occasions
        self._journal.message(
            "AMR ModelModifier: replayed {:} occasion(s) from eids in {:} materialisation pass(es), "
            "active elements {:}".format(len(occasions), passes, len(self._mesh.active())),
            "hadaptivity",
            0,
        )

    def _splitByOccasion(self, data: dict[str, np.ndarray], key: str):
        """Split a flattened per-occasion array back into one list per occasion."""
        out = []
        offset = 0
        for size in data["occasionSizes"]:
            out.append([int(v) for v in data[key][offset : offset + int(size)]])
            offset += int(size)
        return out

    def setRestartData(self, model: FEModel, data: dict[str, np.ndarray]) -> None:
        """Replay every committed occasion, in order, through :meth:`_refineAndMaterialize` --
        exactly the mechanics :meth:`updateModel` uses live, just fed a recorded decision instead
        of evaluating markers -- to reconstruct this instance's topology, then restore the pending
        marks and the cutback guard.

        Must run before :meth:`~edelweissfe.models.femodel.FEModel.readRestart` restores node
        fields/element state variables (see :meth:`ModelModifierBase.setRestartData`): the elements
        this replays are not in ``model.elements`` yet otherwise, so that restore would silently
        skip them.
        """

        # A checkpoint only ever exists after at least one increment converged (checkpoints are
        # written from finalizeIncrement), which means updateModel already ran at least once
        # before it was written -- so this is never truly a first call. Without this, the live
        # path's initialOnly markers would re-evaluate on the next updateModel call as if it were,
        # redundantly re-marking whatever they select (harmless today only because a refined
        # parent is no longer in self._eidToEl for the marker to re-select, and maxLevel/eligibility
        # filtering catches the rest -- not a guarantee every marker implementation shares).
        self._isFirstCall = False

        offset = 0
        self._replayMode = True
        try:
            if "occasionEids" in data and len(data["occasionEids"]):
                self._replayOccasionsByEid(model, data)
            else:
                # legacy checkpoint (no eids recorded): per-occasion materialisation
                for size in data["occasionSizes"]:
                    labels = data["occasionLabels"][offset : offset + int(size)]
                    offset += int(size)
                    eligible = [model.elements[int(label)] for label in labels]
                    self._refineAndMaterialize(model, eligible)
        finally:
            self._replayMode = False

        # Every element this modifier manages had its state restored above by octree eid; tell
        # FEModel.readRestart not to restore them again by (renumbered) element number.
        self.restoredElementLabels = frozenset(el.elNumber for el in self._eidToEl.values())

        self._pendingMarkedElements = {model.elements[int(label)] for label in data["pendingLabels"]}
        lastRefinedTime = float(data["lastRefinedTime"][0])
        self._lastRefinedTime = None if np.isnan(lastRefinedTime) else lastRefinedTime

        # Restore refined elements' material history by octree eid (see getRestartData). Element
        # numbers assigned during the replay above do not match those at checkpoint time, so
        # FEModel.readRestart's number-keyed restore leaves these children virgin; this runs after the
        # full replay, when self._eidToEl maps each checkpointed eid to its (renumbered) live element.
        if "stateEids" in data and len(data["stateEids"]):
            sizes = data["stateSizes"]
            flat = data["stateData"]
            offset = 0
            for i, eid in enumerate(data["stateEids"]):
                size = int(sizes[i])
                chunk = flat[offset : offset + size]
                offset += size
                el = self._eidToEl.get(int(eid))
                if el is not None:
                    el.setStateVars(np.ascontiguousarray(chunk, dtype=float))
