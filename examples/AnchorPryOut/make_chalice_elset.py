#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the ``chalice_refine`` element set: the initial-refinement domain for the explicit
pry-out run.

WHY THIS EXISTS
---------------
The explicit production run ``run_production_v13`` refined the concrete with a *live* marker on
principal stress (67 refinement events over 20 h). Two problems with that:

1. Every mid-run refinement introduces a fresh hanging-node transition *into an already-damaged
   stress field*, so the damage pattern near a transition is not separable from the transition
   itself.
2. The Ensight export of that run is unusable (see ``PLAN_EXPLICIT_PRYOUT.md``): after the first
   live refinement every variable file is shorter than the geometry part it is painted onto, so
   the exported damage field is not aligned with the mesh at all.

This script replaces the live marker with a single, geometry-driven, ``initialOnly`` refinement of
a pry-out-cone-shaped ("chalice") domain about the anchor axis. The mesh is then fixed for the
whole analysis: no mid-run topology change, one transition, placed deliberately outside the
expected breakout cone, and present from increment 0 so its influence is part of the initial
condition rather than an event.

THE DOMAIN
----------
Axis is ``y`` (the anchor axis), origin on the concrete top surface, model is the ``z <= 0`` half
about the ``z = 0`` symmetry plane. Radius is measured about the anchor axis,
``r = sqrt(x**2 + z**2)``. The domain is the union of

* the **cone** (the chalice bowl): ``r <= R_RIM * (y - Y_APEX) / (Y_TOP - Y_APEX)`` over
  ``Y_APEX <= y <= Y_TOP``, i.e. a straight-walled truncated cone from the anchor's embedded tip
  to the concrete surface, and
* the **stem**: ``r <= R_STEM`` over ``Y_STEM <= y <= Y_TOP``, a cylindrical core that keeps the
  borehole/mortar interface refined over the full embedment AND extends two element layers below
  the anchor tip -- deliberately, so that the refinement transition does not sit on the tip stress
  concentration, which is the same mistake in a new place.

An element is marked if **any** of its 20 nodes falls inside the domain, so the refined region
fully contains the analytic domain rather than approximating it.

GEOMETRY AS MESHED (measured, not assumed -- see ``PLAN_EXPLICIT_PRYOUT.md``)
    anchor      y in [-80, +41.8], r = 10       -> embedded tip at y = -80
    mortar      y in [-80, 0],     r = 11       -> bonded embedment is 80 mm
    concrete    500 x 500 x 240, y in [-240, 0], uniform 7.273 mm y-extent, radial 3 .. 64 mm

Run from this directory; writes ``chalice_refine_block.txt`` next to the input file.
"""

import numpy as np

MESH = "./mesh/concrete_edelweissfe.inp"
OUT = "./chalice_refine_block.txt"
ELSET_NAME = "chalice_refine"

#: Cone rim radius at the concrete surface. 1.5 * h_ef with the as-meshed h_ef = 80 mm, i.e. the
#: full standard concrete-cone projection radius -- chosen over the 2.5 * 70 / 2 = 87.5 mm the
#: original request named so that the refinement transition sits *outside* any credible breakout
#: cone rather than inside it.
R_RIM = 120.0
#: Cone apex depth: the anchor's embedded tip.
Y_APEX = -80.0
#: Concrete top surface.
Y_TOP = 0.0
#: Stem radius. Covers the borehole (r = 11) plus margin, so the transition is never on the
#: mortar/concrete tie interface.
R_STEM = 30.0
#: Stem bottom: two 7.273 mm element layers below the anchor tip.
Y_STEM = -94.6


def readConcreteMesh(fileName: str):
    """Read node coordinates and the ``concrete`` element connectivity from an Abaqus-style mesh.

    Returns
    -------
    tuple
        ``(labels, coordinates)`` -- element labels as an ``(nEl,)`` int array, and their node
        coordinates as an ``(nEl, 20, 3)`` float array.
    """
    nodes = {}
    rows = []
    mode = None
    for line in open(fileName):
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        if stripped.startswith("*"):
            upper = stripped.upper()
            mode = "node" if upper.startswith("*NODE") else ("element" if upper.startswith("*ELEMENT") else None)
            continue
        if mode == "node":
            parts = stripped.split(",")
            nodes[int(parts[0])] = [float(x) for x in parts[1:4]]
        elif mode == "element":
            rows.append([int(x) for x in stripped.replace(",", " ").split()])

    coordinateOfLabel = np.full((max(nodes) + 1, 3), np.nan)
    for label, coordinate in nodes.items():
        coordinateOfLabel[label] = coordinate

    labels = np.array([row[0] for row in rows])
    connectivity = np.array([row[1:] for row in rows])
    return labels, coordinateOfLabel[connectivity]


def isInChalice(coordinates: np.ndarray) -> np.ndarray:
    """Point-in-domain test for the chalice, evaluated on every node of every element.

    Parameters
    ----------
    coordinates
        ``(nEl, nNodesPerEl, 3)`` nodal coordinates.

    Returns
    -------
    np.ndarray
        ``(nEl, nNodesPerEl)`` boolean mask, True where the node is inside the domain.
    """
    radius = np.hypot(coordinates[:, :, 0], coordinates[:, :, 2])
    height = coordinates[:, :, 1]

    coneFraction = np.clip((height - Y_APEX) / (Y_TOP - Y_APEX), 0.0, 1.0)
    inCone = (height >= Y_APEX) & (height <= Y_TOP) & (radius <= R_RIM * coneFraction)
    inStem = (height >= Y_STEM) & (height <= Y_TOP) & (radius <= R_STEM)
    return inCone | inStem


def writeElSet(fileName: str, name: str, labels: np.ndarray, header: str) -> None:
    """Write an Abaqus-style ``*ELSET`` block, 10 labels per line, matching the house style of the
    other generated include blocks in this directory."""
    with open(fileName, "w") as f:
        f.write(header)
        f.write(f"*ELSET, ELSET={name}\n")
        for start in range(0, len(labels), 10):
            f.write(", ".join(str(label) for label in labels[start : start + 10]) + ",\n")


if __name__ == "__main__":
    labels, coordinates = readConcreteMesh(MESH)
    marked = isInChalice(coordinates).any(axis=1)
    markedLabels = np.sort(labels[marked])

    cornerCoordinates = coordinates[marked][:, :8, :]
    centroids = cornerCoordinates.mean(axis=1)
    centroidRadius = np.hypot(centroids[:, 0], centroids[:, 2])
    refinedExtent = coordinates[marked].max(axis=1) - coordinates[marked].min(axis=1)

    header = (
        "**\n"
        f"** Initial-refinement element set '{ELSET_NAME}' -- GENERATED, do not edit by hand.\n"
        f"** Regenerate with: python make_chalice_elset.py\n"
        "**\n"
        "** Pry-out-cone ('chalice') shaped domain about the anchor axis, for a single initialOnly\n"
        "** h-adaptivity pass. See make_chalice_elset.py for the geometry and the reasoning.\n"
        "**\n"
        f"**   cone   r <= {R_RIM:.1f} * (y {Y_APEX:+.1f}) / {Y_TOP - Y_APEX:.1f}   for {Y_APEX:.1f} <= y <= {Y_TOP:.1f}\n"
        f"**   stem   r <= {R_STEM:.1f}                        for {Y_STEM:.1f} <= y <= {Y_TOP:.1f}\n"
        f"**   marked if ANY of an element's 20 nodes is inside\n"
        "**\n"
        f"** {len(markedLabels)} of {len(labels)} concrete elements "
        f"({100.0 * len(markedLabels) / len(labels):.1f} %).\n"
        "**\n"
    )
    writeElSet(OUT, ELSET_NAME, markedLabels, header)

    print(f"marked {len(markedLabels)} of {len(labels)} concrete elements " f"({100.0 * len(markedLabels) / len(labels):.1f} %)")
    print(f"  centroid radius   {centroidRadius.min():7.2f} .. {centroidRadius.max():7.2f} mm")
    print(f"  y extent          {coordinates[marked][:, :, 1].min():7.2f} .. {coordinates[marked][:, :, 1].max():7.2f} mm")
    print(f"  smallest marked element extents (x,y,z) {np.round(refinedExtent.min(axis=0), 3)}")
    print(f"  -> after splitFactor=2 the smallest becomes {np.round(refinedExtent.min(axis=0) / 2, 3)}")
    print(f"  active elements after refinement: {len(labels)} -> {len(labels) + 7 * len(markedLabels)}")
    print(f"wrote {OUT}")
