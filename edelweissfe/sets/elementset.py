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
#  Alexander Dummer alexander.dummer@uibk.ac.at
#  Paul Hofer Paul.Hofer@uibk.ac.at
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

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.utils.misc import checkSuccessfulExtension

if checkSuccessfulExtension("edelweissfe.elements.marmotelement.element"):
    from edelweissfe.elements.marmotelement.element import MarmotElementWrapper
else:
    MarmotElementWrapper = None

if checkSuccessfulExtension("edelweissfe.materials.marmot.marmothypoelastic") or checkSuccessfulExtension(
    "edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic"
):
    # MarmotMaterialWrappingElement can drive either point-wise material family; each has its
    # own separately compiled extension, and only one of the two building is enough for the
    # element to be partially usable (see materialdrivers.py's createMaterial methods for the
    # per-family lazy import that fails, cleanly, if its own extension isn't built).
    from edelweissfe.elements.marmotsingleqpelement.element import (
        MarmotMaterialWrappingElement,
    )
else:
    MarmotMaterialWrappingElement = None

from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.sets.orderedset import OrderedSet
from edelweissfe.utils.exceptions import TopologyError
from edelweissfe.utils.meshtools import extractNodesFromElementSet


class ElementSet(OrderedSet):
    """A basic element set.
    It has a label, and a list containing unique elements.

    An element set holds the elements of the set that were **created here**: the model describes its
    mesh as data first and then creates the element objects from it
    (:meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`). In a serial run that is every
    element of the set. A domain-decomposed run may create only part of the mesh in each process; a
    set then holds only its part, and says so through :attr:`isComplete`. Code that needs the whole set
    -- an output, a checkpoint, a marker -- states it by calling :meth:`requireComplete`.

    Parameters
    ----------
    name
        The unique label for this element set.
    elements
        A list of elements.
    mesh
        The :class:`~edelweissfe.models.mesh.Mesh` describing this set under the same name, if the set
        was resolved from a mesh; :attr:`isComplete` is derived from it.
    nodes
        The nodes of the model, by label, if the set was resolved from a mesh: the nodes of the whole
        set (:meth:`extractNodeSet`) are found from them even where only part of the set was
        created.
    """

    def __init__(
        self,
        label: str,
        elements,
        mesh=None,
        nodes: dict = None,
    ):
        self.allowedObjectTypes = [BaseElement]
        self.allowedObjectTypes.append(MarmotElementWrapper) if MarmotElementWrapper is not None else None
        (
            self.allowedObjectTypes.append(MarmotMaterialWrappingElement)
            if MarmotMaterialWrappingElement is not None
            else None
        )

        super().__init__(label, elements)
        self._nodes = None
        #: The mesh describing this set, or None for a set not resolved from a mesh.
        self.mesh = mesh
        #: The nodes of the model, by label, for a set resolved from a mesh.
        self.modelNodes = nodes

        self.elements = self.items

    @property
    def isComplete(self) -> bool:
        """True if this set holds every element of its set in the mesh (always, for a set not resolved
        from a mesh). Derived, not stored, so that no change of the members can leave it stale. Counting
        suffices, since a set resolved from the mesh holds only elements of its set there."""

        if self.mesh is None:
            return True
        numbers = self.mesh.elementSets.get(self.name)
        return numbers is None or len(self.data) == len(numbers)

    def requireComplete(self, reader: str):
        """State that ``reader`` needs every element of this set, not only the part created here.

        A partial set read as if it were whole gives silently wrong results -- an output, a marker
        or a checkpoint computed from part of the set. Every whole-set reader therefore calls this,
        so that such a reading fails loudly instead.

        Parameters
        ----------
        reader
            Who reads the set, for the error message.

        Raises
        ------
        TopologyError
            If this process created only some of the elements of the set.
        """

        if not self.isComplete:
            raise TopologyError(
                "{:} needs the whole element set {:}, but only part of it was created in this process".format(
                    reader, self.name
                )
            )

    def extractNodeSet(
        self,
    ):
        """The nodes of the whole set, without duplicates, in the order the elements list them.

        For a set of which only part was created here, they are taken from the mesh, which
        describes the whole set: the nodes are the same in every process.

        Returns
        -------
        NodeSet
            The nodes.
        """
        if not self._nodes:
            if self.isComplete:
                self._nodes = extractNodesFromElementSet(self)
            else:
                labels = dict.fromkeys(
                    label
                    for number in self.mesh.elementSets[self.name]
                    for label in self.mesh.elements[number].nodeLabels
                )
                self._nodes = NodeSet(self.name, [self.modelNodes[label] for label in labels])
        return self._nodes

    def replaceMembers(self, item_s):
        """Replace all members in-place (see :meth:`OrderedSet.replaceMembers`), additionally
        invalidating the cached :meth:`extractNodeSet` result, which is stale once the element
        membership changes."""
        super().replaceMembers(item_s)
        self._nodes = None
