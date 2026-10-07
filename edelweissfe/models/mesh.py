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
"""The mesh of a model, as data.

``model.mesh`` holds the element connectivity, the element sets and the surfaces. The nodes are
``model.nodes`` (they are not part of the mesh yet), and ``model.elements`` are the element objects
created from the mesh -- never assigned directly.

A model is built in two stages:

1. **Describe the mesh.** The ``*element``/``*elset``/``*surface`` keywords and the mesh generators
   fill the :class:`Mesh` held by the model as ``model.mesh``: per element its number, type, provider
   and node labels; element sets as ordered lists of element numbers; surfaces as element faces. No
   element object exists yet.
2. **Make the elements.** :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh` creates an
   element object for each element of the mesh, and resolves the element sets and surfaces to those
   objects.

Nodes are not part of this description: they are created directly as :class:`~edelweissfe.points.node.Node`
objects in ``model.nodes``, and node sets as :class:`~edelweissfe.sets.nodeset.NodeSet` objects, as before.

Which fields an element has at which of its nodes depends on its type only. The mesh therefore
answers that question per element without any element object of the model, by asking one prototype
element per type (:meth:`Mesh.typeOf`). That is what lets the model activate the fields at its
nodes -- and so lay out its degrees of freedom -- from the mesh alone.
"""

from dataclasses import dataclass

from edelweissfe.utils.exceptions import TopologyError

#: The provider of every auxiliary element (:meth:`Mesh.addAuxiliaryElement`): a contact facet or a
#: point mass, which a model entity makes itself and no provider creates from its record.
AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class ElementTypeInfo:
    """What the mesh needs to know about an element type, without an element of the model.

    Parameters
    ----------
    fields
        The fields of the element, per node: ``fields[i]`` lists the fields at its ``i``-th node.
    ensightType
        The element shape, e.g. ``hexa20``; it determines the faces of the element.
    nDof
        The number of degrees of freedom of the element.
    hasKernels
        Whether the element computes forces and carries a state (see
        :attr:`~edelweissfe.elements.base.baseelement.BaseElement.hasKernels`); a contact facet does not.
    """

    fields: list
    ensightType: str
    nDof: int
    hasKernels: bool


@dataclass(frozen=True)
class SurfaceFace:
    """The elements exposing one face (by face number) of an element-based surface.

    A surface is described by the element set of each face, so a face names an element set of the
    mesh. A model modifier that refines the surface replaces the elements of a face by their children,
    which form no named set: such a face lists its element numbers instead. Exactly one of the two is
    given.

    Parameters
    ----------
    elementSetName
        The name of the element set of the mesh whose elements expose the face.
    elementNumbers
        The numbers of the elements exposing the face, in order.
    """

    elementSetName: str = None
    elementNumbers: tuple = None

    def elementNumbersIn(self, mesh: "Mesh") -> list:
        """The numbers of the elements exposing the face.

        Parameters
        ----------
        mesh
            The mesh holding the element set the face may name.

        Returns
        -------
        list
            The element numbers, in order.
        """

        if self.elementSetName is not None:
            return mesh.elementSets[self.elementSetName]
        return list(self.elementNumbers)


class MeshElement:
    """One element of the mesh, as data.

    Parameters
    ----------
    number
        The element number.
    elType
        The element type, e.g. ``C3D20R``.
    provider
        The element provider, e.g. ``marmot``; ``None`` for the default provider; :data:`AUXILIARY`
        for an auxiliary element (see :meth:`Mesh.addAuxiliaryElement`), which cannot be created
        from this record.
    nodeLabels
        The labels of the element's nodes, in the element's node order, as a tuple of ints.
    ownTypeInfo
        The type information of an auxiliary element; ``None`` for an element whose type
        alone determines it.
    hostElement
        For an auxiliary element, the number of its host element: the element of the mesh
        it lies on (the solid element whose face a contact facet tiles); ``None`` if it has none (the point mass
        of a rigid body), and for every other element.
    """

    __slots__ = ("number", "elType", "provider", "nodeLabels", "ownTypeInfo", "hostElement")

    def __init__(
        self,
        number: int,
        elType: str,
        provider: str,
        nodeLabels: tuple,
        ownTypeInfo: ElementTypeInfo = None,
        hostElement: int | None = None,
    ):
        self.number = number
        self.elType = elType
        self.provider = provider
        self.nodeLabels = nodeLabels
        self.ownTypeInfo = ownTypeInfo
        self.hostElement = hostElement

    @property
    def isAuxiliary(self) -> bool:
        """True for an auxiliary element: one a model entity (a contact surface, a rigid body) made
        itself, e.g. a contact facet or a point mass; it cannot be created from this record. Its
        provider says so (:data:`AUXILIARY`)."""
        return self.provider == AUXILIARY


class Mesh:
    """The mesh of a model: its element connectivity, element sets and surfaces, as data. The nodes
    are held by the model (``model.nodes``), not here.

    Element numbers are never chosen here: the input file gives them, or the model's number
    allocator (:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveElementNumbers`)
    hands them out, so that the mesh, the contact facets and every model modifier share one counter.
    """

    def __init__(self):
        #: The elements, by number, in the order they were described.
        self.elements = {}
        #: The element sets, by name: ordered lists of element numbers without duplicates.
        self.elementSets = {}
        #: The element-based surfaces, by name: per face number, a :class:`SurfaceFace`.
        self.surfaces = {}
        #: Element class and type information per (type, provider), asked once per type.
        self._types = {}

    def addElement(self, number: int, elType: str, provider: str, nodeLabels) -> MeshElement:
        """Describe an element.

        Parameters
        ----------
        number
            The element number, from the input file or the model's number allocator.
        elType
            The element type.
        provider
            The element provider; ``None`` selects the default provider (``marmot``). Not
            :data:`AUXILIARY`: an auxiliary element is described from its object
            (:meth:`addAuxiliaryElement`).
        nodeLabels
            The labels of the element's nodes.

        Returns
        -------
        MeshElement
            The record of the element.
        """

        if provider == AUXILIARY:
            raise TopologyError(
                "element {:}: an auxiliary element is described from its object (Mesh.addAuxiliaryElement)".format(
                    number
                )
            )
        return self._describe(MeshElement(number, elType, provider, tuple(nodeLabels)))

    def _describe(self, record: MeshElement) -> MeshElement:
        """Add the record of an element under its number, which must be new.

        Parameters
        ----------
        record
            The record.

        Returns
        -------
        MeshElement
            The record.
        """

        if record.number in self.elements:
            raise TopologyError(
                "element number {:} is already taken in the mesh -- every element number is described once, "
                "and numbers from TopologyPipeline.reserveElementNumbers() are never recycled".format(record.number)
            )
        self.elements[record.number] = record
        return record

    def addAuxiliaryElement(self, element, hostElement: int | None = None) -> MeshElement:
        """Describe an auxiliary element, from the element object: one a model entity (a contact
        surface, a rigid body) makes itself, instead of the input describing it in the mesh.

        Contact facets and the point masses of rigid bodies are made by the surface or rigid body they belong to
        (with data an element type alone does not carry: a parent face, a mass), on every process.
        They are part of the mesh all the same -- they have numbers, nodes and fields -- so they are
        described here from the object, including the fields it reports.

        Parameters
        ----------
        element
            The element object.
        hostElement
            The number of its host element, the element of the mesh it lies on, or None if it has none.

        Returns
        -------
        MeshElement
            The record of the element.
        """

        return self._describe(
            MeshElement(
                element.elNumber,
                element.elType,
                AUXILIARY,
                tuple(node.label for node in element.nodes),
                ElementTypeInfo(element.fields, element.ensightType, element.nDof, element.hasKernels),
                hostElement,
            )
        )

    def removeElement(self, number: int):
        """Remove an element from the mesh, and from every element set and surface; see
        :meth:`removeElements`.

        Parameters
        ----------
        number
            The element number.
        """

        self.removeElements((number,))

    def removeElements(self, numbers):
        """Remove elements from the mesh, and from every element set and every surface listing them,
        so that a set or a surface never names an element the mesh no longer describes. Their
        numbers are retired, never reissued.

        The sets and surfaces keep their other members, in their order, and their identity: a set
        is changed in place. A surface whose face names an element set follows that set.

        Parameters
        ----------
        numbers
            The element numbers; removed in one pass over the sets, which a model modifier removing
            many elements at once should use.
        """

        removed = set(numbers)
        for number in removed:
            del self.elements[number]

        for members in self.elementSets.values():
            if not removed.isdisjoint(members):
                members[:] = [number for number in members if number not in removed]

        for faces in self.surfaces.values():
            for face, surfaceFace in faces.items():
                listed = surfaceFace.elementNumbers
                if listed is not None and not removed.isdisjoint(listed):
                    faces[face] = SurfaceFace(
                        elementNumbers=tuple(number for number in listed if number not in removed)
                    )

    def setElementSet(self, name: str, numbers):
        """Define (or redefine) an element set.

        Parameters
        ----------
        name
            The name of the set.
        numbers
            The element numbers, in order; repeated numbers are kept once, at their first position.
        """

        numbers = list(dict.fromkeys(int(number) for number in numbers))
        missing = [number for number in numbers if number not in self.elements]
        if missing:
            raise KeyError(
                "element set {:}: element(s) {:} are not in the mesh".format(name, ", ".join(map(str, missing[:10])))
            )
        self.elementSets[name] = numbers

    def addSurface(self, name: str, faceToElementSetName: dict):
        """Define an element-based surface.

        Parameters
        ----------
        name
            The name of the surface.
        faceToElementSetName
            Per face number, the name of the element set of this mesh whose elements expose that face.
        """

        for setName in faceToElementSetName.values():
            if setName not in self.elementSets:
                raise KeyError("surface {:}: element set {:} is not in the mesh".format(name, setName))
        self.surfaces[name] = {
            face: SurfaceFace(elementSetName=setName) for face, setName in faceToElementSetName.items()
        }

    def setSurfaceElements(self, name: str, faceToElementNumbers: dict):
        """Redefine a surface by the element numbers of each face, e.g. after a refinement replaced
        the elements of a face by their children.

        Parameters
        ----------
        name
            The name of the surface.
        faceToElementNumbers
            Per face number, the element numbers, in order.
        """

        self.surfaces[name] = {
            face: SurfaceFace(elementNumbers=tuple(numbers)) for face, numbers in faceToElementNumbers.items()
        }

    def elementNumbersOfSurface(self, name: str) -> dict:
        """The element numbers of each face of a surface.

        Parameters
        ----------
        name
            The name of the surface.

        Returns
        -------
        dict
            Per face number, the element numbers, in order.
        """

        return {face: surfaceFace.elementNumbersIn(self) for face, surfaceFace in self.surfaces[name].items()}

    def elementClassOf(self, record: MeshElement) -> type:
        """The class that creates the element of a record.

        Parameters
        ----------
        record
            The record of the element.

        Returns
        -------
        type
            The element class.
        """

        if record.isAuxiliary:
            raise TopologyError(
                "element {:} ({:}) is an auxiliary element, made by a model entity, and cannot be created from the mesh".format(
                    record.number, record.elType
                )
            )
        return self._typeEntry(record.elType, record.provider)[0]

    def typeOf(self, record: MeshElement) -> ElementTypeInfo:
        """The fields per node, the shape and the size of an element, without an element object of the
        model.

        Parameters
        ----------
        record
            The record of the element.

        Returns
        -------
        ElementTypeInfo
            The type information.
        """

        if record.isAuxiliary:
            return record.ownTypeInfo
        return self._typeEntry(record.elType, record.provider)[1]

    def _typeEntry(self, elType: str, provider: str) -> tuple:
        """The element class and the type information of an element type, from one prototype
        element (:func:`~edelweissfe.config.elementlibrary.createPrototypeElement`) created once per
        (type, provider)."""

        key = (elType, provider)
        entry = self._types.get(key)
        if entry is None:
            from edelweissfe.config.elementlibrary import createPrototypeElement

            prototype = createPrototypeElement(elType, provider)
            entry = (
                type(prototype),
                ElementTypeInfo(prototype.fields, prototype.ensightType, prototype.nDof, prototype.hasKernels),
            )
            self._types[key] = entry
        return entry

    def __getstate__(self):
        """Pickle (and deep-copy) the data only; the prototype elements are recreated on demand."""

        state = self.__dict__.copy()
        state["_types"] = {}
        return state
