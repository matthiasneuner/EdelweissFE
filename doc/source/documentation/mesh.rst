Creating finite element meshes
==============================

.. _mesh_to_elements:

The mesh is data; elements are made from it
-------------------------------------------

What a model holds, up front:

* ``model.mesh`` (:class:`~edelweissfe.models.mesh.Mesh`) holds the **nodes, the node sets, the
  element connectivity, the element sets and the surfaces**, as data -- node labels and reference
  coordinates, node sets as lists of labels, element numbers, types and node labels -- not node or
  element objects.
* ``model.nodes`` holds the **node objects created from the mesh**
  (:class:`~edelweissfe.points.node.Node`), by
  :meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh`, in mesh order; ``model.nodeSets``
  the node sets of the mesh resolved to them (:class:`~edelweissfe.sets.nodeset.NodeSetOfMesh`).
* ``model.elements`` holds the **element objects created from the mesh**. It is never assigned
  directly: an element is described in ``model.mesh`` and created by
  :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`. Preparing the model checks both
  directions -- an element of the mesh never created, or an element of ``model.elements`` the mesh
  does not describe, raises a :class:`~edelweissfe.utils.exceptions.TopologyError` -- and the same for
  the nodes, which must moreover be held in mesh order
  (:meth:`~edelweissfe.models.femodel.FEModel.checkNodesAgreeWithMesh`). The only exception are
  *auxiliary* elements and nodes, which a model entity makes itself -- the contact facets of a
  surface, the point mass of a rigid body, the surface nodes and the reference point of a discrete
  rigid body -- and adds with :meth:`~edelweissfe.models.femodel.FEModel.createAuxiliaryElement` and
  :meth:`~edelweissfe.models.femodel.FEModel.createAuxiliaryNodes`, which describe them in the mesh as
  well.

A model is built in two stages, as in the textbook picture of the finite element method:

1. **Describe the mesh.** The ``*node``, ``*element``, ``*nset``, ``*elset`` and ``*surface``
   keywords below, and every :doc:`mesh generator <generators>`, fill the
   :class:`~edelweissfe.models.mesh.Mesh` held by the model as ``model.mesh``. A node is a label and
   its reference coordinates, held as arrays, one row per node; a node set is an ordered list of node
   labels. An element is a record: its number, type, provider and node labels. An element set is an
   ordered list of element numbers, and a surface names the element set of each of its faces.
2. **Make the nodes and the elements.** Once the mesh is described,
   :meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh` creates a node object for every
   node of the mesh -- in the order of the mesh, each with a copy of its reference coordinates -- and
   resolves the node sets to them; :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh`
   creates an element object for every element of the mesh -- in the order of the mesh -- and resolves
   the element sets and surfaces to those objects. Sections, element properties, contact facets,
   constraints and model modifiers come after and refer to the objects through the sets.

A generator that needs node objects while the mesh is still being described (``executePythonCode``,
``surfaceElementGenerator``, ``discreteRigidBodyGenerator``) makes the nodes described so far itself.
A model modifier that adds nodes (a refinement) describes them in the mesh and calls
:meth:`~edelweissfe.models.femodel.FEModel.createNodesOfMesh` again; one that changes the reference
geometry (a tie adjusting its slave nodes, a snap onto a surface) changes the coordinates in the mesh
too (:meth:`~edelweissfe.models.mesh.Mesh.moveNodes`). The current position of a node carried by a
rigid body is not reference geometry: it is held by the node object only.

The fields at the nodes -- and with them the layout of the degrees of freedom -- follow from the mesh
alone: which fields an element has at which node depends on its type, which the mesh asks one
prototype element per type for (:meth:`~edelweissfe.models.mesh.Mesh.typeOf`). The numbering of the
degrees of freedom can therefore be written down from the mesh, without node objects
(:class:`~edelweissfe.numerics.globaldofnumbering.GlobalDofNumbering`). Element numbers and node
labels are those of the input file, or are drawn from the model's number allocators
(:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveElementNumbers`,
:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveNodeNumbers`) by generators,
contact facets and adaptive refinement alike.

A model can be built the same way from Python:

.. code-block:: python

    from edelweissfe.models.femodel import FEModel

    model = FEModel(2)
    with model.topology.changes():
        model.mesh.addNodes([1, 2, 3, 4], [(0, 0), (1, 0), (1, 1), (0, 1)])
        model.mesh.setNodeSet("bottom", [1, 2])
        model.mesh.addElement(1, "CPE4", "edelweiss", [1, 2, 3, 4])
        model.mesh.setElementSet("all", [1])
        model.createNodesOfMesh()
        model.createElementsOfMesh()

Which elements :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh` creates, the model's
:attr:`~edelweissfe.models.femodel.FEModel.elementDistribution` decides -- every element, unless a
domain-decomposed run distributes the elements over its processes; see
:ref:`domaindecomposition_mesh_to_elements`.

.. automodule:: edelweissfe.models.elementdistribution
   :members:

.. automodule:: edelweissfe.numerics.globaldofnumbering
   :members:

The Abaqus-like keywords
------------------------

Relevant module ``edelweissfe.generators.abqmodelconstructor``

.. automodule:: edelweissfe.generators.abqmodelconstructor
   :members: __doc__

.. literalinclude:: ../../../testfiles/marmot/LinearElasticIsotropic/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/LinearElasticIsotropic/test.inp``
