Creating finite element meshes
==============================

.. _mesh_to_elements:

The mesh is data; elements are made from it
-------------------------------------------

A model is built in two stages, as in the textbook picture of the finite element method:

1. **Describe the mesh.** The ``*node``, ``*element``, ``*nset``, ``*elset`` and ``*surface``
   keywords below, and every :doc:`mesh generator <generators>`, fill the
   :class:`~edelweissfe.models.mesh.Mesh` held by the model as ``model.mesh``. An element is a record:
   its number, type, provider and node labels. An element set is an ordered list of element numbers,
   and a surface names the element set of each of its faces. Nodes are created right away, as
   :class:`~edelweissfe.points.node.Node` objects, and node sets as
   :class:`~edelweissfe.sets.nodeset.NodeSet` objects.
2. **Make the elements.** Once the mesh is described,
   :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh` creates an element object for
   every element of the mesh -- in the order of the mesh -- and resolves the element sets and surfaces
   to those objects. Sections, element properties, contact facets, constraints and model modifiers
   come after and refer to the element objects through the sets.

The fields at the nodes -- and with them the layout of the degrees of freedom -- follow from the mesh
alone: which fields an element has at which node depends on its type, which the mesh asks one
prototype element per type for (:meth:`~edelweissfe.models.mesh.Mesh.typeOf`). Element numbers are
those of the input file, or are drawn from the model's number allocator
(:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveElementNumbers`) by generators,
contact facets and adaptive refinement alike.

A model can be built the same way from Python:

.. code-block:: python

    from edelweissfe.models.femodel import FEModel, everyElement

    model = FEModel(2)
    for label, coordinates in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], start=1):
        model.nodes[label] = Node(label, np.array(coordinates, dtype=float))

    with model.topology.changes():
        model.mesh.addElement(1, "CPE4", "edelweiss", [1, 2, 3, 4])
        model.mesh.setElementSet("all", [1])
        model.createElementsOfMesh(everyElement)

The predicate of :meth:`~edelweissfe.models.femodel.FEModel.createElementsOfMesh` decides which
elements a process creates; see :ref:`domaindecomposition_mesh_to_elements` for why that matters.

The Abaqus-like keywords
------------------------

Relevant module ``edelweissfe.generators.abqmodelconstructor``

.. automodule:: edelweissfe.generators.abqmodelconstructor
   :members: __doc__

.. literalinclude:: ../../../testfiles/marmot/LinearElasticIsotropic/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/LinearElasticIsotropic/test.inp``
