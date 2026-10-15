Sets
====

Currently, EdelweissFE supports nodesets and elementsets.
Since for the usage of nodesets or elementsets
in the outputmanagers or constraints
the ordering of the respective objects is of importance
they are based on a custom ordered set object.

An element set of a model (:class:`~edelweissfe.sets.elementset.ElementSetOfMesh`) holds the
elements of its set in the mesh (``model.mesh.elementSets``), and a node set of the mesh
(:class:`~edelweissfe.sets.nodeset.NodeSetOfMesh`) the :class:`~edelweissfe.points.node.Node`
objects of its set in the mesh (``model.mesh.nodeSets``). A set of the objects it was given -- e.g.
the slave nodes a tie tied -- holds exactly those. See :ref:`mesh_to_elements`.

In a domain-decomposed run (:doc:`domaindecomposition`), a process may hold only some of the elements
of the mesh: its *local* ones. Element sets and node sets are therefore local ordered sets
(:class:`~edelweissfe.sets.orderedset.LocalOrderedSet`): a set of the mesh holds the local members of
its set, and is *complete* if that is all of them -- in a serial run, always. Reading a set that is
not complete as a whole (iterating it, indexing it, ``len()``) raises; a reader of the local part asks
for :meth:`~edelweissfe.sets.elementset.ElementSet.localElements` or
:meth:`~edelweissfe.sets.nodeset.NodeSet.localNodes`. A set of the objects it was given is always
complete.

Relevant modules:

 * ``edelweissfe.sets.orderedset``
 * ``edelweissfe.sets.nodeset``
 * ``edelweissfe.sets.elementset``

.. automodule:: edelweissfe.sets.orderedset
   :members:

.. automodule:: edelweissfe.sets.nodeset
   :members:

.. automodule:: edelweissfe.sets.elementset
   :members:
