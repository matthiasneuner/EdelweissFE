"""A finished model is released: nothing keeps it alive once the caller lets go of it.

A copy of a DOF vector used to be a view of a second DOF vector that owned the copied data. numpy hides
that reference (``base``) from the garbage collector, so the inner vector looked referenced from
outside, and everything its entity map reaches stayed alive (the closure of an expression field output
that captured ``{'model': model}`` was only one of the many references inside that cycle). With a rigid body, that is the whole model:
the point mass of the rigid body is in the entity map, and it refers to the model.
"""

import gc
import os
import shutil
import weakref

import numpy as np

from edelweissfe.numerics.dofvector import DofVector

_RIGID_BODY_DECK = os.path.join(
    os.path.dirname(__file__), "..", "testfiles", "edelweiss-only", "NEDSurfaceToDiscreteRigidBodyContact"
)


class _Entity:
    """An entity of a DOF vector that refers back to the vector, as a rigid body's point mass refers to
    the model holding the vector."""

    def __init__(self):
        self.vector = None


def test_a_copy_of_a_dof_vector_in_a_reference_cycle_is_collected():
    entity = _Entity()
    vector = DofVector(4)
    vector.entitiesInDofVector = {entity: np.array([0, 1])}
    entity.vector = vector.copy()
    entity.vector[entity]  # an entity access, which builds the cached plain view
    released = weakref.ref(entity)
    del entity, vector
    gc.collect()

    assert released() is None


def test_a_copy_of_a_dof_vector_is_the_only_dof_vector_holding_its_data():
    vector = DofVector(4)
    vector[:] = [1.0, 2.0, 3.0, 4.0]

    copy = vector.copy()

    assert type(copy.base) is np.ndarray
    assert np.array_equal(copy, vector)
    assert copy.entitiesInDofVector == vector.entitiesInDofVector
    assert copy.entitiesInDofVector is not vector.entitiesInDofVector


def test_a_model_with_a_rigid_body_is_released_after_its_simulation(tmp_path, monkeypatch):
    from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
    from edelweissfe.utils.inputfileparser import parseInputFile

    deck = tmp_path / "deck"
    shutil.copytree(_RIGID_BODY_DECK, deck)
    monkeypatch.chdir(deck)

    model, fieldOutputController = finiteElementSimulation(
        parseInputFile("test.inp"), verbose=False, suppressPlots=True
    )
    assert model.rigidBodies
    released = weakref.ref(model)
    del model, fieldOutputController
    gc.collect()

    assert released() is None


def test_a_copy_of_a_scatter_vector_is_the_only_scatter_vector_holding_its_data():
    scatterVector = DofVector(4, {"entity": np.array([0, 1])}).createScatterVector()
    scatterVector[:] = [1.0, 2.0]

    copy = scatterVector.copy()

    assert type(copy.base) is np.ndarray
    assert np.array_equal(copy, scatterVector)
    assert copy.offsetMap == scatterVector.offsetMap


def test_the_plain_view_of_a_dof_vector_refers_to_the_plain_buffer_not_to_the_vector():
    from edelweissfe.numerics.dofvector import DofVector

    for vector in (DofVector(5, {}), DofVector(5, {}).copy()):
        assert type(vector.asPlainArray().base) is np.ndarray
