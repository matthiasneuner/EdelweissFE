"""A per-node field output over an empty set is empty: it never falls back to the whole field."""

from edelweissfe.helpers.inputfilehelpers import (
    createFieldOutputFromInputFile,
    fillFEModelFromInputFile,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK = """
*job, name=emptysetjob, domain=2d
*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.15
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=4
y0=0, h=2
elType=CPE4
elProvider=edelweiss
nX=4
nY=2
*section, name=section1, thickness=1.0, material=linearelastic, type=plane
all
*fieldOutput
>>perNode, name=uOfNoElements, elSet=noElements, field=displacement, result=U
>>perNode, name=uOfAllElements, elSet=all, field=displacement, result=U
"""


def test_a_node_field_output_over_an_empty_element_set_is_empty(tmp_path):
    deck = tmp_path / "test.inp"
    deck.write_text(_DECK)
    inputFile = parseInputFile(str(deck))
    model = fillFEModelFromInputFile(FEModel(2), inputFile, Journal(verbose=False))
    model.mesh.setElementSet("noElements", [])
    model.resolveElementSetOfMesh("noElements")
    model.prepareYourself(Journal(verbose=False))
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")

    assert len(model.elementSets["noElements"]) == 0
    fieldOutputs = createFieldOutputFromInputFile(inputFile, model, Journal(verbose=False)).fieldOutputs
    for fieldOutput in fieldOutputs.values():
        fieldOutput.updateResults(model)

    assert fieldOutputs["uOfNoElements"].getLastResult().shape == (0, 2)
    # the same output over every element covers all 15 nodes: the empty set did not read as "no set"
    assert fieldOutputs["uOfAllElements"].getLastResult().shape == (15, 2)
