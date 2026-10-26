"""Embedded reinforcement: truss elements, the embedded region constraint (perfect bond) and the embedded bond
generator (bond-slip), verified against analytical solutions.

Needs the Marmot modules TrussElement, EmbeddedBondElement, LinearElasticBondSlip and ModelCode2010BondSlip.
"""

import math

import numpy as np
import pytest

pytest.importorskip("edelweissfe.elements.marmotelement.element")

from edelweissfe.drivers.inputfiledrivensimulation import (  # noqa: E402
    finiteElementSimulation,
)
from edelweissfe.utils.inputfileparser import parseInputFile  # noqa: E402


def _marmotHas(elementType: str) -> bool:
    from edelweissfe.elements.marmotelement.element import MarmotElementWrapper

    try:
        MarmotElementWrapper(elementType, 1)
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not (_marmotHas("TR2D2FS") and _marmotHas("EB2D2Q4")),
    reason="Marmot is built without the TrussElement / EmbeddedBondElement modules",
)

E_STEEL, A_STEEL, E_CONCRETE, NU_CONCRETE = 200000.0, 0.01, 30000.0, 0.2
PERIMETER = 0.3545  # circular bar of area 0.01


def _run(tmp_path, deck: str, *results):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def _bar(firstLabel: int, start, end, nElements: int, elType: str = "TR2D2"):
    """*node and *element blocks of a straight bar, and its first and last node labels."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    nodes = "\n".join(
        f"{firstLabel + i}, " + ", ".join(f"{c:.12g}" for c in start + (end - start) * i / nElements)
        for i in range(nElements + 1)
    )
    elements = "\n".join(f"{firstLabel + i}, {firstLabel + i}, {firstLabel + i + 1}" for i in range(nElements))
    return f"*node\n{nodes}\n\n*element, type={elType}\n{elements}\n", firstLabel, firstLabel + nElements


def _concreteStrip(nX=8, nY=2, elType="CPE4"):
    return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*modelGenerator, generator=planeRectQuad, name=gen
l=4
h=1
nX={nX}
nY={nY}
elType={elType}

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""


def _rebarsInStrip(nRebarElements=5, y=0.37):
    bar, first, last = _bar(100001, (0.0, y), (4.0, y), nRebarElements)
    return f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarRight
{last}

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars
"""


_UNIAXIAL_STEP = """*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=gen_right, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=RFrebar, nSet=rebarRight, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perElement, name=rebarStrain, elSet=rebars, result=strain, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=corner, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""

#: uniaxial tension of the reinforced plane strain strip at the strain 1e-3: concrete + steel
REACTION_PERFECT_BOND = (E_CONCRETE / (1 - NU_CONCRETE**2) * 1.0 + E_STEEL * A_STEEL) * 1e-3


def test_truss_small_strain_inclined(tmp_path):
    """an inclined linear elastic truss: reaction = EA/L x axial elongation"""
    deck = f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*node
1, 0.0, 0.0
2, 3.0, 4.0

*element, type=TR2D2
1, 1, 2

*elSet, elSet=bar
1

*nSet, nSet=left
1
*nSet, nSet=right
2

*section, name=s, area={A_STEEL}, material=steel, type=truss
bar

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=right, field=displacement, result=P, saveHistory=True,
>>perElement, name=N, elSet=bar, result=normal force, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=right, field=displacement, 1=0.003, 2=0.004
"""
    _, (RF, N) = _run(tmp_path, deck, "RF", "N")
    axialForce = E_STEEL * A_STEEL * 0.005 / 5.0
    np.testing.assert_allclose(RF.getLastResult().ravel(), axialForce * np.array([0.6, 0.8]), rtol=1e-10)
    np.testing.assert_allclose(N.getLastResult().ravel(), axialForce, rtol=1e-10)


def test_truss_finite_strain_frame_invariance(tmp_path):
    """a Neo-Hooke truss stretched by 1.5, once along x and once rotated by 90 degrees: equal normal force, the end
    force along the current axis"""

    def deck(u):
        return f"""*material, name=compressibleneohooke, id=rubber, provider=marmot
3000.0, 1000.0, 1e-9

*node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0

*element, type=TR3D2FS
1, 1, 2

*elSet, elSet=bar
1

*nSet, nSet=left
1
*nSet, nSet=right
2

*section, name=s, area=2.0, material=rubber, type=truss
bar

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=right, field=displacement, result=P, saveHistory=True,
>>perElement, name=N, elSet=bar, result=normal force, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=0.1, minInc=1e-8, maxNumInc=100, maxIter=15, stepLength=1
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=right, nSet=right, field=displacement, 1={u[0]}, 2={u[1]}, 3={u[2]}
"""

    _, (RFx, Nx) = _run(tmp_path, deck((0.5, 0.0, 0.0)), "RF", "N")
    _, (RFy, Ny) = _run(tmp_path, deck((-1.0, 1.5, 0.0)), "RF", "N")
    N = Nx.getLastResult().item()
    assert N > 0
    np.testing.assert_allclose(Ny.getLastResult().item(), N, rtol=1e-10)
    np.testing.assert_allclose(RFx.getLastResult().ravel(), [N, 0, 0], atol=1e-9 * N)
    np.testing.assert_allclose(RFy.getLastResult().ravel(), [0, N, 0], atol=1e-9 * N)


def test_embedded_region_uniaxial_tension(tmp_path):
    """perfect bond: the non-matching rebar follows the host strain, the reaction is concrete + steel"""
    deck = _concreteStrip() + _rebarsInStrip() + """
*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all
""" + _UNIAXIAL_STEP
    _, (RF, RFrebar, strain) = _run(tmp_path, deck, "RF", "RFrebar", "rebarStrain")
    np.testing.assert_allclose(strain.getLastResult(), 1e-3, rtol=1e-10)
    # the end node of the rebar is a slave: its share of the reaction is reported on it
    np.testing.assert_allclose(RF.getLastResult() + RFrebar.getLastResult(), REACTION_PERFECT_BOND, rtol=1e-10)


def test_embedded_region_rejects_unembedded_nodes(tmp_path):
    deck = _concreteStrip() + _rebarsInStrip().replace("100001, 0, 0.37", "100001, -0.5, 0.37") + """
*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all
""" + _UNIAXIAL_STEP
    with pytest.raises(ValueError, match="outside of all host elements"):
        _run(tmp_path, deck, "RF")


@pytest.mark.parametrize("elType", ["CPE4", "CPE8"])
def test_stiff_bond_reproduces_perfect_bond(tmp_path, elType):
    """bond-slip with a very stiff linear bond law converges to the perfect bond solution"""
    deck = _concreteStrip(elType=elType) + _rebarsInStrip() + """
*material, name=LinearElasticBondSlip, id=bond, provider=marmot
1e8, 1e8

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
""" + _UNIAXIAL_STEP
    model, (RF, RFrebar) = _run(tmp_path, deck, "RF", "RFrebar")
    # the free rebar end carries no load, the bond transfers it to the host; the short shear lag at the free rebar
    # ends (1/beta = 0.0075 << 4) makes the response only slightly softer than with perfect bond
    assert abs(RFrebar.getLastResult()) < 1e-9
    assert RF.getLastResult() < REACTION_PERFECT_BOND
    np.testing.assert_allclose(RF.getLastResult(), REACTION_PERFECT_BOND, rtol=3e-4)
    # 8 x 2 host elements, the rebar at y = 0.37 crosses all 8 lower ones, split at the 5 rebar elements' nodes
    assert len(model.elementSets["rebarBond_bond"]) == 8 + 4


def _pullOutDeck(bondMaterial: str, nRebarElements: int, steps: str, L=4.0):
    bar, first, last = _bar(100001, (0.0, 0.37), (L, 0.37), nRebarElements)
    return _concreteStrip(nX=5) + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars

{bondMaterial}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=s, nSet=loadedEnd, field=displacement, result=U, f(x)='sum(x[:,0])', saveHistory=True,

{steps}
"""


def test_pull_out_linear_bond_vs_analytical(tmp_path):
    """pull-out from a rigid host with a linear bond law: F / u = EA beta tanh(beta L), beta = sqrt(K_t p / EA)"""
    Kt = 1000.0
    deck = _pullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n{Kt}, 1e6",
        40,
        """*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
""",
    )
    _, (F,) = _run(tmp_path, deck, "F")
    EA = E_STEEL * A_STEEL
    beta = math.sqrt(Kt * PERIMETER / EA)
    np.testing.assert_allclose(F.getLastResult(), EA * beta * math.tanh(beta * 4.0) * 0.01, rtol=5e-4)


def test_pull_out_model_code_2010(tmp_path):
    """pull-out of a short, stiff bar from a rigid host: nearly uniform slip, so the pull-out force follows the
    bond-slip envelope x perimeter x embedded length, through the peak, the softening and the residual branch"""
    tauMax, L = 2.5 * math.sqrt(30.0), 0.5
    mc2010 = f"*material, name=ModelCode2010BondSlip, id=bond, provider=marmot\n{tauMax}, 1.0, 2.0, 10.0, 0.4, {0.4 * tauMax}, 200.0, 1000.0"
    deck = _pullOutDeck(
        mc2010,
        4,
        """*step, solver=theSolver
maxInc=0.02, minInc=1e-8, maxNumInc=1000, maxIter=15, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=12.0
""",
        L=L,
    ).replace(
        f"area={A_STEEL}", "area=100.0"
    )  # practically rigid bar: uniform slip
    _, (F, s) = _run(tmp_path, deck, "F", "s")

    slip = np.asarray(s.getResultHistory()).ravel()
    force = np.asarray(F.getResultHistory()).ravel()

    def envelope(x):
        s0 = (tauMax / (200.0 * 1.0**0.4)) ** (1 / 0.6)
        return np.select(
            [x <= s0, x <= 1.0, x <= 2.0, x <= 10.0],
            [200.0 * x, tauMax * np.abs(x) ** 0.4, tauMax, tauMax - 0.6 * tauMax * (x - 2.0) / 8.0],
            0.4 * tauMax,
        )

    np.testing.assert_allclose(force, envelope(slip) * PERIMETER * L, rtol=2e-3, atol=1e-6)
    assert force.max() == pytest.approx(tauMax * PERIMETER * L, rel=1e-6)


@pytest.mark.parametrize("barType", ["TR3D2", "TR3D3"])
def test_bond_3d_inclined_bar_matches_embedded_region(tmp_path, barType):
    """3D: an inclined bar crossing hexahedra, split at the host boundaries (splitBars=True); a stiff bond-slip law
    converges to perfect bond of the split bar"""

    def deck(bondStiffness, perfectBond):
        bar, first, last = _bar(100001, (0.15, 0.2, 0.3), (1.85, 0.75, 0.65), 3, "TR3D2")
        if barType == "TR3D3":  # quadratic bars: add the mid nodes
            lines = bar.split("*element, type=TR3D2\n")[1].strip().splitlines()
            nodes, elements = [], []
            for i, line in enumerate(lines):
                a, b = (int(v) for v in line.split(",")[1:])
                xa = np.array([float(v) for v in bar.splitlines()[1 + a - first].split(",")[1:]])
                xb = np.array([float(v) for v in bar.splitlines()[1 + b - first].split(",")[1:]])
                nodes.append(f"{200001 + i}, " + ", ".join(f"{c:.12g}" for c in 0.5 * (xa + xb)))
                elements.append(f"{first + i}, {a}, {b}, {200001 + i}")
            bar = bar.split("*element")[0] + "\n".join(nodes) + "\n\n*element, type=TR3D3\n" + "\n".join(elements)
        coupling = (
            "*constraint, type=embeddedRegion, name=embedded\nembeddedElSet=rebars, hostElSet=gen_all"
            if perfectBond
            else ""
        )
        return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bondStiffness}, {bondStiffness}

*modelGenerator, generator=boxGen, name=gen
lX=2
lY=1
lZ=1
nX=4
nY=2
nZ=2
elType=C3D8

*section, name=concreteSection, material=concrete, type=solid
gen_all

{bar}

*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarNodes, generate=True
{first}, {last}, 1

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
splitBars=True

{coupling}

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=uRebar, nSet=rebarNodes, field=displacement, result=U, saveHistory=True,
>>perNode, name=uHost, elSet=gen_all, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.002, 2=0.001, 3=-0.0005
"""

    # perfect bond of the split bars: zero bond stiffness, embedded region constraint
    _, (uRebarPerfect, uHostPerfect) = _run(tmp_path, deck(0.0, True), "uRebar", "uHost")
    model, (uRebarBond, uHostBond) = _run(tmp_path, deck(1e9, False), "uRebar", "uHost")

    rebars = model.elementSets["rebars"]
    assert len(rebars) > 3  # split at host boundaries
    assert all(len(el.nodes) == (2 if barType == "TR3D2" else 3) for el in rebars)
    scale = 0.002
    np.testing.assert_allclose(uRebarBond.getLastResult(), uRebarPerfect.getLastResult(), atol=1e-4 * scale)
    np.testing.assert_allclose(uHostBond.getLastResult(), uHostPerfect.getLastResult(), atol=1e-4 * scale)


def test_bond_law_property_change_and_shared_nodes(tmp_path):
    """a changeMaterialProperty step action drives the bond-slip law of the generated bond elements; a bar sharing
    nodes with its host is rejected by the bond generator"""
    Kt = 1000.0
    deck = _pullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n{Kt}, 1e6",
        8,
        """*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
>>changeMaterialProperty, name=chProp, material=bond, index=0, f(t)='1000 + 3000 * t'
""",
    )
    _, (F,) = _run(tmp_path, deck, "F")
    _, (FRef,) = _run(
        tmp_path, deck.replace(f"\n{Kt}, 1e6", "\n4000.0, 1e6").replace(">>changeMaterialProperty", "**"), "F"
    )
    np.testing.assert_allclose(F.getLastResult(), FRef.getLastResult(), rtol=1e-10)

    # a bar along the bottom edge of the strip, using the host nodes
    shared = _concreteStrip(nX=2, nY=1) + """*element, type=TR2D2
100001, 1, 3

*elSet, elSet=rebars
100001

*section, name=rebarSection, area=0.01, material=steel, type=truss
rebars

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
1000.0, 1e6

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
""" + _UNIAXIAL_STEP.replace(">>perNode, name=RFrebar", "**").replace(">>perElement, name=rebarStrain", "**")
    with pytest.raises(ValueError, match="are nodes of host element"):
        _run(tmp_path, shared, "RF")


@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_ensight_deformed_geometry(tmp_path, scale):
    """>>configuration, deformedGeometry=displacement writes the geometry in the deformed configuration: in the
    Ensight file, the concrete and the rebar coordinates move by their displacements"""
    vtkEnSight = pytest.importorskip("vtkmodules.vtkIOEnSight")
    from vtkmodules.util.numpy_support import vtk_to_numpy

    deck = (
        _concreteStrip()
        + _rebarsInStrip()
        + """
*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all
"""
        + _UNIAXIAL_STEP.replace(
            "*step,",
            """*output, type=ensight, name=esExport
>>perNode, fieldOutput=uAll
>>configuration, overwrite=yes, deformedGeometry=displacement, deformationScaleFactor={scale}

*step,""".format(scale=scale),
        ).replace("*fieldOutput\n", "*fieldOutput\n>>perNode, name=uAll, elSet=all, field=displacement, result=U\n")
    )
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        model, _ = _run(tmp_path, deck, "RF")
    finally:
        os.chdir(cwd)

    reader = vtkEnSight.vtkGenericEnSightReader()
    reader.SetCaseFileName(str(tmp_path / "esExport.case"))
    reader.UpdateInformation()
    times = reader.GetTimeSets().GetItem(0)
    reader.SetTimeValue(times.GetValue(times.GetNumberOfTuples() - 1))
    reader.Update()
    blocks = reader.GetOutput()
    names = [reader.GetDescription(i) for i in range(reader.GetNumberOfOutputPorts())]
    for i in range(blocks.GetNumberOfBlocks()):
        block = blocks.GetBlock(i)
        if block is None or block.GetPointData().GetArray("uAll") is None:
            continue
        X = vtk_to_numpy(block.GetPoints().GetData())[:, :2]
        U = vtk_to_numpy(block.GetPointData().GetArray("uAll"))[:, :2]
        labels = (
            vtk_to_numpy(block.GetPointData().GetArray("Node Ids"))
            if block.GetPointData().GetArray("Node Ids")
            else None
        )
        if labels is not None:
            X0 = np.array([model.nodes[int(n)].coordinates for n in labels])
            np.testing.assert_allclose(X - X0, scale * U, atol=1e-6 * scale)
        # the right edge moved by 0.004 (scaled)
        assert X[:, 0].max() == pytest.approx(4.0 + scale * 0.004, abs=1e-6 * scale)
        return
    pytest.fail(f"no part with the displacement found in {names}")


def _largeSlipPullOutDeck(bondMaterial: str, generatorOptions: str, pull: float, maxInc: float = 0.05):
    """a rigid bar along y = 0.37 from x = 0 to 9 (18 elements), bonded in the strip x = 0..4 (rigid host)"""
    bar, first, last = _bar(100001, (0.0, 0.37), (9.0, 0.37), 18)
    return _concreteStrip(nX=5) + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=barNodes, generate=True
{first}, {last}, 1

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, area=100.0, material=steel, type=truss
rebars

{bondMaterial}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}
{generatorOptions}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=s, nSet=loadedEnd, field=displacement, result=U, f(x)='sum(x[:,0])', saveHistory=True,

*step, solver=theSolver
maxInc={maxInc}, minInc=1e-8, maxNumInc=10000, maxIter=15, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=barY, nSet=barNodes, field=displacement, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1={pull}
"""


def test_large_slip_equals_small_slip_for_small_slip(tmp_path):
    """for slips small compared to the bonded length, the large-slip bond gives the small-slip response"""
    law = (
        "*material, name=ModelCode2010BondSlip, id=bond, provider=marmot\n13.69, 1.0, 2.0, 10.0, 0.4, 5.48, 200.0, 1e4"
    )
    _, (FSmall,) = _run(tmp_path, _largeSlipPullOutDeck(law, "", 0.02, 0.25), "F")
    _, (FLarge,) = _run(tmp_path, _largeSlipPullOutDeck(law, "largeSlip=True\nmaxSlip=1.0", 0.02, 0.25), "F")
    np.testing.assert_allclose(
        np.asarray(FLarge.getResultHistory()), np.asarray(FSmall.getResultHistory()), rtol=2e-2, atol=1e-9
    )


def test_large_slip_complete_pull_out(tmp_path):
    """a rigid bar pulled completely out of a rigid host with a linear bond law: the bonded length shrinks as
    L_b - s, so F = K_t s p (L_b - s), and the force vanishes once the bar has left the host"""
    Kt, Lb = 10.0, 4.0
    deck = _largeSlipPullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n{Kt}, 1e4",
        "largeSlip=True\nmaxSlip=6.0\nnIntegrationPoints=3",
        6.0,
    )
    model, (F, s) = _run(tmp_path, deck, "F", "s")
    assert any(el.elType.startswith("EBLS") for el in model.elementSets["rebarBond_bond"])
    slip, force = np.asarray(s.getResultHistory()).ravel(), np.asarray(F.getResultHistory()).ravel()
    for target in (0.5, 1.0, 2.0, 3.0):
        i = np.argmin(np.abs(slip - target))
        expected = Kt * slip[i] * PERIMETER * (Lb - slip[i])
        # the bond fades smoothly over the tributary parts at the bar end: a small deviation from the exact length
        np.testing.assert_allclose(force[i], expected, rtol=3e-2, err_msg=f"at s = {slip[i]}")
    assert np.all(np.abs(force[slip > Lb + 0.1]) < 1e-6)  # Newton tolerance
    assert force.max() == pytest.approx(Kt * PERIMETER * Lb**2 / 4, rel=3e-2)
