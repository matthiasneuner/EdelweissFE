# LGMRES to 1e-10 around the Schur complement of the embedded bars: the dofs of the interior nodes of the bar and the
# bond elements are condensed (solved exactly); the remaining dofs get blockamg's field preconditioners in two block
# Gauss-Seidel sweeps over the diagonally scaled Schur complement.
from edelweissfe.config.markerlibrary import getMarkerClass
from edelweissfe.linsolve.composition import (
    LGMRES,
    DiagonalScaling,
    FixedTolerance,
    Multiplicative,
    OnField,
    SchurComplement,
    gaussSeidelOrder,
)
from edelweissfe.linsolve.composition.presets import blockAMGFieldPreconditioner

bars = getMarkerClass("elementSet").fromOptions({"elSet": "rebars"})
bonds = getMarkerClass("elementSet").fromOptions({"elSet": "rebarBond_bond"})


def fieldCorrections(context):
    corrections = [OnField(field.name, blockAMGFieldPreconditioner(field, context)) for field in context.fields]
    return gaussSeidelOrder(corrections, sweeps=2)


blockGaussSeidel = Multiplicative(fieldCorrections)
linearSolver = LGMRES(
    SchurComplement(condense=[bars, bonds], preconditionS=DiagonalScaling(blockGaussSeidel)),
    tolerance=FixedTolerance(1e-10),
)
