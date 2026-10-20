#!/bin/bash
# verify_refactor.sh WORKTREE LABEL [quick|full]
#
# Bitwise verification of an EdelweissFE worktree against the frozen references of 804d3817
# (mpi_explicit_benchmarks/refactor_reference_804d3817/; REF=.../refactor_reference for the old base 6e08a2d7,
# whose checkpoint is format version 1 -- 804d3817 writes version 8, #196/#198/#199). Every check must print
# BITWISE / PASS. All references except the checkpoint are bitwise identical between the two bases.
#
#   quick: serial NED decks; testfiles/mpi at 1,2,3,4 procs and 4x2; the materialParameterFromField deck
#          at serial/2/3 procs; tests/test_domaindecomposition.py.                       (~5 min)
#   full : quick + c1_150 verify deck with NEDParallel 32 threads and NEDMPI 8x4 (bitwise + timing),
#          + c1_150 checkpoint at 250 written by NEDMPI 8x4 (== NED's) and resumed by NEDMPI 8x4 to
#          350 (== straight NED).                                                         (~25 min)
#
# Outputs under mpi_explicit_benchmarks/refactor_runs/LABEL/. Uses ~64 cores in the full stage:
# do NOT run two full stages at the same time on rabbit (timings become meaningless, memory ~150 GB).
set -u
W=$(readlink -f $1); L=$2; MODE=${3:-quick}
B=/home/matthias/constitutive_modeling/next_v2611/mpi_explicit_benchmarks
REF=${REF:-$B/refactor_reference_marmot_e992a3f4}; T=$B/tools; K=$B/c1_150/verify
O=$B/refactor_runs/$L; mkdir -p $O
source /home/matthias/constitutive_modeling/miniforge3/etc/profile.d/conda.sh && conda activate next_v2611_mpi
export PYTHONPATH=$W PYTHON_GIL=0 MKL_CBWR=AUTO,STRICT PYTHONUNBUFFERED=1
cd $W  # the current directory precedes PYTHONPATH on sys.path
python -c "import edelweissfe,sys; p=edelweissfe.__file__; print('CODE', p); sys.exit(0 if p.startswith('$W/') else 1)" || { echo "FAIL: edelweissfe not imported from $W"; exit 1; }
MPIRUN="mpirun --bind-to none -x PYTHONPATH -x PYTHON_GIL -x MKL_CBWR -x OMP_NUM_THREADS -x PYTHONUNBUFFERED"
SER="edelweiss-only/NED edelweiss-only/NEDContact edelweiss-only/NEDRestart1Write edelweiss-only/NEDRestart2Resume edelweiss-only/NEDSurfaceContact edelweiss-only/NEDSurfaceToDiscreteRigidBodyContact edelweiss-only/TieNED marmot/GCDPNEDExplicit marmot/GCDPNEDExplicitAMR marmot/GCDPNEDExplicitHyperbolic marmot/GCDPNEDExplicitHyperbolicAMR marmot/NED marmot/NEDLiveAMR marmot/NEDLiveAMRStepEndsOffInterval marmot/NEDParallel"

decks() {  # decks CONFIG NPROC THREADS
  c=$1; n=$2; th=$3; D=$O/$c; mkdir -p $D; rm -rf $D/out
  if [ $n -eq 0 ]; then
    cp -r $W/testfiles/edelweiss-only $W/testfiles/marmot $D/
    cp -r $W/testfiles/mpi/marmot/NEDInitialStressPressure $D/marmot/
    # NEDRestart1Write/2Resume were removed upstream by #201 (3ba063c5): take them from 6e08a2d7 if absent
    for r in NEDRestart1Write NEDRestart2Resume; do [ -d $D/edelweiss-only/$r ] && continue
      mkdir -p $D/edelweiss-only/$r; git -C $W show 6e08a2d7:testfiles/edelweiss-only/$r/test.inp > $D/edelweiss-only/$r/test.inp; done
    sed -i 's/solver=NEDMPI/solver=NED/' $D/marmot/NEDInitialStressPressure/test.inp
    dirs=""; for t in $SER marmot/NEDInitialStressPressure; do dirs="$dirs $D/$t"; done
    OMP_NUM_THREADS=$th python $T/dumpUPV.py $D/out $dirs > $D/log.txt 2>&1
  else
    cp -r $W/testfiles/mpi/edelweiss-only $W/testfiles/mpi/marmot $D/
    # configs named *poison*: debug-poison-stale-solution (NaN wherever a process must not read), same references
    case $c in *poison*) sed -i '/^\*solver,.*NEDMPI/a debug-poison-stale-solution=True' $D/*/*/test.inp ;; esac
    dirs=$(ls -d $D/edelweiss-only/* $D/marmot/* | grep -v Restart2; ls -d $D/edelweiss-only/NEDRestart2Resume)
    OMP_NUM_THREADS=$th $MPIRUN -n $n python $T/dumpUPV.py $D/out $dirs > $D/log.txt 2>&1
  fi
  # More than one thread: pure-Python materials race in the threaded element loop (pre-existing, also in
  # serial NEDParallel, nondeterministic, ~1e-3..1e-2), so only the Marmot decks are compared there.
  filter=cat; [ $th -gt 1 ] && filter="grep ^marmot__"
  res=$(python $T/cmp.py $REF/decks $D/out | $filter); nt=$(echo "$res" | wc -l); r=$(echo "$res" | grep -vc BITWISE)
  [ "$r" -eq 0 ] && echo "decks $c: BITWISE ($nt/$nt)" || { echo "decks $c: FAIL"; echo "$res" | grep -v BITWISE; }
}

fieldmat() {  # fieldmat CONFIG NPROC
  c=$1; n=$2; D=$O/fm_$c; mkdir -p $D/deck/fieldmat; rm -rf $D/out; cp $REF/fieldmat.inp $D/deck/fieldmat/test.inp
  if [ $n -eq 0 ]; then sed -i 's/solver=NEDMPI/solver=NED/' $D/deck/fieldmat/test.inp; OMP_NUM_THREADS=1 python $T/dumpUPV.py $D/out $D/deck/fieldmat > $D/log.txt 2>&1
  else OMP_NUM_THREADS=1 $MPIRUN -n $n python $T/dumpUPV.py $D/out $D/deck/fieldmat > $D/log.txt 2>&1; fi
  python -c "
import numpy as np, os
a=np.load('$REF/deck__fieldmat.npy'); f='$D/out/deck__fieldmat.npy'
print('fieldmat $c:', 'BITWISE' if os.path.exists(f) and np.array_equal(a, np.load(f)) else 'FAIL (see $D/log.txt)')"
}

echo "=== quick checks ($L)"
decks serial 0 1 & decks mpi1 1 1 & decks mpi2 2 1 & decks mpi3 3 1 & decks mpi4 4 1 & decks mpi4x2 4 2 & decks mpi3poison 3 1 &
fieldmat serial 0 & fieldmat mpi2 2 & fieldmat mpi3 3 & wait
(cd $W && OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/test_domaindecomposition.py 2>&1 | tail -1 | sed 's/^/pytest test_domaindecomposition: /')

[ "$MODE" = full ] || exit 0

echo "=== full checks ($L)"
env -u PYTHON_GIL BASELINE=${BASELINE:-$K/baseline.npy} $K/verify.sh $W ${L}_serial32 0 2>&1 | grep -E 'vs baseline|^ \| increment .* 251 '
env -u PYTHON_GIL BASELINE=${BASELINE:-$K/baseline.npy} $K/verify.sh $W ${L}_8x4 8 2>&1 | grep -E 'vs baseline|^ \| increment .* 251 '
echo "(reference: NEDParallel 32 threads ~0.092 s/increment at 804d3817 (was ~0.104 at 6e08a2d7), NEDMPI 8x4 ~0.091 s/increment at the old base, rabbit idle)"

# checkpoint written by NEDMPI 8x4 at 250 must equal NED's; resuming it with NEDMPI 8x4 to 350 must equal straight NED
c150() {  # c150 NAME MAXINC [RESUMEFROM]
  D=$O/c150_$1; mkdir -p $D; rm -f $D/restart_*.h5 $D/UPV.npy; ln -sfn $K/deck/mesh $D/mesh
  python - "$D" "$2" "${3:-}" <<'PY'
import sys
d, m, frm = sys.argv[1:4]
s = open("/home/matthias/constitutive_modeling/next_v2611/mpi_explicit_benchmarks/c1_150/verify/deck/test.inp").read()
s = s.replace("maxNumInc=250", "maxNumInc=" + m).replace("*solver, solver=NEDParallel", "*solver, solver=NEDMPI")
extra = ("*restart, readFrom=" + frm + "\n\n") if frm else "*output, type=restart, name=restart\nwriteInterval=1, baseName=restart, numberOfFilesToKeep=1\n\n"
s = s.replace("*step, type=adaptiveForExplicitSimulations", extra + "*step, type=adaptiveForExplicitSimulations", 1)
open(d + "/test.inp", "w").write(s)
PY
  (cd $D && OMP_NUM_THREADS=4 $MPIRUN -n 8 python $K/run_and_dump.py $D > log.txt 2>&1)
}
c150 write 250
python $T/h5cmp.py $REF/c1_150_ckpt250_NED.h5 $O/c150_write/restart_0.h5 | sed 's/^/checkpoint NEDMPI 8x4 vs NED: /'
cp $O/c150_write/restart_0.h5 $O/c150_ckpt250.h5
c150 resume 350 $O/c150_ckpt250.h5
python -c "
import numpy as np, os
a=np.load('$REF/c1_150_straight350_UPV.npy'); f='$O/c150_resume/UPV.npy'
print('resume NEDMPI 8x4 250->350 vs straight NED:', 'BITWISE' if os.path.exists(f) and np.array_equal(a, np.load(f)) else 'FAIL')"
grep -l 'searching afresh' $O/c150_resume/log.txt && echo "FAIL: resume searched afresh"
true
