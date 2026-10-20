#!/bin/bash
# verify.sh WORKTREE LABEL [NPROC]: run the c1_150 verification deck (250 increments from t=0,
# BASELINE=/path/to/other.npy overrides the baseline (e.g. for a branch on an older base).
# topology check every 50) with WORKTREE's code; save final U,P,V as LABEL.npy; compare bitwise to
# baseline.npy; print the timing rows. NPROC>1 runs NEDMPI under mpirun (deck solver replaced).
set -e
K=$(dirname $(readlink -f $0)); W=$1; L=$2; N=${3:-0}
source /home/matthias/constitutive_modeling/miniforge3/etc/profile.d/conda.sh && conda activate next_v2611_mpi
export PYTHONPATH=$W PYTHON_GIL=0 MKL_CBWR=AUTO,STRICT PYTHONUNBUFFERED=1
D=$K/runs/$L; rm -rf $D; mkdir -p $D; cp $K/deck/test.inp $D/; ln -s $K/deck/mesh $D/mesh
cd $D
if [ "$N" -gt 0 ]; then
  sed -i 's/\*solver, solver=NEDParallel/*solver, solver=NEDMPI/' test.inp
  OMP_NUM_THREADS=4 mpirun -n $N --bind-to none -x PYTHONPATH -x PYTHON_GIL -x MKL_CBWR -x OMP_NUM_THREADS python $K/run_and_dump.py $D > log.txt 2>&1
else
  OMP_NUM_THREADS=32 python $K/run_and_dump.py $D > log.txt 2>&1
fi
python - <<PY
import numpy as np
a=np.load("${BASELINE:-$K/baseline.npy}"); b=np.load("$D/UPV.npy")
print("$L vs baseline:", "BITWISE" if a.shape==b.shape and np.array_equal(a,b) else "DIFFERENT (shape %s vs %s, maxdiff %s)"%(a.shape,b.shape, np.max(np.abs(a-b)) if a.shape==b.shape else "-"))
PY
grep -E "^ \| (topology update|  AMR|  refresh mesh|build equation|increment|constraint connectivity)|Job computation" log.txt | cut -c1-90 || true
