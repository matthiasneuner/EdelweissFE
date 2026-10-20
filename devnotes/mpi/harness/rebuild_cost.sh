#!/bin/bash
# rebuild_cost.sh WORKTREE LABEL NPROC THREADS [MAXINC] [CONTACTFREQ]: tools/rebuild_cost.py under mpirun
W=$(readlink -f $1); L=$2; N=$3; TH=$4; M=${5:-60}; C=${6:-10}
B=/home/matthias/constitutive_modeling/next_v2611/mpi_explicit_benchmarks
source /home/matthias/constitutive_modeling/miniforge3/etc/profile.d/conda.sh && conda activate next_v2611_mpi
export PYTHONPATH=$W PYTHON_GIL=0 MKL_CBWR=AUTO,STRICT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$TH
D=$B/rebuild_cost/$L; rm -rf $D; mkdir -p $D; cd $W
mpirun -n $N --bind-to none -x PYTHONPATH -x PYTHON_GIL -x MKL_CBWR -x OMP_NUM_THREADS -x PYTHONUNBUFFERED python $B/tools/rebuild_cost.py $D $M $C > $D/log.txt 2>&1
grep -E "^MEM|^INTERFACE|^CODE|^REBUILD|^   |^   peak|^OUTSIDE|^PEAK|^processes|Error|error" $D/log.txt | head -150
