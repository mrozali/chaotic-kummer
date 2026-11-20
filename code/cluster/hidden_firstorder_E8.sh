#!/bin/bash
#SBATCH --job-name=r2g_hidden_E8
#SBATCH --account=<account>
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=0-08:00:00
#SBATCH --output=%x-%j.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.
set -uo pipefail
module load StdEnv/2023 python/3.12
source "<path to virtual environment>/bin/activate"
export FLEXIBLAS=OPENBLAS OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
echo "job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GATE: E=6 on this node (leaves 899 + 451 with 0.5242 / 0.5189 along qgen, as in the Letter)"
python r2g_hidden_firstorder.py 6 --dirs=qgen,qgen2 > "$KUMMER_OUT"/fir_hidden_firstorder_E6.log 2>&1
EC=$?; tail -n 6 "$KUMMER_OUT"/fir_hidden_firstorder_E6.log
[ $EC -eq 0 ] || { echo "E=6 gate run exited $EC"; exit $EC; }
grep -q "leaves 899: 0.5242 451: 0.5189" "$KUMMER_OUT"/fir_hidden_firstorder_E6.log || { echo "E=6 gate FAILED"; exit 2; }
echo "PRODUCTION: hidden Z_2 of the trivial character at E=8 from first-order data, 22 directions"
/usr/bin/time -v python r2g_hidden_firstorder.py 8 --save-gram > "$KUMMER_OUT"/fir_hidden_firstorder_E8.log 2>&1
EC=$?; echo "python exited $EC at $(date)"; tail -n 50 "$KUMMER_OUT"/fir_hidden_firstorder_E8.log; exit $EC
