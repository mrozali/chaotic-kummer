#!/bin/bash
#SBATCH --job-name=r2g_lean_E7_qrand
#SBATCH --account=<account>
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.
set -uo pipefail
module load StdEnv/2023 python/3.12
source "<path to virtual environment>/bin/activate"
export FLEXIBLAS=OPENBLAS OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
echo "job $SLURM_JOB_ID ${SLURM_ARRAY_TASK_ID:-} starting on $(hostname) at $(date)"
DIRS=(qrand106 qrand111 qrand112); D=${DIRS[$SLURM_ARRAY_TASK_ID]}
echo "PRODUCTION: lean probe, $D at E=7, all irreps"
LOG="$KUMMER_OUT"/fir_lean_${D}_E7.log
/usr/bin/time -v python r2g_generic_probe_mf_lean.py $D 7 --validate > $LOG 2>&1
EC=$?; echo "python exited $EC at $(date)"; tail -n 25 $LOG; exit $EC
