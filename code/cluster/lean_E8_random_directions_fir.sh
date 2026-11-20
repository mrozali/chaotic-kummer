#!/bin/bash
#SBATCH --job-name=r2g_lean_E8_qrand
#SBATCH --account=<account>
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --array=0-11
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
DIRS=(qrand106 qrand111 qrand112); IRREPS=(0 2 3 6); D=${DIRS[$((SLURM_ARRAY_TASK_ID / 4))]}; R=${IRREPS[$((SLURM_ARRAY_TASK_ID % 4))]}
echo "PRODUCTION: lean probe, $D at E=8, irrep $R"
LOG="$KUMMER_OUT"/fir_lean_${D}_E8_rho$R.log
/usr/bin/time -v python r2g_generic_probe_mf_lean.py $D 8 --irreps=$R --validate > $LOG 2>&1
EC=$?; echo "python exited $EC at $(date)"; tail -n 25 $LOG; exit $EC
