#!/bin/bash
#SBATCH --job-name=r2g_lean_E7_qrand20
#SBATCH --account=<account>
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-17
#SBATCH --output=%x-%A_%a.out

# Job script as run on the Nibi cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.
set -uo pipefail
module load StdEnv/2023 python/3.12
source "<path to virtual environment>/bin/activate"
export FLEXIBLAS=OPENBLAS OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
echo "job $SLURM_JOB_ID ${SLURM_ARRAY_TASK_ID:-} starting on $(hostname) at $(date)"
# The 17 directions of the random sample qrand101-120 other than 106, 111, 112 (those were run on Fir) at E = 7.
# Task 17 reruns qrand106 against the Fir values (cross-machine gate).
DIRS=(qrand101 qrand102 qrand103 qrand104 qrand105 qrand107 qrand108 qrand109 qrand110 qrand113 qrand114 qrand115 qrand116 qrand117 qrand118 qrand119 qrand120)
if [ "$SLURM_ARRAY_TASK_ID" -lt 17 ]; then
  D=${DIRS[$SLURM_ARRAY_TASK_ID]}
  echo "PRODUCTION: lean probe, $D at E=7, all irreps"
  LOG="$KUMMER_OUT"/nibi_lean_${D}_E7.log
  python r2g_generic_probe_mf_lean.py $D 7 --validate > $LOG 2>&1
else
  echo "VALIDATION: lean probe, qrand106 at E=7 vs Fir values"
  LOG="$KUMMER_OUT"/nibi_lean_qrand106_E7_check.log
  python r2g_generic_probe_mf_lean.py qrand106 7 --ref="$KUMMER_OUT"/fir_ref_qrand106_E7_lean_values.npz --tag=_nibicheck > $LOG 2>&1
fi
EC=$?; echo "python exited $EC at $(date)"; tail -n 25 $LOG; exit $EC
