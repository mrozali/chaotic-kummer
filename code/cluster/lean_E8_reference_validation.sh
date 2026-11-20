#!/bin/bash
#SBATCH --job-name=r2g_lean_E8_qgen_rho2
#SBATCH --account=<account>
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=%x-%j.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.
set -uo pipefail
module load StdEnv/2023 python/3.12
source "<path to virtual environment>/bin/activate"
export FLEXIBLAS=OPENBLAS OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
echo "job $SLURM_JOB_ID ${SLURM_ARRAY_TASK_ID:-} starting on $(hostname) at $(date)"
echo "VALIDATION GATE: lean probe, qgen at E=8, irrep 2, against the stored values of an earlier Fir run"
/usr/bin/time -v python r2g_generic_probe_mf_lean.py qgen 8 --irreps=2 --validate --ref="$KUMMER_OUT"/Generic_probe_qgen_E8_mf_values.npz > "$KUMMER_OUT"/fir_lean_qgen_E8_rho2_validate.log 2>&1
EC=$?; echo "python exited $EC at $(date)"; tail -n 25 "$KUMMER_OUT"/fir_lean_qgen_E8_rho2_validate.log; exit $EC
