#!/bin/bash
#SBATCH --job-name=r2g_E8_qgen
#SBATCH --account=<account>
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.

set -uo pipefail
module load python/3.12
source "<path to virtual environment>/bin/activate"
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
mkdir -p "$KUMMER_OUT"
export OMP_NUM_THREADS=48
echo "job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "generic direction qgen, first order at E=8, matrix-free probe with --validate (dense checks at th=5,6)"
/usr/bin/time -v python r2g_generic_probe_mf.py qgen 8 --validate --chunk=400 > "$KUMMER_OUT"/fir_generic_probe_qgen_E8.log 2>&1
EC=$?; echo "python exited $EC at $(date)"
tail -n 30 "$KUMMER_OUT"/fir_generic_probe_qgen_E8.log
exit $EC
