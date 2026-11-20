#!/bin/bash
#SBATCH --job-name=r2g_L7_fo_qgen
#SBATCH --account=<account>
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-06:00:00
#SBATCH --output=%x-%j.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.

# The first-order step of sweep_L7_reference.sh, run as a second job because that job reached its 36 h limit
# after the sweep. It finds the leaves from the saved matrices, then the first-order spectra level by level
# (10.9 h in all).
set -uo pipefail
module load python/3.12
source "<path to virtual environment>/bin/activate"
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
export OMP_NUM_THREADS=32
echo "job $SLURM_JOB_ID starting on $(hostname) at $(date)"
/usr/bin/time -v python r2g_first_order_from_sweep.py qgen 7 > "$KUMMER_OUT"/fir_generic_firstorder_qgen_L7.log 2>&1
EC=$?; echo "first-order-from-sweep exited $EC at $(date)"
tail -n 30 "$KUMMER_OUT"/fir_generic_firstorder_qgen_L7.log
exit $EC
