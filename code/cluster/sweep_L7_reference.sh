#!/bin/bash
#SBATCH --job-name=r2g_L7_qgen
#SBATCH --account=<account>
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-12:00:00
#SBATCH --output=%x-%j.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.

set -uo pipefail
module load python/3.12
source "<path to virtual environment>/bin/activate"
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
mkdir -p "$KUMMER_OUT"
export OMP_NUM_THREADS=32
echo "job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "generic direction qgen, th_max=7 (Lambda=3.5): sweep -> first order level by level"
# The original job ran one further diagnostic between the two steps (statistics of the truncated Hamiltonian at
# finite coupling).  Its script is not part of this repository and its output is not used by the Letter, so the
# command is left out here.
/usr/bin/time -v python r2g_generic_sweep.py qgen 7 > "$KUMMER_OUT"/fir_generic_sweep_qgen_L7.log 2>&1
EC=$?; echo "sweep exited $EC at $(date)"
if [ $EC -eq 0 ]; then
  /usr/bin/time -v python r2g_first_order_from_sweep.py qgen 7 > "$KUMMER_OUT"/fir_generic_firstorder_qgen_L7.log 2>&1
  echo "first-order-from-sweep exited $? at $(date)"
fi
tail -n 25 "$KUMMER_OUT"/fir_generic_sweep_qgen_L7.log
exit $EC
