#!/bin/bash
#SBATCH --job-name=r2c_L8
#SBATCH --account=<account>
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.

set -uo pipefail

module load python/3.12
source "<path to virtual environment>/bin/activate"

cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
mkdir -p "$KUMMER_OUT"

export OMP_NUM_THREADS=16

echo "job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "th_max=8 mode=prim"

/usr/bin/time -v python r2c_fir_run.py 8 prim > "$KUMMER_OUT"/fir_L8.log 2>&1
EC=$?

echo "python exited with code $EC at $(date)"
tail -n 30 "$KUMMER_OUT"/fir_L8.log
exit $EC
