#!/bin/bash
#SBATCH --job-name=r2g_scan_L6
#SBATCH --account=<account>
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=101-116
#SBATCH --output=%x-%A_%a.out

# Job script as run on the Fir cluster of the Digital Research Alliance of Canada (Slurm).
# Site-specific values are placeholders in angle brackets. Run from a clone of this repository.

set -uo pipefail
module load python/3.12
source "<path to virtual environment>/bin/activate"
cd "<path to repository>/code/kummer"
export KUMMER_OUT="<path to a record folder for this run>"
mkdir -p "$KUMMER_OUT"
export OMP_NUM_THREADS=8
D=qrand${SLURM_ARRAY_TASK_ID}
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID direction $D starting on $(hostname) at $(date)"
echo "direction scan at Lambda=6: sweep -> first order level by level"
# The original job ran two further diagnostics in this chain (statistics of the truncated Hamiltonian at finite
# coupling, and a 100-sample chiral GOE reference).  Their scripts are not part of this repository and their
# output is not used by the Letter, so the two commands are left out here.
python r2g_generic_sweep.py $D 6 > "$KUMMER_OUT"/fir_scan_${D}_L6.log 2>&1 && \
python r2g_first_order_from_sweep.py $D 6 >> "$KUMMER_OUT"/fir_scan_${D}_L6.log 2>&1
EC=$?; echo "chain exited $EC at $(date)"
rm -f "$KUMMER_OUT"/Generic_sweep_${D}_L6_matrices.npz
tail -n 12 "$KUMMER_OUT"/fir_scan_${D}_L6.log
exit $EC
