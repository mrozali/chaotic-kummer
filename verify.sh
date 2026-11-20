#!/bin/sh
# verify.sh: one-command check of the reproducible results of this repository.
#
# usage: ./verify.sh [quick|full]
#
#   quick   (default) Regenerates Figs. 1, 2, 3 (two-column layout), 4, S1 and S2 from the stored spectra and
#           compares them with the Letter's figures, recomputes every number quoted in the Letter from the
#           stored spectra, rebuilds the first-order spectra at E = 3, 4, 5 from the free-field code for 50
#           directions, rebuilds the hidden Z_2 at E = 4 from first-order data at two directions and its
#           explicit form at E = 2 along the singlet, and checks the chiral GOE reference.  About 3 minutes,
#           under 1 GB of memory.
#   full    The same, then Fig. 3 in the single-column layout, the E = 4 analysis, the unfolding control,
#           the hidden Z_2 at E = 6 by both methods, the first-order spectra at E = 6 for 50 directions, and
#           the reference direction at E = 7.  About 30 minutes in all, up to 10 GB of memory.
#
# Environment: PYTHON names the interpreter, with the packages of environment/requirements.txt.
# The default is .venv/bin/python if it exists, otherwise python3.
#
# Each check is reported OK or FAIL.  A check is OK when its script exits normally and prints
# RESULT: PASS.  Without pdftoppm the figure scripts cannot compare with the Letter's figures, and
# their OK lines say so.  The full output of every check goes to
# code/figure-reproduction/generated/verify_logs/.  The exit status is the number of failed checks.

set -u

MODE=${1:-quick}
case "$MODE" in
  quick|full) ;;
  -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
  *) echo "usage: ./verify.sh [quick|full]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")" || exit 99

if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

FR=code/figure-reproduction
LOG=$FR/generated/verify_logs
mkdir -p "$LOG"
fails=0

run() {
  name=$1
  script=$2
  shift 2
  start=$(date +%s)
  if "$PYTHON" -B "$FR/$script.py" "$@" > "$LOG/$name.txt" 2>&1 && grep -q "RESULT: PASS" "$LOG/$name.txt"; then
    if grep -q "pixel comparison with the Letter's figure skipped" "$LOG/$name.txt"; then
      echo "OK    $name ($(( $(date +%s) - start )) s), but the pixel comparison was skipped: install pdftoppm"
    else
      echo "OK    $name ($(( $(date +%s) - start )) s)"
    fi
  else
    echo "FAIL  $name (output in $LOG/$name.txt)"
    tail -n 3 "$LOG/$name.txt" | sed 's/^/      /'
    fails=$((fails + 1))
  fi
}

echo "verify.sh $MODE, interpreter $PYTHON"
for s in fig01_spectrum_direction fig02_gap_ratios fig03_spacings_twocol fig04_hidden_z2 \
         figS1_kummer_sectors figS2_mirror_geometry check_paper_numbers check_first_order_rebuild \
         check_hidden_z2_firstorder check_hidden_z2_E2 check_null_sigma; do
  run "$s" "$s"
done
if [ "$MODE" = full ]; then
  run fig03_spacings_onecol fig03_spacings_onecol
  run check_E4_deficit check_E4_deficit
  run check_unfolding_control check_unfolding_control
  run check_hidden_z2_firstorder_E6 check_hidden_z2_firstorder 6
  run check_hidden_z2_truncated check_hidden_z2_truncated qrand105
  run check_first_order_rebuild_E6 check_first_order_rebuild 6
  run check_lean_probe_E7 check_lean_probe_E7
fi

echo "$fails failed"
exit "$fails"
