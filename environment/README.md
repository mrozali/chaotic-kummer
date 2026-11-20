# Environment

This file is the source of truth for recreating the computational environment of this repository.

## Summary

- Tested platform: macOS 26.5.2 on Apple silicon (M4 Pro, 14 cores, 48 GB), 2026-11-16 and 2026-11-20.
- Primary toolchain: Python 3.14.3. TeX Live 2025 (pdfTeX and BibTeX) to compile the paper.
- Environment required: yes, a Python virtual environment with four pinned packages.
- External or proprietary requirements: none. The levels $E\le7$ and every check of `./verify.sh` run on a workstation. Recomputing $E=8$ and the singlet through $E=8$ needs a cluster (see `code/figure-reproduction/README.md`).

## Dependency files

- `environment/requirements.txt` pins the Python packages: numpy 2.4.6, scipy 1.17.1, matplotlib 3.10.9 and mpmath 1.3.0. All other imports come from the Python standard library or from the modules in `code/kummer/` and `code/figure-reproduction/`. The dependencies of matplotlib are not pinned. The validation below ran with contourpy 1.4.0, cycler 0.12.1, fonttools 4.68.0, kiwisolver 1.5.1, packaging 26.3, pillow 13.0.0, pyparsing 3.3.3 and python-dateutil 4.0.0 and tzdata 2026.6.

## Setup from the repository root

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r environment/requirements.txt
```

Run every script from the repository root with the prefix `.venv/bin/python`, for example:

```sh
.venv/bin/python code/figure-reproduction/check_paper_numbers.py
```

Newer releases of the packages were not tried.

`./verify.sh` runs, in one command, the figure scripts and the quick checks of `code/figure-reproduction/`, and reports each as OK or FAIL (about 3 minutes, under 1 GB of memory). `./verify.sh full` adds the longer checks, about 30 minutes in all with up to 10 GB of memory. It uses `.venv/bin/python` when that exists, and the variable `PYTHON` names another interpreter. GitHub Actions runs `./verify.sh quick` on every push, on Ubuntu with Python 3.14 and the same pinned packages (`.github/workflows/verify.yml`).

Two variables control where the code reads and writes:

- `KUMMER_OUT` names the record folder of the pipeline modules and of the figure and check scripts. The default is `data/r2_out`.
- The linear-algebra thread count follows the usual variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`). The checks do not set them, except `check_hidden_z2_truncated.py`, which uses 4 threads, and `check_lean_probe_E7.py`, which uses all cores.

## Other tools

| Software | Version tested | Used for | Required? |
|---|---|---|---|
| pdfTeX, BibTeX | TeX Live 2025 | compiling `paper/chaotic-kummer-v1.tex` | only to rebuild the PDF |
| pdftoppm (Poppler) | 26.04.0 | the pixel comparison at the end of each figure script | optional, the scripts skip the comparison without it |
| Slurm | | the job scripts in `code/cluster/` | only for the cluster-scale runs |

To rebuild the paper, run from `paper/`:

```sh
pdflatex chaotic-kummer-v1 && bibtex chaotic-kummer-v1 && pdflatex chaotic-kummer-v1 && pdflatex chaotic-kummer-v1 && pdflatex chaotic-kummer-v1
```

The result has 17 pages, with no errors and no undefined references. The source uses the `revtex4-2` class.

## Gitignored local artifacts

- `.venv/` is the virtual environment.
- `code/figure-reproduction/generated/` holds all outputs of the figure and check scripts, including the sandboxes of the checks that run a pipeline module.
- `code/kummer/_cache/` holds the group tables that `r2c_group.py` writes on first use.
- LaTeX build files, if the paper is compiled inside `paper/`.

## Validation evidence

All commands were run from the repository root of a fresh copy, with the environment above.

| Command | Result | Notes |
|---|---|---|
| `python3.14 -m venv .venv` and `.venv/bin/pip install -r environment/requirements.txt` | passed | 2026-11-16 and 2026-11-20, Python 3.14.3, pinned versions installed in seconds |
| `./verify.sh quick` | `0 failed` | 11 checks in 2.6 minutes, peak memory 0.8 GB |
| `./verify.sh full` | `0 failed` | 18 checks in 27 minutes, peak memory 9.2 GB (2026-11-20, with the final text of the paper) |
| `.venv/bin/python code/figure-reproduction/fig03_spacings_twocol.py`, and `fig03_spacings_onecol.py` | `RESULT: PASS` | about 50 s each. 39 of 157500 and 15 of 64940 pixels differ, all on the $E=8$ curve of $\Sigma^2(L)$ (see `code/figure-reproduction/README.md`) |
| `.venv/bin/python code/figure-reproduction/check_paper_numbers.py` | `RESULT: PASS` | 133 checks in 12 s, 0.1 GB |
| `.venv/bin/python code/figure-reproduction/check_first_order_rebuild.py` | `RESULT: PASS` | 27 s, 0.6 GB |
| `.venv/bin/python code/figure-reproduction/check_first_order_rebuild.py 6` | `RESULT: PASS` | 3.8 to 4.4 minutes, 3.5 GB, largest difference $10^{-9}$ |
| `.venv/bin/python code/figure-reproduction/check_hidden_z2_firstorder.py 6` | `RESULT: PASS` | 54 s, 3.8 GB |
| `.venv/bin/python code/figure-reproduction/check_hidden_z2_truncated.py qrand105` | `RESULT: PASS` | 5.9 to 6.2 minutes on 4 threads, 5.4 GB |
| `.venv/bin/python code/figure-reproduction/check_hidden_z2_E2.py` | `RESULT: PASS` | 2 s. With the argument `6`: 18 s, 1.0 GB |
| `.venv/bin/python code/figure-reproduction/check_null_sigma.py` | `RESULT: PASS` | 35 s |
| `.venv/bin/python code/figure-reproduction/check_E4_deficit.py` | `RESULT: PASS` | 68 s, 1.1 GB |
| `.venv/bin/python code/figure-reproduction/check_unfolding_control.py` | `RESULT: PASS` | 2.4 minutes, 797 numbers equal to the stored record |
| `.venv/bin/python code/figure-reproduction/check_lean_probe_E7.py` | `RESULT: PASS` | 9.4 to 9.9 minutes on 14 cores, 9.2 to 9.4 GB, largest difference 0 |
| the build of the paper | passed | in a copy of `paper/`: 17 pages, 0 errors, 0 undefined references. The journal layout gives 9 pages for the Letter and 9 for the Supplemental Material. |

The runs left every shipped file unchanged (`diff -r` against a second copy).

On Linux, the workflow of `.github/workflows/verify.yml` was run on 2026-11-16 in a private test copy of this repository on GitHub (Ubuntu 26.04, x86-64, Python 3.14, the same pinned packages): `./verify.sh quick` passed 11 of 11 checks in 3.5 minutes. The checks of `./verify.sh full` were run there too, one by one, on the standard runner of a private repository (2 cores, 7 GB): all passed except the $E=7$ rebuild, which was not run because it needs 9.4 GB. The $E=6$ rebuild took 15 minutes and the truncated-Hamiltonian check 10.5 minutes. Results agree with the Mac except in the last digits: the regenerated chiral GOE samples differ by up to $10^{-13}$, the unfolding control by up to $5\times10^{-3}$, and $\Sigma^2(L)$ at $E=8$ in the third decimal (`code/figure-reproduction/README.md` explains why). On 2026-11-20 the workflow ran again on the release candidate of this repository, with the final text of the paper: `./verify.sh quick` passed 11 of 11 checks in 5 minutes, and Fig. 3 differed from the Letter's in 25 of 157500 pixels, against 39 on the Mac.
