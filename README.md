# Onset of quantum chaos in a family of two-dimensional conformal field theories

[Felix M. Haehl](https://github.com/felixhaehl) (University of Southampton) and [Moshe Rozali](https://github.com/mrozali) (University of British Columbia)

This Letter exhibits random matrix statistics in the scaling dimensions of non-BPS primary operators in a family of explicit two-dimensional conformal field theories: the K3 sigma model near the Kummer orbifold point, deformed by an exactly marginal operator. Along a generic direction of deformation, and with every exact symmetry resolved, the first-order anomalous dimensions follow the Gaussian orthogonal ensemble, while special directions such as the well-known singlet deformation do not.

arXiv: [2611.26694](https://arxiv.org/abs/2611.26694). The paper is also in [`paper/`](paper/), with the compiled PDF at [`paper/chaotic-kummer-v1.pdf`](paper/chaotic-kummer-v1.pdf). Its source is that of the first arXiv version, with the LaTeX comments removed.

## Talk to this paper

This paper is published with an AI agent, following the [Agentic Publication Protocol](https://github.com/LionSR/AgenticPublicationProtocol). Clone this repository and open it in an AI coding agent to ask questions about the paper, check its numbers and reproduce its figures.

**Claude Code:** clone and open the repository, and it reads `AGENTS.md` automatically. With the Agentic Publication Protocol plugin installed, `/load-paper https://github.com/mrozali/chaotic-kummer` also works.

**Codex or other agents:** clone and open the repository. Any agent that reads `AGENTS.md` picks up the paper context.

## What is here

- `paper/` holds the LaTeX source of the Letter and of its Supplemental Material, the bibliography, the figure files and the compiled PDF. This is the ground truth.
- `code/kummer/` holds the code that builds the first-order matrices of the deformation from free-field data, resolves their symmetries and computes the statistics. `code/README.md` describes it.
- `code/figure-reproduction/` holds one script per figure and the checks of the Letter's numerical statements.
- `code/cluster/` holds the job scripts of the computations that were run on a cluster.
- `data/` holds the stored first-order spectra: the reference direction, 20 random directions and the singlet direction up to level $E=8$, and direction scans at lower levels, 24 MB in all.
- `environment/` gives the tested platform and the pinned Python packages.
- `supplementary/` holds the validation report and a test of the paper's agent. It is secondary to the paper.

## Figures

The authoritative map from each figure, the table and each quoted number to its script and status is [`code/figure-reproduction/README.md`](code/figure-reproduction/README.md).

## Reproducing results

### Setup

[`environment/README.md`](environment/README.md) gives the tested platform, the dependency file, the setup commands and the command prefix for running scripts. In short, from the repository root:

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r environment/requirements.txt
.venv/bin/python code/figure-reproduction/check_paper_numbers.py
```

The last command recomputes every number quoted in the Letter from the stored spectra, in about 12 seconds. `./verify.sh` runs all quick checks in one go, in about 3 minutes: it redraws the figures, repeats the check above, and rebuilds the spectra at the levels $E=3,4,5$ from the free-field code. GitHub Actions runs it on every push.

### Run figures

Use the commands in [`code/figure-reproduction/README.md`](code/figure-reproduction/README.md). Every script writes to `code/figure-reproduction/generated/` and compares its figure with the one in the Letter.

### Full computation

`./verify.sh full` recomputes on a workstation, in about 30 minutes and with up to 10 GB of memory, the spectra at $E=3,\dots,6$ for 50 directions, the hidden $\mathbb Z_2$ at $E=4$ and $6$, and the reference direction at $E=7$. The other directions at $E=7$ can be recomputed one at a time in about 10 minutes each. The level $E=8$, the singlet direction at the higher levels, and the truncated-Hamiltonian run behind the table were computed on a cluster, in jobs that requested up to 500 GB of memory. Their stored spectra are checked here, and `code/cluster/` holds the job scripts of the original runs.

## Data

[`data/README.md`](data/README.md) describes the stored spectra, their names and how they were made. No download is needed.

## Citation

Please cite the paper. `CITATION.cff` holds the same information in machine-readable form.

```bibtex
@article{Haehl:2026onset,
    author = "Haehl, Felix M. and Rozali, Moshe",
    title = "{Onset of quantum chaos in a family of two-dimensional conformal field theories}",
    eprint = "2611.26694",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    year = "2026"
}
```

## License

The whole repository (paper, code, data and supplementary material) is released under the Creative Commons Attribution 4.0 International license. See [`LICENSE`](LICENSE).
