# Validation report

This file is secondary context. It records the full validation of this repository against the Agentic Publication Protocol (APP) 1.1.0, for release 1.0.0.

- **Stage:** full.
- **Result:** passed.
- **Date:** 2026-11-20.
- **Validated by:** Claude (Anthropic, Opus model), working for the authors. The paper-agent test used a separate fresh agent (Sonnet model).
- **Release:** `mrozali/chaotic-kummer`, tag `v1.0.0`, version `1.0.0`.
- **Paper:** the authors' final text of the Letter and its Supplemental Material, as submitted to arXiv. The sources in `paper/` are the authors' files with the LaTeX comments removed (see `paper/README.md`).

## Checks passed

**Structure.**
- The top level holds `paper/`, `code/`, `data/`, `environment/`, `supplementary/`, `AGENTS.md`, `CLAUDE.md` (`@AGENTS.md`), `README.md`, `LICENSE`, `.gitignore`, `.github/workflows/verify.yml`, `CITATION.cff` and `verify.sh`. The last two sit at the root by convention. `verify.sh` is the one-command entry point to the checks.
- `CITATION.cff` is valid under the Citation File Format schema 1.2.0 (checked with `cffconvert --validate`).
- There is one canonical paper source, `paper/chaotic-kummer-v1.tex`, which includes the Supplemental Material `paper/chaotic-kummer-SM.tex`. Both are declared as `latex` in the `AGENTS.md` frontmatter.
- No hidden, cached or generated files are committed. Script outputs, the virtual environment, the group-table cache and LaTeX build files are gitignored.

**Paths and commands.**
- Every path named in `AGENTS.md`, `README.md` and the READMEs under `paper/`, `code/`, `data/` and `environment/` resolves from the repository root. The only exceptions are a few abbreviations such as `fig01_...py` inside sentences that also give the full name.
- Every shell block in those files parses.
- Every label of the paper that the docs cite (`eq:Mc`, `fig:firstorderE`, `app:symmetries` and others) exists in the source.
- The external link to the protocol repository returns HTTP 200. The repository URL itself resolves once the repository is public.
- The paper cites this repository by its release URL, https://github.com/mrozali/chaotic-kummer/releases/tag/v1.0.0 (reference `HaehlRozali:2026code`, in the paragraph on code and data). It matches the repository and tag of this release.

**Figure and table map.**
- `code/figure-reproduction/README.md` lists every figure of the Letter and of the Supplemental Material (Figs. 1 to 4, S1 and S2, Fig. 3 in both layouts) and the table of the End Matter, each with a script, its inputs and outputs, and a final status. All are `reproduced` from the stored spectra.
- The figure scripts compare their output with the paper's figure files, rendered at 100 dpi. Figs. 1, 2, 4, S1 and S2 regenerate with 0 differing pixels. Fig. 3 differs in 39 of 157500 pixels (arXiv layout, 25 on Linux) and 15 of 64940 (journal layout), all on the $E=8$ curve of the number variance, whose third decimal depends on the numerics of the unfolding. The figure map explains this.
- Recomputing the spectra at $E=8$ and those of the singlet through $E=8$ is `blocked-heavy-compute`. The figure map gives the cluster resources of the original jobs, and `code/cluster/` holds the job scripts. The stored spectra are checked instead, and the same code is run afresh at $E\le7$.

**Quick claim checks.** Every number quoted in the Letter and the Supplemental Material has a check, listed in the figure map with its command, expected output and evidence level. `check_paper_numbers.py` recomputes the quoted numbers from the stored spectra in seconds (133 checks), including the headline: the table of gap ratios of the reference direction and its agreement with the GOE value to within $1.4\,\sigma$ at every level except $E=4$. Each hard-coded value of the paper in that script was compared with the final text: all 37 occur in it.

**Commands run.** All were run from the root of a fresh copy of the repository, with the pinned environment of `environment/README.md` (Python 3.14.3 on macOS, Apple silicon), on 2026-11-20:

| Command | Result |
|---|---|
| `python3.14 -m venv .venv` and `.venv/bin/pip install -r environment/requirements.txt` | pinned versions installed |
| `./verify.sh full` (the eleven quick checks, then seven longer ones) | `0 failed`, 18 checks in 27 minutes, peak memory 9.2 GB |

On Linux, GitHub Actions ran `./verify.sh quick` on the release candidate (Ubuntu 26.04, Python 3.14, the same pinned packages): 11 of 11 checks passed, and Fig. 3 differed in 25 of 157500 pixels.

The checks reproduce the paper's figures, recompute every quoted number, rebuild the first-order spectra from the free-field code at $E=3,\dots,6$ for 50 directions and at $E=7$ for the reference direction, and rebuild the hidden $\mathbb Z_2$ at $E=2$, $4$ and $6$. The runs left every shipped file unchanged.

**Code identity.** The modules in `code/kummer/` are the authors' modules. Their syntax trees equal those of the authors' working copies, except for three documented substitutions that point the record folder, a cache and a state file into the repository. Only comments and docstrings were edited.

**Data.**
- `data/README.md` describes the stored spectra and summaries in `data/r2_out/`, their naming and the run that made each.
- All data are in the repository, 24 MB, and no download is needed. Each record equals the authors' record, except that a run-time `doc` field was removed from some JSON files and a machine name from one note.

**Paper build.** `paper/chaotic-kummer-v1.tex` compiles with the build of `environment/README.md` to 17 pages, with no errors and no undefined references or citations. The text of the PDF equals that of the authors' PDF. The journal layout gives 9 pages for the Letter and 9 for the Supplemental Material.

**Consistency.**
- Title, authors and BibTeX entry agree between `AGENTS.md` and `README.md`, and `CITATION.cff` names the same paper and authors.
- `version` 1.0.0 matches the tag `v1.0.0`.
- The paper summary and key results in `AGENTS.md` were checked against the final text, and so were the paper's statements quoted in the figure map.
- Statements about reproduction status agree across `README.md`, `AGENTS.md`, `environment/README.md`, the figure map and this report.

**Privacy.**
- No credentials, private paths, user names, hostnames or IP addresses appear in the files. The job scripts in `code/cluster/` keep site-specific values as placeholders. They name the public clusters on which they ran, as the paper's acknowledgments do.
- No e-mail addresses appear outside `LICENSE`. The only people named are the authors, the people thanked in the paper's acknowledgments, and authors of cited work.

**License.** `LICENSE` is the CC BY 4.0 legal code, verbatim, and covers the paper, code, data and supplementary material. The authors chose this license for the whole repository.

**Paper-agent test.** A fresh agent with only this repository answered eight reader questions, recorded in `paper-agent-test.md`. It set up the environment from the docs, ran four checks, and reported observed values for every numerical question, including the headline. It stated the limits the paper states. All eight answers were correct. The three documentation points it raised were corrected.

## Limits recorded, not blockers

- The spectra at $E=8$ and the singlet's spectra through $E=8$ are stored and checked, but recomputing them needs a cluster (up to 250 to 500 GB of memory in the original jobs).
- Fig. 3 regenerates with a few differing pixels, explained above. No number quoted in the paper is affected.
- The explicit form of the hidden $\mathbb Z_2$ at $E=2$ is checked up to the relative signs inside $|v\rangle$, which depend on the phases of the states.
- The analytic statements (sectors, mirrors, rank bound, symmetry class) are derived in the paper. The scripts check their consequences for the numbers (counts of mirrors, chambers and group orders, the rank bound on the stored matrices), not the derivations.

## Public-release blockers

None for APP compliance.
