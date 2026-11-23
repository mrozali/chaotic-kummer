# Paper-agent test

This file is secondary context. It records a test of how an AI agent that opens this repository answers a reader's questions.

## How the test was run

On 2026-11-20 a fresh agent (Claude, Sonnet model, as a general-purpose coding agent) was given a clean copy of this repository as its working directory and nothing else: no conversation history, no virtual environment, and no files from outside the copy. It was told to start from `CLAUDE.md`, as an agent opening the repository would, to set up the environment from the repository's own docs if it needed to run code, and not to start any command the docs describe as heavy. It then answered eight reader questions. Its answers were checked against the paper by the agent that prepared the repository (Claude, Opus model).

The repository tested was the release candidate of version 1.0.0, with the authors' final text of the Letter. It differs from the release by this file, `validation-report.md` and the corrections listed at the end. Release 1.0.1 added the arXiv identifier of the Letter, 2611.26694, which the answer to question 8 below could not yet give.

**Setup it performed.** It read `CLAUDE.md`, `AGENTS.md`, `README.md`, `CITATION.cff`, `verify.sh`, `LICENSE`, `code/figure-reproduction/README.md`, `environment/README.md`, `paper/README.md`, `data/README.md`, the Letter's source and parts of the Supplemental Material. It then ran `python3.14 -m venv .venv` and `.venv/bin/pip install -r environment/requirements.txt`, which installed the pinned versions.

**Commands it ran,** all from the repository root, each ending with `RESULT: PASS`:
- `.venv/bin/python code/figure-reproduction/check_paper_numbers.py` (133 checks, 0 failed, 15 s)
- `.venv/bin/python code/figure-reproduction/check_hidden_z2_firstorder.py` (6 checks, 8 s)
- `.venv/bin/python code/figure-reproduction/check_hidden_z2_E2.py` (8 checks, 2 s)
- `.venv/bin/python code/figure-reproduction/fig03_spacings_twocol.py` (56 s)

## Questions and answers

The answers are given as the agent gave them, shortened. The key output lines are kept.

**1. What in this repository should I treat as authoritative, and what is only secondary?**

> The paper, code and data: `paper/chaotic-kummer-v1.tex` (the Letter and its End Matter), `paper/chaotic-kummer-SM.tex`, the code in `code/kummer/` and the stored spectra in `data/r2_out/`. `code/figure-reproduction/README.md` is the authoritative map from claims to commands, and `environment/README.md` is the source of truth for setup. The summary and key results in `AGENTS.md` are a digest, to be checked against the tex. Second order and the truncated spectrum are not part of the repository ("presented elsewhere").

**2. In one paragraph: what does the paper claim, and what are the limits of that claim as the paper states them?**

> Near the Kummer orbifold of the K3 sigma model, at the free-fermion radius and in the $\mathbb Z_2^4$-invariant sector, the exactly marginal blow-up deformations form a family whose directions are unit quaternions. The first-order splitting of each degenerate level is given by the singular values of a matrix $M_{\rm c}$ built from free-field twist amplitudes (`eq:Mc`, `eq:Mc-chiral`). Along a generic direction, with every symmetry resolved (the characters of the stabilizer, $Z_R$, and a hidden $\mathbb Z_2$ at the even levels), the non-zero shifts follow the GOE through $E=8$, with no ensemble average: gap ratios (Fig. 2 and the table), the Wigner surmise at $E=8$ and the number variance up to $L=10$ at $E=7,8$ (Fig. 3). The singlet direction is not generic. Limits: first order only, so the statement holds as $\lambda\to0^+$, and at finite coupling only while the second-order shifts stay small. Only the invariant sector. Finite levels $E\le8$. Random matrix statistics at weak coupling do not by themselves imply ergodicity (footnote in the Conclusions). The hidden $\mathbb Z_2$ was found only numerically. The novelty claim is made "to our knowledge". At $E=6$ the number variance rises above the GOE for $L\gtrsim4$, which the paper attributes to the small sub-blocks.

**3. The paper says the gap ratio along the reference direction agrees with the GOE value to within 1.4 sigma at every level except E = 4. Check this from what is here and tell me the numbers you observe.**

```
.venv/bin/python code/figure-reproduction/check_paper_numbers.py
  E = 3: <r>_ref over 92 ratios: 0.541195, paper 0.541
  E = 4: <r>_ref over 294 ratios: 0.471062, paper 0.471
  E = 5: <r>_ref over 1468 ratios: 0.5417, paper 0.542
  E = 6: <r>_ref over 5438 ratios: 0.531761, paper 0.532
  E = 7: <r>_ref over 14136 ratios: 0.528169, paper 0.528
  E = 8: <r>_ref over 36830 ratios: 0.530098, paper 0.530
  (<r>_ref - 0.5307)/sigma for E = 3..8: +0.33, -3.31, +1.37, +0.25, -0.99, -0.37
  ok    every level except E = 4 within 1.4 sigma of the GOE value: largest 1.37
```

> The claim holds on the stored spectra, with little margin at $E=5$ (1.37). $E=4$ is at $-3.31\,\sigma$, which is my observation: the paper says only that the deviation is significant. The rebuild of the spectra at $E\le5$ from the free-field code is a separate script, `check_first_order_rebuild.py`, which I did not run.

**4. What is the evidence for the "hidden Z_2", is its origin understood, and which script would let me check it?**

> The evidence is numerical (End Matter, Fig. 4). At $E=6$ the unresolved sub-block of the trivial character with $Z_R=+1$ has $\langle r\rangle=0.432$, close to $0.441\pm0.009$ for two superposed GOE spectra in the ratio 899 : 451, and close levels almost always lie in different eigenspaces. The eigenspaces were found as jointly invariant subspaces of $H_0$ and the deformation. A method based on the first-order data at two directions gives the same split, 50 + 26 at $E=4$, 899 + 451 at $E=6$ and 6137 + 3079 at $E=8$, along every direction tested, and at $E=4$ and $6$ it equals the truncated-Hamiltonian result. An explicit form is given only at $E=2$ along the singlet (`eq:Sreflection`). The origin is not understood: the paper says it would be interesting to understand it. Scripts: `check_hidden_z2_firstorder.py` (run, `same values in each eigenspace as found from the truncated Hamiltonian: largest difference 0.0e+00`), `check_hidden_z2_E2.py` (run, `|v> has the weights of 2|osc> + |m+w> - |mw>: 4/6, 1/12, 1/12, 1/6`, `S = -1 on the twisted state`), `check_hidden_z2_truncated.py` (about 6 minutes, 5.4 GB, not run), and section (C) of `check_paper_numbers.py` for the stored splits at $E=6$ and $8$.

**5. What does the paper say about the low gap ratio at E = 4? Quote the numbers it gives, and only those.**

> $\langle r\rangle_{\rm ref}=0.471$ with $\sigma=0.018$ (table, with $d_+=729$, $d_-=784$, rank 304). Over 3000 random directions more than $4^\circ$ from every mirror, the pooled gap ratio is $0.495$ on average. In the three sub-blocks without a hidden $\mathbb Z_2$, the ten gap ratios at the top of each sub-block average $0.40$ and the remaining ones $0.53$. The two eigenspaces of the hidden $\mathbb Z_2$ give $0.45$ each. The paper calls the deficit a property of the level and not of the reference direction, describes the isolated top level with two nearly degenerate levels below it, and the singlet's nondegenerate top level with an eightfold degenerate one below it. It found no comparable structure at $E=3,5,6$ and no further exact symmetry in these sub-blocks.

**6. How do I reproduce Fig. 3, and how long does it take? Which results in the paper can I not recompute on a laptop, and why?**

> After the setup, `.venv/bin/python code/figure-reproduction/fig03_spacings_twocol.py` (arXiv layout, 56 s here, under 1 GB), or `fig03_spacings_onecol.py` for the journal layout. It reads the stored spectra and compares its PDF with the Letter's figure:
> ```
> reference direction, E = 8: 33157 spacings, <s^2> = 1.286
> Sigma^2 at L = [ 1.  2.  5. 10.]: reference direction, E = 8 [0.443 0.585 0.778 0.896]
> pixel comparison with paper/fig_spacings_singlet_twocol.pdf at 100 dpi: 39 of 157500 pixels differ
> RESULT: PASS
> ```
> The figure is reproduced from stored spectra. The spectra at $E=8$ (the reference direction, the random directions and the hidden $\mathbb Z_2$) and the singlet through $E=8$ are `blocked-heavy-compute`: they were computed on a cluster with up to 250 to 500 GB of memory, and `code/cluster/` holds the job scripts. The rebuilds at $E=6$ (minutes, 3.5 GB) and $E=7$ (about 10 minutes, 9.4 GB) fit a workstation.

**7. Is the singlet deformation chaotic according to this paper?**

> Not in the sense of the generic-direction result. The singlet has enhanced symmetry (nine mirrors meet there, stabilizer of order 384). Fully resolved, its gap ratio lies below the GOE value at every level (the script gives $-2.9$, $-5.3$, $-2.1$, $-2.2$ standard deviations at $E=5,\dots,8$, the paper says well below at $E=5,6$ and about two standard deviations at $E=7,8$). It keeps exact degeneracies: 583 of 10721 non-zero values repeat at $E=8$, 560 of them at $\nu=\pi$. Its number variance deviates strongly from the GOE ($\Sigma^2(10)=7.69$ at $E=8$ against $0.909$). The paper applies no verdict "chaotic" to a single direction, and it says that first-order random matrix statistics do not by themselves imply ergodicity.

**8. How should I cite this work, and under what license can I reuse the code?**

> Cite the Letter, Haehl and Rozali, "Onset of quantum chaos in a family of two-dimensional conformal field theories" (2026), with the BibTeX entry `Haehl:2026onset` of `README.md` and `AGENTS.md`. There is no arXiv identifier yet. The whole repository, code included, is under CC BY 4.0, so the code can be reused with attribution.

## Assessment

All eight answers are correct and grounded in the staged files. The agent set up the environment from the docs alone, ran four checks and reported observed values for every numerical question. It kept its own observations apart from the paper's statements (the $-3.31\,\sigma$ at $E=4$, the singlet's deviations in standard deviations), and it quoted for $E=4$ only the numbers the Letter gives. It stated the limits the paper states: first order, the invariant sector, finite levels, no ergodicity claim, and the hidden $\mathbb Z_2$ found numerically.

The agent reported three points in the documentation, all corrected before the release:
- `environment/README.md` still gave 120 checks for `check_paper_numbers.py` and 18 pages for the compiled paper. With the final text these are 133 checks and 17 pages.
- The figure map said that Fig. 3 regenerates with 0 differing pixels on the tested Mac. With the final figure files it is 39 of 157500 pixels (arXiv layout), along the $E=8$ curve of the number variance, well within the tolerance of the comparison (see the note on the unfolding in `code/figure-reproduction/README.md`).

It also noted that CC BY 4.0, which covers the whole repository, is not the license Creative Commons recommends for software. This is the authors' choice.

One slip on the agent's side: it wrote a copy of one script's output to a temporary file outside the repository copy. No file of the repository was changed.
