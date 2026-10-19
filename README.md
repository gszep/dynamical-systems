# Reviving dynamical landscape research

Research package (2026-10-19) investigating a revival of the inverse-bifurcation research program from Grisha Szep's PhD thesis *Inferring Bifurcations Between Phenotypes* (LaTeX source dated 2021-10-29; KCL award 2022-09-01) and the NeurIPS 2021 paper *Parameter Inference with Bifurcation Diagrams* ([BifurcationInference.jl](https://github.com/bifurcationkit/BifurcationInference.jl)): extending inverse bifurcation to limit cycles and nonlinear PDE patterns, efficient qualitative landscape mapping, language and architecture options (Julia, Python, Rust/wgpu/Wasm; CUDA host not accessed), agentic-coding evidence, Lean interoperability and soundness, and a staged falsifiable research agenda.

## Contents

- **Active experiments** — [EXPERIMENTS.md](EXPERIMENTS.md): authorized experiments 1–3, acceptance gates, linked builders, execution budget and overnight checkpoint.
- **Final report** — [reports/Reviving dynamical landscape research.md](reports/Reviving%20dynamical%20landscape%20research.md): synthesis of all six notes; separates empirical facts, recommendations and unknowns; includes reading-coverage disclosure.
- **Research notes** — [research_notes/Reviving dynamical landscape research/](research_notes/Reviving%20dynamical%20landscape%20research/):
  - [thesis.md](research_notes/Reviving%20dynamical%20landscape%20research/thesis.md) — thesis aims, mathematics, assumptions, limitations and future-work headings (read from full LaTeX source; embedded publication PDFs not read).
  - [inference.md](research_notes/Reviving%20dynamical%20landscape%20research/inference.md) — the NeurIPS 2021 paper and BifurcationInference.jl source: objective, gradients, paper/code discrepancies, successors and maintenance.
  - [literature.md](research_notes/Reviving%20dynamical%20landscape%20research/literature.md) — 2021–2026 literature: snaking and localized states, deflation, periodic-orbit continuation, Conley–Morse, certification, data-driven methods.
  - [frameworks.md](research_notes/Reviving%20dynamical%20landscape%20research/frameworks.md) — Julia/Python/Rust-wgpu stack comparison, browser deployment and precision boundaries, proposed benchmark gates.
  - [lean_agents.md](research_notes/Reviving%20dynamical%20landscape%20research/lean_agents.md) — Julia agentic-coding evidence, Lean FFI/bindings, proof-validation ladder, Mathlib maturity, first milestones.
  - [audit.md](research_notes/Reviving%20dynamical%20landscape%20research/audit.md) — one-round selective audit (2026-10-19) whose corrections **govern all notes above**; read it alongside them.

Provenance: all notes were researched and audited on 2026-10-19; the report is the synthesis of record. No code was installed, built or benchmarked during that initial research pass. Subsequent implementation and actual execution evidence are tracked separately in `EXPERIMENTS.md`.
