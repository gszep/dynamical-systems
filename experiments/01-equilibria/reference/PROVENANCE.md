# Provenance — pinned upstream sources for experiment 01

## Original inference package (BifurcationInference.jl)

* Upstream repository: https://github.com/bifurcationkit/BifurcationInference.jl
  (formerly `gszep/BifurcationFit.jl`, which redirects here)
* Pinned commit: `534ee7c23af16d542a248f181f5e6be7056e30b6` (master HEAD on
  2026-10-19; commit date 2024-05-29T18:18:08Z, message "remove test deps")
* Package version at that commit: **0.1.4**, UUID `7fe238d6-d31e-4646-aa16-9d8429fd6da8`
* Verification that the registered release is byte-identical to the commit:
  * tree hash of commit `534ee7c` (computed via `git rev-parse HEAD^{tree}` on a
    fresh shallow clone): `edd1bc83e6e0d571af55039882f6a27037df2524`
  * Julia General registry
    `B/BifurcationInference/Versions.toml` entry for `["0.1.4"]`:
    `git-tree-sha1 = "edd1bc83e6e0d571af55039882f6a27037df2524"` — **exact match**
  * therefore `Pkg.add(PackageSpec(name="BifurcationInference", version="0.1.4"))`
    resolves to the identical source snapshot in `reference/upstream-534ee7c…/`;
    no substitution occurred.
* Snapshot in this tree: `reference/upstream-534ee7c23af16d542a248f181f5e6be7056e30b6/`
  (`Project.toml`, `README.md`, `src/{BifurcationInference,Objectives,Gradients,Structures,Utils}.jl`)
  fetched from `raw.githubusercontent.com` at the pinned commit on 2026-10-19
  and read in full before any test was written.

## Paper (parameter inference with bifurcation diagrams)

* Szep, Dalchau, Csikász-Nagy, "Parameter Inference with Bifurcation
  Diagrams", NeurIPS 2021; arXiv:2106.04243 (**v3**, revised 2021-10-27).
* The arXiv v3 LaTeX source (`article.tex`, `appendix.tex`; e-print tarball
  fetched 2026-10-19, files dated 2021-10-27/28) was read directly for the
  exact equations implemented in `src/EquilibriaProbes.jl`:
  Eq (2) `eq:predictions`, Eq (4) `eq:error`, Eq (5) `eq:measure`,
  Eq (6) `eq:tangent`, Eq (7) `eq:total-measure`, Eq (8) `eq:loss`,
  Eq (9) `eq:gradient-loss`, Eq (10) `eq:gradient-bifurcation`;
  Supp B `eq:conditions`/`eq:determinant`/`eq:measure-conditions` (B.1/B.6),
  Supp C Leibniz rule + normal deformation (C.8),
  Supp D `eq:measure-tanh` (D.1), Supp E `eq:hopf-measure` (E.1).
* Paper macro `\Det := |∂F_θ/∂u|` (absolute determinant) — the released code
  uses the raw `det(∂Fu)` in `distance`/`velocity`; noted in the probes.

## Runtime

* Julia 1.10.11, tarball
  `julia-1.10.11-mac64.tar.gz` (julialang-s3), SHA-256
  `bdc6f52ee4fdcf13f0d0e2480733742b7eefb7fbfc66c518daca6e19e804f34b`
  (verified against the pinned SHA before use; shared read-only binary at
  `dynamical-systems-research-20261019/.tools/julia-1.10.11`).
* Package environment: `pkgenv/Project.toml` + `pkgenv/Manifest.toml` (committed)
  record the exact resolution, including the `KrylovKit = "0.7.1"` pin that was
  required to make the declared dependency range loadable on Julia 1.10
  (see `results/01-equilibria/setup_record.toml` and the experiment README).

## Related-source verification performed during setup

* `KrylovKit` 0.8.3 `src/deprecated.jl` contains
  `Base.@deprecate(RecursiveVec(args...), tuple(args...))` — `RecursiveVec`
  is a deprecated constructor **function** in 0.8, no longer a type, while
  `BifurcationKit` 0.3.6 `src/BorderedArrays.jl:69` defines
  `rmul!(A::RecursiveVec, a::Number)` treating it as a type: the exact root
  cause of the load failure with 0.8.3 (see README "blockers").
* Offending line presence was confirmed by tag-raw fetch for **all** released
  `BifurcationKit` 0.3.x (0.3.0–0.3.6) and the struct declaration for 0.3.0 /
  0.3.3 was diff-checked against 0.3.6.
