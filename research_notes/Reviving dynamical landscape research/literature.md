# Mapping qualitative dynamical landscapes beyond local linear Turing analysis — literature & software (searched through 2026-10-19)

Scope: continuation-based mapping of nonlinear PDE pattern landscapes (localized states, homoclinic snaking), periodic orbit continuation, deflation-based branch discovery, Conley/Morse global basin topology, and data-driven pattern descriptors / active exploration. Anchor: [BifurcationKit organization](https://bifurcationkit.github.io). Throughout, **detection/classification** (data-driven, heuristic) is kept separate from **mathematical certification** (validated numerics, computer-assisted proofs, rigorous outer approximations).

Review provenance (2026-10-19): one selective audit, not an exhaustive novelty search. Original references are retained; many remain abstract-level checks as disclosed below. Negative search results delimit this corpus, not the entire literature. Proposed integrations are research hypotheses, not priority claims; see `audit.md`.

## KQ1: What do primary sources establish about nonlinear pattern landscapes (localized states, snaking) beyond local linear Turing analysis, and which software computes them?

### Takeaway
Localized patterns and their parameter organization (homoclinic snaking, snakes-and-ladders) are a well-established, actively extended nonlinear phenomenon spanning model PDEs, shear flows, and shell buckling; mature continuation software (BifurcationKit.jl ecosystem, pde2path, AUTO-07p) computes these multi-branch landscapes directly, which linear Turing analysis cannot represent.

### Cited Findings
- Burke & Knobloch's review establishes homoclinic snaking as the characteristic "snakes-and-ladders" organization of localized states, with the one-dimensional origin of the structure reviewed — [Chaos 17, 037102 (2007)](https://pubmed.ncbi.nlm.nih.gov/17903009/).
- In the Swift–Hohenberg equation, snaking proceeds by the localized state adding rolls symmetrically at both ends, becoming more and more like a periodic wavetrain — [Burke & Knobloch (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0375960106013612).
- Recent theory extends snaking to contact defects in reaction–diffusion systems, understood through homoclinic/heteroclinic dynamics — [Roberts et al., Nonlinearity (2025)](https://iopscience.iop.org/article/10.1088/1361-6544/ae1376/pdf).
- Snaking occurs beyond model PDEs: steady states in plane Couette flow grow additional structure smoothly at their fronts under parameter continuation (localized turbulence precursors) — [J. Fluid Mech. (2016)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/homoclinic-snaking-in-plane-couette-flow-bending-skewing-and-finitesize-effects/6484E2538F10F3DC307271D531D61256); fully localized post-buckling dimples of cylindrical shells snake in the azimuthal direction under axial load — [Proc. R. Soc. A (2017)](https://royalsocietypublishing.org/rspa/article/473/2205/20170177/57457/Fully-localized-post-buckling-states-of).
- Radial/curvature variants of snaking describe localized circular patterns (2D) via front dynamics between homogeneous states — [Gomila, IMA J. Appl. Math. (2021)](https://academic.oup.com/imamat/article/86/5/1094/6352381).
- The [BifurcationKit organization](https://bifurcationkit.github.io) provides a Julia ecosystem: BifurcationKit.jl (core, with tutorials for PDE equilibria bifurcation and a GPU tutorial for 2D Swift–Hohenberg), HclinicBifurcationKit.jl (homoclinic orbits), MultiParamContinuation.jl (multi-parameter continuation), GridapBifurcationKit.jl (FEM PDEs via Gridap), DDEBifurcationKit.jl (delay equations), AsymptoticNumericalMethod.jl, PeriodicSchurBifurcationKit.jl; ModelingToolkit.jl and Catalyst.jl rely on BifurcationKit.jl for bifurcation-diagram computation — [bifurcationkit.github.io](https://bifurcationkit.github.io).
- pde2path is a Matlab continuation/bifurcation package for 2D elliptic PDE systems with arbitrary many components (cited 300+), with tutorials for pattern formation in 1D/2D/3D including branch-switching functions — [Uecker, NMTMA (2014)](https://www.cambridge.org/core/journals/numerical-mathematics-theory-methods-and-applications/article/pde2path-a-matlab-package-for-continuation-and-bifurcation-in-2d-elliptic-systems/11B12FB6196D66ADCBE37A84944D7DC2); [Uecker (2019) arXiv:1908.05211](https://arxiv.org/abs/1908.05211); advanced tutorials cover Turing–Hopf pattern mixing and Swift–Hohenberg on a disk — [pde2path tutorials](https://pde2path.uol.de/tutorials.html).
- A 2021/2022 book and review consolidate algorithms, software, and applications of continuation and bifurcation in nonlinear PDEs around pde2path (FEM discretization + continuation/bifurcation techniques) — [MAA book review (2021)](https://maa.org/book-reviews/numerical-continuation-and-bifurcation-in-nonlinear-pdes/); [Uecker, survey (2022)](https://link.springer.com/article/10.1365/s13291-021-00241-5).

### Inferences
- The snaking literature demonstrates that the objects a "qualitative landscape" must contain (multiplicity of stable/unstable branches, folds, disconnected components, localized vs extended states) are global, nonlinear features unreachable by linear stability/Turing-mode arguments alone; continuation software is therefore the primary mapping instrument for these landscapes.
- The BifurcationKit ecosystem's homoclinic-orbit and multi-parameter components target exactly the structures (homoclinics, 2-parameter manifolds) that organize snaking and cross-branch connectivity, but the org's own page documents scope by component; PDE-scale snaking in practice is still mostly done with direct steady-PDE continuation (pde2path, BifurcationKit FEM/GPU tutorials).

### Gaps
- Several primary papers were verified only via abstracts/snippets (paywalled full texts); e.g., the azimuthal shell-snaking mechanism details were not read in full.
- A dedicated recent review unifying snaking across fluids/solids/reaction–diffusion was not found in this session's searches (Burke & Knobloch 2007 remains the canonical entry point found).

## KQ2: What automated branch-discovery methods exist (deflation, multi-parameter continuation), with what demonstrated capabilities and limits?

### Takeaway
Deflated continuation is one established way to find additional, possibly disconnected solution branches. Deflation modifies the nonlinear solve rather than supplying a completeness guarantee. The cited 2025–2026 papers illustrate recent combinations with reduced-order/adaptive methods; this search does not establish that those combinations began in 2025.

### Cited Findings
- Deflated continuation is proposed for computing **disconnected** bifurcation diagrams, overcoming the classical limitations of standard continuation — [Farrell, Beentjes & Birkisson, arXiv:1603.00809 (2016)](https://arxiv.org/html/1603.00809v1).
- Applied to Carrier's problem, deflated continuation followed by backward arclength continuation revealed an intricate bifurcation diagram — [Chapman, IMA J. Appl. Math. (2017)](https://www.jstor.org/stable/45093272).
- Deflation "enables the discovery of disconnected branches" and can be combined with natural or tangent continuation for computing multiple solutions of nonlinear systems — [Farrell (2025), Oxford ORA](https://ora.ox.ac.uk/objects/uuid:6812f36b-2b32-460c-b742-1878ac339be2).
- Deflation-based certified greedy algorithms and adaptivity have been developed for tailored reduced-order models of bifurcating nonlinear parametric PDEs with multiple coexisting solutions — [Pichi & Strazzullo, Commun. Nonlinear Sci. Numer. Simul. (2025)](https://www.sciencedirect.com/science/article/pii/S1007570425003521).
- Adaptive Deflation Continuation (ADC) has been proposed for bifurcation detection and post-buckling analysis of hyperelastic structures (preprint) — [SSRN 6922884](https://papers.ssrn.com/sol3/Delivery.cfm/218892ea-d1f0-46d3-b94c-61997f105da0-MECA.pdf?abstractid=6922884&mirid=1).
- A 2026 preprint proposes a bifurcation-curve-detection strategy combining arclength continuation and deflation — [arXiv:2602.12940 (2026, preprint)](https://arxiv.org/html/2602.12940v2).
- Multi-parameter continuation (solution-manifold covering beyond a single parameter) is available in the BifurcationKit ecosystem — [MultiParamContinuation.jl](https://github.com/bifurcationkit/MultiParamContinuation.jl), via the [anchor page](https://bifurcationkit.github.io).

### Inferences
- Capability limit common to all deflation variants: deflation finds *additional, possibly disconnected* branches given a seed and a deflation operator, but no source found here proves completeness (i.e., "all branches in a region"); branch discovery remains partly heuristic, which is precisely why certification layers (KQ5) remain separate.
- Proposed reading of the sampled papers: adaptive/ROM combinations may help automate branch discovery. The sample does not support a bibliometric claim about the relative size of the peer-reviewed and preprint literatures; a repository record alone does not establish peer-review status.

### Gaps
- Whether deflation-based discovery has been combined with active/exploratory sampling (KQ6) in a single published pipeline was not found in these searches (see open directions).
- The SSRN and arXiv:2602.12940 items are preprints; their claims await peer review.

## KQ3: What is the state of periodic orbit continuation tooling?

### Takeaway
Periodic orbit continuation is a mature, codified capability (AUTO-07p with collocation and cycle-bifurcation detection since decades, now with a modern Python layer), mirrored in the BifurcationKit ecosystem and extended to time-periodic PDE patterns by pde2path.

### Cited Findings
- AUTO is continuation and bifurcation software for ODEs, originally written in 1980 and widely used (SIAM dsweb entry cited 1200+) — [SIAM dsweb archive](https://archive-dsweb.siam.org/Software/auto-07p.html); maintained on GitHub — [auto-07p](https://github.com/auto-07p/auto-07p).
- The AUTO-07p manual documents continuation of equilibria and **periodic orbits** and bifurcation analysis as the package's core — [Doedel et al., AUTO-07p manual (2012)](https://depts.washington.edu/bdecon/workshop2012/auto-tutorial/documentation/auto07p%20manual.pdf).
- HomCont (bundled with AUTO) handles homoclinic/heteroclinic continuation, used in graduate bifurcation courses for codim-1/2 homoclinic bifurcations — [Utrecht master-class page](https://www.staff.science.uu.nl/~kouzn101/mcnbds.html); original AUTO97+HomCont reference — [ResearchGate record](https://www.researchgate.net/publication/2290436_AUTO_97_Continuation_And_Bifurcation_Software_For_Ordinary_Differential_Equations_with_HomCont).
- auto-AUTO (AUTO2), a JOSS 2025 paper, provides a Python layer for automatically running AUTO-07p — [Demaeyer, JOSS (2025)](https://joss.theoj.org/papers/10.21105/joss.08079.pdf).
- BifurcationKit.jl reproduces classic AUTO periodic-orbit workflows: e.g., its "pp2" tutorial computes a periodic orbit branch emerging from a Hopf point — [BifurcationKit stable docs](https://docs.sciml.ai/BifurcationKit/stable/tutorials/ode/tutorialPP2/).
- pde2path supports continuation of time-periodic patterns (Turing–Hopf interactions) and disk-domain Swift–Hohenberg problems in its tutorial suite — [pde2path tutorials](https://pde2path.uol.de/tutorials.html); methods reviewed in — [Uecker (2022)](https://link.springer.com/article/10.1365/s13291-021-00241-5).

### Inferences
- Periodic-orbit continuation for ODEs is essentially commodity infrastructure; the active frontier relevant to landscapes is periodic (and modulated) patterns in PDEs and in control/experiment settings, where tooling is thinner and mostly tutorial-level documentation rather than turnkey products.

### Gaps
- MatCont (Matlab ODE continuation alternative) was not directly verified in this session's searches and is therefore not characterized here.
- Continuation of periodic orbits directly in large-scale PDE solvers outside pde2path/BifurcationKit tutorials was not surveyed (search budget).

## KQ4: What do Conley index / Morse decomposition methods deliver for global basin and landscape topology, and at what scale?

### Takeaway
Conley–Morse theory, made computational via rigorous outer approximations, is the mainline framework that *certifies* global dynamics (isolated invariant sets, Morse decompositions, Conley–Morse graphs) rather than merely detecting it; demonstrated capability is strongest for maps and moderate-dimensional systems, with a database infrastructure (Conley–Morse Database / CHomP) and persistence-style refinements, but PDE-scale applications are scarce.

### Cited Findings
- Conley indices and Morse decompositions of flows can be computed by algorithms that rigorously analyze discrete dynamical systems; discretization strategies are systematized — [Mischaikow, J. Comput. Dyn. (2016)](https://www.aimsciences.org/article/doi/10.3934/jcd.2016001).
- Combinatorial multivector dynamics is linked to classical dynamics: Conley index and Morse decompositions computed combinatorially are provably correct for the underlying classical systems, with worked Conley–Morse graphs and Poincaré polynomials — [Batko et al., Found. Comput. Math. (2020)](https://link.springer.com/article/10.1007/s10208-020-09444-1).
- A database schema (Conley–Morse Database) supports analysis of global dynamics of maps via Conley–Morse graphs (cited 140+) — [Arai, Kalies et al.](https://www.math.fau.edu/files/kalies_papers/dsagdms.pdf); user's guide — [CHomP / Rutgers](https://chomp.rutgers.edu/Projects/survey/cmdbSurvey.pdf).
- Conley–Morse graphs persist under combinatorial refinement, enabling multi-resolution exploration of global dynamics — [SIAM J. Appl. Dyn. Syst. (2022)](https://epubs.siam.org/doi/10.1137/21M143162X).
- Computational realization of Conley's decomposition theorem (separating chain-recurrent dynamics from gradient-like parts) was established in the ASME Comp. Nonlinear Dyn. literature — [ASME page](https://asmedigitalcollection.asme.org/computationalnonlinear/article/1/4/312/475324/A-Computational-Approach-to-Conley-s-Decomposition).
- Hierarchical refinement pipelines for Morse-set classification have been proposed for computational pipelines — [Oregon State report](https://web.engr.oregonstate.edu/~zhange/images/hierMorse.pdf).

### Inferences
- A rigorous outer approximation can support certified statements about isolated invariant sets/Morse decompositions under its theorem hypotheses. Do not interpret every graph edge as an actual connecting orbit or its cells as exact basin boundaries without a specific certification theorem. The reviewed examples do not establish a universal dimensional cutoff or absence of PDE applications.
- The CHomP/Conley-Morse Database line is explicitly a *certification* tool (results are mathematically valid for the underlying flow given the outer approximation), unlike all methods in KQ6.

### Gaps
- No application of Conley–Morse database technology to PDE pattern-formation landscapes (e.g., 2D Swift–Hohenberg) was found in this session's searches.
- Set-oriented alternatives (e.g., GAIO-style almost-invariant-set computation) were not verified this session; not characterized here.

## KQ5: How should detection/classification be separated from mathematical certification, and what certification tools exist?

### Takeaway
A useful reporting scheme separates numerical discovery, theorem-backed certification, and formal proof checking. These are guarantees attached to particular outputs, not mutually exclusive method families: validated continuation already combines path-following with certification, while certified local results need not establish landscape completeness.

### Cited Findings
- The radii-polynomial approach is an a-posteriori validation method based on contraction of a quasi-Newton operator; a recent survey reviews computer-assisted proofs (CAPs) in dynamical systems — [arXiv:2605.07500 (2026)](https://arxiv.org/html/2605.07500v1).
- AD for Fourier series + radii polynomials yields CAPs of existence of periodic solutions, including non-polynomial nonlinearities — [Lessard, Physica D (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0167278916000294).
- A general framework for **validated continuation** of periodic orbits in arbitrary polynomial vector fields exists (parametrized Newton–Kantorovich) — [van den Berg et al., J. Comput. Dyn. (2021)](https://www.aimsciences.org/article/doi/10.3934/jcd.2021004).
- CAPs have been extended to periodic solutions under state-dependent delayed perturbations — [SIAM J. Appl. Dyn. Syst.](https://epubs.siam.org/doi/10.1137/22M1499418).
- CAPs are described as a powerful methodology for "turning numerical evidence into" proofs in delayed and infinite-dimensional dynamical systems (CISM advanced course) — [CISM (2025)](https://www.cism.it/event/computer-assisted-proofs-in-delayed-and-infinite-dimensional-dynamical-systems-8447).
- By contrast, data-driven landscape signals — persistent homology classification of pattern parameter-dependence ([Spector & Harrington, Bull. Math. Biol. (2025)](https://link.springer.com/article/10.1007/s11538-025-01552-9)) and basin-entropy detection of bifurcations ([Wagemakers et al. (2023)](https://arxiv.org/abs/2303.16996)) — are detection/classification methods with no proof guarantees attached.

### Inferences
- Certification can establish object-level or regional/global statements, depending on the theorem, outer approximation and domain. This survey does not quantify its cost or establish that all whole-landscape maps are uncertified. A certificate for a discretized model also needs a separate continuum-error argument to imply a PDE theorem.
- Deflation (KQ2) sits in tier (ii): it broadens discovery but does not certify; Conley–Morse (KQ4) and radii-polynomial methods (this KQ) sit in tier (iii) and are the only tools found here that produce mathematical statements.

### Gaps
- No source found this session certifies a full snaking diagram or a PDE localized-state branch (validated continuation of periodic orbits in polynomial ODE fields is the closest demonstrated capability).
- Cost/scaling comparisons between certification methods were not found in searched sources.

## KQ6: What recent data-driven methods address pattern descriptors, bifurcation inference/design, nonlinear pattern comparison, and active exploration?

### Takeaway
Since 2021 a fast-growing cluster of data-driven tools addresses the landscape problem *without* requiring equations or exhaustive sweeps: SINDyCP discovers parameterized pattern-forming dynamics and extrapolates bifurcation predictions far from onset; Bayesian/active schemes (optimization, basin entropy, control-based continuation) target bifurcations and phase behavior directly; topological (persistent-homology) descriptors classify pattern landscapes; ML inverse problems recover pattern-forming parameters from noisy experimental patterns; all are detection/inference tools, none certify.

### Cited Findings
- SINDyCP (sparse identification of nonlinear dynamics with control parameters) discovers parameterized pattern-forming dynamics from data (discrete maps up to PDEs), recovers universal pattern-formation equations and their bifurcation dependencies, and extrapolates predictions beyond the weakly nonlinear regime near instability onset; a weak formulation mitigates noise — [Nicolaou, Huo, Chen, Brunton & Kutz, Phys. Rev. Research 5, L042017 (2023)](https://link.aps.org/doi/10.1103/PhysRevResearch.5.L042017).
- Bifurcation diagrams can be used as inference targets for parameters; the approach notes the unsupervised-search difficulty that when bifurcations are absent no distance is defined between data and prediction — [NeurIPS 2021, "Parameter Inference with Bifurcation Diagrams"](https://proceedings.neurips.cc/paper/2021/file/2c6ae45a3e88aee548c0714fad7f8269-Paper.pdf).
- An active-learning/Bayesian-optimization framework discovers saddle-node or Hopf bifurcations when large numbers of evaluations are impractical — [Psarellis et al. (2024) arXiv:2406.11141](https://arxiv.org/abs/2406.11141); published as [Chaos 35, 053159 (2025)](https://pubs.aip.org/aip/cha/article/35/5/053159/3347820/Active-search-for-bifurcations).
- Basin entropy — an equation-free measure of basin unpredictability — can be used to explore parameter space and detect bifurcation-induced basin transformations — [Wagemakers et al. (2023) arXiv:2303.16996](https://arxiv.org/abs/2303.16996).
- Persistent homology plus topological clustering classifies how pattern topology depends on parameters in Turing systems (chlorite–iodide–malonic acid demonstration) — [Spector & Harrington, Bull. Math. Biol. (2025)](https://link.springer.com/article/10.1007/s11538-025-01552-9); [arXiv:2409.20491 (2024)](https://arxiv.org/abs/2409.20491).
- Persistent homology quantitatively extracts structural information from Turing patterns and biological skin patterns, enabling structure-focused analysis — [Kanamori, IJAT (2025)](https://www.fujipress.jp/ijat/au/ijate001900040630/).
- Machine learning solves the inverse problem of identifying parameters of artificial and experimental Turing patterns from corrupted/noisy data — [Matas-Gil et al., iScience (2024)](https://www.sciencedirect.com/science/article/pii/S2589004224010447).
- Physics-informed neural networks have been used to design Turing systems (preprint) — [Kho (2022) arXiv:2211.13464](https://arxiv.org/pdf/2211.13464).
- Inverse *design* uses of Turing patterns: converting anisotropic Turing-pattern material distributions into 3D-printable inflatable shape-morphing structures — [Tanaka et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9916983/); reaction–diffusion-generated Turing patterns for inverse design of microchannel fluid flow networks — [Dede, Struct. Multidisc. Optim. (2020)](https://link.springer.com/article/10.1007/s00158-020-02580-w).
- Control-based continuation enables systematic experimental exploration of bifurcations in physical nonlinear experiments without a model (cited 88) — [Barton (2012) arXiv:1209.3713](https://arxiv.org/abs/1209.3713).
- Bayesian optimization supports turn-key constrained exploration of input/phase spaces in chemistry contexts (cited 47) — [Roussel, Nat. Commun. (2021)](https://www.nature.com/articles/s41467-021-25757-3); and Bayesian active learning accelerates CALPHAD phase-diagram discovery in materials science (2026) — [BALPI, ScienceDirect (2026)](https://www.sciencedirect.com/org/science/article/pii/S2635098X26001452).
- Deep learning can predict tipping-point occurrence, with training libraries extended by simulated data exhibiting the target bifurcation types — [PMC12303098](https://pmc.ncbi.nlm.nih.gov/articles/PMC12303098/).

### Inferences
- Convergent trend across 2021–2026: replacing dense parameter sweeps with *query-efficient, uncertainty-aware* exploration of qualitative structure (BO for bifurcations; basin entropy as a scalar landscape observable; active learning for phase diagrams), while *interpretable* models (SINDyCP-style) retain the ability to extrapolate bifurcations outside the training regime — a capability generic black-box surrogates were not found to demonstrate for bifurcation diagrams.
- Pattern *comparison* is currently dominated by topological descriptors (persistent homology), which classify patterns by topology rather than by amplitude/phase statistics; no source found here provides a cross-model metric linking a measured pattern back to a specific branch of a computed landscape (see open directions).

### Gaps
- Neural-operator / PDE-surrogate literature surfaced only general solvers (e.g., [Zhou (2025)](https://www.sciencedirect.com/science/article/pii/S0045782525002622)), none with demonstrated bifurcation-diagram fidelity claims verified this session.
- Optimal-transport or other metric pattern-comparison schemes were not found in these searches.
- The tipping-point deep-learning work and PH pattern papers were verified via abstracts; quantitative performance figures were not extracted.

## KQ7: Annotated primary bibliography (dates, demonstrated capabilities, limits)

### Takeaway
The primary anchors below (each verified via the linked source this session) span theory, software, certification, and data-driven inference; capability statements are scoped to what the sources themselves demonstrate.

### Cited Findings

**Localized states / snaking (theory & applications)**
- [Burke & Knobloch, Chaos (2007)](https://pubmed.ncbi.nlm.nih.gov/17903009/) — review; snakes-and-ladders organization of localized states; 1D structure origin. *Limits:* theory/review; 1D spatial dynamics; model equations.
- [Burke & Knobloch (2007), Phys. Lett. A](https://www.sciencedirect.com/science/article/abs/pii/S0375960106013612) — snaking adds rolls symmetrically at both ends toward a periodic wavetrain. *Limits:* Swift–Hohenberg model.
- [Roberts et al., Nonlinearity (2025)](https://iopscience.iop.org/article/10.1088/1361-6544/ae1376/pdf) — contact-defect snaking in reaction–diffusion systems. *Limits:* 1D theory.
- [Plane Couette snaking, JFM (2016)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/homoclinic-snaking-in-plane-couette-flow-bending-skewing-and-finitesize-effects/6484E2538F10F3DC307271D531D61256) — snaking of steady localized structures in a Navier–Stokes shear flow. *Limits:* exact coherent structures; high-dimensional computation; no experimental realization.
- [Cylindrical-shell localized post-buckling, Proc. R. Soc. A (2017)](https://royalsocietypublishing.org/rspa/article/473/2205/20170177/57457/Fully-localized-post-buckling-states-of) — azimuthal snaking of dimples under axial load. *Limits:* shell model; quasi-static.
- [Gomila, IMA J. Appl. Math. (2021)](https://academic.oup.com/imamat/article/86/5/1094/6352381) — radial snaking; curvature effects on localized circular fronts. *Limits:* 2D axisymmetric theory/numerics.

**Software (continuation & bifurcation)**
- [BifurcationKit organization (anchor, accessed 2026-10-19)](https://bifurcationkit.github.io) — Julia ecosystem: core BK, homoclinics (HclinicBifurcationKit.jl), multi-parameter continuation (MultiParamContinuation.jl), FEM PDEs (GridapBifurcationKit.jl), delay DDEs, ANM, periodic Schur; SciML (ModelingToolkit/Catalyst) integration; GPU 2D Swift–Hohenberg tutorial. *Limits:* ecosystem is young relative to AUTO; docs-level evidence.
- [Doedel et al., AUTO-07p manual (2012)](https://depts.washington.edu/bdecon/workshop2012/auto-tutorial/documentation/auto07p%20manual.pdf); [dsweb entry](https://archive-dsweb.siam.org/Software/auto-07p.html) — continuation of equilibria/periodic orbits, bifurcation detection; in use since 1980. *Limits:* equation-based ODEs; C/Fortran era ergonomics.
- [Demaeyer, JOSS (2025)](https://joss.theoj.org/papers/10.21105/joss.08079.pdf) — auto-AUTO/AUTO2 Python layer over AUTO-07p. *Limits:* wrapper, not new mathematics.
- [Uecker, NMTMA (2014)](https://www.cambridge.org/core/journals/numerical-mathematics-theory-methods-and-applications/article/pde2path-a-matlab-package-for-continuation-and-bifurcation-in-2d-elliptic-systems/11B12FB6196D66ADCBE37A84944D7DC2) — pde2path, 2D elliptic PDE continuation (cited 300+). *Limits:* Matlab; FEM-specific workflows.
- [Uecker, arXiv:1908.05211 (2019)](https://arxiv.org/abs/1908.05211) — tutorial: 1D/2D/3D patterns, branch switching. *Limits:* tutorial scope.
- [Uecker, survey (2022)](https://link.springer.com/article/10.1365/s13291-021-00241-5) — algorithms + applications of continuation/bifurcation in nonlinear PDEs. *Limits:* review; software examples centered on pde2path.

**Deflation & branch discovery**
- [Farrell, Beentjes & Birkisson, arXiv:1603.00809 (2016)](https://arxiv.org/html/1603.00809v1) — deflated continuation for disconnected diagrams. *Limits:* no completeness guarantee.
- [Chapman (2017)](https://www.jstor.org/stable/45093272) — deflation + arclength on Carrier's problem. *Limits:* single model.
- [Farrell (2025), Oxford ORA](https://ora.ox.ac.uk/objects/uuid:6812f36b-2b32-460c-b742-1878ac339be2) — deflation for multiple solutions of nonlinear systems with continuation. *Limits:* repository record; abstract-level verification.
- [Pichi & Strazzullo, CNSNS (2025)](https://www.sciencedirect.com/science/article/pii/S1007570425003521) — deflation-based certified greedy + adaptive ROMs for bifurcating parametric PDEs. *Limits:* "certified" refers to reduced-order treatment; model problems.
- [SSRN ADC preprint](https://papers.ssrn.com/sol3/Delivery.cfm/218892ea-d1f0-46d3-b94c-61997f105da0-MECA.pdf?abstractid=6922884&mirid=1) — adaptive deflation for hyperelastic post-buckling. *Limits:* preprint.
- [arXiv:2602.12940 (2026)](https://arxiv.org/html/2602.12940v2) — bifurcation-curve detection via arclength + deflation. *Limits:* preprint; abstract-level verification.

**Global topology (Conley/Morse)**
- [Mischaikow, JCD (2016)](https://www.aimsciences.org/article/doi/10.3934/jcd.2016001) — rigorous discretization strategies for Conley indices/Morse decompositions. *Limits:* cost grows with dimension.
- [Batko et al., FoCM (2020)](https://link.springer.com/article/10.1007/s10208-020-09444-1) — combinatorial↔classical dynamics bridge (multivector maps). *Limits:* combinatorial machinery; small examples.
- [Arai, Kalies et al.](https://www.math.fau.edu/files/kalies_papers/dsagdms.pdf) — Conley–Morse database schema for global dynamics of maps. *Limits:* maps, not PDEs.
- [CHomP user's guide](https://chomp.rutgers.edu/Projects/survey/cmdbSurvey.pdf) — Conley–Morse Database usage. *Limits:* research software.
- [SIAM ADS (2022)](https://epubs.siam.org/doi/10.1137/21M143162X) — persistence of Conley–Morse graphs under refinement. *Limits:* combinatorial setting.

**Certification (validated numerics / CAP)**
- [Lessard, Physica D (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0167278916000294) — AD-Fourier radii polynomials; existence proofs for periodic solutions. *Limits:* per-solution; polynomial-in-time scaling challenges.
- [van den Berg et al., JCD (2021)](https://www.aimsciences.org/article/doi/10.3934/jcd.2021004) — validated continuation of periodic orbits in polynomial vector fields. *Limits:* polynomial vector fields.
- [SIADS (2023/24)](https://epubs.siam.org/doi/10.1137/22M1499418) — CAP for periodic orbits under state-dependent delays. *Limits:* delayed perturbation setting.
- [arXiv:2605.07500 (2026)](https://arxiv.org/html/2605.07500v1) — survey of CAPs in dynamical systems (radii polynomials). *Limits:* survey.
- [CISM course (2025)](https://www.cism.it/event/computer-assisted-proofs-in-delayed-and-infinite-dimensional-dynamical-systems-8447) — CAPs in delayed/infinite-dimensional systems. *Limits:* educational event page.

**Data-driven descriptors, inference/design, active exploration**
- [Nicolaou et al., PRR (2023)](https://link.aps.org/doi/10.1103/PhysRevResearch.5.L042017) — SINDyCP: parameterized dynamics discovery + bifurcation extrapolation beyond weakly nonlinear onset; weak form for noise. *Limits:* needs adequate training data near/onset regimes; detection/inference tier (no certification).
- [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/2c6ae45a3e88aee548c0714fad7f8269-Paper.pdf) — parameter inference from bifurcation diagrams. *Limits:* unsupervised-search difficulty when no bifurcations present.
- [Psarellis et., Chaos (2025) / arXiv 2024](https://pubs.aip.org/aip/cha/article/35/5/053159/3347820/Active-search-for-bifurcations) — BO active search for saddle-node/Hopf bifurcations. *Limits:* designed for expensive-evaluation settings; local bifurcation types.
- [Wagemakers et al. (2023)](https://arxiv.org/abs/2303.16996) — basin entropy as equation-free bifurcation explorer. *Limits:* requires basin sampling; heuristic.
- [Spector & Harrington (2024/2025)](https://link.springer.com/article/10.1007/s11538-025-01552-9) — persistent-homology classification of Turing-pattern parameter dependence (CIMA). *Limits:* classification, not mechanism; model data.
- [Kanamori, IJAT (2025)](https://www.fujipress.jp/ijat/au/ijate001900040630/) — PH analysis of Turing and biological skin patterns. *Limits:* structural descriptors only.
- [Matas-Gil et al., iScience (2024)](https://www.sciencedirect.com/science/article/pii/S2589004224010447) — ML inverse problem for experimental and artificial Turing patterns under noise. *Limits:* parameter recovery, not landscape reconstruction.
- [Kho (2022), arXiv:2211.13464](https://arxiv.org/pdf/2211.13464) — PINN design of Turing systems. *Limits:* preprint; three-pattern demonstration.
- [Tanaka et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9916983/) — Turing-pattern-based inverse design of inflatable shape-morphing structures. *Limits:* fabrication-oriented, no bifurcation analysis.
- [Dede, SMO (2020)](https://link.springer.com/article/10.1007/s00158-020-02580-w) — reaction–diffusion pattern generation for microchannel inverse design. *Limits:* geometry application, not dynamical landscape analysis.
- [Barton (2012), arXiv:1209.3713](https://arxiv.org/abs/1209.3713) — control-based continuation for experiments. *Limits:* requires real-time control hardware; equilibrium branches primarily.
- [Roussel, Nat. Commun. (2021)](https://www.nature.com/articles/s41467-021-25757-3) — turn-key constrained BO exploration (chemistry phase behavior). *Limits:* chemistry domain; objective-based, not dynamical-systems-specific.
- [BALPI (2026)](https://www.sciencedirect.com/org/science/article/pii/S2635098X26001452) — Bayesian active learning for CALPHAD phase-diagram discovery. *Limits:* materials thermodynamics; adjacent-domain evidence of trend.
- [Deep learning tipping prediction, PMC12303098](https://pmc.ncbi.nlm.nih.gov/articles/PMC12303098/) — train on simulated bifurcation libraries to predict tipping. *Limits:* needs representative simulated data; detection tier.

### Inferences
- These categories organize the reviewed uses, not impermeable divisions of the literature. Label the guarantee for each output; absence of a certified-ML example in this search is not evidence that certified-ML methods do not exist.

### Gaps
- Some items verified only at abstract/preprint level (flagged inline); full-text capabilities and limitations for those are not fully extracted.

## KQ8: What plausible open directions follow from the cited gaps?

### Takeaway
Three directions are directly grounded in the cited literature's demonstrated capabilities and explicit gaps: (1) coupling active data-driven discovery with deflated continuation and certification; (2) scaling certified Conley–Morse global databases to PDE pattern problems; (3) establishing descriptor-to-branch grounding (metrics/benchmarks) for nonlinear pattern comparison. These are framed as plausible next steps; no novelty claims beyond the sources.

### Cited Findings (basis for each direction)
- **Direction 1 — Active discovery → deflated continuation → certification pipeline.** Active BO locates bifurcations cheaply ([Psarellis, Chaos 2025](https://pubs.aip.org/aip/cha/article/35/5/053159/3347820/Active-search-for-bifurcations)); SINDyCP extrapolates bifurcation structure beyond weakly nonlinear training data ([Nicolaou et al., PRR 2023](https://link.aps.org/doi/10.1103/PhysRevResearch.5.L042017)); deflated continuation then computes the discovered branches and their disconnected relatives ([Farrell et al. 2016](https://arxiv.org/html/1603.00809v1); [Farrell 2025](https://ora.ox.ac.uk/objects/uuid:6812f36b-2b32-460c-b742-1878ac339be2); [Pichi & Strazzullo 2025](https://www.sciencedirect.com/science/article/pii/S1007570425003521)); validated continuation can certify selected periodic-orbit branches ([van den Berg et al. 2021](https://www.aimsciences.org/article/doi/10.3934/jcd.2021004)). No published integration of all three stages was found in this session's searches — the coupling is a plausible direction, not an existing artifact.
- **Direction 2 — Conley–Morse global topology at PDE scale.** Rigorous Conley–Morse computation is established for maps and moderate dimensions ([Mischaikow 2016](https://www.aimsciences.org/article/doi/10.3934/jcd.2016001); [Batko et al. 2020](https://link.springer.com/article/10.1007/s10208-020-09444-1); [Conley–Morse Database](https://chomp.rutgers.edu/Projects/survey/cmdbSurvey.pdf)); GPU-scale continuation for 2D patterns exists in the anchor ecosystem ([BifurcationKit GPU tutorial](https://bifurcationkit.github.io)); but no source found applies Conley–Morse databases to discretized PDE pattern landscapes — the scaling gap is documented by absence in these searches and by the methods' demonstrated map-level scope.
- **Direction 3 — Descriptor-to-branch grounding for pattern comparison.** PH descriptors classify pattern topology vs parameters ([Spector & Harrington 2025](https://link.springer.com/article/10.1007/s11538-025-01552-9); [Kanamori 2025](https://www.fujipress.jp/ijat/au/ijate001900040630/)); basin entropy flags qualitative changes near bifurcations ([Wagemakers 2023](https://arxiv.org/abs/2303.16996)); ML inverse problems recover parameters from real experimental patterns ([Matas-Gil 2024](https://www.sciencedirect.com/science/article/pii/S2589004224010447)); yet no metric or benchmark found this session connects a measured pattern to a specific branch/basin within a computed landscape across models — establishing such grounded benchmarks (e.g., on CIMA-type experimental data as in the PH and ML papers) is a plausible direction.

### Inferences
- Tentative opportunities, not established novelty: test integration cost and coverage for Direction 1, state-space/truncation scalability for Direction 2, and descriptor ambiguity plus branch/basin ground truth for Direction 3. Failure to find a publication or metric is not itself a technical blocker and cannot establish priority.

### Gaps
- These directions are reasoned syntheses of cited capabilities and observed absences; a dedicated search for in-progress work (very recent arXiv postings after the searched corpus, workshop proceedings) might find partial realizations this session did not surface.
