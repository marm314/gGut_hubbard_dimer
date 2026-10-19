# Graph Report - gGut_mau  (2026-10-19)

## Corpus Check
- Corpus is ~29,616 words - fits in a single context window. You may not need a graph.

## Summary
- 308 nodes · 646 edges · 11 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.83)
- Token cost: 187,028 input · 0 output

## Community Hubs (Navigation)
- gGA Theory & DeltaV Convention
- Second-Order Stationarity Test
- Spectral-Function Plotting & Videos
- Exact Diagonalization with Site-Splitting DeltaV
- Direct-Minimization Solver (NotImpDimer)
- Ng=2 DeltaV=2 Direct-Minimization Scan
- Ng=2 DeltaV=0 Direct-Minimization Scan
- Ng=4 DeltaV=2 Scan
- Ng=4 DeltaV=0 Scan
- Exact Diagonalization (DeltaV=0)
- ED Pole Comparison Concepts

## God Nodes (most connected - your core abstractions)
1. `Ghost Gutzwiller Approximation for the Half-Filled Hubbard Dimer (code.pdf)` - 33 edges
2. `NotImpDimer` - 23 edges
3. `run_gga()` - 20 edges
4. `analysis()` - 17 edges
5. `Ghost Gutzwiller by Direct Minimization: not_imp_gga_dimer.py (not_impurity.pdf)` - 16 edges
6. `main()` - 12 edges
7. `Exact Spectral Function of the Half-Filled Hubbard Dimer with Site-Splitting Potential DeltaV (ED_deltaV.pdf)` - 10 edges
8. `build()` - 9 edges
9. `imp_terms()` - 9 edges
10. `compute_spectral_function()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `compute_spectral_function()` --semantically_similar_to--> `Exact 4-pole spectral function via Lehmann representation`  [INFERRED] [semantically similar]
  gga_dimer.py → scan_ed_deltaV/ED_deltaV.pdf
- `qp_state()` --implements--> `Quasiparticle Hamiltonian H_qp(R, lambda)`  [EXTRACTED]
  gga_dimer.py → doc/code.pdf
- `FCIDUMP text-quantization emulation for V and lambda^c (determinism vs reference)` --rationale_for--> `impurity_solve()`  [EXTRACTED]
  doc/code.pdf → gga_dimer.py
- `gGut_hubbard_dimer Project README` --references--> `run_gga()`  [INFERRED]
  README.md → gga_dimer.py
- `Ghost Gutzwiller by Direct Minimization: not_imp_gga_dimer.py (not_impurity.pdf)` --references--> `NotImpDimer`  [EXTRACTED]
  doc/not_impurity.pdf → not_imp_gga_dimer.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Python gGA routines ported from reference C++ implementation** — gga_dimer_run_gga, gga_dimer_grad_s, gga_dimer_embedding_lambda_c, gga_dimer_apply_s_inv, ext_cpp_reference_ggut [EXTRACTED 1.00]
- **Worked examples validating gga_dimer.py against the C++ reference** — examples_u10_readme, examples_u2_backward_readme, examples_u2_forward_readme, gga_dimer_run_gga, ext_cpp_reference_ggut [INFERRED 0.85]
- **Sections building the exact DeltaV spectral-function derivation** — concept_molecular_orbitals_gu, concept_ph_symmetry_deltav, concept_exact_spectral_function_lehmann [EXTRACTED 1.00]

## Communities (11 total, 0 thin omitted)

### Community 0 - "gGA Theory & DeltaV Convention"
Cohesion: 0.06
Nodes (68): check_hessian.py (Hessian / second-order test of converged points), one_iteration(), check_stationarity.py (checks Euler-Lagrange stationarity of converged points), Bath-potential operator-ordering sign convention in H_imp (annihilation-before-creation fix), Constrained variational Lagrangian (five Lagrange multipliers), DeltaV placement convention: undressed H_loc shift vs dressed H_qp field, and why undressed was kept, Site-splitting potential DeltaV (mu0=U/2+DeltaV, mu1=U/2-DeltaV), Direct minimization (no impurity problem) approach (+60 more)

### Community 1 - "Second-Order Stationarity Test"
Cohesion: 0.07
Nodes (50): analyse(), basis_sym(), build(), constraint_values(), constraint_vector(), curve_test(), energy(), feasible_curve_energy() (+42 more)

### Community 2 - "Spectral-Function Plotting & Videos"
Cohesion: 0.07
Nodes (37): Overlay the exact (ED) spectral poles of the half-filled Hubbard dimer on the…, Are the lowest-energy solutions of the direct minimization (scan_ng2_not_imp/)…, collections, contextlib, glob, io, Delta V/t=2.0 analogue of make_A_omega_video.py, for scan_ng2_imp_dV2/,…, make_video() (+29 more)

### Community 3 - "Exact Diagonalization with Site-Splitting DeltaV"
Cohesion: 0.09
Nodes (30): Exact 4-pole spectral function via Lehmann representation, Bonding/antibonding molecular orbitals g,u (DeltaV-dependent), Site-exchange x particle-hole product symmetry (survives DeltaV!=0), build_c_ops(), build_hamiltonian(), eigensectors(), _kron_list(), exact_dimer_deltaV.py (brute-force ED with DeltaV, 16-dim Fock space) (+22 more)

### Community 4 - "Direct-Minimization Solver (NotImpDimer)"
Cohesion: 0.14
Nodes (13): argparse, _F_inv_sqrt(), NotImpDimer, gradL(), Ghost Gutzwiller for the half-filled Hubbard dimer WITHOUT the impurity…, I selects the fragment's own H_loc (self.Hloc[I] = U n_up n_dn - mu_I n, mu_I =…, Starting point from an (R, lambda) guess, as read by gga_dimer.py: X = occupied…, Hessian of the Lagrangian L = E + mu.c (mu fixed by grad E + J^T mu = 0) with… (+5 more)

### Community 5 - "Ng=2 DeltaV=2 Direct-Minimization Scan"
Cohesion: 0.17
Nodes (19): branch_color(), continued_point(), converge(), ed_ground_state_energy(), ed_poles(), first_point(), ok(), U-continuation scans (Ng=2, Delta V/t=2.0) with the direct minimization of… (+11 more)

### Community 6 - "Ng=2 DeltaV=0 Direct-Minimization Scan"
Cohesion: 0.21
Nodes (13): branch_color(), converge(), ed_poles(), gga_dimer_envelope(), U-continuation scans (Ng=2) with the direct minimization of…, prime: optional list of U values traversed first WITHOUT saving., Lower envelope of the E_var of the two gga_dimer.py scans…, SLSQP passes until feasible and stationary; returns (z, passes, |c|, tangent… (+5 more)

### Community 7 - "Ng=4 DeltaV=2 Scan"
Cohesion: 0.23
Nodes (10): branch_color(), converge(), embed_ng2_in_ng4(), finalize(), U scan with FOUR ghosts (Ng=4, Neff=5), Delta V/t=2.0, by the direct…, Minimize from every candidate start; keep the lowest feasible result., solve_point(), start_candidates() (+2 more)

### Community 8 - "Ng=4 DeltaV=0 Scan"
Cohesion: 0.23
Nodes (9): branch_color(), converge(), embed_ng2_in_ng4(), finalize(), U scan with FOUR ghosts (Ng=4, Neff=5) by the direct minimization of…, Minimize from every candidate start; keep the lowest feasible result., solve_point(), start_candidates() (+1 more)

### Community 9 - "Exact Diagonalization (DeltaV=0)"
Cohesion: 0.43
Nodes (6): build_c_ops(), build_hamiltonian(), _kron_list(), Exact diagonalization of the 2-site Hubbard dimer (4 orbitals: 2 sites x 2…, Jordan-Wigner annihilation operators c_i, i = 0..n_modes-1., solve()

### Community 10 - "ED Pole Comparison Concepts"
Cohesion: 0.60
Nodes (5): DeltaV site-splitting potential (Hubbard dimer), Exact diagonalization (ED) spectral-function poles, ED pole energy vs U (DeltaV=0 vs DeltaV=2), ED quasi-particle pole branch (DeltaV=0 and DeltaV=2), ED satellite pole branch (DeltaV=0 and DeltaV=2)

## Knowledge Gaps
- **13 isolated node(s):** `check_stationarity.py (checks Euler-Lagrange stationarity of converged points)`, `check_hessian.py (Hessian / second-order test of converged points)`, `Ghost Gutzwiller Approximation (gGA)`, `Bonding/antibonding molecular orbitals g,u (DeltaV-dependent)`, `Site-exchange x particle-hole product symmetry (survives DeltaV!=0)` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 108 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Ghost Gutzwiller Approximation for the Half-Filled Hubbard Dimer (code.pdf)` connect `gGA Theory & DeltaV Convention` to `Exact Diagonalization with Site-Splitting DeltaV`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `NotImpDimer` connect `Direct-Minimization Solver (NotImpDimer)` to `gGA Theory & DeltaV Convention`, `Ng=2 DeltaV=2 Direct-Minimization Scan`, `Ng=2 DeltaV=0 Direct-Minimization Scan`, `Ng=4 DeltaV=2 Scan`, `Ng=4 DeltaV=0 Scan`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `run_gga()` connect `gGA Theory & DeltaV Convention` to `Second-Order Stationarity Test`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `analysis()` (e.g. with `modC()` and `modD()`) actually correct?**
  _`analysis()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `check_stationarity.py (checks Euler-Lagrange stationarity of converged points)`, `check_hessian.py (Hessian / second-order test of converged points)`, `Ghost Gutzwiller Approximation (gGA)` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `gGA Theory & DeltaV Convention` be split into smaller, more focused modules?**
  _Cohesion score 0.058385093167701865 - nodes in this community are weakly interconnected._
- **Should `Second-Order Stationarity Test` be split into smaller, more focused modules?**
  _Cohesion score 0.06623376623376623 - nodes in this community are weakly interconnected._