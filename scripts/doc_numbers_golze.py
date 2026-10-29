"""Numbers for doc/cederbaum.tex at V/g = 1/sqrt2, sqrt2 (rho/Omega = 0.5, 1.0). Prints everything the text/tables quote."""
import numpy as np
import exact_cederbaum as ed
import plot_cederbaum_energies as pe
import plot_cederbaum_spectra as ps

VS = pe.VS
E = pe.load_all(); B5 = pe.load_bc5(); S = ps.load_all()
print("== ED ground state (E0+2Ec+2g, 2-nu_c, rho^cv) and poles")
for V in VS:
    for tc in (0.5, 0.2, 0.1, 0.05):
        r = ed.solve(1.0, 10.0, V, tc, ht="cv")
        H, c, n = ed.build_hamiltonian(1.0, 10.0, V, tc, "cv")
        w, U = np.linalg.eigh(H); gs = U[:, 0]
        rcv = gs @ (c[ed.C_UP].T @ c[ed.V_UP]) @ gs
        print(f"V={V:.4f} tc={tc}: dE0={r['E0']+22:.3e} 2-nu={2-r['nu_c']:.3e} rho_cv={rcv:.3e}")
    for tc in (0.0, 0.5, 0.1, 0.05):
        r = ed.solve(1.0, 10.0, V, tc, ht="cv"); b = r["weights"] > 1e-3
        print(f"   poles tc={tc}: {np.round(r['poles'][b],4)} w {np.round(r['weights'][b],4)}")
print("\n== energies: fits and ranges")
for V in VS:
    r = np.array([E[f"V{V}_tc{t}"] for t in pe.TCS]); t = np.array(pe.TCS)
    e1 = r[:, 2] - r[:, 0]; e3 = r[:, 4] - r[:, 0]
    sl = np.polyfit(np.log(t), np.log(e1), 1)[0]
    print(f"V={V:.4f}: B1 dE/(V^2 tc^2) range {np.min(e1/(V**2*t**2)):.3e}..{np.max(e1/(V**2*t**2)):.3e} slope {sl:.2f}; "
          f"B3 dE {e3.min():.2e}..{e3.max():.2e}; |dnu3| max {np.abs(r[:,5]-r[:,1]).max():.1e}; |dnu1|/tc^2 {np.min(np.abs(r[:,3]-r[:,1])/t**2):.2e}..{np.max(np.abs(r[:,3]-r[:,1])/t**2):.2e}; "
          f"cons max {r[:,7].max():.0e} kkt max {r[:,8].max():.1e}; B1 conv {int(r[:,6].sum())}/{len(t)}; gain frac {np.min((e1-(r[:,4]-r[:,0]))/e1):.4f}..{np.max((e1-(r[:,4]-r[:,0]))/e1):.4f}")
print("\n== B_c=5")
for V in VS:
    keys = [f"V{V}_tc{t}_E" for t in pe.TCS if f"V{V}_tc{t}_E" in B5]
    a = np.array([B5[k] for k in keys])
    print(f"V={V:.4f}: points {len(keys)}; dE5 {np.min(a[:,4]-a[:,0]):.2e}..{np.max(a[:,4]-a[:,0]):.2e}; E5-E3 {np.min(a[:,4]-a[:,2]):.1e}..{np.max(a[:,4]-a[:,2]):.1e}; "
          f"|dnu5| max {np.max(np.abs(a[:,5]-a[:,1])):.1e}; kkt max {a[:,6].max():.1e}; converged runs {int(a[:,7].min())}..{int(a[:,7].max())}; random ok total {int(a[:,9].sum())}; spread max {a[:,8].max():.1e}")
print("\n== main poles (t_c=0.1, 0.05)")
for V in VS:
    for tc in (0.1, 0.05):
        k = f"V{V}_tc{tc}"
        r = ed.solve(1.0, 10.0, V, tc, ht="cv"); i = np.argmax(r["weights"])
        row = [f"ED {r['poles'][i]:.4f} {r['weights'][i]:.4f}"]
        for lab, p, w in (("B1", S[k+"_g1_poles"], S[k+"_g1_w"]), ("B3", S[k+"_gg_poles"], S[k+"_gg_w"]),
                          ("B5", B5.get(k+"_poles"), B5.get(k+"_w"))):
            if p is None: row.append(lab+" n/a"); continue
            j = np.argmax(w); sat = np.sort(w)[:-1].max() if len(w) > 1 else 0
            row.append(f"{lab} {p[j]:.4f} {w[j]:.4f} (max other w {sat:.1e})")
        print(f"V={V:.4f} tc={tc}: " + " | ".join(row))
        print(f"     ED satellites weights: {np.round(r['weights'][r['weights']>1e-4],4)} at {np.round(r['poles'][r['weights']>1e-4],3)}")
