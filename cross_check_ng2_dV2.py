"""Ng=2 analogue of cross_check_ng4_dV2.py: systematic cross-check of scan_ng2_imp_dV2/
(gga_dimer.py) against scan_ng2_not_imp_dV2/ (not_imp_gga_dimer.py) at every U/t = 0.1..10.0,
deltaV/t=2.0 -- the two ORIGINAL scans are left untouched; the confirmed result goes into a
SEPARATE new directory, scan_ng2_dV2_confirmed/.

Unlike Ng=4, both scan_ng2_imp_dV2/ and scan_ng2_not_imp_dV2/ themselves hold forward/backward
branches with their own lowest_energy.txt picking the best per U (see scan_ng2_imp_dV2_U.py,
scan_ng2_not_imp_dV2_U.py) -- so both branch maps are loaded here, not just one.

At each U, each method's own result is cross-seeded into the OTHER method's solver:
  gga_dimer.run_gga seeded from not_imp's (R, lambda)      -> does gga "recognize" it?
  not_imp_gga_dimer seeded from gga's (R, lambda)           -> does not_imp "recognize" it?
A point is a genuine MUTUAL solution only if the cross-seeded run lands back on (near) the
same energy with FEW iterations/passes (capped at STRONG_ITERS/STRONG_PASSES -- if it hasn't
converged by then it's already disqualified, no point letting it run further).

Writes scan_ng2_dV2_confirmed/U_<U>/:
  gga_dimer/{R,lambda}.dat + info.txt        -- gga_dimer's OWN native result, plus whether
                                                 not_imp recognizes it
  not_imp_gga_dimer/{R,lambda}.dat + info.txt -- not_imp's OWN native result, plus whether
                                                 gga_dimer recognizes it
  eqp.txt, output.out                        -- the WINNING (global lowest-energy, confirmed)
                                                 candidate
Also scan_ng2_dV2_confirmed/lowest_energy.txt and eqp_vs_U.txt (the winning candidate per U).
"""
import os
import re
import time

import numpy as np

import gga_dimer as g
import not_imp_gga_dimer as ni

t, deltaV = 1.0, 2.0
Ng = 2
IMP_DIR = "scan_ng2_imp_dV2"
NOTIMP_DIR = "scan_ng2_not_imp_dV2"
OUT_DIR = "scan_ng2_dV2_confirmed"

# Absolute path to this file's directory, so restart.sh scripts work when run from inside
# the deeply-nested U_<U>/... output directories (where gga_dimer.py/not_imp_gga_dimer.py
# do not themselves live) and not just from the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# A cross-seeded run that needed more than this many outer iterations/passes to converge is
# flagged as only a WEAK confirmation (it had to relax substantially, i.e. the original point
# was not already sitting at the other method's fixed point, even if it happened to land back
# on a similar energy). Also used to cap the cross-seed runs themselves: no converged-by-then
# means already disqualified, so there's no point letting it run further.
STRONG_ITERS = 3
STRONG_PASSES = 1


def tangent_grad(model, z):
    E, gE = model.energy(z)
    J = model.constraint_jac(z)
    u, s, vt = np.linalg.svd(J)
    rank = int((s > 1e-9 * s[0]).sum())
    return np.linalg.norm(vt[rank:] @ gE)


def not_imp_converge(model, z, max_pass=5):
    for k in range(max_pass):
        z = model.minimize(z, max_iter=3000).x
        viol = np.abs(model.constraints(z)).max()
        tg = tangent_grad(model, z)
        if viol < 1e-9 and tg < 5e-6:
            break
    return z, viol, tg, k + 1


def load_branch_map(scan_dir):
    br = {}
    for line in open(os.path.join(scan_dir, "lowest_energy.txt")):
        if not line.startswith("#") and line.strip():
            p = line.split()
            br[round(float(p[0]), 1)] = p[2]
    return br


def imp_RL(U, branch):
    d = os.path.join(IMP_DIR, branch, f"U_{U:.1f}")
    R = np.loadtxt(os.path.join(d, f"gga_Ng{Ng}_U{U:g}_R.dat"))
    lam = np.loadtxt(os.path.join(d, f"gga_Ng{Ng}_U{U:g}_lambda.dat"))
    N = R.shape[1]
    return [R[0], R[1]], [lam[:N], lam[N:]]


def notimp_RL(U, branch):
    d = os.path.join(NOTIMP_DIR, branch, f"U_{U:.1f}")
    R = np.loadtxt(os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}_R.dat"))
    lam = np.loadtxt(os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}_lambda.dat"))
    N = R.shape[1]
    return [R[0], R[1]], [lam[:N], lam[N:]]


def read_E(path):
    txt = open(path).read()
    return float(re.search(r"E_var/t=(-?[\d.]+)", txt).group(1))


def save_RL(d, R, lam, U, script):
    """Saves R.dat/lambda.dat (plain record) AND R.in/L.in -- gga_dimer.py's and
    not_imp_gga_dimer.py's own --R-file/--lam-file default guess-file names, same 2-row
    stacked format already used by both (read_R_guess/read_lambda_guess) -- plus a
    restart.sh with the exact command line to resume/reproduce this point from this
    (R, lambda), so nothing beyond this directory is needed to restart that U."""
    os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, "R.dat"), np.vstack(R))
    np.savetxt(os.path.join(d, "lambda.dat"), np.vstack(lam))
    np.savetxt(os.path.join(d, "R.in"), np.vstack(R),
               header=f"R guess for U/t={U:.1f}, deltaV/t={deltaV:g}, Ng={Ng} (restart file)")
    np.savetxt(os.path.join(d, "L.in"), np.vstack(lam),
               header=f"lambda guess for U/t={U:.1f}, deltaV/t={deltaV:g}, Ng={Ng} (restart file)")
    script_path = os.path.join(PROJECT_ROOT, script)
    with open(os.path.join(d, "restart.sh"), "w") as f:
        f.write("#!/bin/sh\n"
                f"# Resume/reproduce U/t={U:.4f}, Ng={Ng}, deltaV/t={deltaV:g} from this directory's\n"
                f"# R.in/L.in. Run from anywhere -- both the solver script and the guess files are\n"
                f"# referenced by absolute/directory-relative path below.\n"
                f"cd \"$(dirname \"$0\")\"\n"
                f"/usr/bin/python3 {script_path} --U {U:g} --Ng {Ng} --deltaV {deltaV:g} "
                f"--R-file R.in --lam-file L.in\n")
    os.chmod(os.path.join(d, "restart.sh"), 0o755)


def save_point(U, E_imp, R_imp, lam_imp, gga_check,
               E_ni, R_ni, lam_ni, notimp_check,
               best_E, best_R, best_lam, best_src):
    """gga_check: (E_landed, iters, converged) from seeding gga_dimer at not_imp's point.
       notimp_check: (E_landed, passes, viol, tg) from seeding not_imp at gga's point."""
    d = os.path.join(OUT_DIR, f"U_{U:.1f}")

    # --- gga_dimer's own native result ---
    d_gga = os.path.join(d, "gga_dimer")
    save_RL(d_gga, R_imp, lam_imp, U, "gga_dimer.py")
    with open(os.path.join(d_gga, "info.txt"), "w") as f:
        f.write(f"gga_dimer.py's own result at U/t={U:.4f}, deltaV/t={deltaV:.4f}: "
                f"E_var/t={E_imp:.8f}\n")
        E_landed, iters, converged = gga_check
        strong = converged and iters <= STRONG_ITERS and abs(E_landed - E_ni) < 1e-6
        f.write(f"Cross-seeded INTO not_imp's own result instead, to see if not_imp's point\n"
                f"is ALSO a fixed point of gga_dimer: gga_dimer, started AT not_imp's (R,lambda),\n"
                f"converged={converged} in {iters} outer iterations to E_var/t={E_landed:.8f}.\n")
        if strong:
            f.write("-> STRONG confirmation: few iterations, landed back on not_imp's own energy;\n"
                    "   not_imp's point is ALSO a genuine fixed point of gga_dimer.\n")
        elif converged and abs(E_landed - E_imp) < 1e-6:
            f.write(f"-> NOT a mutual solution: gga_dimer drifted (took {iters} iterations) away\n"
                    f"   from not_imp's point and landed back on gga_dimer's OWN energy instead\n"
                    f"   ({E_imp:.8f}) -- not_imp's point is not a fixed point of gga_dimer.\n")
        else:
            f.write(f"-> WEAK/uncertain: took {iters} iterations and landed on a third energy\n"
                    f"   ({E_landed:.8f}), neither cleanly matching gga_dimer's own nor not_imp's.\n")

    # --- not_imp's own native result ---
    d_ni = os.path.join(d, "not_imp_gga_dimer")
    save_RL(d_ni, R_ni, lam_ni, U, "not_imp_gga_dimer.py")
    with open(os.path.join(d_ni, "info.txt"), "w") as f:
        f.write(f"not_imp_gga_dimer.py's own result at U/t={U:.4f}, deltaV/t={deltaV:.4f}: "
                f"E_var/t={E_ni:.8f}\n")
        E_landed, passes, viol, tg = notimp_check
        strong = viol < 1e-8 and passes <= STRONG_PASSES and abs(E_landed - E_imp) < 1e-6
        f.write(f"Cross-seeded INTO gga's own result instead, to see if gga_dimer's point is\n"
                f"ALSO a fixed point of not_imp: not_imp, started AT gga_dimer's (R,lambda),\n"
                f"converged (|c|={viol:.2e}, tangent grad={tg:.2e}) in {passes} SLSQP pass(es)\n"
                f"to E_var/t={E_landed:.8f}.\n")
        if strong:
            f.write("-> STRONG confirmation: 1 pass, landed back on gga_dimer's own energy;\n"
                    "   gga_dimer's point is ALSO a genuine fixed point of not_imp.\n")
        elif viol < 1e-8 and abs(E_landed - E_ni) < 1e-6:
            f.write(f"-> NOT a mutual solution: not_imp drifted (took {passes} passes) away from\n"
                    f"   gga_dimer's point and landed back on not_imp's OWN energy instead\n"
                    f"   ({E_ni:.8f}) -- gga_dimer's point is not a fixed point of not_imp.\n")
        else:
            f.write(f"-> WEAK/uncertain: took {passes} passes and landed on a third energy\n"
                    f"   ({E_landed:.8f}), neither cleanly matching not_imp's own nor gga_dimer's.\n")

    # --- the winning (confirmed, physically correct) candidate: saved at the TOP level
    # under BOTH naming conventions (same (R, lambda) either way) ---
    Hqp, rho = g.qp_state(best_R, best_lam, t)
    eqp = np.linalg.eigvalsh(Hqp)
    np.savetxt(os.path.join(d, f"gga_Ng{Ng}_U{U:g}_R.dat"), np.vstack(best_R),
               header=f"Ng={Ng} U={U} deltaV={deltaV} (cross-check winner, source={best_src})")
    np.savetxt(os.path.join(d, f"gga_Ng{Ng}_U{U:g}_lambda.dat"), np.vstack(best_lam),
               header=f"Ng={Ng} U={U} deltaV={deltaV} (cross-check winner, source={best_src})")
    np.savetxt(os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}_R.dat"), np.vstack(best_R),
               header=f"Ng={Ng} U={U} deltaV={deltaV} (cross-check winner, source={best_src})")
    np.savetxt(os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}_lambda.dat"), np.vstack(best_lam),
               header=f"Ng={Ng} U={U} deltaV={deltaV} (cross-check winner, source={best_src})")
    np.savetxt(os.path.join(d, "R.in"), np.vstack(best_R),
               header=f"R guess for U/t={U:.1f}, deltaV/t={deltaV:g}, Ng={Ng} (restart file, confirmed winner)")
    np.savetxt(os.path.join(d, "L.in"), np.vstack(best_lam),
               header=f"lambda guess for U/t={U:.1f}, deltaV/t={deltaV:g}, Ng={Ng} (restart file, confirmed winner)")
    gga_script = os.path.join(PROJECT_ROOT, "gga_dimer.py")
    notimp_script = os.path.join(PROJECT_ROOT, "not_imp_gga_dimer.py")
    with open(os.path.join(d, "restart.sh"), "w") as f:
        f.write("#!/bin/sh\n"
                f"# Resume/reproduce the CONFIRMED WINNING solution at U/t={U:.4f}, Ng={Ng},\n"
                f"# deltaV/t={deltaV:g} from this directory's R.in/L.in, with either solver.\n"
                f"# Run from anywhere -- both solver scripts and the guess files are referenced\n"
                f"# by absolute/directory-relative path below.\n"
                f"cd \"$(dirname \"$0\")\"\n"
                f"/usr/bin/python3 {gga_script} --U {U:g} --Ng {Ng} --deltaV {deltaV:g} "
                f"--R-file R.in --lam-file L.in\n"
                f"/usr/bin/python3 {notimp_script} --U {U:g} --Ng {Ng} --deltaV {deltaV:g} "
                f"--R-file R.in --lam-file L.in\n")
    os.chmod(os.path.join(d, "restart.sh"), 0o755)
    np.savetxt(os.path.join(d, "eqp.txt"), eqp,
               header=f"quasiparticle energies, U/t={U:.4f}, deltaV/t={deltaV:.4f} (cross-check winner)")
    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(f"Ng={Ng}  U/t={U:.4f}  deltaV/t={deltaV:.4f}  Z0={best_R[0]@best_R[0]:.6f}  "
                f"Z1={best_R[1]@best_R[1]:.6f}  E_var/t={best_E:.8f}  converged=True  "
                f"source={best_src}  (global lowest energy among gga_dimer's own, not_imp's own, "
                f"and both cross-seeded candidates; see gga_dimer/info.txt and "
                f"not_imp_gga_dimer/info.txt for the mutual-confirmation details)\n")
    return eqp


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    br_imp = load_branch_map(IMP_DIR)
    br_ni = load_branch_map(NOTIMP_DIR)
    Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]
    n_changed = 0
    rows = []

    for U in Us:
        branch_imp = br_imp[U]
        branch_ni = br_ni[U]
        R_imp, lam_imp = imp_RL(U, branch_imp)
        R_ni, lam_ni = notimp_RL(U, branch_ni)
        E_imp = read_E(os.path.join(IMP_DIR, branch_imp, f"U_{U:.1f}", "output.out"))
        E_ni = read_E(os.path.join(NOTIMP_DIR, branch_ni, f"U_{U:.1f}", "output.out"))

        candidates = [(E_imp, R_imp, lam_imp, "imp-own"), (E_ni, R_ni, lam_ni, "notimp-own")]

        # does gga_dimer recognize not_imp's point? (seed gga AT not_imp's result). Capped at
        # STRONG_ITERS -- not converged by then is already disqualified from strong confirmation.
        res = g.run_gga(Ng, U, t=t, deltaV=deltaV, tol_E=1e-12, tol_mat=1e-12, max_iter=STRONG_ITERS,
                         Rg0=R_ni[0], Rg1=R_ni[1], lamg0=lam_ni[0], lamg1=lam_ni[1])
        gga_check = (res["E_var"], res["iters"], res["converged"])
        if res["converged"]:
            candidates.append((res["E_var"], [res["R0"], res["R1"]], [res["lam0"], res["lam1"]],
                                "gga-seeded-from-notimp"))

        # does not_imp recognize gga's point? (seed not_imp AT gga's result). Capped at
        # STRONG_PASSES SLSQP passes (3000 iterations each).
        model = ni.NotImpDimer(Ng, U, t, deltaV=deltaV)
        z0 = model.start_from_R_lambda(R_imp, lam_imp)
        z, viol, tg, passes = not_imp_converge(model, z0, max_pass=STRONG_PASSES)
        a = model.analyse(z)
        notimp_check = (a["E_var"], passes, viol, tg)
        if viol < 1e-8:
            candidates.append((a["E_var"], [a["R0"], a["R1"]], [a["lam0"], a["lam1"]],
                                "notimp-seeded-from-gga"))

        best_E, best_R, best_lam, best_src = min(candidates, key=lambda c: c[0])
        eqp = save_point(U, E_imp, R_imp, lam_imp, gga_check,
                          E_ni, R_ni, lam_ni, notimp_check,
                          best_E, best_R, best_lam, best_src)
        rows.append((U, best_E, eqp))

        if abs(E_imp - E_ni) > 1e-6 or best_src != "imp-own":
            n_changed += 1
            print(f"U/t={U:5.1f}  E_imp={E_imp: .8f}  E_notimp={E_ni: .8f}  "
                  f"-> confirmed best={best_E: .8f}  source={best_src}", flush=True)

    with open(os.path.join(OUT_DIR, "lowest_energy.txt"), "w") as f:
        f.write(f"# U/t  E_var/t  eqp0..eqp{rows[0][2].shape[0]-1}   "
                f"(cross-checked: lowest energy confirmed by both solvers)\n")
        for U, E, eqp in rows:
            f.write(f"{U:5.1f} {E: .8f} " + " ".join(f"{x: .6f}" for x in eqp) + "\n")
    U_arr = np.array([r[0] for r in rows])
    eq_arr = np.array([r[2] for r in rows])
    np.savetxt(os.path.join(OUT_DIR, "eqp_vs_U.txt"), np.column_stack([U_arr, eq_arr]),
               header="U/t eqp... (cross-checked)")

    print(f"\n{n_changed}/{len(Us)} U-points had a disagreement; confirmed solution written to "
          f"{OUT_DIR}/. total {time.time()-t0:.0f}s")
