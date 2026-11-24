#!/usr/bin/env python3
"""(g)Gutzwiller scan of the half-filled 4-site Hubbard model with the C++ run_ggut
(rsimp_molecule, four 1-site fragments), OBC chain and PBC ring, U/t = 0..20.

Forward sweep (U=0 -> 10, seeded at U=0 with R=1, lambda=0) and backward sweep
(Umax -> 0, seeded at Umax with the forward Umax solution); every other point is
seeded with the previous point's final_R/final_lambda. The lowest-energy branch
per U is written to lowest_energy.txt.

Conventions (same as the dimer runs):
  FCIDUMP_loc_i: U n_up n_dn - (U/2) n   -> E_var is the energy of H - (U/2) N, as printed
                 by the C++ and as in the dimer runs (= E_FCI - 2U for hubbard4_fci energies.dat).
  docc (site average) from E_loc = sum_i <U n_iup n_idn - U/2 n_i>:
                 D = (E_loc + 2U) / (4U)   (n_i = 1 at half filling)
  w in the GF is relative to mu = U/2.

    ~/.venv/bin/python scripts/scan_4site_cpp.py [--Ng 0] [--bc obc pbc apbc]

Output: scan_4site_ng{Ng}/{obc,pbc,apbc}/{forward,backward}/U_xx.xx/ (inputs, final_*, gg.out,
run.log, A.npz with A_ij(w) = -Im G_ij / pi) and per-bc summary.txt, lowest_energy.txt.
"""
import argparse
import os
import re
import shutil
import subprocess

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUT = "/home/mrodmay/Installers/gGut/build/main/run_ggut"
TEMPLATE = os.path.join(REPO, "outside_code", "forward", "gGut.input")
L = 4
ENV = os.environ.copy()
ENV["LC_ALL"] = "C"


def make_input(ng):
    """4-fragment version of the dimer gGut.input; GF on the FCI grid, all orbitals."""
    txt = open(TEMPLATE).read()
    rep = [
        (r"^Nfragments_cell .*", "Nfragments_cell 4"),
        (r"^Norbs_phys_cell .*", "Norbs_phys_cell 4"),
        (r"^FragNOrbs .*", "FragNOrbs 1 1 1 1"),
        (r"^mol_nups .*", "mol_nups 2"),
        (r"^mol_ndos .*", "mol_ndos 2"),
        (r"^Nghosts .*", "Nghosts " + " ".join([str(ng)] * L)),
        (r"^loc_fcidump_2 .*", "\n".join(f"loc_fcidump_{i} FCIDUMP_loc_{i}.dat" for i in range(2, L + 1))),
        (r"^gg_Rfile2 .*", "\n".join(f"gg_Rfile{i} guess_R{i}.dat" for i in range(2, L + 1))),
        (r"^gg_lambdfile2 .*", "\n".join(f"gg_lambdfile{i} guess_L{i}.dat" for i in range(2, L + 1))),
        (r"^gg_maxits .*", "gg_maxits 500"),
        (r"^gg_abs_tol_E .*", "gg_abs_tol_E 1.E-8"),
        (r"^gg_abs_tol_mat .*", "gg_abs_tol_mat 1.E-8"),
        (r"^gf_eta .*", "gf_eta 0.05"),
        (r"^gf_wmin .*", "gf_wmin -20.0"),
        (r"^gf_wmax .*", "gf_wmax 20.0"),
        (r"^gf_nws .*", "gf_nws 8001"),
        (r"^gf_orbs .*", "gf_orbs " + " ".join(str(i) for i in range(L))),
    ]
    for pat, new in rep:
        txt, n = re.subn(pat, new, txt, flags=re.M)
        assert n == 1, pat
    return txt


def write_files(work, U, bc, seed, ng):
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "gGut.input"), "w") as f:
        f.write(make_input(ng))
    for i in range(1, L + 1):
        with open(os.path.join(work, f"FCIDUMP_loc_{i}.dat"), "w") as f:
            f.write(f"    1    1    1    1    {U:22.8e}\n")
            f.write(f"    1    1    0    0    {-U / 2:22.8e}\n")
    # (i, j, t_ij); APBC flips the sign of the closing bond (closed-shell L=4 ring)
    bonds = [(i, i + 1, -1.0) for i in range(1, L)]
    if bc in ("pbc", "apbc"):
        bonds.append((L, 1, -1.0 if bc == "pbc" else 1.0))
    with open(os.path.join(work, "FCIDUMP_env.dat"), "w") as f:
        for i, j, tij in bonds:
            f.write(f"   {i}    {j}    0    0   {tij:22.16e}\n")
            f.write(f"   {j}    {i}    0    0   {tij:22.16e}\n")
    for i in range(1, L + 1):
        shutil.copy(seed["R"][i - 1], os.path.join(work, f"guess_R{i}.dat"))
        shutil.copy(seed["L"][i - 1], os.path.join(work, f"guess_L{i}.dat"))


def cold_seed(root, ng):
    """R = [1, 0...], lambda = 0 (exact U=0 Gutzwiller solution for Ng=0)."""
    os.makedirs(root, exist_ok=True)
    n = 1 + ng
    r = np.zeros((1, n)); r[0, 0] = 1.0
    np.savetxt(os.path.join(root, "seed_R.dat"), r, fmt="%22.15e")
    np.savetxt(os.path.join(root, "seed_L.dat"), np.zeros((n, n)), fmt="%22.15e")
    return dict(R=[os.path.join(root, "seed_R.dat")] * L, L=[os.path.join(root, "seed_L.dat")] * L)


def dimer_seed(root, src):
    """Seed from a converged dimer (R.in, L.in, e.g. examples/U10): dimer atom 0 -> sites
    1, 3 and atom 1 -> sites 2, 4. R.in/L.in may hold one shared atom or both atoms."""
    os.makedirs(root, exist_ok=True)
    rows = [l.split() for l in open(os.path.join(src, "R.in")) if l.strip() and l[0] != "#"]
    lam = [l.split() for l in open(os.path.join(src, "L.in")) if l.strip() and l[0] != "#"]
    n = len(rows[0])
    seed = dict(R=[], L=[])
    for a in range(2):
        a_r = min(a, len(rows) - 1)
        a_l = min(a, len(lam) // n - 1)
        for key, txt in (("R", [rows[a_r]]), ("L", lam[a_l * n:(a_l + 1) * n])):
            fn = os.path.join(root, f"seed_{key}{a}.dat")
            with open(fn, "w") as f:
                f.write("".join("  ".join(r) + "\n" for r in txt))
    for i in range(L):
        seed["R"].append(os.path.join(root, f"seed_R{i % 2}.dat"))
        seed["L"].append(os.path.join(root, f"seed_L{i % 2}.dat"))
    return seed


def final_seed(work):
    return dict(R=[os.path.join(work, f"final_R_{i}.dat") for i in range(1, L + 1)],
                L=[os.path.join(work, f"final_lambda_{i}.dat") for i in range(1, L + 1)])


def run(work, U):
    p = subprocess.run([GGUT, "gGut.input"], cwd=work, capture_output=True, text=True,
                       timeout=1800, env=ENV)
    log = p.stdout + p.stderr
    open(os.path.join(work, "run.log"), "w").write(log)
    gg = open(os.path.join(work, "gg.out")).read()
    last = re.findall(r"Eloc: ([-\d.eE+]+), Evar: ([-\d.eE+]+)", gg)[-1]
    Eloc, Evar = float(last[0]), float(last[1])
    conv = "converged after" in log
    nit = re.search(r"after (\d+) iterations", log)
    # |R_i| = sqrt(R_i R_i^+) (= R_i for Ng=0), so Z_i = |R_i|^2
    R = [np.linalg.norm(np.loadtxt(os.path.join(work, f"final_R_{i}.dat")))
         for i in range(1, L + 1)]
    # GF -> A_ij(w), then drop the bulky text files
    g = np.loadtxt(os.path.join(work, "gGut_gf.dat"))
    w = g[:, 0]
    G = (g[:, 2::2] + 1j * g[:, 3::2]).reshape(len(w), L, L)
    np.savez_compressed(os.path.join(work, "A.npz"), w=w, eta=g[0, 1], A=-G.imag / np.pi)
    for f in ("gGut_gf.dat", "gGut_aw.dat", "gGut_GFdet.dat", "gGut_GFqpdet.dat"):
        os.remove(os.path.join(work, f))
    D = (Eloc + 2 * U) / (4 * U) if U > 0 else 0.25
    return dict(U=U, E=Evar, D=D, R=R, conv=conv, nit=int(nit.group(1)) if nit else -1)


def fmt(r):
    return (f"{r['U']:6.2f} {r['E']:14.8f} {r['D']:12.8f} "
            + " ".join(f"{x:11.8f}" for x in r["R"]) + f" {int(r['conv'])} {r['nit']:4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ng", type=int, default=0)
    ap.add_argument("--bc", nargs="+", default=["obc", "pbc"])
    ap.add_argument("--umax", type=float, default=20.0)
    ap.add_argument("--du", type=float, default=0.2)
    ap.add_argument("--seed-fwd", default=None,
                    help="dimer R.in/L.in dir seeding the forward sweep at U=0 "
                         "(default: R=1, lambda=0; for Ng>0: examples/U2_backward)")
    ap.add_argument("--seed-bwd", default=None,
                    help="dimer R.in/L.in dir seeding the backward sweep at Umax "
                         "(default: forward Umax solution; for Ng>0: examples/U10)")
    args = ap.parse_args()
    if args.Ng > 0:  # a cold start lands on the inert-ghost (= Ng=0) branch
        args.seed_fwd = args.seed_fwd or os.path.join(REPO, "examples", "U2_backward")
        args.seed_bwd = args.seed_bwd or os.path.join(REPO, "examples", "U10")
    Us = np.round(np.arange(0, args.umax + args.du / 2, args.du), 10)
    top = os.path.join(REPO, f"scan_4site_ng{args.Ng}")
    hdr = "U  E_var(H-U/2 N)  docc  R_1  R_2  R_3  R_4  converged  n_iter"

    for bc in args.bc:
        res = {}
        for sweep, Uorder in (("forward", Us), ("backward", Us[::-1])):
            print(f"=== {bc.upper()} {sweep} ===", flush=True)
            src = args.seed_fwd if sweep == "forward" else args.seed_bwd
            if src:
                seed = dimer_seed(os.path.join(top, bc, f"seed_{sweep}"), src)
            elif sweep == "forward":
                seed = cold_seed(os.path.join(top, bc), args.Ng)
            else:
                seed = final_seed(os.path.join(top, bc, "forward", f"U_{Us[-1]:05.2f}"))
            out = []
            for U in Uorder:
                work = os.path.join(top, bc, sweep, f"U_{U:05.2f}")
                write_files(work, U, bc, seed, args.Ng)
                try:
                    r = run(work, U)
                except Exception as e:  # crashed run: record it, keep the previous seed
                    print(f"{bc} {sweep:8s} U={U:.2f} FAILED: {e!r}", flush=True)
                    out.append(dict(U=U, E=np.nan, D=np.nan, R=[np.nan] * L, conv=False, nit=-1))
                    continue
                out.append(r)
                print(f"{bc} {sweep:8s} " + fmt(r), flush=True)
                seed = final_seed(work)
            res[sweep] = {r["U"]: r for r in out}
            with open(os.path.join(top, bc, f"summary_{sweep}.txt"), "w") as f:
                f.write(f"# {bc.upper()} Ng={args.Ng} {sweep}\n# {hdr}\n")
                for r in sorted(out, key=lambda r: r["U"]):
                    f.write(fmt(r) + "\n")
        with open(os.path.join(top, bc, "lowest_energy.txt"), "w") as f:
            f.write(f"# {bc.upper()} Ng={args.Ng} lowest converged of forward/backward\n# {hdr}  branch\n")
            for U in Us:
                cands = [(s, res[s][U]) for s in ("forward", "backward") if res[s][U]["conv"]]
                if not cands:
                    f.write(f"# U={U:.2f}: no converged branch\n")
                    continue
                s, r = min(cands, key=lambda c: c[1]["E"])
                f.write(fmt(r) + f"  {s}\n")
        print(f"wrote {top}/{bc}/", flush=True)


if __name__ == "__main__":
    main()
