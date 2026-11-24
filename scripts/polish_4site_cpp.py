#!/usr/bin/env python3
"""Remove branch jumps from the 4-site (g)Gutzwiller scans of scan_4site_cpp.py by
neighbour seeding: alternate ascending and descending sweeps over the U grid in which
each U is rerun from the current best solution at the previous U of the sweep; a lower
converged energy replaces the best one. Stops when a full up+down pass changes nothing.

Starting point: the lowest converged of forward/backward per U (those runs are not touched).
Accepted improvements are stored in scan_4site_ng{Ng}/{bc}/polish/U_xx.xx/, and
lowest_energy.txt is rewritten (branch column: forward / backward / polish).

    ~/.venv/bin/python scripts/polish_4site_cpp.py --Ng 2 [--bc obc pbc apbc] [--maxpass 6]
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_4site_cpp import REPO, final_seed, fmt, run, write_files

TOL = 1e-6  # energy gain needed to accept (C++ prints E_var to ~6 digits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ng", type=int, default=0)
    ap.add_argument("--bc", nargs="+", default=["obc", "pbc", "apbc"])
    ap.add_argument("--maxpass", type=int, default=6)
    args = ap.parse_args()
    top = os.path.join(REPO, f"scan_4site_ng{args.Ng}")
    hdr = "U  E_var(H-U/2 N)  docc  R_1  R_2  R_3  R_4  converged  n_iter"

    for bc in args.bc:
        base = os.path.join(top, bc)
        # current best per U: (result dict, run directory, branch label)
        best = {}
        for sweep in ("forward", "backward"):
            for l in open(os.path.join(base, f"summary_{sweep}.txt")):
                if l.startswith("#"):
                    continue
                p = l.split()
                U = round(float(p[0]), 6)
                r = dict(U=U, E=float(p[1]), D=float(p[2]), R=[float(x) for x in p[3:7]],
                         conv=p[7] == "1", nit=int(p[8]))
                if not r["conv"]:
                    continue
                if U not in best or r["E"] < best[U][0]["E"]:
                    best[U] = (r, os.path.join(base, sweep, f"U_{U:05.2f}"), sweep)
        Us = sorted(set(round(float(l.split()[0]), 6) for l in open(os.path.join(base, "summary_forward.txt"))
                        if not l.startswith("#")))
        tmp = os.path.join(base, "polish_tmp")
        for ipass in range(args.maxpass):
            changed = 0
            for order in (Us, Us[::-1]):
                for prev, U in zip(order[:-1], order[1:]):
                    if prev not in best:
                        continue
                    shutil.rmtree(tmp, ignore_errors=True)
                    write_files(tmp, U, bc, final_seed(best[prev][1]), args.Ng)
                    try:
                        r = run(tmp, U)
                    except Exception as e:
                        print(f"{bc} pass {ipass} U={U:.2f} from {prev:.2f} FAILED: {e!r}", flush=True)
                        continue
                    old = best[U][0]["E"] if U in best else np.inf
                    if r["conv"] and r["E"] < old - TOL:
                        dest = os.path.join(base, "polish", f"U_{U:05.2f}")
                        shutil.rmtree(dest, ignore_errors=True)
                        shutil.copytree(tmp, dest)
                        best[U] = (r, dest, "polish")
                        changed += 1
                        print(f"{bc} pass {ipass} U={U:5.2f} from {prev:5.2f}: "
                              f"{old:.6f} -> {r['E']:.6f}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{bc} pass {ipass}: {changed} points improved", flush=True)
            if not changed:
                break
        with open(os.path.join(base, "lowest_energy.txt"), "w") as f:
            f.write(f"# {bc.upper()} Ng={args.Ng} lowest converged of forward/backward/polish\n"
                    f"# {hdr}  branch\n")
            for U in Us:
                if U not in best:
                    f.write(f"# U={U:.2f}: no converged branch\n")
                    continue
                r, _, lab = best[U]
                f.write(fmt(r) + f"  {lab}\n")
        print(f"wrote {base}/lowest_energy.txt", flush=True)


if __name__ == "__main__":
    main()
