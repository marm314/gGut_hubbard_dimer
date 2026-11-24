"""Exact (FCI) ground-state energies and spectral functions of the half-filled
4-site Hubbard model, open (chain) and periodic (ring) boundary conditions,
for U = Umin..Umax in steps of dU, using PySCF's FCI kernel in the site basis.

    H = -t sum_<ij>,s c+_is c_js + U sum_i n_iup n_idn

Every (N_up, N_dn) sector of N = 4, 3, 5 electrons is diagonalized exactly
(dimension <= 36), so degenerate ground states (e.g. the PBC ring at U = 0)
are treated as an equal-weight ensemble over the whole degenerate manifold,
all Sz sectors included.

Green's function (Lehmann), per spin, ensemble-averaged over the ground manifold:

    G_ij(w) = sum_n <0|c_i|n><n|c+_j|0> / (w - (E_n^{N+1} - E_0) + i eta)
            + sum_m <0|c+_j|m><m|c_i|0> / (w - (E_0 - E_m^{N-1}) + i eta)

    A_ij(w) = -Im G_ij(w) / pi

Pole energies are stored relative to mu = (E_0^{N+1} - E_0^{N-1}) / 2 (= U/2 by
particle-hole symmetry), so the spectrum is centered at w = 0.

Output (in --out, default hubbard4_fci/):
  {obc,pbc}/energies.dat     U, E0(N), E0(N+1), E0(N-1), mu, gap, ground degeneracy
  {obc,pbc}/poles_U*.npz     raw poles + residue matrices (site basis, per spin avg)
  {obc,pbc}/spectra.npz      broadened A(w) on a grid for all U:
                               A_loc[U, site, w]  local, per spin
                               A_tot[U, w]        sum over sites, per spin
                               A_orb[U, orb, w]   in the U=0 one-body eigenbasis
                               A_k[U, k, w]       (pbc only) k = 0, pi/2, pi, 3pi/2
  {obc,pbc}/A_loc_U*.dat     w, A_site1..4, A_tot (text, per U)

Run e.g.
    nohup ~/.venv/bin/python scripts/hubbard4_fci.py > hubbard4_fci.log 2>&1 &
"""
import argparse
import os
import time

import numpy as np
from pyscf.fci import addons, cistring, direct_spin1

L = 4
NHALF = L  # half filling: N = L electrons


def hopping(t, pbc):
    h = np.zeros((L, L))
    for i in range(L - 1):
        h[i, i + 1] = h[i + 1, i] = -t
    if pbc:
        h[0, L - 1] = h[L - 1, 0] = -t
    return h


def onsite_eri(U):
    eri = np.zeros((L, L, L, L))
    for i in range(L):
        eri[i, i, i, i] = U
    return eri


def sector_eig(h1e, eri, nelec):
    """Full diagonalization of the (na, nb) sector. Returns (E, C[dim, na_str, nb_str])."""
    na = cistring.num_strings(L, nelec[0])
    nb = cistring.num_strings(L, nelec[1])
    dim = na * nb
    h2e = direct_spin1.absorb_h1e(h1e, eri, L, nelec, 0.5)
    H = np.empty((dim, dim))
    for k in range(dim):
        v = np.zeros(dim)
        v[k] = 1.0
        H[:, k] = direct_spin1.contract_2e(h2e, v.reshape(na, nb), L, nelec).ravel()
    assert np.allclose(H, H.T, atol=1e-10), f"non-symmetric H in sector {nelec}"
    E, V = np.linalg.eigh(0.5 * (H + H.T))
    return E, V.T.reshape(dim, na, nb)


def all_sectors(h1e, eri, N):
    out = {}
    for na in range(max(0, N - L), min(L, N) + 1):
        nb = N - na
        out[(na, nb)] = sector_eig(h1e, eri, (na, nb))
    return out


def lowest(sectors):
    return min(E[0] for E, _ in sectors.values())


def solve(U, t, pbc, degtol=1e-8):
    h1e = hopping(t, pbc)
    eri = onsite_eri(U)
    sec0 = all_sectors(h1e, eri, NHALF)
    secp = all_sectors(h1e, eri, NHALF + 1)
    secm = all_sectors(h1e, eri, NHALF - 1)
    E0 = lowest(sec0)
    Ep = lowest(secp)
    Em = lowest(secm)
    mu = 0.5 * (Ep - Em)

    # ground manifold: all states within degtol of E0, in all Sz sectors
    ground = []
    for nel, (E, C) in sec0.items():
        for n in np.where(E - E0 < degtol)[0]:
            ground.append((nel, C[n]))
    ng = len(ground)

    # accumulate poles per spin; residues R[p, i, j] = <0|c_i|p><p|c+_j|0> (particle)
    #                                              or <0|c+_j|p><p|c_i|0> (hole)
    poles = {s: [] for s in ("up", "dn")}
    resid = {s: [] for s in ("up", "dn")}
    for (na, nb), c0 in ground:
        for s in ("up", "dn"):
            if s == "up":
                cre, des = addons.cre_a, addons.des_a
                nelp, nelm = (na + 1, nb), (na - 1, nb)
                okp, okm = na < L, na > 0
            else:
                cre, des = addons.cre_b, addons.des_b
                nelp, nelm = (na, nb + 1), (na, nb - 1)
                okp, okm = nb < L, nb > 0
            if okp:
                Ep_s, Cp = secp[nelp]
                # a[n, j] = <n|c+_j|0>
                a = np.array([[np.vdot(Cp[n], cre(c0, L, (na, nb), j))
                               for j in range(L)] for n in range(len(Ep_s))])
                poles[s].append(Ep_s - E0 - mu)
                resid[s].append(np.einsum("ni,nj->nij", a, a) / ng)
            if okm:
                Em_s, Cm = secm[nelm]
                # b[m, i] = <m|c_i|0>
                b = np.array([[np.vdot(Cm[m], des(c0, L, (na, nb), i))
                               for i in range(L)] for m in range(len(Em_s))])
                poles[s].append(E0 - Em_s - mu)
                resid[s].append(np.einsum("mi,mj->mij", b, b) / ng)

    res = dict(E0=E0, Ep=Ep, Em=Em, mu=mu, gap=Ep + Em - 2 * E0, ndeg=ng)
    for s in ("up", "dn"):
        w = np.concatenate(poles[s])
        R = np.concatenate(resid[s])
        keep = np.abs(np.einsum("nii->n", R)) > 1e-12   # drop zero-weight poles
        res[f"poles_{s}"] = w[keep]
        res[f"resid_{s}"] = R[keep]
    return res


def broaden(poles, resid, wgrid, eta):
    """A_ij(w) = -Im G_ij / pi with Lorentzian broadening eta. Returns [w, i, j]."""
    lor = (eta / np.pi) / ((wgrid[:, None] - poles[None, :]) ** 2 + eta ** 2)
    return np.einsum("wp,pij->wij", lor, resid)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--umin", type=float, default=0.0)
    ap.add_argument("--umax", type=float, default=10.0)
    ap.add_argument("--du", type=float, default=0.2)
    ap.add_argument("--eta", type=float, default=0.05, help="Lorentzian broadening")
    ap.add_argument("--wmin", type=float, default=-12.0)
    ap.add_argument("--wmax", type=float, default=12.0)
    ap.add_argument("--nw", type=int, default=4801)
    ap.add_argument("--bc", choices=["obc", "pbc", "both"], default="both")
    ap.add_argument("--out", default="hubbard4_fci")
    args = ap.parse_args()

    Us = np.round(np.arange(args.umin, args.umax + 0.5 * args.du, args.du), 10)
    wgrid = np.linspace(args.wmin, args.wmax, args.nw)
    bcs = ["obc", "pbc"] if args.bc == "both" else [args.bc]

    # U=0 one-body eigenbasis for orbital-resolved spectra
    for bc in bcs:
        pbc = bc == "pbc"
        d = os.path.join(args.out, bc)
        os.makedirs(d, exist_ok=True)
        eps0, phi = np.linalg.eigh(hopping(args.t, pbc))
        if pbc:  # plane waves, c_k = 1/sqrt(L) sum_j e^{-ikj} c_j
            ks = 2 * np.pi * np.arange(L) / L
            Fk = np.exp(1j * np.outer(ks, np.arange(L))) / np.sqrt(L)  # [k, j]

        A_loc = np.zeros((len(Us), L, args.nw))
        A_tot = np.zeros((len(Us), args.nw))
        A_orb = np.zeros((len(Us), L, args.nw))
        A_k = np.zeros((len(Us), L, args.nw)) if pbc else None

        fe = open(os.path.join(d, "energies.dat"), "w")
        fe.write(f"# 4-site Hubbard, {bc.upper()}, half filling, t={args.t}, FCI (pyscf)\n")
        fe.write("# U  E0(N)  E0(N+1)  E0(N-1)  mu  gap=E(N+1)+E(N-1)-2E(N)  ndeg\n")
        print(f"=== {bc.upper()} ===", flush=True)
        for iu, U in enumerate(Us):
            t0 = time.time()
            r = solve(U, args.t, pbc)
            fe.write(f"{U:6.2f} {r['E0']:18.12f} {r['Ep']:18.12f} {r['Em']:18.12f} "
                     f"{r['mu']:14.10f} {r['gap']:14.10f} {r['ndeg']:3d}\n")
            fe.flush()

            # spin-averaged spectral matrix (up/dn identical after ensemble averaging)
            Aw = 0.5 * (broaden(r["poles_up"], r["resid_up"], wgrid, args.eta)
                        + broaden(r["poles_dn"], r["resid_dn"], wgrid, args.eta))
            A_loc[iu] = np.einsum("wii->iw", Aw)
            A_tot[iu] = A_loc[iu].sum(axis=0)
            A_orb[iu] = np.einsum("ia,wij,ja->aw", phi, Aw, phi)
            if pbc:
                A_k[iu] = np.einsum("ki,wij,kj->kw", Fk, Aw, Fk.conj()).real

            # sum rule check: total weight per site and spin = 1
            sw = np.einsum("nii->i", r["resid_up"])
            occ = sum(np.einsum("nii->i", r["resid_up"][r["poles_up"] < 0]))
            np.savez(os.path.join(d, f"poles_U{U:05.2f}.npz"), U=U, t=args.t, mu=r["mu"],
                     E0=r["E0"], Ep=r["Ep"], Em=r["Em"], ndeg=r["ndeg"],
                     poles_up=r["poles_up"], resid_up=r["resid_up"],
                     poles_dn=r["poles_dn"], resid_dn=r["resid_dn"])
            np.savetxt(os.path.join(d, f"A_loc_U{U:05.2f}.dat"),
                       np.column_stack([wgrid, A_loc[iu].T, A_tot[iu]]),
                       header=f"U={U} eta={args.eta} (w relative to mu={r['mu']:.10f}), "
                              "per spin\nw  A_1  A_2  A_3  A_4  A_tot", fmt="%.8e")
            print(f"U={U:5.2f}  E0={r['E0']:.10f}  gap={r['gap']:.8f}  ndeg={r['ndeg']}  "
                  f"sumrule={sw.min():.10f}..{sw.max():.10f}  n_up={occ:.8f}  "
                  f"({time.time() - t0:.2f}s)", flush=True)
        fe.close()

        out = dict(U=Us, w=wgrid, eta=args.eta, t=args.t, A_loc=A_loc, A_tot=A_tot,
                   A_orb=A_orb, eps0=eps0, phi0=phi)
        if pbc:
            out.update(A_k=A_k, k=ks)
        np.savez(os.path.join(d, "spectra.npz"), **out)
        print(f"wrote {d}/", flush=True)


if __name__ == "__main__":
    main()
