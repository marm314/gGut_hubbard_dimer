"""Impurity loop (lowest embedding state, follow=False) started from the direct solution in the gauge where Phi_c is the H_imp ground state (V/g=1/sqrt2, t_c/g=1, B_c=3, t=4): E, dDelta, dR, resid per iteration for MIX=1, 0.1, 0.01. MIX=.. NIT=.. GHOST_EPS=1e-7 PYTHONPATH=scripts python3 scripts/loop_trajectory.py"""
import sys, os
src = open("scripts/loop_at_gauge.py").read().split("    ts = np.arange")[0]
sys.argv = ["x", "0.7071067811865476", "1.0"]; os.environ["GHOST_EPS"] = "1e-7"
exec(src.replace("for tc in TCS:", "for tc in TCS[:1]:"))
t = 4.0; lam = lam_of(mu0 + t * d); lamc = [ls[I] - lam[I] for I in range(3)]
state0 = dict(R=R, lam=lam, nu=q[0]["nu"], K=q[0]["K"], rho=rho, hyb=hyb, lamc=lamc, Dl=Dl)
s = g.run(1.0, 10.0, V, tc, Bc, max_iter=int(os.environ.get("NIT","12")), mix=float(os.environ.get("MIX","1.0")), tol=1e-10, follow=False, phi0=st["phis"], state0=state0, shift=EPS)
print("E_direct", E)
for h in s["hist"]: print(f"it={h[0]:2d} E_var-E_direct={h[1]-E:+.3e} dDelta={h[3]:.2e} dR={h[4]:.2e} resid={h[5]:.2e}")
