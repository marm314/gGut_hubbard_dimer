"""gGut (direct minimization) vs exact for the REDUCED model of the best U>=0 point at V=2:
Cederbaum model with Delta=(eps_v-eps_o)/2, E_c=-eps_c, core U, one c-v hopping; dropped: U_o,U_v,t_ov,t_co.
Ng=0 <-> B_c=1, Ng=2 <-> B_c=3. Each converged solution is saved immediately."""
import os, sys, time
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import numpy as np
import ucore_cederbaum as uc
import not_imp_gga_cederbaum as N
V,tc,U,Delta,Ec,eps=2.0,0.5,9.95,1.7725,13.2,1e-7
out="../hubbard_sat_data/gut"; os.makedirs(out,exist_ok=True)
r=uc.ed_U(V,tc,U,Delta=Delta,Ec=Ec); s=r["weights"]>1e-3
print(f"ED: E0={r['E0']:.6f} nu_c={r['nu_c']:.5f} ndeg={r['ndeg']} poles {np.round(r['poles'][s],3)} w {np.round(r['weights'][s],4)}",flush=True)
np.savez(f"{out}/ed.npz",poles=r["poles"],w=r["weights"],E0=r["E0"],nu=r["nu_c"])
for Bc in (1,3):
    t0=time.time(); m=uc.model_U(V,tc,Bc,U,eps,Delta=Delta,Ec=Ec); P=N.Problem(m); rng=np.random.default_rng(11+Bc); best=None
    for i in range(40):
        res=N.minimize_once(P,N.random_start(P,rng),maxiter=600); cons=np.abs(P.constraints(res.x)).max()
        ok=False
        if cons<1e-9:
            o=N.recover(P,res.x); ok=o["comm"]<1e-4
            if ok and (best is None or res.fun<best[0].fun): best=(res,o)
        print(f"  Bc={Bc} start {i}: E={res.fun:.6f} cons={cons:.1e} accepted={ok}",flush=True)
        if best is not None and best[0].fun-r["E0"]<1e-3: break
    if best is None: print(f"Bc={Bc} Ng={Bc-1}: no accepted solution ({time.time()-t0:.0f}s)",flush=True); continue
    res,o=best
    print(f"gGut Ng={Bc-1} (B_c={Bc}): E-E_ED={res.fun-r['E0']:.3e} nu_c={o['nu']:.5f} kkt={o['kkt']:.1e} comm={o['comm']:.1e} poles {np.round(o['poles'],3)} w {np.array2string(o['weights'],precision=4)} ({time.time()-t0:.0f}s)",flush=True)
    np.savez(f"{out}/Bc{Bc}.npz",z=res.x,E=res.fun,E_ED=r["E0"],nu=o["nu"],poles=o["poles"],w=o["weights"])
