"""Gauge check of the gGut poles for the reduced model at V=2: several random starts, compare E, poles, weights."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, ucore_cederbaum as uc, not_imp_gga_cederbaum as N
V,tc,U,Delta,Ec,eps=2.0,0.5,9.95,1.7725,13.2,1e-7
E0=uc.ed_U(V,tc,U,Delta=Delta,Ec=Ec)["E0"]
for Bc in (3,1):
    m=uc.model_U(V,tc,Bc,U,eps,Delta=Delta,Ec=Ec); P=N.Problem(m); rng=np.random.default_rng(100+Bc); sols=[]
    for i in range(10):
        res=N.minimize_once(P,N.random_start(P,rng),maxiter=600)
        if np.abs(P.constraints(res.x)).max()<1e-9:
            o=N.recover(P,res.x)
            if o["comm"]<1e-4:
                sols.append((res.fun-E0,o["poles"],o["weights"])); print(f"Bc={Bc} start {i}: dE={res.fun-E0:.2e} poles {np.round(o['poles'],3)} w {np.array2string(o['weights'],precision=4)}",flush=True)
    np.save(f"../hubbard_sat_data/gut/gauge_Bc{Bc}.npy",np.array([np.concatenate([[s[0]],s[1],s[2]]) for s in sols]))
