"""Maximize satellite weight of the c line, strictly local U_i>=0, by differential evolution (FCI each eval)."""
import numpy as np, sys
from scipy.optimize import differential_evolution
from fci_trimer import spectrum
def T(a,b,c): return [[0,a,c],[a,0,b],[c,b,0]]
bounds=[(-20,-3),(-4,0),(0,6),(0,3),(0,3),(0,3),(0,10),(0,10),(0,10)]  # vc vo vv tco tov tcv Uc Uo Uv
def metrics(x):
    vc,vo,vv,tco,tov,tcv,Uc,Uo,Uv=x
    E0,dg,n,om,wt=spectrum([vc,vo,vv],T(tco,tov,tcv),[Uc,Uo,Uv])
    ip=-om; wc=wt[:,0]; tot=wc.sum(); main=np.argmax(wc)
    far=abs(ip-ip[main])>0.8; sat=wc[far].sum()/tot
    cl=wc>1e-2; clean=wt[cl,1:].sum()/wc[cl].sum()
    wo=wt[:,1]; ipo=ip[np.argmax(wo)]
    return dict(sat=sat,Z=2*wc[main]/tot,clean=clean,n=n,ipc=ip[main],ipo=ipo,ipmin=ip[wt.sum(1)>1e-3].min(),dg=dg,ip=ip,wt=wt)
def cost(x):
    m=metrics(x); pen=0
    pen+=max(0,1.9-m['n'][0])*10+max(0,m['n'][2]-0.3)*10+max(0,m['clean']-0.1)*10
    pen+=max(0,3-(m['ipc']-m['ipo']))+max(0,0.2-m['ipmin'])*5+(m['dg']>1)*5
    return -m['sat']+pen
if __name__=="__main__":
    best=None
    for seed in range(6):
        r=differential_evolution(cost,bounds,seed=seed,popsize=30,maxiter=400,tol=1e-8,polish=False)
        m=metrics(r.x); print(f"seed {seed}: cost={r.fun:.4f} sat={m['sat']:.3f} Z={m['Z']:.3f} clean={m['clean']:.3f} n={np.round(m['n'],2)} IPc={m['ipc']:.2f} IPo={m['ipo']:.2f}  x={np.round(r.x,3)}",flush=True)
        if best is None or r.fun<best.fun: best=r
    np.save('hubbard_sat_data/opt_best.npy',best.x)
