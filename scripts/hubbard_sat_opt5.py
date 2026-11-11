"""Best U>=0 point (opt2) + Cederbaum term V. (1) V scan at fixed params; (2) DE re-optimization at fixed V (U>=0, t_co,t_cv<=0.5)."""
import numpy as np, sys
sys.path.insert(0,'scripts')
from scipy.optimize import differential_evolution
from fci_trimer import spectrum
def T(a,b,c): return [[0,a,c],[a,0,b],[c,b,0]]
def metrics(x,V):
    vc,vo,vv,tco,tov,tcv,Uc,Uo,Uv=x
    E0,dg,n,om,wt=spectrum([vc,vo,vv],T(tco,tov,tcv),[Uc,Uo,Uv],Vc=V)
    ip=-om; wc=wt[:,0]; tot=wc.sum(); main=np.argmax(wc)
    cl=wc>1e-2; clean=wt[cl,1:].sum()/wc[cl].sum()
    return dict(sat=wc[ip>ip[main]+1].sum()/tot,Z=2*wc[main]/tot,clean=clean,n=n,ipc=ip[main],ipo=ip[np.argmax(wt[:,1])],ipmin=ip[wt.sum(1)>1e-3].min(),dg=dg,ip=ip,wt=wt)
def cost(x,V):
    m=metrics(x,V); pen=max(0,1.9-m['n'][0])*10+max(0,m['n'][2]-0.3)*10+max(0,m['clean']-0.1)*10
    pen+=max(0,3-(m['ipc']-m['ipo']))+max(0,0.5-m['ipmin'])*5+(m['dg']>1)*5
    return -m['sat']+pen
if __name__=="__main__":
    for seed in (0,4):
        x0=np.load(f'hubbard_sat_data/opt2_seed{seed}.npy'); print('fixed params from opt2 seed',seed,np.round(x0,3))
        for V in [0,0.25,0.5,1,1.5,2]:
            m=metrics(x0,V); print(f"  V={V}: sat={m['sat']:.3f} Z={m['Z']:.3f} clean={m['clean']:.3f} n={np.round(m['n'],2)} IPc={m['ipc']:.2f} IPo={m['ipo']:.2f}",flush=True)
    bounds=[(-20,-3),(-5,0),(0,8),(0,0.5),(0,4),(0,0.5),(0,10),(0,10),(0,10)]
    for V in [0.5,1,2]:
        for seed in range(3):
            r=differential_evolution(cost,bounds,args=(V,),seed=seed,popsize=30,maxiter=300,tol=1e-8,polish=False)
            m=metrics(r.x,V); print(f"opt V={V} seed{seed}: sat={m['sat']:.3f} Z={m['Z']:.3f} clean={m['clean']:.3f} n={np.round(m['n'],2)} IPc={m['ipc']:.2f} IPo={m['ipo']:.2f} x={np.round(r.x,3)}",flush=True)
            np.save(f'hubbard_sat_data/opt5_V{V}_seed{seed}.npy',r.x)
