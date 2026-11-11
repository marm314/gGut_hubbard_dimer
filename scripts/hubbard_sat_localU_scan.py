"""FCI scan, strictly local U_i (no U'). Core-clean criterion: weight on o,v within c-lines < 10%."""
import numpy as np, itertools, time
from fci_trimer import spectrum
def T(a,b,c): return [[0,a,c],[a,0,b],[c,b,0]]
grid=list(itertools.product([-14,-8],[-1],[0.5,2,4],[0.3,0.6,1,1.5,2.5],[0.5,1,2],[0,0.5,1,2],[-4,0,4,8],[-3,0,3,6],[-3,0,3,6]))
rows=[]; t0=time.time()
for k,(vc,vo,vv,tco,tov,tcv,Uc,Uo,Uv) in enumerate(grid):
    E0,dg,n,om,wt=spectrum([vc,vo,vv],T(tco,tov,tcv),[Uc,Uo,Uv])
    if dg>1: continue
    ip=-om; w=wt.sum(1); wc=wt[:,0]; tot=wc.sum()
    main=np.argmax(wc); far=abs(ip-ip[main])>0.8
    Z=2*wc[main]/tot; sat=wc[far].sum()/tot
    clean=(wt[wc>1e-2,1:].sum()/wc[wc>1e-2].sum())    # o,v weight inside c lines
    rows.append((sat,Z,clean,n[0],n[1],n[2],ip.min(),vc,vo,vv,tco,tov,tcv,Uc,Uo,Uv))
    if k%2000==0: print(k,len(grid),round(time.time()-t0),flush=True)
R=np.array(rows); np.save('hubbard_sat_data/localU_scan.npy',R); print('saved',R.shape,flush=True)
