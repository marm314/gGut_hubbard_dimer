import numpy as np, itertools
from fci_trimer import spectrum
def T(tco,tov,tcv): return [[0,tco,tcv],[tco,0,tov],[tcv,tov,0]]
res=[]
grid=itertools.product([-12,-8,-5],[-2,-1,0],[1,2,4],[0.3,0.6,1,1.5],[0.3,0.6,1],[0,0.3],[2,4,8],[1,2,4],[1,2,4])
for vc,vo,vv,tco,tov,tcv,Uc,Uo,Uv in grid:
    if vv<=vo+0.5: continue
    E0,dg,n,om,wt=spectrum([vc,vo,vv],T(tco,tov,tcv),[Uc,Uo,Uv])
    if n[0]<1.9 or n[2]>0.3 or n[1]<1.7: continue
    ip=-om; w=wt.sum(1); m=w>1e-3
    if (ip[m]<0.2).any(): continue
    wc=wt[:,0]; main=np.argmax(wc); tot=wc.sum()
    far=abs(ip-ip[main])>0.8
    sat=wc[far].sum()/tot
    res.append((sat,vc,vo,vv,tco,tov,tcv,Uc,Uo,Uv,ip[main],*np.round(n,2)))
res.sort(key=lambda r:-r[0])
for r in res[:12]: print("sat=%.3f vc=%g vo=%g vv=%g tco=%g tov=%g tcv=%g Uc=%g Uo=%g Uv=%g IPmain=%.2f n=%.2f %.2f %.2f"%r)
print(len(res))
