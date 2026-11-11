import numpy as np, itertools
from fci_trimer import spectrum
def T(a,b,c): return [[0,a,c],[a,0,b],[c,b,0]]
def UP(a,b,c): return [[0,a,b],[a,0,c],[b,c,0]]
res=[]
for vc,eo,ev,tco,tov,Uc,Uo,Uv,Ucv,Uco,Uov in itertools.product([-30,-22],[-3,-1],[0.5,2.5],[0.2,0.4,0.8],[0.5,1,1.5],[2,4],[1,3],[1,3],[0,2,4,6],[2,4,6],[0,1,2]):
    if ev<=eo+0.4: continue
    vo=eo-2*Uco; vv=ev-2*Ucv
    E0,dg,n,om,wt=spectrum([vc,vo,vv],T(tco,tov,0),[Uc,Uo,Uv],Up=UP(Uco,Ucv,Uov))
    if n[0]<1.95 or n[2]>0.35 or n[1]<1.6: continue
    ip=-om; w=wt.sum(1)
    if (ip[w>1e-3]<0.2).any(): continue
    wc=wt[:,0]; tot=wc.sum(); main=np.argmax(wc)
    far=abs(ip-ip[main])>0.8
    res.append((wc[far].sum()/tot,vc,eo,ev,tco,tov,Uc,Uo,Uv,Uco,Ucv,Uov,ip[main],*n))
res.sort(key=lambda r:-r[0])
for r in res[:10]: print("sat=%.3f vc=%g eo=%g ev=%g tco=%g tov=%g Uc=%g Uo=%g Uv=%g U'co=%g U'cv=%g U'ov=%g IPc=%.2f n=%.2f %.2f %.2f"%r)
print(len(res))
