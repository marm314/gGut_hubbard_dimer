"""FCI scans of the c-o-v chain with U, U'. Saves hubbard_sat_data/*.npz and prints progress.
Effective levels: v_o = eo-2U'co, v_v = ev-2U'cv (keeps o,v levels fixed when c is full)."""
import numpy as np, sys
from fci_trimer import spectrum
def T(a,b,c): return [[0,a,c],[a,0,b],[c,b,0]]
def UP(a,b,c): return [[0,a,b],[a,0,c],[b,c,0]]
REF=dict(vc=-22,eo=-3,ev=0.5,tco=0.4,tov=1.0,tcv=0.0,Uc=2,Uo=3,Uv=1,Uco=2,Ucv=6,Uov=1)
def run(**kw):
    p=dict(REF); p.update(kw)
    E0,dg,n,om,wt=spectrum([p['vc'],p['eo']-2*p['Uco'],p['ev']-2*p['Ucv']],T(p['tco'],p['tov'],p['tcv']),
                           [p['Uc'],p['Uo'],p['Uv']],Up=UP(p['Uco'],p['Ucv'],p['Uov']))
    ip=-om; wc=wt[:,0]; tot=wc.sum(); main=np.argmax(wc)
    sat=wc[abs(ip-ip[main])>0.8].sum()/tot
    return dict(E0=E0,n=n,ip=ip,wt=wt,Zmain=2*wc[main]/tot,sat=sat,IPmain=ip[main])
if __name__=="__main__":
    out='hubbard_sat_data/'
    r=run(); np.savez(out+'ref.npz',ip=r['ip'],wt=r['wt'],n=r['n'],**REF); print('ref',r['Zmain'],r['n'],flush=True)
    # 2D: U'cv x t_ov, for two ev
    Ucvs=np.arange(0,9,1.0); tovs=np.array([0.5,0.75,1,1.25,1.5,2.0])
    for ev in (0.5,2.5):
        Z=np.zeros((len(Ucvs),len(tovs))); S=Z.copy(); NV=Z.copy()
        for a,u in enumerate(Ucvs):
            for b,t in enumerate(tovs):
                r=run(ev=ev,Ucv=u,tov=t); Z[a,b]=r['Zmain']; S[a,b]=r['sat']; NV[a,b]=r['n'][2]
            print(f'ev={ev} Ucv={u}: Z={np.round(Z[a],3)} nv={np.round(NV[a],3)}',flush=True)
        np.savez(out+f'scan_Ucv_tov_ev{ev}.npz',Ucv=Ucvs,tov=tovs,Z=Z,sat=S,nv=NV)
    # 1D: t_co and U'co, Uc
    for name,vals in (('tco',[0.1,0.2,0.4,0.8,1.2,1.6]),('Uco',[0,1,2,3,4,6]),('Uc',[0,1,2,4,6,8])):
        Z=[];NV=[]
        for x in vals:
            r=run(**{name:x}); Z.append(r['Zmain']); NV.append(r['n'][2])
            print(f'{name}={x}: Z={Z[-1]:.3f} nv={NV[-1]:.3f}',flush=True)
        np.savez(out+f'scan_{name}.npz',x=vals,Z=Z,nv=NV)
