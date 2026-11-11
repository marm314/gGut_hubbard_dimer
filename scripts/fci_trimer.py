"""3-site (c,o,v) Hubbard chain, exact (Fock-space) ED; N=4 ground state, N=3 removal spectrum.
H = sum_i v_i n_i + sum_{ij,s} t_ij c+_is c_js + sum_i U_i n_iu n_id
Spin-orbital index = 2*site+spin."""
import numpy as np, itertools, sys
ns=6; dim=1<<ns
def cop(p):
    M=np.zeros((dim,dim))
    for s in range(dim):
        if s>>p&1:
            sign=(-1)**bin(s&((1<<p)-1)).count('1'); M[s^(1<<p),s]=sign
    return M
C=[cop(p) for p in range(ns)]; Cd=[c.T for c in C]
def ham(v,t,U,Up=None,Vc=0.0):
    H=np.zeros((dim,dim))
    for i in range(3):
        for s in range(2):
            H+=v[i]*Cd[2*i+s]@C[2*i+s]
        H+=U[i]*Cd[2*i]@C[2*i]@Cd[2*i+1]@C[2*i+1]
        for j in range(3):
            if i!=j and t[i][j]!=0:
                for s in range(2): H+=t[i][j]*Cd[2*i+s]@C[2*j+s]
    if Up is not None:
        for i in range(3):
            for j in range(i+1,3):
                if Up[i][j]!=0: H+=Up[i][j]*(Cd[2*i]@C[2*i]+Cd[2*i+1]@C[2*i+1])@(Cd[2*j]@C[2*j]+Cd[2*j+1]@C[2*j+1])
    if Vc!=0:   # Cederbaum term  -Vc sum_{s,s'} (1-n_cs)(v+_s' o_s' + h.c.)
        one=np.eye(dim)
        for sc in range(2):
            for so in range(2):
                hop=Cd[4+so]@C[2+so]+Cd[2+so]@C[4+so]
                H-=Vc*(one-Cd[sc]@C[sc])@hop
    return H
Nop=sum(Cd[p]@C[p] for p in range(ns)); Nd=np.diag(Nop)
def spectrum(v,t,U,N=4,Up=None,Vc=0.0):
    H=ham(v,t,U,Up,Vc); idx=np.where(Nd==N)[0]
    w,V=np.linalg.eigh(H[np.ix_(idx,idx)]); E0=w[0]
    g=np.zeros(dim); g[idx]=V[:,0]
    degen=np.sum(abs(w-E0)<1e-9)
    idm=np.where(Nd==N-1)[0]; wm,Vm=np.linalg.eigh(H[np.ix_(idm,idm)])
    wt=np.zeros((len(wm),3))
    for i in range(3):
        for s in range(2):
            a=(C[2*i+s]@g)[idm]; wt[:,i]+=(Vm.T@a)**2
    n=[g@(Cd[2*i]@C[2*i]+Cd[2*i+1]@C[2*i+1])@g for i in range(3)]
    return E0,degen,n,E0-wm,wt   # removal energies = E0-E_m (ionization = -that)
def report(v,t,U,thr=2e-3,Up=None):
    E0,dg,n,om,wt=spectrum(v,t,U,Up=Up)
    print(f"v={v} U={U} t_co={t[0][1]} t_ov={t[1][2]} t_cv={t[0][2]}  E0={E0:.5f} degen={dg} n_c,o,v={np.round(n,3)}")
    # group degenerate levels
    for e,w in sorted(zip(om,wt.tolist()),key=lambda x:x[0]):
        if sum(w)>thr: print(f"   IP={-e:8.4f}  w_c={w[0]:.4f} w_o={w[1]:.4f} w_v={w[2]:.4f}")
if __name__=="__main__":
    def T(tco,tov,tcv): return [[0,tco,tcv],[tco,0,tov],[tcv,tov,0]]
    report([-10,0,4],T(0.5,1,0),[6,6,6])
