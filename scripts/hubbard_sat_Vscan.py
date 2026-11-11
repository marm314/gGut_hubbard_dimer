import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'scripts')
from hubbard_sat_opt5 import metrics
Vs=np.arange(0,3.001,0.05); x0=np.load('hubbard_sat_data/opt2_seed0.npy')  # best U>=0 without V
L=[]
for V in Vs:
    m=metrics(x0,V); ip,wt=m['ip'],m['wt']; k=wt[:,0]/2>2e-3; seen=[]
    for e,w in sorted(zip(ip[k],wt[k,0])):
        if not any(abs(e-s)<1e-7 for s in seen): seen.append(e); L.append((V,e,w))
L=np.array(L); np.save('hubbard_sat_data/Vscan_seed0.npy',L)
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].scatter(L[:,0],L[:,1],s=2+200*L[:,2],c='C3',alpha=.6,edgecolor='none'); ax[0].set_xlabel('V'); ax[0].set_ylabel('IP of c lines (size = weight)')
Zs=[];Ss=[]
for V in Vs:
    s=L[np.isclose(L[:,0],V)]; w=s[:,2]; j=np.argmax(w); Zs.append(w[j]); Ss.append(w[s[:,1]>s[j,1]+1].sum())
ax[1].plot(Vs,Zs,label='qp (strongest line)'); ax[1].plot(Vs,Ss,label='satellites > qp+1'); ax[1].set_xlabel('V'); ax[1].set_ylabel('weight'); ax[1].legend()
fig.suptitle('best U>=0 point: eps=(%.1f,%.1f,%.2g) t=(%.2f,%.2f,%.2f) U=(%.2f,%.2f,%.2f)'%(x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8]),fontsize=9)
plt.tight_layout(); plt.savefig('doc/hubbard_sat_Vscan.png',dpi=140)
