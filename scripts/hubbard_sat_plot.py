import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d='hubbard_sat_data/'; r=np.load(d+'ref.npz')
ip=r['ip']; wt=r['wt']
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
a=ax[0]
for k,(lab,c) in enumerate(zip('cov',['C3','C0','C2'])):
    m=wt[:,k]>3e-3; a.vlines(ip[m],0,wt[m,k]/2,color=c,lw=2,label=f'weight on {lab}')  # doublet counted once
a.set_xlabel('IP'); a.set_ylabel('spectral weight (per spin)'); a.legend(fontsize=8); a.set_title('reference point')
for ev,ls in ((0.5,'-'),(2.5,'--')):
    s=np.load(d+f'scan_Ucv_tov_ev{ev}.npz')
    for b,t in enumerate(s['tov']):
        if t in (0.5,1.0,2.0): ax[1].plot(s['Ucv'],s['Z'][:,b],ls,color={0.5:'C0',1.0:'C1',2.0:'C2'}[t],label=f"$t_{{ov}}$={t}, $\\varepsilon_v$={ev}")
ax[1].axvline(2,color='gray',lw=.8,ls=':'); ax[1].set_xlabel("$U'_{cv}$"); ax[1].set_ylabel('$Z$ (main c line)'); ax[1].legend(fontsize=7)
for name,a in (('Uco',ax[2]),):
    s=np.load(d+'scan_Uco.npz'); a.plot(s['x'],s['Z'],'o-'); a.set_xlabel("$U'_{co}$ (at $U'_{cv}=6$)"); a.set_ylabel('$Z$')
plt.tight_layout(); plt.savefig('doc/hubbard_sat_spectrum.png',dpi=150)
