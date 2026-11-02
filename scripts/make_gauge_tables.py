"""Writes doc/tab_*.tex (tables \\input by doc/cederbaum.tex) from the saved data: cederbaum_data/ghost_placed/V*.npz, scan_ghost_tc_eps1e-07/, gauge_logs/*.log. python3 scripts/make_gauge_tables.py"""
import re, ast, numpy as np
Vs = [0.7071067811865476, 1.4142135623730951]
def sci(x, p=1):
    if x == 0: return "0"
    m, e = f"{x:.{p}e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
out = []
# ---- Tab: eps dependence
out.append(r"""\begin{table}[h]
\centering\small
\begin{tabular}{r r r r l}
\toprule
$\varepsilon$ & $E-E_{\rm ED}$ & projected gradient & $\|[H_{qp},\rho]\|$ & outcome\\
\midrule""")
for line in open("cederbaum_data/gauge_logs/eps_dependence.log"):
    m = re.match(r"eps=(\S+): dE=(\S+) \|cons\|=(\S+) projected grad=(\S+) comm=(\S+) smallest sv\(J_phi\)=(\S+)", line)
    if m:
        eps, dE, c, pg, cm, sv = m.groups(); note = "converged" if float(pg) < 1e-5 else ("stalls" if float(eps) == 1e-9 else "no convergence")
        out.append(f"{sci(float(eps),0)} & {sci(float(dE))} & {sci(float(pg))} & {sci(float(cm))} & {note}\\\\")
out.append(r"""\bottomrule
\end{tabular}
\caption{Re-minimization of the stored $\varepsilon=10^{-9}$ solution at other shifts ($V/g=\sqrt2$, $t_c/g=1$, $B_c=3$; \texttt{eps\_dependence.py}). The projected gradient is the part of $\nabla E$ not cancelled by the constraint normals.}
\label{tab:eps}
\end{table}
""")
# ---- Tab: commutator ranges
cm9 = {}
for line in open("cederbaum_data/gauge_logs/comm_scan_eps1e-9.log"):
    m = re.match(r"(\S+) points with .*?: (\{.*\})", line)
    if m:
        d = ast.literal_eval(m.group(2)); cm9[float(m.group(1))] = {B: [c for _, c in d.get(B, [])] for B in (3, 5)}
out.append(r"""\begin{table}[h]
\centering\small
\begin{tabular}{r r r r r}
\toprule
$V/g$ & $B_c$ & $\varepsilon=10^{-9}$: min--max & $\varepsilon=10^{-7}$: min--max & points\\
\midrule""")
for V in Vs:
    d = np.load(f"cederbaum_data/ghost_placed/V{V}.npz")
    for B in (3, 5):
        c7 = [float(d[k]) for k in d.files if k.endswith(f"B{B}_comm_before")]; c9 = cm9[V][B]
        out.append(f"{V:.3f} & {B} & {sci(min(c9))}--{sci(max(c9))} & {sci(min(c7))}--{sci(max(c7))} & {len(c7)}\\\\")
out.append(r"""\bottomrule
\end{tabular}
\caption{$\|[H_{qp},\rho]\|$ at the minimum-norm gauge over the $25$ values of $t_c/g$ (\texttt{comm\_scan.py}; \texttt{ghost\_placed}).}
\label{tab:comm}
\end{table}
""")
# ---- Tab: placed poles vs tc
sel = (0.2, 1.0, 2.0, 3.0, 4.0, 5.0)
out.append(r"""\begin{table}[p]
\centering\scriptsize
\begin{tabular}{r r r l r r r}
\toprule
$t_c/g$ & ED main & gGut QP & ED satellites: position $(p)$ & $B_c=3$ ghost & $B_c=5$ ghosts & $B_c=5$ empty ghost\\
\midrule""")
for V in Vs:
    d = np.load(f"cederbaum_data/ghost_placed/V{V}.npz"); out.append(f"\\multicolumn{{7}}{{l}}{{$V/g={V:.3f}$}}\\\\")
    for t in sel:
        k3, k5 = f"tc{t}_B3_", f"tc{t}_B5_"; sats = ", ".join(f"{x:.3f} ({w:.3f})" for x, w in zip(d[k5 + "sats"], d[k5 + "satw"]))
        out.append(f"{t:.1f} & {float(d[k5+'main']):.3f} & {float(d[k3+'qp_after']):.3f} & {sats} & {d[k3+'ghost_after'][0]:.3f} & {', '.join(f'{x:.3f}' for x in d[k5+'ghost_after'])}\\\\")
    out.append(r"\midrule" if V == Vs[0] else "")
out.append(r"""\bottomrule
\end{tabular}
\caption{Satellites of the exact $G^h_{cc}$ (clusters of poles with weight $>10^{-3}$ below the main pole, summed weight in parentheses, heaviest first) and the positions of the weightless ghost poles after they are placed on them (Sec.~\ref{sec:place}); the last column is the free empty ghost level of $B_c=5$, placed on the second heaviest core particle-side pole of ED. All in units of $g$.}
\label{tab:placed}
\end{table}
""")
# ---- Tab: invariants
out.append(r"""\begin{table}[h]
\centering\scriptsize
\begin{tabular}{r r r r r r r r r}
\toprule
$V/g$ & $B_c$ & $\max|\epsilon_{\rm ghost}-\text{target}|$ & $\max|\epsilon_{\rm empty}-\text{target}|$ & $\max|\Delta\epsilon_{\rm QP}|$ & $\max|\Delta w_{\rm QP}|$ & $\max|n_k-\{0,1\}|$ & $\max\|[H_{qp},\rho]\|$ & aufbau\\
\midrule""")
for V in Vs:
    d = np.load(f"cederbaum_data/ghost_placed/V{V}.npz")
    for B in (3, 5):
        ks = [k for k in d.files if k.endswith(f"B{B}_targets")]; P = lambda s: [k[:-len("targets")] + s for k in ks]; g = lambda s: np.array([d[x] for x in P(s)])
        mr = max(np.abs(np.array(d[k[:-7] + "ghost_after"]) - np.array(d[k])).max() for k in ks)
        emp = (sci(float(np.nanmax(np.abs(g('empty_after')-g('empty_target'))))) if B == 5 else '--'); out.append(f"{V:.3f} & {B} & {sci(mr)} & {emp} & {sci(np.abs(g('qp_after')-g('qp_before')).max())} & {sci(np.abs(g('wq_after')-g('wq_before')).max())} & {sci(g('occdev_after').max())} & {sci(g('comm_after').max())} & {int(g('aufbau_after').sum())}/{len(ks)}\\\\")
out.append(r"""\bottomrule
\end{tabular}
\caption{Worst case over the $25$ values of $t_c/g$ of the placement: distance of the occupied ghost levels (and, at $B_c=5$, of the free empty ghost level) to the targets, change of the QP pole and weight, distance of the occupations of the $H_{qp}$ levels to $0$ or $1$, commutator after the shift, and number of points obeying aufbau.}
\label{tab:inv}
\end{table}
""")
# ---- Tab: rank of Phi_c in H_imp
out.append(r"""\begin{table}[h]
\centering\small
\begin{tabular}{r r r r r}
\toprule
$V/g$ & $B_c$ & dim $H^{imp}_c$ & rank of $\Phi_c$, min-norm gauge & rank of $\Phi_c$, ghosts placed\\
\midrule""")
for V in Vs:
    d = np.load(f"cederbaum_data/ghost_placed/V{V}.npz")
    for B in (3, 5):
        ks = [k for k in d.files if k.endswith(f"B{B}_targets")]; g = lambda s: np.array([d[k[:-len("targets")] + s] for k in ks]); rb, ra = g("rankPhi_before"), g("rankPhi_after")
        out.append(f"{V:.3f} & {B} & {36 if B==3 else 400} & {rb.min()}--{rb.max()} (median {int(np.median(rb))}) & {ra.min()}--{ra.max()} (median {int(np.median(ra))})\\\\")
out.append(r"""\bottomrule
\end{tabular}
\caption{Position of $\Phi_c$ in the spectrum of $H^{imp}_c$ (index $0$ is the ground state) over the $25$ values of $t_c/g$.}
\label{tab:rank}
\end{table}
""")

# ---- Tab: eps dependence of the levels at small t_c
out.append(r"""\begin{table}[h]
\centering\scriptsize
\begin{tabular}{r r r r r r r r}
\toprule
$V/g$ & $t_c/g$ & $\varepsilon$ & $E-E_{\rm ED}$ & QP core (weight) & $o$ level & $v$ level & $\|[H_{qp},\rho]\|$\\
\midrule""")
last = None
for line in open("cederbaum_data/gauge_logs/eps_levels.log"):
    m = re.match(r"V/g=(\S+) tc=(\S+) eps=(\S+): E-E_ED (\S+); QP core (\S+) \(w (\S+)\); o level (\S+) \(R_o\^2 (\S+)\); v level (\S+); comm ([^;\s]+)", line)
    if m:
        V, tc, eps, dE, qp, w, ol, ro, vl, cm = m.groups()
        if last and (V, tc) != last: out.append(r"\midrule")
        last = (V, tc)
        out.append(f"{float(V):.3f} & {tc} & {sci(float(eps),0)} & {sci(float(dE))} & {float(qp):.4f} ({float(w):.4f}) & {float(ol):.4f} & {float(vl):.4f} & {sci(float(cm))}\\\\")
out.append(r"""\bottomrule
\end{tabular}
\caption{Dependence of the $B_c=3$ solution on the regularizing shift $\varepsilon$ at small $t_c$ (re-minimization, \texttt{eps\_levels.py}). The exact $o$ and $v$ peaks are at $-1$ and $+1$.}
\label{tab:epslev}
\end{table}
""")
# ---- Tab: placement at small t_c (eps=1e-9)
out.append(r"""\begin{table}[h]
\centering\scriptsize
\begin{tabular}{r r r l l r r r}
\toprule
$V/g$ & $t_c/g$ & $B_c$ & targets (ED satellites) & ghost poles after & empty ghost after & $\|[H_{qp},\rho]\|$ & rank of $\Phi_c$\\
\midrule""")
for V in Vs:
    d = np.load(f"cederbaum_data/ghost_placed_eps1e-09/V{V}.npz")
    for t in (0.1, 0.05):
        for B in (3, 5):
            k = f"tc{t}_B{B}_"
            out.append(f"{V:.3f} & {t} & {B} & {', '.join(f'{x:.3f}' for x in d[k+'targets'])} & {', '.join(f'{x:.3f}' for x in d[k+'ghost_after'])} & {(f'{float(d[k+chr(101)+chr(109)+chr(112)+chr(116)+chr(121)+chr(95)+chr(97)+chr(102)+chr(116)+chr(101)+chr(114)]):.3f}' if B == 5 else '--')} & {sci(float(d[k+'comm_after']))} & {int(d[k+'rankPhi_before'])}$\\to${int(d[k+'rankPhi_after'])}\\\\")
out.append(r"""\bottomrule
\end{tabular}
\caption{Placement of the ghost poles on the exact satellites for the $\varepsilon=10^{-9}$ solutions of Sec.~\ref{sec:results} ($t_c/g=0.1$, $0.05$). QP pole, QP weight and occupations are unchanged (to $\le1.2\times10^{-7}$, $3.2\times10^{-7}$ and $6\times10^{-13}$); the commutator is the same before and after the shift. The empty ghost of $B_c=5$ is placed on the second heaviest ED core particle-side pole ($3.000$, $3.002$ at $t_c/g=0.05$, $0.1$).}
\label{tab:small}
\end{table}
""")

txt = "\n".join(out)
for b in re.split(r"\n(?=\\begin\{table\})", txt):
    mm = re.search(r"\\label\{tab:(\w+)\}", b)
    if mm:
        if mm.group(1) in ("placed", "inv", "small", "epslev"): b = b.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
        open(f"doc/tab_{mm.group(1)}.tex", "w").write(b.strip() + "\n"); print("wrote doc/tab_%s.tex" % mm.group(1))
