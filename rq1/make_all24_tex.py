#!/usr/bin/env python3
"""Regenerate the archived RQ1 sweep tables from rq1/rq1_all24_table.csv.

  rq1/rq1_all24.tex   full-disclosure table over all swept tasks
                              (+ per-cell K_max and effective confidence 1-K*delta)
  rq1/rq1_attack.tex  per-setting r / rho for the QUALIFIED tasks only

Qualification comes from the CSV's ``qualified`` column and reflects the
saturation-only rule (n_c >= 90 on both backends). Cells with <5 conclusive points
show n_c but no correlation; the MobileNetV2/inductor cell is absent from the
artifact. All emitted LaTeX is wrapped in red."""
import csv, math, collections
rows=list(csv.DictReader(open("rq1/rq1_all24_table.csv")))
cells={(int(r["task_id"]), r["backend"]): r for r in rows if r["backend"] in ("inductor","tensorrt")}
tasks=sorted({int(r["task_id"]) for r in rows})
meta={int(r["task_id"]):(r["model"],r["dataset"],r["qualified"]=="1") for r in rows}
nq=sum(1 for t in tasks if meta[t][2])
DELTA=0.01

def fmt(c):
    if c is None: return "-- & -- & -- & --"
    n=f"{c['n_conclusive']}/{c['n_total']}"
    k=f"{c['K_max']}" if c.get("K_max") not in (None,"","0") else "--"
    if c["pearson_r"]=="": return f"{n} & -- & -- & {k}"
    return f"{n} & {float(c['pearson_r']):.2f} & {float(c['spearman_rho']):.2f} & {k}"

def agg(backend,pred):
    sub=[c for (t,b),c in cells.items() if b==backend and pred(t) and c["pearson_r"]!=""]
    naive=sum(float(c["pearson_r"]) for c in sub)/len(sub)
    zs=[(math.atanh(float(c["pearson_r"])), int(c["n_conclusive"])-3) for c in sub]
    W=sum(w for _,w in zs); fz=math.tanh(sum(z*w for z,w in zs)/W)
    nrho=sum(float(c["spearman_rho"]) for c in sub)/len(sub)
    return naive,fz,nrho

kmax_all=max(int(c["K_max"]) for c in cells.values() if c.get("K_max") not in (None,""))
kmin_all=min(int(c["K_max"]) for c in cells.values() if c.get("K_max") not in (None,""))

# ---------------------------------------------------------------- all-24 ----
L=[]
L.append(r"\begin{table}[t]")
L.append(r"{\color{red}%")
L.append(r"\centering\scriptsize")
L.append(r"\setlength{\tabcolsep}{3.6pt}")
L.append(r"\caption{Full disclosure of the RQ1 sweep: all 24 swept (model, dataset) tasks per backend. "
         r"$n_{\mathrm{c}}$ = conclusive certificates (found, $\epsilon_{\mathrm{Hard}}\le100$) out of the verified attack-success inputs; "
         r"$r$/$\rho$ = Pearson/Spearman correlation with the attack budget on the conclusive pairs (blank if $n_{\mathrm{c}}<5$); "
         r"$K$ = maximum number of certification probes issued by the $\epsilon$-search in that cell, so the reported radii carry an "
         r"effective union-bound confidence of at least $1-K\delta$ ($\delta=0.01$; range " + f"{1-kmax_all*DELTA:.2f}--{1-kmin_all*DELTA:.2f}" + r" over all cells). "
         r"Qual.\ (\ding{51}) marks the " + str(nq) + r" tasks that pass the pre-specified saturation-only rule "
         r"$n_{\mathrm{c}}\ge90$ on \emph{both} backends; the significance of $r$ is not a criterion. "
         r"Aggregates are reported unweighted and Fisher-$z$ weighted (weights $n_{\mathrm{c}}-3$).}")
L.append(r"\label{tab:all24}")
L.append(r"\begin{tabular}{llc rccr rccr}")
L.append(r"\toprule")
L.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Dataset} & \multirow{2}{*}{Qual.} & \multicolumn{4}{c}{inductor} & \multicolumn{4}{c}{tensorrt} \\")
L.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
L.append(r"& & & $n_{\mathrm{c}}$ & $r$ & $\rho$ & $K$ & $n_{\mathrm{c}}$ & $r$ & $\rho$ & $K$ \\")
L.append(r"\midrule")
for t in tasks:
    m,d,q=meta[t]
    qm=r"\ding{51}" if q else ""
    L.append(f"{m} & {d} & {qm} & {fmt(cells.get((t,'inductor')))} & {fmt(cells.get((t,'tensorrt')))} \\\\")
L.append(r"\midrule")
for label,pred in ((f"Mean ({nq} qualified)",lambda t:meta[t][2]),("Mean (all swept cells)",lambda t:True)):
    ni,fi,ri=agg("inductor",pred); nt,ft,rt=agg("tensorrt",pred)
    L.append(f"\\multicolumn{{3}}{{l}}{{{label}, unweighted / Fisher-$z$}} & & {ni:.2f}\\,/\\,{fi:.2f} & {ri:.2f} & & & {nt:.2f}\\,/\\,{ft:.2f} & {rt:.2f} & \\\\")
L.append(r"\bottomrule")
L.append(r"\end{tabular}%")
L.append(r"}")
L.append(r"\end{table}")
open("rq1/rq1_all24.tex","w").write("\n".join(L)+"\n")

# ------------------------------------------------------------ qualified ----
Q=[]
Q.append(r"\begin{table}[t]")
Q.append(r"{\color{red}%")
Q.append(r"\centering\small")
Q.append(r"\setlength{\tabcolsep}{3pt}")
Q.append(r"\caption{Per-setting correlation (Pearson $r$ / Spearman $\rho$) between \tool's certified hardness "
         r"$\epsilon_{\mathrm{Hard}}$ and the white-box attack's minimal $L_2$ flip budget, for the " + str(nq) +
         r" tasks that pass the saturation-only qualification rule of Section~\ref{sec:setup} ($n_{\mathrm{c}}\ge90$ of 100 on both backends), per compiler backend. "
         r"Every qualified cell is reported regardless of significance; $p$-values and confidence intervals are in the artifact CSV.}")
Q.append(r"\label{tab:robustness}")
Q.append(r"\begin{tabular}{llcccc}")
Q.append(r"\toprule")
Q.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Dataset} & \multicolumn{2}{c}{inductor} & \multicolumn{2}{c}{tensorrt} \\")
Q.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
Q.append(r"& & Pearson & Spearman & Pearson & Spearman \\")
Q.append(r"\midrule")
def f2(c,key):
    return f"{float(c[key]):.2f}" if c and c[key]!="" else "--"
# group rows by model for \multirow like the original table
by_model=collections.OrderedDict()
for t in tasks:
    if meta[t][2]: by_model.setdefault(meta[t][0],[]).append(t)
for m,ts in by_model.items():
    for i,t in enumerate(ts):
        ci=cells.get((t,"inductor")); ct=cells.get((t,"tensorrt"))
        head=(f"\\multirow{{{len(ts)}}}{{*}}{{{m}}}" if len(ts)>1 else m) if i==0 else ""
        Q.append(f"{head} & {meta[t][1]} & {f2(ci,'pearson_r')} & {f2(ci,'spearman_rho')} & {f2(ct,'pearson_r')} & {f2(ct,'spearman_rho')} \\\\")
    Q.append(r"\midrule")
ni,fi,ri=agg("inductor",lambda t:meta[t][2]); nt,ft,rt=agg("tensorrt",lambda t:meta[t][2])
Q.append(f"\\multicolumn{{2}}{{l}}{{Mean (unweighted)}} & {ni:.2f} & {ri:.2f} & {nt:.2f} & {rt:.2f} \\\\")
Q.append(r"\bottomrule")
Q.append(r"\end{tabular}%")
Q.append(r"}")
Q.append(r"\end{table}")
open("rq1/rq1_attack.tex","w").write("\n".join(Q)+"\n")
print(f"qualified tasks: {nq}; K range {kmin_all}-{kmax_all}; wrote rq1/rq1_all24.tex and rq1/rq1_attack.tex")
