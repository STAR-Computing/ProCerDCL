"""Summarise rq1/pointwise24/*.csv: per backend, verdict counts, pooled
pointwise and reference-stable densities, q, and draws. Writes
rq1/pointwise24_summary.csv (one row per task and backend)."""
import csv, glob, statistics as st, collections
rows = []
for f in sorted(glob.glob("rq1/pointwise24/t*.csv")):
    rs = list(csv.DictReader(open(f)))
    if not rs: continue
    v = collections.Counter(r["verdict"] for r in rs)
    gen = sum(int(r["n_generated"]) for r in rs)
    rows.append(dict(task_id=rs[0]["task_id"], model=rs[0]["model"], dataset=rs[0]["dataset"],
                     backend=rs[0]["backend"], n=len(rs), yes=v.get("Yes", 0), no=v.get("No", 0),
                     none=v.get("None", 0),
                     pointwise_density=f"{sum(int(r['n_pointwise_gen']) for r in rs)/gen:.2e}",
                     joint_density=f"{sum(int(r['n_joint_gen']) for r in rs)/gen:.2e}",
                     q_min=min(float(r["q_ref_stable"]) for r in rs),
                     mean_kdraws=round(st.mean(int(r["n_samples"]) for r in rs) / 1e3, 1)))
w = csv.DictWriter(open("rq1/pointwise24_summary.csv", "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
for b in ("inductor", "tensorrt"):
    sub = [r for r in rows if r["backend"] == b]
    if not sub: continue
    print(b, "tasks", len(sub), "inputs", sum(r["n"] for r in sub), "Yes", sum(r["yes"] for r in sub),
          "No", sum(r["no"] for r in sub), "None", sum(r["none"] for r in sub),
          "pointwise density range", min(float(r["pointwise_density"]) for r in sub), max(float(r["pointwise_density"]) for r in sub),
          "q_min", min(r["q_min"] for r in sub), "mean kdraws", round(st.mean(r["mean_kdraws"] for r in sub), 1))
for r in rows:
    if r["yes"] < r["n"]: print("not all Yes:", r)
