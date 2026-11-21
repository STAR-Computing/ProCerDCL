#!/usr/bin/env python3
"""Generate the parameter-study table from
rq4/param_study.csv, so the table cannot drift from the measurements.

One subject, ResNet-18/CIFAR-10 on Inductor, 100 inputs per configuration,
sweeping one guarantee parameter at a time around the default.

Emitted as a wraptable with no surrounding \\vspace: a negative space inside
the box makes wrapfig under-count its height and the body text then overprints
the table. The caller must also keep the \\input out of any { } group, which
wrapfig cannot start a wrap from.

The table is a wraptable: the paper is single-column acmsmall, where a
full-width float leaves a band of whitespace beside it. Only the varied
parameter gets a column, the other two staying at their defaults, which is
what keeps it narrow enough for text to flow alongside.
"""
import csv, os

ROWS = "rq4/param_study.csv"
OUT = "rq4/param_study.tex"

AXIS = {"default": "default", "theta": r"$\theta$", "delta": r"$\delta$",
        "eta": r"$\eta$"}

rows = list(csv.DictReader(open(ROWS)))
order = ["def", "th0p005", "th0p02", "th0p05",
         "dl0p001", "dl0p05", "et0p0025", "et0p01"]
by = {r["config"]: r for r in rows}


def varied_value(r):
    """The value of the parameter this row varies, formatted."""
    axis = r["axis"]
    if axis == "theta":
        return "%g\\%%" % (float(r["theta"]) * 100)
    if axis == "eta":
        return "%g\\%%" % (float(r["eta"]) * 100)
    if axis == "delta":
        return "%g\\%%" % ((1 - float(r["delta"])) * 100)
    return "---"


L = []
A = L.append
A(r"\begin{wraptable}{r}{0.56\textwidth}")
A(r"{%")
A(r"\centering\scriptsize")
A(r"\setlength{\tabcolsep}{4pt}")
A(r"\caption{Parameter study; the setup is described in the text.}")
A(r"\label{tab:param}")
A(r"\begin{tabular}{llrrr}")
A(r"\toprule")
A(r"varied & value & concl. & radius & draws \\")
A(r"\midrule")
for cfg in order:
    r = by.get(cfg)
    if not r:
        continue
    frac = f"{float(r['conclusive_frac'])*100:.0f}\\%"
    A("%s & %s & %s & %.2f & %s \\\\" % (
        AXIS[r["axis"]], varied_value(r), frac,
        float(r["median_eps"]), f"{int(r['median_samples']):,}"))
A(r"\bottomrule")
A(r"\end{tabular}%")
A(r"}")
A(r"\end{wraptable}")

os.makedirs("rq4", exist_ok=True)
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} with {len(order)} configurations")
