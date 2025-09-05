import pathlib
from datetime import datetime

OUT_DIR = pathlib.Path("analysis_outputs")
REPORT  = OUT_DIR / "plots_report.md"

imgs = [
    "papers_per_year.png",
    "publication_trend.png",
    "top_journals.png",
    "top_authors.png",
    "length_distributions.png",
]

lines = [
    f"# Plot Report ({datetime.now():%Y-%m-%d})\n",
    "This document summarizes the descriptive statistics figures produced from the Elsevier dataset.\n",
]

for name in imgs:
    p = OUT_DIR / name
    if p.exists():
        title = name.replace("_", " ").replace(".png", "").title()
        lines += [f"## {title}\n", f"![{title}]({p.name})\n\n"]
    else:
        lines += [f"## {name} (not generated)\n\n"]

REPORT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote report: {REPORT.resolve()}")
