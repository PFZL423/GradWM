"""Summarize the rope solver matrix CSV."""

import csv
import math
from pathlib import Path


CSV_PATH = Path("logs/rope_solver_matrix.csv")
OUT_PATH = Path("analysis/rope_solver_probe/summary.md")


def parse_float(value, default=math.nan):
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_grad_nan(value):
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return 1.0
    if "/" in text:
        num, den = text.split("/", 1)
        den_v = parse_float(den, 0.0)
        return 1.0 if den_v == 0 else parse_float(num, den_v) / den_v
    if text.endswith("%"):
        return parse_float(text[:-1], 100.0) / 100.0
    value_f = parse_float(text, 1.0)
    return value_f / 100.0 if value_f > 1.0 else value_f


def fmt_float(value, digits=3):
    value = parse_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def verdict_for(row):
    status = (row.get("grad_status") or "").strip()
    if status and status != "ok":
        return "blocked"
    fps = parse_float(row.get("fwd_fps"))
    sag = parse_float(row.get("sag_mm"))
    grad_nan = parse_grad_nan(row.get("grad_nan"))
    if grad_nan == 0.0 and fps >= 30.0 and sag >= 5.0:
        return "recommended"
    if grad_nan < 0.10 and fps >= 10.0:
        return "conditional"
    return "blocked"


def markdown_table(rows):
    headers = [
        "solver",
        "n_particles",
        "fwd_fps",
        "bwd_s",
        "grad_norm",
        "grad_nan",
        "sag_mm",
        "peak_mem_mb",
        "grad_status",
        "verdict",
    ]
    table_rows = []
    for row in rows:
        grad_nan_pct = parse_grad_nan(row.get("grad_nan")) * 100.0
        table_rows.append(
            {
                "solver": row.get("solver", ""),
                "n_particles": row.get("n_particles", ""),
                "fwd_fps": fmt_float(row.get("fwd_fps"), 2),
                "bwd_s": fmt_float(row.get("bwd_s"), 2),
                "grad_norm": fmt_float(row.get("grad_norm"), 4),
                "grad_nan": "nan" if not math.isfinite(grad_nan_pct) else f"{grad_nan_pct:.1f}%",
                "sag_mm": fmt_float(row.get("sag_mm"), 2),
                "peak_mem_mb": fmt_float(row.get("peak_mem_mb"), 1),
                "grad_status": row.get("grad_status", ""),
                "verdict": verdict_for(row),
            }
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def verdict_paragraph(row):
    solver = row.get("solver", "unknown")
    verdict = verdict_for(row)
    fps = fmt_float(row.get("fwd_fps"), 2)
    sag = fmt_float(row.get("sag_mm"), 2)
    grad_nan = parse_grad_nan(row.get("grad_nan"))
    finite_pct = max(0.0, (1.0 - grad_nan) * 100.0)
    status = row.get("grad_status", "") or "ok"
    nan_text = "NaN-free" if grad_nan == 0.0 else f"{grad_nan * 100.0:.1f}% non-finite gradient"
    return (
        f"{solver}: backward {nan_text}, {finite_pct:.1f}% finger.grad finite, "
        f"fps {fps}, sag {sag} mm, status {status} - {verdict}."
    )


def main():
    if not CSV_PATH.exists():
        message = f"No matrix CSV found at {CSV_PATH}. Run the solver probes first."
        print(message)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(message + "\n")
        return 0

    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        message = f"No rows found in {CSV_PATH}."
        print(message)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(message + "\n")
        return 0

    table = markdown_table(rows)
    verdicts = "\n\n".join(verdict_paragraph(row) for row in rows)
    summary = f"# Rope Solver Matrix Summary\n\n{table}\n\n{verdicts}\n"

    print(table)
    print()
    print(verdicts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
