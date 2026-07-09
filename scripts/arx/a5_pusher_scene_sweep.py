"""Orchestrate A5 Pusher-like scene sweeps in subprocesses.

Genesis is more reliable when each scene build happens in a fresh process, so
this script does not import Genesis. It calls a5_pusher_forward_sanity.py for
each candidate and aggregates the resulting JSON files into one CSV/JSON.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "arx" / "a5_pusher_forward_sanity.py"
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_pusher_scene_sweep.json")
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/a5_pusher_scene_sweep.csv")
DEFAULT_RUN_DIR = Path("analysis/2026-07-09_arx_pusher/a5_scene_sweep_runs")
DEFAULT_QPOS = "0.0,1.4,-0.4,0.5,0.0,0.0"


def _parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def _qvel(speed):
    return f"{speed},0.0,0.0,0.0,0.0,0.0"


def _run_one(args, obj_y, obj_z, speed, push_steps, idx):
    out = args.run_dir / f"run_{idx:03d}_y{obj_y:.4f}_z{obj_z:.4f}_s{speed:.2f}_n{push_steps}.json"
    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "--no-capture-output",
        "python",
        str(SCRIPT),
        "--obj-x",
        f"{args.obj_x:.6f}",
        "--obj-y",
        f"{obj_y:.6f}",
        "--obj-z",
        f"{obj_z:.6f}",
        "--qpos",
        args.qpos,
        "--qvel",
        _qvel(speed),
        "--settle-steps",
        str(args.settle_steps),
        "--push-steps",
        str(push_steps),
        "--out",
        str(out),
    ]
    if args.requires_grad:
        cmd.append("--requires-grad")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    row = {
        "idx": idx,
        "status": "ok" if proc.returncode == 0 and out.exists() else f"error:{proc.returncode}",
        "obj_x": args.obj_x,
        "obj_y": obj_y,
        "obj_z": obj_z,
        "speed": speed,
        "push_steps": push_steps,
        "requires_grad": args.requires_grad,
        "out": str(out),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-5:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-5:]),
    }
    if out.exists():
        payload = json.loads(out.read_text())
        result = payload["result"]
        horizontal = float(result["horizontal_disp"])
        vertical = float(result["vertical_disp"])
        row.update(
            {
                "horizontal_disp": horizontal,
                "vertical_disp": vertical,
                "abs_vertical_disp": abs(vertical),
                "vertical_over_horizontal": abs(vertical) / (horizontal + 1e-12),
                "max_total_contact_count": int(result["max_total_contact_count"]),
                "initial_pos": json.dumps(result["initial_pos"]),
                "final_pos": json.dumps(result["final_pos"]),
                "displacement": json.dumps(result["displacement"]),
            }
        )
        row["score"] = horizontal - args.vertical_weight * abs(vertical)
    else:
        row.update(
            {
                "horizontal_disp": 0.0,
                "vertical_disp": 0.0,
                "abs_vertical_disp": 0.0,
                "vertical_over_horizontal": 0.0,
                "max_total_contact_count": 0,
                "initial_pos": "",
                "final_pos": "",
                "displacement": "",
                "score": -1e9,
            }
        )
    return row


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "idx",
        "status",
        "score",
        "horizontal_disp",
        "vertical_disp",
        "abs_vertical_disp",
        "vertical_over_horizontal",
        "max_total_contact_count",
        "obj_x",
        "obj_y",
        "obj_z",
        "speed",
        "push_steps",
        "requires_grad",
        "out",
        "initial_pos",
        "final_pos",
        "displacement",
        "stdout_tail",
        "stderr_tail",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="genesis")
    parser.add_argument("--requires-grad", action="store_true")
    parser.add_argument("--obj-x", type=float, default=0.306)
    parser.add_argument("--obj-y-values", default="0.074,0.076,0.078,0.080,0.082")
    parser.add_argument("--obj-z-values", default="0.118,0.120,0.122")
    parser.add_argument("--speeds", default="1.0,1.5,2.0")
    parser.add_argument("--push-steps", default="170,220")
    parser.add_argument("--qpos", default=DEFAULT_QPOS)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--vertical-weight", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="0 means run all combinations")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    obj_y_values = _parse_float_list(args.obj_y_values)
    obj_z_values = _parse_float_list(args.obj_z_values)
    speeds = _parse_float_list(args.speeds)
    push_steps_values = _parse_int_list(args.push_steps)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for obj_z in obj_z_values:
        for obj_y in obj_y_values:
            for speed in speeds:
                for push_steps in push_steps_values:
                    if args.limit and idx >= args.limit:
                        break
                    idx += 1
                    row = _run_one(args, obj_y, obj_z, speed, push_steps, idx)
                    rows.append(row)
                    print(
                        f"[sweep:{idx:03d}] y={obj_y:.4f} z={obj_z:.4f} speed={speed:.2f} "
                        f"steps={push_steps} h={row['horizontal_disp']:.6f} "
                        f"v={row['vertical_disp']:.6f} score={row['score']:.6f} {row['status']}",
                        flush=True,
                    )
                if args.limit and idx >= args.limit:
                    break
            if args.limit and idx >= args.limit:
                break
        if args.limit and idx >= args.limit:
            break

    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    payload = {
        "description": "A5 Pusher-like scene sweep",
        "requires_grad": args.requires_grad,
        "qpos": args.qpos,
        "vertical_weight": args.vertical_weight,
        "num_rows": len(rows),
        "best": ranked[0] if ranked else None,
        "top10": ranked[:10],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.csv, ranked)
    print(f"[sweep] wrote {args.out} and {args.csv}", flush=True)
    if ranked:
        best = ranked[0]
        print(
            f"[sweep] best y={best['obj_y']:.4f} z={best['obj_z']:.4f} "
            f"speed={best['speed']:.2f} steps={best['push_steps']} "
            f"h={best['horizontal_disp']:.6f} v={best['vertical_disp']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
