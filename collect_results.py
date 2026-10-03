"""
把 results/experiments/ 底下所有 result.json 彙整成一張表(summary.csv),每組實驗一列。

用法:
    python collect_results.py                    # 輸出 results/experiments/summary.csv
    python collect_results.py --root results/xxx
"""

import argparse
import csv
import glob
import json
import os

COLUMNS = [
    "task", "algo", "tag", "seed", "total_timesteps", "train_seconds",
    "det_success_rate", "det_mean_steps_to_success", "det_mean_return",
    "sto_success_rate", "sto_mean_steps_to_success",
    "steps_to_eval_success_50", "steps_to_eval_success_90",
    "steps_to_train_success_50", "steps_to_train_success_90",
    "run_dir",
]


def collect(root: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "result.json"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        det, sto = r["final_eval_deterministic"], r["final_eval_stochastic"]
        rows.append({
            "task": r["task"], "algo": r["algo"], "tag": r["tag"], "seed": r["seed"],
            "total_timesteps": r["total_timesteps"], "train_seconds": r["train_seconds"],
            "det_success_rate": det["success_rate"],
            "det_mean_steps_to_success": det["mean_steps_to_success"],
            "det_mean_return": det["mean_return"],
            "sto_success_rate": sto["success_rate"],
            "sto_mean_steps_to_success": sto["mean_steps_to_success"],
            "steps_to_eval_success_50": r["steps_to_eval_success_50"],
            "steps_to_eval_success_90": r["steps_to_eval_success_90"],
            "steps_to_train_success_50": r["steps_to_train_success_50"],
            "steps_to_train_success_90": r["steps_to_train_success_90"],
            "run_dir": os.path.dirname(path),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/experiments")
    args = parser.parse_args()

    rows = collect(args.root)
    out = os.path.join(args.root, "summary.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig:Excel 開中文不會亂碼
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"彙整 {len(rows)} 組實驗 → {out}")
