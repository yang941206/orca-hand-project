"""
階段 4:讀取 results/experiments/ 與 results/final/ 的實驗結果,產生論文/講義用的圖表與表格。

產生(都存在 analysis/figures/ 與 analysis/tables.md):
- fig1_phaseA_curves.png     :階段 A 學習曲線(定期 deterministic 評估成功率,3 seeds 平均 + 範圍)
- fig2_efficiency_vs_time.png:樣本效率(首次 ≥90% 的步數)vs 牆鐘時間(每組訓練分鐘數),分兩張圖、不用雙軸
- fig3_phaseB_seeds.png      :階段 B 各設定「首次 ≥90% 步數」的每個 seed 與平均 → 看 seed 變異 vs 設定差異
- fig4_ppo_pinch_instability.png:PPO 捏合的 deterministic 評估 vs 訓練中(隨機)成功率,說明 E18 的疑問
- tables.md                  :階段 A、B、正式版的彙整表,含「穩定度」指標

「穩定度」指標(為什麼需要):「首次 ≥90% 的步數」只記第一次碰到門檻,不管之後有沒有掉下來。
PPO 捏合 seed 1 在 4 萬步評估 100%,之後 17–39 萬步卻全是 0%(見 fig4)。所以另外計算:
  - stable_after_first90:第一次 ≥90% 之後的所有定期評估裡,仍 ≥90% 的比例
  - last5_eval_mean:最後 5 次定期評估成功率的平均

用法:
    python analysis/plot_results.py
"""

import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "experiments")
FINAL = os.path.join(ROOT, "results", "final")
FIG_DIR = os.path.join(ROOT, "analysis", "figures")
TABLES = os.path.join(ROOT, "analysis", "tables.md")

TASK_NAMES = {"fist": "握拳", "pinch": "捏合"}
ALGO_ORDER = ["ppo", "sac", "td3"]
# 類別色(固定順序,用 dataviz 驗證器檢查過色盲可分辨度,見 實驗紀錄)
ALGO_COLOR = {"ppo": "#2a78d6", "sac": "#eb6834", "td3": "#1baf7a"}
NEUTRAL = "#8a8985"
INK = "#3d3c38"
PHASE_B_ORDER = ["default", "lr3e-4", "lr3e-3", "net64", "net256", "batch128", "batch512"]
PHASE_B_LABEL = {
    "default": "預設", "lr3e-4": "學習率 3e-4", "lr3e-3": "學習率 3e-3", "net64": "網路 64×64",
    "net256": "網路 256×256", "batch128": "batch 128", "batch512": "batch 512",
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK,
    "xtick.color": "#5e5d59", "ytick.color": "#5e5d59",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


# ---- 讀資料 ----
def load_runs(root: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "result.json"), recursive=True)):
        run_dir = os.path.dirname(path)
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            r["config"] = json.load(f)
        with open(os.path.join(run_dir, "curve.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        to_f = lambda v: float(v) if v not in ("", None) else None  # noqa: E731
        r["curve"] = [{k: to_f(v) for k, v in row.items()} for row in rows]
        r["evals"] = [(row["timesteps"], row["eval_success_rate"]) for row in r["curve"]
                      if row["eval_success_rate"] is not None]
        runs.append(r)
    return runs


def stability(run: dict) -> tuple[float | None, float | None]:
    """回傳 (第一次 ≥90% 之後仍 ≥90% 的比例, 最後 5 次定期評估的平均)。"""
    evals = [v for _, v in run["evals"]]
    last5 = float(np.mean(evals[-5:])) if evals else None
    idx = next((i for i, v in enumerate(evals) if v >= 0.9), None)
    if idx is None:
        return None, last5
    after = evals[idx:]
    return float(np.mean([v >= 0.9 for v in after])), last5


def group(runs, **match):
    return [r for r in runs if all(r[k] == v for k, v in match.items())]


def step_grid(run, grid):
    """把某次實驗的定期評估結果對齊到共同的步數格點(取該步數以前最近一次評估)。"""
    xs = [t for t, _ in run["evals"]]
    ys = [v for _, v in run["evals"]]
    out = []
    for g in grid:
        i = np.searchsorted(xs, g, side="right") - 1
        out.append(ys[i] * 100 if i >= 0 else np.nan)
    return np.array(out)


def minutes_to_first90(run):
    """首次定期評估 ≥90% 時,已經過的訓練牆鐘時間(分鐘;含定期評估的時間)。沒達到回傳 None。"""
    for row in run["curve"]:
        if row["eval_success_rate"] is not None and row["eval_success_rate"] >= 0.9:
            return row["elapsed_s"] / 60
    return None


def fmt_k(x):
    return "—" if x is None else f"{x / 1000:.0f}k"


# ---- 圖 1:階段 A 學習曲線 ----
def fig_phase_a_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, task in zip(axes, ["fist", "pinch"]):
        budget = max(r["total_timesteps"] for r in group(runs, task=task, tag="default"))
        freq = group(runs, task=task, tag="default")[0]["config"]["eval_freq"]
        grid = np.arange(freq, budget + 1, freq)
        for algo in ALGO_ORDER:
            rs = group(runs, task=task, algo=algo, tag="default")
            Y = np.vstack([step_grid(r, grid) for r in rs])
            mean, lo, hi = np.nanmean(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0)
            ax.fill_between(grid / 1e6, lo, hi, color=ALGO_COLOR[algo], alpha=0.15, linewidth=0)
            ax.plot(grid / 1e6, mean, color=ALGO_COLOR[algo], lw=2, label=f"{algo.upper()}(3 seeds 平均,陰影=最小~最大)")
        ax.set(title=f"{TASK_NAMES[task]}:定期評估成功率(deterministic,10 回合)",
               xlabel="訓練步數(百萬)", ylabel="成功率 (%)", ylim=(-3, 103))
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_phaseA_curves.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---- 圖 2:樣本效率 vs 牆鐘時間(兩張圖,不用雙軸)----
def fig_efficiency_vs_time(runs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharey="row")
    for row, task in enumerate(["fist", "pinch"]):
        for col, (key, xlabel) in enumerate([("steps90", "首次達到 90% 所需訓練步數(千步,對數軸)"),
                                             ("minutes", "每組訓練牆鐘時間(分鐘,對數軸)")]):
            ax = axes[row, col]
            for yi, algo in enumerate(ALGO_ORDER):
                rs = group(runs, task=task, algo=algo, tag="default")
                if key == "steps90":
                    vals = [(r["steps_to_eval_success_90"] or r["total_timesteps"]) / 1000 for r in rs]
                else:
                    vals = [r["train_seconds"] / 60 for r in rs]
                jitter = np.linspace(-0.15, 0.15, len(vals))  # 數值相同的 seed 上下錯開,不會疊成一個點
                ax.scatter(vals, yi + jitter, s=70, color=ALGO_COLOR[algo], edgecolor="#fcfcfb", linewidth=2, zorder=3)
                ax.plot([np.mean(vals)] * 2, [yi - 0.28, yi + 0.28], color=INK, lw=2)
                unit = "k 步" if key == "steps90" else " 分"
                ax.annotate(f"平均 {np.mean(vals):.0f}{unit}", (np.mean(vals), yi + 0.32), fontsize=9, color=INK, ha="center")
            ax.set_xscale("log")
            ticks = [5, 10, 25, 50, 100, 200, 500, 1000]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks(range(3), [a.upper() for a in ALGO_ORDER])
            ax.set_ylim(-0.6, 2.7)
            ax.grid(axis="x", color="#e6e5e0", lw=0.8)
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(TASK_NAMES[task])
    axes[0, 0].set_title("樣本效率:越左越好")
    axes[0, 1].set_title("牆鐘時間:越左越好")
    fig.suptitle("階段 A:樣本效率與牆鐘時間的結論相反(點 = 各 seed,短線 = 平均)", fontsize=12, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_efficiency_vs_time.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---- 圖 3:階段 B,seed 變異 vs 設定差異 ----
def fig_phase_b_seeds(runs, winners):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, task in zip(axes, ["fist", "pinch"]):
        for yi, tag in enumerate(PHASE_B_ORDER):
            rs = group(runs, task=task, algo="td3", tag=tag)
            vals = [(r["steps_to_eval_success_90"] or r["total_timesteps"]) / 1000 for r in rs]
            color = ALGO_COLOR["td3"] if tag == winners[task] else NEUTRAL
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(vals, yi + jitter, s=55, color=color, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
            ax.plot([np.mean(vals)] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=2)
        ax.set_yticks(range(len(PHASE_B_ORDER)), [PHASE_B_LABEL[t] + (" ★" if t == winners[task] else "")
                                                  for t in PHASE_B_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("首次達到 90% 所需訓練步數(千步)")
        ax.set_title(f"{TASK_NAMES[task]} × TD3(點 = 各 seed,短線 = 平均,★ = 自動選出)")
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_phaseB_seeds.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---- 圖 4:PPO 捏合不穩定 ----
def fig_ppo_pinch(runs):
    rs = group(runs, task="pinch", algo="ppo", tag="default")
    fig, axes = plt.subplots(1, len(rs), figsize=(13, 3.8), sharey=True)
    for ax, r in zip(axes, rs):
        tr = [(row["timesteps"], row["train_success_rate"] * 100) for row in r["curve"] if row["train_success_rate"] is not None]
        ax.plot([t / 1000 for t, _ in tr], [v for _, v in tr], color=NEUTRAL, lw=2, label="訓練中(隨機動作,最近 100 回合)")
        ax.plot([t / 1000 for t, _ in r["evals"]], [v * 100 for _, v in r["evals"]], color=ALGO_COLOR["ppo"], lw=2,
                marker="o", ms=3, label="定期評估(deterministic,10 回合)")
        det = r["final_eval_deterministic"]["success_rate"] * 100
        ax.set_title(f"seed {r['seed']}:最終 deterministic {det:.0f}%")
        ax.set_xlabel("訓練步數(千步)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    axes[0].set_ylabel("成功率 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle("PPO × 捏合:隨機動作穩定進步,但「固定動作」版本反覆在 0% 與 100% 之間震盪", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = os.path.join(FIG_DIR, "fig4_ppo_pinch_instability.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---- 表格 ----
def table_rows(rs):
    det = [r["final_eval_deterministic"]["success_rate"] * 100 for r in rs]
    eps = [r["final_eval_deterministic"]["mean_steps_to_success"] for r in rs]
    s90 = [r["steps_to_eval_success_90"] for r in rs]
    stab = [stability(r) for r in rs]
    mins = [r["train_seconds"] / 60 for r in rs]
    return {
        "det": "/".join(f"{d:.0f}" for d in det) + f"(平均 {np.mean(det):.1f})",
        "eps": "/".join("—" if e is None else f"{e:.1f}" for e in eps),
        "s90": "/".join(fmt_k(s) for s in s90),
        "stable": "/".join("—" if s is None else f"{s * 100:.0f}%" for s, _ in stab),
        "last5": "/".join("—" if l5 is None else f"{l5 * 100:.0f}%" for _, l5 in stab),
        "min": f"{np.mean(mins):.0f}",
    }


def write_tables(runs, final_runs, selections):
    lines = ["# 實驗結果彙整表(由 analysis/plot_results.py 自動產生)", "",
             "各欄的「a/b/c」依序為 seed 0/1/2。",
             "- **最終 det 成功率**:訓練結束時的模型,20 回合(seed 0–19)deterministic 評估",
             "- **首次 ≥90%**:定期評估(10 回合)第一次達到 90% 的訓練步數",
             "- **≥90% 後的穩定度**:第一次 ≥90% 之後,所有定期評估中仍 ≥90% 的比例(越接近 100% 越穩定)",
             "- **最後 5 次評估平均**:訓練最後 5 次定期評估成功率的平均", ""]
    head = "| {} | 最終 det 成功率 % | 平均成功步數 | 首次 ≥90% | ≥90% 後的穩定度 | 最後 5 次評估平均 | 每組用時(分) |"
    sep = "|---|---|---|---|---|---|---|"
    for task in ["fist", "pinch"]:
        lines += [f"## 階段 A:{TASK_NAMES[task]}(演算法比較,預設超參數)", "", head.format("演算法"), sep]
        for algo in ALGO_ORDER:
            t = table_rows(group(runs, task=task, algo=algo, tag="default"))
            lines.append(f"| {algo.upper()} | {t['det']} | {t['eps']} | {t['s90']} | {t['stable']} | {t['last5']} | {t['min']} |")
        lines += ["", f"自動選擇:**{selections['A'][task]['winner'].upper()}** — {selections['A'][task]['reason']}", ""]
    lines += ["## 階段 A:首次 ≥90% 所需的「步數」與「牆鐘時間」", "",
              "牆鐘時間取自 curve.csv 的 elapsed_s(從訓練開始計時,含定期評估;多組同時跑,有 CPU 競爭)。", "",
              "| 任務 | 演算法 | 首次 ≥90% 步數 | 平均 | 首次 ≥90% 牆鐘時間(分) | 平均 | 整組訓練時間(分) |",
              "|---|---|---|---|---|---|---|"]
    for task in ["fist", "pinch"]:
        for algo in ALGO_ORDER:
            rs = group(runs, task=task, algo=algo, tag="default")
            s90 = [r["steps_to_eval_success_90"] for r in rs]
            m90 = [minutes_to_first90(r) for r in rs]
            lines.append(f"| {TASK_NAMES[task]} | {algo.upper()} | {'/'.join(fmt_k(x) for x in s90)} | {fmt_k(np.mean(s90))} | "
                         f"{'/'.join(f'{x:.1f}' for x in m90)} | {np.mean(m90):.1f} | {np.mean([r['train_seconds'] for r in rs]) / 60:.0f} |")
    lines.append("")
    for task in ["fist", "pinch"]:
        lines += [f"## 階段 B:{TASK_NAMES[task]} × TD3(超參數比較)", "", head.format("設定"), sep]
        for tag in PHASE_B_ORDER:
            t = table_rows(group(runs, task=task, algo="td3", tag=tag))
            star = " ★" if tag == selections["B"][task]["winner"] else ""
            lines.append(f"| {PHASE_B_LABEL[tag]}{star} | {t['det']} | {t['eps']} | {t['s90']} | {t['stable']} | {t['last5']} | {t['min']} |")
        lines += ["", f"自動選擇:**{selections['B'][task]['winner']}** — {selections['B'][task]['reason']}", ""]
    lines += ["## 正式版模型(最佳設定 × 2 倍預算,seed 0)", "",
              "| 任務 | 設定 | 訓練步數 | 最終 det 成功率 | 平均成功步數 | 首次 ≥90% | ≥90% 後的穩定度 | 最後 5 次評估平均 | 用時(分) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(final_runs, key=lambda r: r["task"]):
        s, l5 = stability(r)
        det = r["final_eval_deterministic"]
        lines.append(f"| {TASK_NAMES[r['task']]} | {r['algo'].upper()} {r['config']['hyperparams']} | {r['total_timesteps']:,} | "
                     f"{det['success_rate'] * 100:.0f}% | {det['mean_steps_to_success']:.1f} | {fmt_k(r['steps_to_eval_success_90'])} | "
                     f"{s * 100:.0f}% | {l5 * 100:.0f}% | {r['train_seconds'] / 60:.0f} |")
    with open(TABLES, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return TABLES


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs(EXP)
    final_runs = load_runs(FINAL)
    selections = {ph: {} for ph in "AB"}
    for ph in "AB":
        for task in ["fist", "pinch"]:
            with open(os.path.join(EXP, f"selection_{ph}_{task}.json"), encoding="utf-8") as f:
                selections[ph][task] = json.load(f)
    winners = {t: selections["B"][t]["winner"] for t in ["fist", "pinch"]}
    print(f"讀到 {len(runs)} 組實驗、{len(final_runs)} 組正式版")
    for p in [fig_phase_a_curves(runs), fig_efficiency_vs_time(runs), fig_phase_b_seeds(runs, winners),
              fig_ppo_pinch(runs), write_tables(runs, final_runs, selections)]:
        print("已存:", os.path.relpath(p, ROOT))
