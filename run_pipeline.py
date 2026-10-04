"""
全自動實驗流程:階段 A(演算法比較)→ 自動選最佳演算法 → 階段 B(超參數比較)→ 自動選最佳設定 → 正式版模型。

決策(使用者確認,2026-10-03):
- 每個任務各自選最佳演算法、最佳超參數
- 階段 B 完整版:3 個超參數 × 2 個比較值 × 3 seeds(預設值那組沿用階段 A)
- 正式版模型:最佳設定 × 2 倍預算(握拳 200 萬步、捏合 100 萬步)

「最佳」的判斷規則(跑之前就固定,寫在 select_best):
  1. 最終 deterministic 成功率(20 回合,seeds 平均)最高
  2. 跟最高者差距在 5 個百分點以內的,比「定期評估成功率首次 ≥90% 的步數」(越少越好)
  3. 還一樣,比「平均成功步數」(越少越好)

特性:
- 可中斷續跑:已有 result.json 的實驗直接跳過
- 依 CPU 容量同時跑多組(PPO 佔 8、SAC/TD3 佔 4,總容量 16;依 E11/E12/E16 實測)
- 失敗自動重試一次;進度寫在 <out_root>/pipeline.log

用法:
    python run_pipeline.py --dry-run          # 只列出要跑的實驗與時間估算
    python run_pipeline.py                    # 正式執行
    python run_pipeline.py --scale 0.005 --out-root results/_pipeline_smoke   # 縮小步數測試整條流程
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from rl_framework import DEFAULT_TIMESTEPS, ExperimentConfig

TASK_LIST = ["fist", "pinch"]
ALGO_LIST = ["ppo", "sac", "td3"]
SEEDS = [0, 1, 2]
SUCCESS_MARGIN = 0.05
FINAL_BUDGET_MULTIPLIER = 2

# 階段 B 的比較值:以各演算法的 SB3 預設值為基準(實測確認:PPO lr 3e-4、n_steps 2048、64×64;
# SAC lr 3e-4、batch 256、256×256;TD3 lr 1e-3、batch 256、400×300),一次只改一個因素
PHASE_B_GRID = {
    "ppo": {
        "lr1e-4": {"learning_rate": 1e-4},
        "lr1e-3": {"learning_rate": 1e-3},
        "nsteps1024": {"n_steps": 1024},
        "nsteps4096": {"n_steps": 4096},
        "net128": {"policy_kwargs": {"net_arch": [128, 128]}},
        "net256": {"policy_kwargs": {"net_arch": [256, 256]}},
    },
    "sac": {
        "lr1e-4": {"learning_rate": 1e-4},
        "lr1e-3": {"learning_rate": 1e-3},
        "net64": {"policy_kwargs": {"net_arch": [64, 64]}},
        "net400x300": {"policy_kwargs": {"net_arch": [400, 300]}},
        "batch128": {"batch_size": 128},
        "batch512": {"batch_size": 512},
    },
    "td3": {
        "lr3e-4": {"learning_rate": 3e-4},
        "lr3e-3": {"learning_rate": 3e-3},
        "net64": {"policy_kwargs": {"net_arch": [64, 64]}},
        "net256": {"policy_kwargs": {"net_arch": [256, 256]}},
        "batch128": {"batch_size": 128},
        "batch512": {"batch_size": 512},
    },
}

# 排程用的資源成本與執行緒(總容量 16)
CAPACITY = 16
COST = {"ppo": 8, "sac": 4, "td3": 4}
TORCH_THREADS = {"ppo": 2, "sac": 4, "td3": 4}
# 時間估算用的速度(steps/s,同時跑時每組的速度;E11 PPO 握拳 3 組同時 ~1,900、E12 SAC 4 組同時 122、TD3 推估 150)
EST_SPEED = {"ppo": 1_900, "sac": 122, "td3": 150}


# ---- 紀錄 ----
class Logger:
    def __init__(self, out_root: str):
        os.makedirs(out_root, exist_ok=True)
        self.path = os.path.join(out_root, "pipeline.log")
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{datetime.now():%m-%d %H:%M:%S}] [已過 {(time.time() - self.t0) / 3600:5.2f} h] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---- 實驗工作 ----
def make_cfg(task, algo, seed, tag, hyperparams, timesteps, out_root) -> ExperimentConfig:
    return ExperimentConfig(task=task, algo=algo, seed=seed, tag=tag, hyperparams=hyperparams,
                            total_timesteps=timesteps, torch_threads=TORCH_THREADS[algo],
                            out_root=out_root).resolved()


def result_path(cfg: ExperimentConfig) -> str:
    return os.path.join(cfg.run_dir, "result.json")


def run_jobs(cfgs: list[ExperimentConfig], log: Logger, phase: str) -> None:
    """依容量平行執行;已完成的跳過;失敗重試一次。"""
    todo = [c for c in cfgs if not os.path.exists(result_path(c))]
    skipped = len(cfgs) - len(todo)
    if skipped:
        log(f"{phase}:{skipped} 組已有結果,跳過")
    # 先排最久的(LPT),縮短總時間
    todo.sort(key=lambda c: c.total_timesteps / EST_SPEED[c.algo], reverse=True)
    queue = [(c, 1) for c in todo]
    running = []  # (proc, cfg, attempt, start_time, log_file)
    done = 0
    total = len(todo)

    while queue or running:
        used = sum(COST[c.algo] for _, c, _, _, _ in running)
        i = 0
        while i < len(queue):
            cfg, attempt = queue[i]
            if used + COST[cfg.algo] <= CAPACITY:
                queue.pop(i)
                os.makedirs(cfg.run_dir, exist_ok=True)
                job_cfg = os.path.join(cfg.run_dir, "job_config.json")
                fields = {k: v for k, v in cfg.__dict__.items()}
                with open(job_cfg, "w", encoding="utf-8") as f:
                    json.dump(fields, f, ensure_ascii=False, indent=2)
                # 用附加模式:重試時保留上一次失敗的錯誤訊息(E19:第一次失敗的原因因為被覆蓋而遺失)
                out = open(os.path.join(cfg.run_dir, "stdout.txt"), "a", encoding="utf-8")
                out.write(f"\n===== 第 {attempt} 次執行,{datetime.now():%m-%d %H:%M:%S} =====\n")
                out.flush()
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
                proc = subprocess.Popen([sys.executable, "train_experiment.py", "--config", job_cfg],
                                        stdout=out, stderr=subprocess.STDOUT, env=env)
                running.append((proc, cfg, attempt, time.time(), out))
                used += COST[cfg.algo]
            else:
                i += 1

        time.sleep(5)
        still = []
        for proc, cfg, attempt, start, out in running:
            if proc.poll() is None:
                still.append((proc, cfg, attempt, start, out))
                continue
            out.close()
            name = f"{cfg.task}×{cfg.algo}×{cfg.tag}×seed{cfg.seed}"
            minutes = (time.time() - start) / 60
            if os.path.exists(result_path(cfg)):
                done += 1
                with open(result_path(cfg), encoding="utf-8") as f:
                    r = json.load(f)
                det = r["final_eval_deterministic"]
                log(f"{phase} [完成 {done}/{total}] {name} 用時 {minutes:.1f} 分 | "
                    f"det 成功率 {det['success_rate'] * 100:.0f}% | ≥90% 於 {r['steps_to_eval_success_90']} 步")
            elif attempt < 2:
                log(f"{phase} [失敗,重試] {name}(exit {proc.returncode},見 {cfg.run_dir}/stdout.txt)")
                queue.insert(0, (cfg, attempt + 1))
            else:
                done += 1
                log(f"{phase} [失敗兩次,放棄] {name}(exit {proc.returncode})")
        running = still


# ---- 選最佳 ----
def summarize_group(cfgs: list[ExperimentConfig]) -> dict | None:
    """一組設定(多個 seed)的平均指標;沒達到 90% 的 seed 以「預算 + 一次評估間隔」計。"""
    rs = []
    for c in cfgs:
        if os.path.exists(result_path(c)):
            with open(result_path(c), encoding="utf-8") as f:
                rs.append((c, json.load(f)))
    if not rs:
        return None
    success = [r["final_eval_deterministic"]["success_rate"] for _, r in rs]
    steps90 = [r["steps_to_eval_success_90"] if r["steps_to_eval_success_90"] is not None
               else c.total_timesteps + c.eval_freq for c, r in rs]
    ep_steps = [r["final_eval_deterministic"]["mean_steps_to_success"] or c.max_episode_steps for c, r in rs]
    return {
        "n_seeds": len(rs),
        "success_mean": float(np.mean(success)), "success_std": float(np.std(success)),
        "success_per_seed": success,
        "steps90_mean": float(np.mean(steps90)), "steps90_per_seed": steps90,
        "ep_steps_mean": float(np.mean(ep_steps)),
    }


def select_best(groups: dict[str, dict]) -> tuple[str, str]:
    """依固定規則選最佳,回傳 (名稱, 理由)。"""
    valid = {k: v for k, v in groups.items() if v is not None}
    best_success = max(v["success_mean"] for v in valid.values())
    close = {k: v for k, v in valid.items() if v["success_mean"] >= best_success - SUCCESS_MARGIN}
    winner = min(close, key=lambda k: (close[k]["steps90_mean"], close[k]["ep_steps_mean"]))
    if len(close) == 1:
        reason = f"最終成功率最高({best_success * 100:.1f}%),其他都低超過 {SUCCESS_MARGIN * 100:.0f} 個百分點"
    else:
        reason = (f"最終成功率在最高值 {best_success * 100:.1f}% 的 {SUCCESS_MARGIN * 100:.0f} 個百分點內的有 "
                  f"{sorted(close)};其中達到 90% 所需步數最少(平均 {close[winner]['steps90_mean']:.0f} 步)")
    return winner, reason


def save_selection(out_root, name, groups, winner, reason, log):
    path = os.path.join(out_root, f"selection_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"winner": winner, "reason": reason, "rule": __doc__.split("「最佳」的判斷規則")[1].split("特性")[0].strip(),
                   "groups": groups}, f, ensure_ascii=False, indent=2)
    table = " | ".join(f"{k}: {v['success_mean'] * 100:.0f}%/{v['steps90_mean']:.0f}步" for k, v in groups.items() if v)
    log(f"選擇 {name} → **{winner}**({reason})| {table}")


# ---- 主流程 ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="results/experiments")
    parser.add_argument("--final-root", default="results/final")
    parser.add_argument("--scale", type=float, default=1.0, help="訓練步數倍率(測試流程用)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    budget = {t: max(int(DEFAULT_TIMESTEPS[t] * args.scale), 1000) for t in TASK_LIST}
    log = Logger(args.out_root)

    phase_a = {t: {a: [make_cfg(t, a, s, "default", {}, budget[t], args.out_root) for s in SEEDS]
                   for a in ALGO_LIST} for t in TASK_LIST}

    if args.dry_run:
        def est(cfgs):
            return sum(c.total_timesteps / EST_SPEED[c.algo] * COST[c.algo] for c in cfgs) / CAPACITY / 3600
        a_cfgs = [c for t in phase_a.values() for cs in t.values() for c in cs]
        print(f"階段 A:{len(a_cfgs)} 組,估計約 {est(a_cfgs):.1f} 小時(平行)")
        for algo in ALGO_LIST:
            b = [make_cfg(t, algo, s, tag, hp, budget[t], args.out_root)
                 for t in TASK_LIST for tag, hp in PHASE_B_GRID[algo].items() for s in SEEDS]
            print(f"階段 B 若兩個任務都是 {algo}:{len(b)} 組,估計約 {est(b):.1f} 小時")
        return

    log(f"===== 開始全自動流程(scale={args.scale},預算 {budget}) =====")

    # 階段 A
    run_jobs([c for t in phase_a.values() for cs in t.values() for c in cs], log, "階段A")
    best_algo = {}
    for t in TASK_LIST:
        groups = {a: summarize_group(cs) for a, cs in phase_a[t].items()}
        best_algo[t], reason = select_best(groups)
        save_selection(args.out_root, f"A_{t}", groups, best_algo[t], reason, log)

    # 階段 B(各任務用自己的最佳演算法;預設值那組沿用階段 A)
    phase_b = {t: {tag: [make_cfg(t, best_algo[t], s, tag, hp, budget[t], args.out_root) for s in SEEDS]
                   for tag, hp in PHASE_B_GRID[best_algo[t]].items()} for t in TASK_LIST}
    run_jobs([c for t in phase_b.values() for cs in t.values() for c in cs], log, "階段B")
    best_cfg = {}
    for t in TASK_LIST:
        groups = {"default": summarize_group(phase_a[t][best_algo[t]])}
        groups.update({tag: summarize_group(cs) for tag, cs in phase_b[t].items()})
        winner, reason = select_best(groups)
        best_cfg[t] = {} if winner == "default" else PHASE_B_GRID[best_algo[t]][winner]
        save_selection(args.out_root, f"B_{t}", groups, winner, reason, log)

    # 正式版
    finals = [make_cfg(t, best_algo[t], 0, "final", best_cfg[t], budget[t] * FINAL_BUDGET_MULTIPLIER, args.final_root)
              for t in TASK_LIST]
    run_jobs(finals, log, "正式版")
    for c in finals:
        log(f"正式版模型:{c.task} = {c.algo} {c.hyperparams or '(預設超參數)'} → {c.run_dir}")

    subprocess.run([sys.executable, "collect_results.py", "--root", args.out_root])
    subprocess.run([sys.executable, "collect_results.py", "--root", args.final_root])
    log("===== 全部完成 =====")


if __name__ == "__main__":
    main()
