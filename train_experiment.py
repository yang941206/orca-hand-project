"""
統一訓練腳本:用命令列參數或 JSON 設定檔指定一組實驗,跑完自動存模型、評估結果、學習曲線。

用法範例:
    # 握拳 × PPO × 預設超參數 × seed 0(預算用任務預設值:握拳 100 萬步)
    python train_experiment.py --task fist --algo ppo --seed 0

    # 捏合 × SAC × 改學習率,用 GPU
    python train_experiment.py --task pinch --algo sac --tag lr1e-4 --hp learning_rate=1e-4 --device cuda

    # 改網路大小(值用 JSON 格式)
    python train_experiment.py --task fist --algo ppo --tag net256 --hp "policy_kwargs={\"net_arch\": [256, 256]}"

    # 從 JSON 設定檔讀(欄位同 rl_framework.ExperimentConfig;命令列有給的參數會覆蓋設定檔)
    python train_experiment.py --config configs/example.json

結果位置:results/experiments/<task>/<algo>/<tag>_seed<seed>/
"""

import argparse
import json

from rl_framework import ALGOS, TASKS, ExperimentConfig, run_experiment


def parse_hp(items: list[str]) -> dict:
    """把 ["learning_rate=1e-4", "policy_kwargs={...}"] 轉成 dict;值先試著用 JSON 解析。"""
    hp = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            hp[key] = json.loads(raw)
        except json.JSONDecodeError:
            hp[key] = raw
    return hp


def main():
    p = argparse.ArgumentParser(description="ORCA Hand 統一訓練 + 評估")
    p.add_argument("--config", help="JSON 設定檔(欄位同 ExperimentConfig)")
    p.add_argument("--task", choices=list(TASKS))
    p.add_argument("--algo", choices=list(ALGOS))
    p.add_argument("--seed", type=int)
    p.add_argument("--tag")
    p.add_argument("--timesteps", type=int, dest="total_timesteps")
    p.add_argument("--hp", nargs="*", default=[], help="超參數,格式 key=value")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--n-envs", type=int, dest="n_envs")
    p.add_argument("--torch-threads", type=int, dest="torch_threads")
    p.add_argument("--reset-noise", type=float, dest="reset_noise")
    p.add_argument("--eval-freq", type=int, dest="eval_freq")
    p.add_argument("--log-freq", type=int, dest="log_freq")
    p.add_argument("--out-root", dest="out_root")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    fields = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            fields.update(json.load(f))
    for key in ["task", "algo", "seed", "tag", "total_timesteps", "device", "n_envs",
                "torch_threads", "reset_noise", "eval_freq", "log_freq", "out_root"]:
        value = getattr(args, key)
        if value is not None:
            fields[key] = value
    if args.hp:
        fields["hyperparams"] = {**fields.get("hyperparams", {}), **parse_hp(args.hp)}
    if "task" not in fields or "algo" not in fields:
        p.error("必須指定 --task 與 --algo(或在設定檔裡給)")

    cfg = ExperimentConfig(**fields)
    print(f"開始:{cfg.task} × {cfg.algo} × {cfg.tag} × seed {cfg.seed}", flush=True)
    r = run_experiment(cfg, verbose=args.verbose)
    det, sto = r["final_eval_deterministic"], r["final_eval_stochastic"]
    print(f"完成:訓練 {r['train_seconds']:.0f}s | 最終評估 det 成功率 {det['success_rate'] * 100:.0f}% "
          f"(平均 {det['mean_steps_to_success']} 步) | 隨機 {sto['success_rate'] * 100:.0f}% | "
          f"eval ≥50% 於 {r['steps_to_eval_success_50']} 步、≥90% 於 {r['steps_to_eval_success_90']} 步", flush=True)
    print(f"結果:{cfg.resolved().run_dir}", flush=True)


if __name__ == "__main__":
    main()
