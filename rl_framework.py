"""
統一的訓練 + 評估框架:同一套程式碼跑「任務 × 演算法 × 超參數 × device × seed」的任意組合。

設計重點:
- 任務、演算法都用登錄表(TASKS / ALGOS),新增一個只要加一行
- 所有可調的東西都在 ExperimentConfig 裡,存成 config.json,任何一組實驗都能重現
- 評估流程對每一組設定完全相同(固定的評估 seed),比較才公平
- 每組實驗各自寫自己的資料夾,平行跑多組也不會互相覆蓋;彙整用 collect_results.py

輸出(results/experiments/<task>/<algo>/<tag>_seed<seed>/):
- config.json :完整設定
- model.zip   :訓練結束時的模型(不挑最佳 checkpoint,避免選擇偏差)
- curve.csv   :學習曲線(訓練中的成功率/回合長度/回合 reward + 定期 deterministic 評估)
- result.json :最終評估結果、學習效率指標、訓練時間
"""

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from gymnasium.wrappers import RescaleAction, TimeLimit
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from fist_task_v2 import OrcaFistTaskV2
from pinch_task_v2 import OrcaPinchTaskV2


# ---- 登錄表 ----
TASKS = {
    "fist": OrcaFistTaskV2,
    "pinch": OrcaPinchTaskV2,
}

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

ON_POLICY = {"ppo"}

# 各任務的訓練預算(依 E11 / E13 的 PPO 試驗決定,見 實驗計畫_階段1.md)
DEFAULT_TIMESTEPS = {"fist": 1_000_000, "pinch": 500_000}

# 定期評估頻率:捏合學得快(E15:SAC 幾千步就學會),間隔太大會分不出學習速度
DEFAULT_EVAL_FREQ = {"fist": 25_000, "pinch": 10_000}

# 評估用的 seed:定期評估從 1000 起、最終評估從 0 起(兩批不重疊),每一組設定都用同一批
PERIODIC_EVAL_SEED_START = 1000
FINAL_EVAL_SEED_START = 0


@dataclass
class ExperimentConfig:
    task: str
    algo: str
    seed: int = 0
    tag: str = "default"                  # 這組超參數的名字,例如 "lr1e-4"
    total_timesteps: int | None = None    # None → 用 DEFAULT_TIMESTEPS[task]
    hyperparams: dict = field(default_factory=dict)  # 直接傳給演算法建構子,例如 {"learning_rate": 1e-4}
    device: str = "cpu"                   # "cpu" 或 "cuda";E02/E12 實測 CPU 較快
    n_envs: int | None = None             # None → PPO 8 個、SAC/TD3 1 個
    torch_threads: int | None = None      # None → PyTorch 預設;平行跑多組時可設 4(E12)
    reset_noise: float = 0.05             # 初始姿勢雜訊(rad),見 E14
    max_episode_steps: int = 300
    eval_freq: int | None = None          # 每幾步做一次定期評估;None → DEFAULT_EVAL_FREQ[task]
    periodic_eval_episodes: int = 10
    final_eval_episodes: int = 20
    log_freq: int = 10_000                # 每幾步記一次訓練中統計
    out_root: str = "results/experiments"

    def resolved(self) -> "ExperimentConfig":
        """把 None 的欄位換成實際用到的預設值,並檢查參數。"""
        if self.task not in TASKS:
            raise ValueError(f"未知任務 {self.task!r},可用:{list(TASKS)}")
        if self.algo not in ALGOS:
            raise ValueError(f"未知演算法 {self.algo!r},可用:{list(ALGOS)}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device 只能是 cpu 或 cuda,收到 {self.device!r}")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定 device=cuda,但 torch.cuda.is_available() 是 False")
        cfg = ExperimentConfig(**asdict(self))
        if cfg.total_timesteps is None:
            cfg.total_timesteps = DEFAULT_TIMESTEPS[cfg.task]
        if cfg.eval_freq is None:
            cfg.eval_freq = DEFAULT_EVAL_FREQ[cfg.task]
        if cfg.n_envs is None:
            cfg.n_envs = 8 if cfg.algo in ON_POLICY else 1
        return cfg

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_root, self.task, self.algo, f"{self.tag}_seed{self.seed}")


# ---- 環境 ----
def make_env(task: str, reset_noise: float, max_episode_steps: int):
    """回傳一個建立環境的函式(SubprocVecEnv 需要)。動作一律正規化到 [-1, 1]。"""
    def _init():
        env = TASKS[task](reset_noise=reset_noise)
        ones = np.ones(env.action_space.shape, dtype=np.float32)
        env = RescaleAction(env, -ones, ones)
        return TimeLimit(env, max_episode_steps=max_episode_steps)
    return _init


def make_vec_env(cfg: ExperimentConfig):
    fns = [make_env(cfg.task, cfg.reset_noise, cfg.max_episode_steps) for _ in range(cfg.n_envs)]
    vec = SubprocVecEnv(fns) if cfg.n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(vec)


# ---- 評估 ----
def evaluate_policy(model, cfg: ExperimentConfig, seeds, deterministic: bool) -> dict:
    """
    用指定的 seed 逐回合評估。回傳成功率、平均成功步數(只算成功回合)、平均回合 reward。
    評估環境跟訓練環境設定相同(同樣的雜訊、回合上限、動作正規化)。
    """
    env = make_env(cfg.task, cfg.reset_noise, cfg.max_episode_steps)()
    successes, success_steps, returns = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        ep_return = 0.0
        for t in range(1, cfg.max_episode_steps + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        ok = bool(info.get("is_success", False))
        successes.append(ok)
        returns.append(ep_return)
        if ok:
            success_steps.append(t)
    env.close()
    return {
        "episodes": len(seeds),
        "success_rate": float(np.mean(successes)),
        "mean_steps_to_success": float(np.mean(success_steps)) if success_steps else None,
        "mean_return": float(np.mean(returns)),
    }


class CurveCallback(BaseCallback):
    """
    記錄學習曲線,兩種資料:
    - 每 log_freq 步:最近 100 個訓練回合的成功率、平均長度、平均 reward(訓練中的策略,含探索)
    - 每 eval_freq 步:用固定 seed 做 deterministic 評估(跟最終評估同一套流程)
    用「步數」而不是「rollout」當間隔,PPO 跟 SAC/TD3 的曲線才能放在同一個橫軸比較。
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.rows = []
        self._next_log = cfg.log_freq
        self._next_eval = cfg.eval_freq

    def _train_stats(self) -> dict:
        succ = self.model.ep_success_buffer
        infos = self.model.ep_info_buffer
        return {
            "train_success_rate": float(np.mean(succ)) if len(succ) else None,
            "train_ep_len": float(np.mean([e["l"] for e in infos])) if infos else None,
            "train_ep_return": float(np.mean([e["r"] for e in infos])) if infos else None,
        }

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_log or self.num_timesteps >= self._next_eval:
            row = {"timesteps": self.num_timesteps, "elapsed_s": round(time.perf_counter() - self._t0, 1)}
            row.update(self._train_stats())
            if self.num_timesteps >= self._next_eval:
                seeds = range(PERIODIC_EVAL_SEED_START, PERIODIC_EVAL_SEED_START + self.cfg.periodic_eval_episodes)
                ev = evaluate_policy(self.model, self.cfg, seeds, deterministic=True)
                row.update({"eval_success_rate": ev["success_rate"],
                            "eval_mean_steps_to_success": ev["mean_steps_to_success"],
                            "eval_mean_return": ev["mean_return"]})
                while self._next_eval <= self.num_timesteps:
                    self._next_eval += self.cfg.eval_freq
            while self._next_log <= self.num_timesteps:
                self._next_log += self.cfg.log_freq
            self.rows.append(row)
        return True

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()


def first_reaching(rows: list[dict], key: str, threshold: float):
    """學習效率:某個指標第一次 >= threshold 的步數;沒達到回傳 None。"""
    for r in rows:
        v = r.get(key)
        if v is not None and v >= threshold:
            return r["timesteps"]
    return None


def write_curve_csv(rows: list[dict], path: str) -> None:
    keys = ["timesteps", "elapsed_s", "train_success_rate", "train_ep_len", "train_ep_return",
            "eval_success_rate", "eval_mean_steps_to_success", "eval_mean_return"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


# ---- 主流程 ----
def run_experiment(cfg: ExperimentConfig, verbose: int = 0) -> dict:
    cfg = cfg.resolved()
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)

    hyperparams = dict(cfg.hyperparams)
    # 多個環境的 off-policy:每收 n_envs 筆資料更新 n_envs 次,維持「每筆資料更新 1 次」(UTD = 1,同 E12)
    if cfg.algo not in ON_POLICY and cfg.n_envs > 1:
        hyperparams.setdefault("gradient_steps", cfg.n_envs)

    env = make_vec_env(cfg)
    model = ALGOS[cfg.algo]("MlpPolicy", env, device=cfg.device, seed=cfg.seed, verbose=verbose, **hyperparams)
    callback = CurveCallback(cfg)

    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    train_seconds = time.perf_counter() - t0
    env.close()

    model.save(os.path.join(cfg.run_dir, "model"))  # 只存模型,不存 replay buffer(可達數百 MB)
    write_curve_csv(callback.rows, os.path.join(cfg.run_dir, "curve.csv"))

    final_seeds = range(FINAL_EVAL_SEED_START, FINAL_EVAL_SEED_START + cfg.final_eval_episodes)
    result = {
        "task": cfg.task, "algo": cfg.algo, "tag": cfg.tag, "seed": cfg.seed,
        "total_timesteps": model.num_timesteps,
        "train_seconds": round(train_seconds, 1),
        "final_eval_deterministic": evaluate_policy(model, cfg, final_seeds, deterministic=True),
        # 注意:TD3 的策略本身是確定性的,predict(deterministic=False) 不會加雜訊,所以這欄會等於上一欄
        "final_eval_stochastic": evaluate_policy(model, cfg, final_seeds, deterministic=False),
        "steps_to_eval_success_50": first_reaching(callback.rows, "eval_success_rate", 0.5),
        "steps_to_eval_success_90": first_reaching(callback.rows, "eval_success_rate", 0.9),
        "steps_to_train_success_50": first_reaching(callback.rows, "train_success_rate", 0.5),
        "steps_to_train_success_90": first_reaching(callback.rows, "train_success_rate", 0.9),
        "versions": {"torch": torch.__version__, "device": cfg.device,
                     "torch_threads": torch.get_num_threads(), "n_envs": cfg.n_envs},
    }
    with open(os.path.join(cfg.run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
