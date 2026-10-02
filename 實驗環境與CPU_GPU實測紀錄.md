# 實驗環境與 CPU vs GPU 實測紀錄(2026-10-02)

## 實驗環境

| 項目 | 內容 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core Ultra 7 265K(20 核 / 20 執行緒) |
| RAM | 31.3 GB |
| GPU | NVIDIA GeForce RTX 5050(Blackwell,compute capability 12.0) |
| VRAM | 8 GB(torch 回報 7.96 GiB) |
| NVIDIA 驅動 | 610.88(驅動支援 CUDA 13.3) |
| PyTorch | 2.14.0+cu130(CUDA 13.0、cuDNN 9.24.0) |
| Python / 套件 | Python 3.11.16、stable-baselines3 2.9.0、gymnasium 1.3.0、mujoco 3.13.0、orca_sim 0.1.0、numpy 2.4.6、tensorboard 2.21.0、pandas 3.0.6、matplotlib 3.11.2(sb3-contrib 未安裝) |

PyTorch 安裝說明:原本裝的是 CPU-only 版(`2.14.0+cpu`),`torch.cuda.is_available()` 回傳 False。
依官方 wheel index / RELEASE.md,torch 2.14 提供 CUDA 12.6 / 13.0 / 13.2 三種版本;
RTX 50 系列(sm_120)需要 CUDA 12.8 以上,所以 cu126 不能用,改裝 cu130:

```
pip install --force-reinstall --no-deps torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

## CPU vs GPU 實測(`bench_device.py`)

設定:捏合任務、PPO `MlpPolicy`(SB3 預設 64×64 網路、n_steps=2048、batch_size=64、n_epochs=10)、
`total_timesteps=20_000`、seed=0,各設定依序執行(不同時跑)。
註:SB3 會把步數補滿到 `n_steps × n_envs` 的整數倍,所以 8 個環境實際跑了 32,768 步,1 個環境跑了 20,480 步。

| device | 環境數 | 實際步數 | 總時間 (s) | learn() 時間 (s) | 吞吐量 (steps/s) |
|---|---|---|---|---|---|
| cpu  | 8 (SubprocVecEnv) | 32,768 | 17.14 / 16.26 | 11.60 / 11.26 | ~2,870 |
| cuda | 8 (SubprocVecEnv) | 32,768 | 38.64 / 38.95 | 33.53 / 33.74 | ~975 |
| cpu  | 1 (DummyVecEnv)   | 20,480 | 17.54 | 16.00 | 1,280 |
| cuda | 1 (DummyVecEnv)   | 20,480 | 43.93 | 41.57 | 493 |

(8 環境兩組各跑兩次,前後兩次差距 < 4%。)

**結論:GPU 比 CPU 慢約 2.6–2.9 倍。正式實驗用 CPU + SubprocVecEnv。**

## 原因分析

1. **物理模擬只在 CPU 上跑**:MuJoCo 的 step 全部在 CPU,GPU 完全幫不上忙。
2. **網路太小**:策略/價值網路只有兩層 64 個神經元,一次前向傳遞的計算量極小,
   GPU 的平行運算能力用不到;反而每次呼叫 CUDA kernel 的啟動開銷(launch overhead)比計算本身還久。
3. **每一步都要在 CPU ↔ GPU 間搬資料**:蒐集經驗時每一步都要把觀測值從 CPU 搬到 GPU 做推論,
   再把動作搬回 CPU 給 MuJoCo。8 個環境的批次也只有 8 筆,傳輸延遲遠大於計算時間。
4. **更新階段的 minibatch 很小**(64 筆),同樣是小批次 + 多次 kernel 呼叫,GPU 吃不飽。
5. 8 環境時 CPU 版吞吐量是 1 環境的 2.2 倍,代表瓶頸是「環境模擬」,
   加速應該靠多開平行環境(SubprocVecEnv),不是靠 GPU。
   SB3 本身在 GPU 跑 MlpPolicy 時也會跳出警告,建議改用 CPU。

GPU 什麼時候才有用:用影像當觀測值(CNN policy)、網路很大、或物理模擬本身也放在 GPU
(例如 MuJoCo MJX / Isaac Gym 這種上千個環境一起在 GPU 跑)的情況。
