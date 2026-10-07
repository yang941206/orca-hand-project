# Goal-Conditioned 強化學習（GCRL）講義

> 依據：Eric Liu，〈Goal-conditioned 的強化學習調研〉，知乎專欄，2022（https://zhuanlan.zhihu.com/p/460539223）
> 對應論文：Minghuan Liu, Menghui Zhu, Weinan Zhang, *Goal-Conditioned Reinforcement Learning: Problems and Solutions*, IJCAI 2022 Survey Track（arXiv:2201.08299）
> 演算法與環境整理表：https://github.com/apexrl/GCRL-Collection
>
> 說明：標示「📌 補充」的段落是講義額外補充的背景知識，不屬於原文內容。

---

## 0. 學習目標

讀完本講義後，你應該能夠：

1. 說明 GCRL 與一般 RL 在問題設定上的差異，並寫出它的目標函數。
2. 區分 desired goal、achieved goal、behavior goal 三種 goal 概念。
3. 說出 GCRL 的兩大核心挑戰，並解釋為什麼「用距離當獎勵」不一定好。
4. 列舉 goal 的常見表示方式。
5. 用「選 goal → 採樣 → relabel → 優化」四步驟框架，把各類方法歸位。
6. 理解 HER（Hindsight Experience Replay）的核心想法並能寫出偽代碼。

---

## 1. 問題背景

GCRL 也稱為 goal-oriented RL 或 multi-goal RL，和機器人學淵源很深。機器人領域希望一個策略能對「一類任務」泛化：例如「把物體推到某個位置」，目標位置每次不同，但本質是同一類任務。把「不同的目標位置」建模成 **goal**，就得到 goal-conditioned 強化學習問題。

原文作者強調，這篇調研聚焦在 RL 學派的方法論，與具體機器人任務的連結較少。

---

## 2. 問題定義

### 2.1 核心想法

一般 RL 的策略是 $\pi(a \mid s)$；GCRL 的策略則同時看狀態與目標：

$$
\pi(a \mid s, g)
$$

實作上通常把 goal **拼接（augment）到 state** 上一起輸入網路。這讓策略需要具備兩種能力：

| 能力 | 意義 | 直觀圖像 |
|---|---|---|
| 多任務決策 | 一個策略就能完成多個不同的 goal | 從不同起點，到達不同終點 |
| 任務分解 | 把遙遠、難以達成的 goal 拆成容易達成的子目標（sub-goal） | 起點 → 子目標 → 子目標 → 終點 |

### 2.2 目標函數

相較一般 RL，GCRL 的目標函數多了一層對 goal 分布的期望：

$$
J(\pi) = \mathbb{E}_{\substack{a_t \sim \pi(\cdot \mid s_t, g),\; g \sim p_g \\ s_{t+1} \sim \mathcal{T}(\cdot \mid s_t, a_t)}}
\left[ \sum_t \gamma^t \, r(s_t, a_t, g) \right]
$$

- $p_g$：環境給定的目標分布
- $\mathcal{T}$：狀態轉移函數
- $r(s_t, a_t, g)$：與目標相關的獎勵

### 2.3 狀態到目標的映射

通常存在一個映射 $\phi: \mathcal{S} \rightarrow \mathcal{G}$，把狀態對應到目標空間。常見形式：

- **恆等映射**：$\phi(s) = s$，目標就是一個完整狀態
- **取部分維度**：例如只取物體座標 $(x, y, z)$ 當目標

因此才會有「goal state」的說法：某個狀態經 $\phi$ 映射後若等於 $g$，就是到達了目標。

### 2.4 三種 Goal 概念（重要）

| 名稱 | 定義 | 備註 |
|---|---|---|
| **Desired goal**（期望目標） | 需要解決的任務，可以由環境提供，也可以由智能體自己內在產生 | 對應環境的任務分布 |
| **Achieved goal**（已達成目標） | 智能體在當前狀態實際達成的 goal，即 $\phi(s_t)$ | 就是「目前所在的 goal state」 |
| **Behavior goal**（行為目標） | rollout 採樣時，策略實際拿來參考決策的 goal | 測試時等於 desired goal；**訓練時可由演算法改寫**以提升效率 |

> 💡 記憶要點：behavior goal 是演算法的「操作空間」。許多 GCRL 方法的創新，其實就是在設計「訓練時要讓智能體去追哪個 goal」。

---

## 3. 兩大核心挑戰

### 挑戰 1：策略的多任務泛化性

學到的策略要能遷移到不同的 goal $g$ 上，包括訓練時沒見過的 goal。

### 挑戰 2：稀疏獎勵下的探索

GCRL 的獎勵通常是已知的「完成任務才給分」：

$$
r_g(s_t, a_t, g) = \mathbb{1}(\text{the goal is reached})
$$

這種獎勵**極度稀疏**：大部分軌跡拿到的獎勵全是 0，策略難以學習。

**為什麼不直接用距離當獎勵？**
直覺做法是 reward shaping：$r = -\lVert \phi(s) - g \rVert$。但原文指出這可能帶來**新的局部最優**。例如要到達目標，有時必須先「繞遠路」（距離先變大、再變小），像迷宮中隔著一道牆的目標。距離型獎勵會懲罰繞路，讓智能體卡在牆邊。

> 📌 補充：實務上常寫成 $r = \mathbb{1}(\lVert \phi(s_{t+1}) - g \rVert \le \epsilon)$，或 HER 原論文採用的 $r \in \{-1, 0\}$ 形式（未達成 −1，達成 0）。

---

## 4. Goal 的表示方式

原文歸納了三種典型表示：

| 類型 | 說明 | 例子 |
|---|---|---|
| **特徵向量** | 刻畫某個狀態的向量 | 目標位置 $(x_G, y_G)$、目標速度 |
| **圖片** | 任務完成時的畫面 | 物塊被推到指定位置的影像、門打開後的畫面 |
| **自然語言** | 人類看得懂的指令 | 「把紅色磚塊移到藍色圓球上」 |

### 其他可視為 goal 的設定

有些工作並非解決一般性的 GCRL 問題，但可以視為**特殊類型的 goal**：

- **以獎勵為 goal**：要求智能體在剩餘的 rollout 中取得指定的累積獎勵。
- **Decision Transformer**（offline RL）：以 return-to-go 作為條件，概念相近。
- **Upside-Down RL**（Schmidhuber 團隊）：以「期望獎勵 $R$ + 期望步數 $H$」作為指令，要求智能體在 $H$ 步內拿到 $R$ 分。

---

## 5. 解決方案總覽：四步驟框架

原文先把 GCRL 智能體的學習過程拆成四步，再把所有方法對應到各步驟：

```
① 選擇 behavior goal ──► ② 與環境互動採樣 ──► ③ Relabel ──► ④ 優化與學習
   (Goal Selection)        (Sample)            (改寫 buffer 中的 goal)   (Value / Policy)
        ▲                                                                    │
        └────────────────────────────────────────────────────────────────────┘
```

1. **選擇 behavior goal**：通常是子任務（sub-goal），可由環境提供，或由智能體自行生成、挑選。
2. **環境互動**：用選定的 behavior goal 收集經驗。
3. **Relabel**：學習之前，重新標註 replay buffer 中的 goal。
4. **優化**：更新價值函數與策略。

下面依序說明三大類方法：**優化（④）**、**子目標選擇（①）**、**Relabeling（③）**。

---

## 6. 優化類方法（Optimization）

| 角度 | 核心想法 |
|---|---|
| **Universal Value Function** | 把價值函數擴展到 goal 上：$V(s, g)$、$Q(s, a, g)$，讓標準 RL 演算法可以直接優化 |
| **Reward Shaping** | 修改獎勵函數來緩解探索困難（但要小心第 3 節提到的局部最優） |
| **Self-Imitation Learning** | 從已採樣的資料中挑出好的軌跡來模仿 |
| **Planning & Model-based RL** | 借助環境模型減少真實互動、提高樣本效率 |

> 📌 補充：Universal Value Function Approximator（UVFA，Schaul et al., 2015）是此方向的基礎。對應的 goal-conditioned Bellman 方程為
> $$Q(s, a, g) = r(s, a, g) + \gamma \, \mathbb{E}_{s'}\left[ \max_{a'} Q(s', a', g) \right]$$
> 只要把 $g$ 當成輸入的一部分，DQN、DDPG、SAC 等演算法幾乎不用改就能套用。
>
> 📌 補充：Self-imitation 方向的代表包括 GCSL（Ghosh et al., 2021）：把「任意軌跡在事後看都是到達自己終點的示範」，直接做監督式模仿學習。

---

## 7. 子目標選擇類方法（Sub-goal Selection）

把環境原本的任務換成一系列子任務來加速學習。這時 goal 不再取自環境分布 $p_g$，而是取自演算法設計的分布 $f$：

$$
J(\pi) = \mathbb{E}_{\substack{a_t \sim \pi(\cdot \mid s_t, g),\; g \sim f \\ s_{t+1} \sim \mathcal{T}(\cdot \mid s_t, a_t)}}
\left[ \sum_t \gamma^t \, r(s_t, a_t, g) \right]
$$

原文歸納的選擇原則：

| 原則 | 說明 |
|---|---|
| **Intermediate Difficulty** | 評估 goal 對當前策略的難度，在不同階段挑選「不太難也不太簡單」的 goal（課程學習的概念） |
| **Exploration Awareness** | 選擇能增強探索的 sub-goal |
| **Searching from Experience** | 從過去的經驗中挑選合適的子任務 |
| **Model-Based Planning** | 用環境模型做規劃，選出能拆解任務的子任務 |
| **Learn from Experts** | 用專家資料中的路徑點（waypoints）拆解任務 |

> 📌 補充代表作：Goal GAN（Florensa et al., 2018）用 GAN 生成中等難度的 goal；MEGA（Pitis et al., 2020）挑選已達成目標中密度低的區域，推動探索邊界；Skew-Fit（Pong et al., 2020）讓目標分布朝向更均勻的狀態覆蓋。

---

## 8. Relabeling 類方法

透過**替換經驗池中的 goal** 來提高資料利用率。設 $h$ 為 relabel 函數，它把一筆轉移 $(s_t, a_t, r_t, s_{t+1}, g)$ 中的 $g$ 換成 $g' = h(\cdot)$，並重新計算獎勵 $r'_t = r(s_t, a_t, g')$。

| 類型 | 說明 |
|---|---|
| **Hindsight（事後）** | 經典的 HER 及其延伸：從失敗的經驗學習，把失敗軌跡「在某種意義上當作成功」 |
| **Foresight（預見）** | 借助模型，依據當前策略推演，得知策略已經能完成哪些新目標，並用來學習 |
| **Relabeling by Learning** | 從過去經驗中**學習**一個好的 relabel 函數，而不是用固定規則 |

### 8.1 HER 的核心直覺

> 想射中紅心卻射到了左上角。這一箭對「射紅心」是失敗，但對「射左上角」是完美的示範。

把軌跡中**實際達成的狀態**當成 goal 重新標註，稀疏獎勵的失敗軌跡就變成了有正獎勵的成功資料。

### 8.2 HER 偽代碼（📌 補充）

```text
for episode = 1..M:
    取樣 desired goal g，重置環境得到 s_0
    for t = 0..T-1:
        a_t ← π(s_t, g) + 探索噪聲
        執行 a_t，得到 s_{t+1}
    for t = 0..T-1:
        存入 (s_t, a_t, r(s_t, a_t, g), s_{t+1}, g)            # 原始 goal
        取樣額外的 goals G' ← 策略 S（例如 future）
        for g' in G':
            存入 (s_t, a_t, r(s_t, a_t, g'), s_{t+1}, g')      # relabel 後的 goal
    從 buffer 取樣並用 off-policy 演算法（如 DDPG）更新
```

**常見的 relabel 策略 $S$：**

- `final`：用該軌跡最後一個狀態的 achieved goal
- `future`：用同一軌跡中**時間 $t$ 之後**某個隨機狀態的 achieved goal（最常用、效果通常最好）
- `episode`：用同一軌跡中任意狀態
- `random`：用整個 buffer 中任意狀態

> ⚠️ 注意：relabel 會改變資料的 goal 分布，因此 HER 需搭配 **off-policy** 演算法使用。

---

## 9. 未來展望

| 方向 | 說明 |
|---|---|
| **Learning Totally Intrinsic Skills** | 環境不提供 desired goal，智能體自己學出不同技能：不同的 $g$ 對應不同的行為策略。代表作 DIAYN（*Diversity Is All You Need*，Eysenbach et al., 2018）※原文誤寫為 DIYAN |
| **Learn from Offline Datasets** | 從離線資料直接學一個多任務策略，不需要與環境互動 |

文末留言區的延伸討論：GCRL 被作者歸類在 multi-task / skill learning 之下；與分層強化學習（HRL，含 option 框架）關係密切；在多智能體與自動駕駛上的應用也有讀者關注。

---

## 10. 方法分類總覽（一頁複習）

```
GCRL 解法
├── ④ 優化（Optimization）
│   ├── Universal Value Function
│   ├── Reward Shaping
│   ├── Self-Imitation Learning
│   └── Planning & Model-based RL
├── ① 子目標選擇（Sub-goal Selection）
│   ├── Intermediate Difficulty
│   ├── Exploration Awareness
│   ├── Searching from Experience
│   ├── Model-Based Planning
│   └── Learn from Experts
└── ③ Relabeling
    ├── Hindsight（HER 系列）
    ├── Foresight
    └── Relabeling by Learning
```

---

## 11. 名詞對照表

| 英文 | 中文 | 一句話解釋 |
|---|---|---|
| Goal-conditioned policy | 目標條件策略 | $\pi(a \mid s, g)$ |
| Desired goal | 期望目標 | 要解決的任務 |
| Achieved goal | 已達成目標 | $\phi(s_t)$ |
| Behavior goal | 行為目標 | 採樣時策略追的目標 |
| Sparse reward | 稀疏獎勵 | 只有完成任務才有獎勵 |
| Reward shaping | 獎勵塑形 | 修改獎勵以引導學習 |
| Relabel | 重新標註 | 換掉 buffer 中的 goal |
| UVFA | 通用價值函數近似 | $V(s, g)$ |
| HER | 事後經驗回放 | 把失敗當成另一個目標的成功 |
| Curriculum | 課程學習 | 由易到難安排目標 |

---

## 12. 自我檢測題

**Q1.** GCRL 的目標函數比一般 RL 多了什麼？
<details><summary>答案</summary>多了對 goal 分布的期望 $g \sim p_g$，且策略與獎勵都以 $g$ 為條件。</details>

**Q2.** 測試時 behavior goal 等於什麼？訓練時為什麼要改它？
<details><summary>答案</summary>測試時等於 desired goal。訓練時改寫它（例如挑中等難度或利於探索的子目標）可以提升學習效率。</details>

**Q3.** 舉一個「用到目標的距離當獎勵」會失敗的情境。
<details><summary>答案</summary>目標在牆後的迷宮：必須先遠離目標繞過牆才能到達，距離型獎勵會讓智能體停在牆邊的局部最優。</details>

**Q4.** HER 屬於四步驟中的哪一步？為什麼必須用 off-policy 演算法？
<details><summary>答案</summary>屬於 ③ Relabel。relabel 後的資料不是在該 goal 下由當前策略產生的，資料分布與行為策略不一致，因此需要 off-policy 演算法。</details>

**Q5.** Decision Transformer 與 GCRL 有什麼關聯？
<details><summary>答案</summary>它以 return-to-go 為條件生成動作，可視為「以獎勵作為 goal」的特殊 GCRL 設定。</details>

**Q6.** HER 的 `future` 策略與 `final` 策略差在哪？
<details><summary>答案</summary>`final` 只用軌跡最後一個狀態當新目標；`future` 從當前時間步之後的狀態中隨機挑選，產生更多樣、與當前狀態更相關的目標。</details>

---

## 13. 延伸閱讀（📌 補充）

1. Liu, Zhu, Zhang. *Goal-Conditioned Reinforcement Learning: Problems and Solutions.* IJCAI 2022.（本講義主要來源）
2. Schaul et al. *Universal Value Function Approximators.* ICML 2015.
3. Andrychowicz et al. *Hindsight Experience Replay.* NeurIPS 2017.
4. Florensa et al. *Automatic Goal Generation for Reinforcement Learning Agents（Goal GAN）.* ICML 2018.
5. Eysenbach et al. *Diversity Is All You Need（DIAYN）.* ICLR 2019.
6. Ghosh et al. *Learning to Reach Goals via Iterated Supervised Learning（GCSL）.* ICLR 2021.
7. Chen et al. *Decision Transformer.* NeurIPS 2021.
8. 演算法與環境整理：https://github.com/apexrl/GCRL-Collection
