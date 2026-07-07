# Contact-VJP-Repair：Genesis Forward + 接触帧学习式 VJP

> 定位：本研究是 GradWM 框架下的一条 **task-specialized 独立线**。共享 GradWM 的 anchor / perturbation / state-restore 基础设施，但主 claim 收窄到 contact-rich manipulation 的 object-loss-to-action 梯度修补，不学 full transition response。

---

## 1. 背景与动机

在 contact-rich manipulation 中（推箱子、抓绳子、插销），任务 reward 通常挂在 **object 状态** 上（位姿、形变、抓取稳定性），但可优化的参数只作用在 **robot 侧动作** 上。二者之间必须经过接触：

$$
\underbrace{\pi_\phi}_{\text{policy}}
\longrightarrow
\underbrace{a_t}_{\text{robot action}}
\longrightarrow
\underbrace{\text{contact}}_{\text{Genesis 求解}}
\longrightarrow
\underbrace{s_{\text{obj},t+1}}_{\text{object state}}
\longrightarrow
\underbrace{L}_{\text{task loss}}
$$

Genesis 的正向模拟在 object 状态上准确、稳定、快，可以视为高精度 forward oracle：

$$s_{t+1} = E(s_t, a_t)$$

但当我们希望使用 analytic policy gradient / BPTT 优化策略时，需要：

$$
\frac{\partial L}{\partial a_t}
=
\left(\frac{\partial s_{\text{obj},t+1}}{\partial a_t}\right)^\top
\lambda^{\text{obj}}_{t+1},
\qquad
\lambda^{\text{obj}}_{t+1} = \frac{\partial L}{\partial s_{\text{obj},t+1}}
$$

这条 backward path 必须穿过接触求解器。已有实验（详见 `notes/GENESIS_CONTACT_GRADIENT_REPORT.md`）显示：Genesis 1.1.1 标准 rigid backward pipeline 在接触帧上给出的 `∂s_obj / ∂a_robot` 与真实有限差分响应严重偏离（two-box 探针：FD slope ≠ 0，但 analytic pusher grad ≡ 0）。**Genesis 正向可用，接触帧反向不可信。**

同时，我们不希望：

- **学一个完整可微仿真器**（如传统 world model $\hat f_\theta(s,a) \to s'$）：多步 rollout 会漂，接触状态预测错，且和 OrbiSim / GNS 等已有工作重叠。
- **学一个全 transition 的局部响应模型**（GradWM 主线）：覆盖 all steps、all state dims，工程量大，且监督信号维度过高、样本效率低。

本研究希望：

> **保留 Genesis 作为高精度 forward oracle 和数据采样引擎；只在接触帧上，直接学习一个 object-loss-conditioned 的 learned VJP 模块，作为策略优化时的 backward 替代。**

---

## 2. 核心问题

优化目标：

$$
J(\phi)
=
\mathbb{E}_{\tau \sim \pi_\phi}
\left[
\sum_{t=0}^{T-1} L(s_t, a_t)
\right],
\qquad
L\ \text{主要依赖}\ s_\text{obj}
$$

标准 BPTT 递推：

$$
\lambda_t
=
\left(\frac{\partial s_t}{\partial s_{t-1}}\right)^\top \lambda_{t+1}
+
\frac{\partial L}{\partial s_t},
\qquad
\frac{\partial J}{\partial a_t}
=
\left(\frac{\partial s_{t+1}}{\partial a_t}\right)^\top \lambda_{t+1}
$$

其中：

- 非接触帧 $t \notin \mathcal{W}$：Genesis autograd 可信，直接使用。
- 接触帧 $t \in \mathcal{W}$：`∂s_obj / ∂a_robot` 不可靠，是本方案要修补的对象。

核心问题：

> 如何在保持 forward rollout 完全走 Genesis 的前提下，只在接触帧上构造一个可用于策略优化的 object-centric VJP surrogate？

---

## 3. 不推荐的直接方案

**方案 A：直接学 next-state**

$$\hat s_{t+1} = f_\theta(s_t, a_t)$$

在 forward 上替换 Genesis，退化为传统 world model，正向精度掉档，多步 rollout 漂移，跟 OrbiSim / GNS 撞车，放弃。

**方案 B：学 full-transition 局部响应**（GradWM 主线）

$$m_\theta(z, \Delta s, \Delta a) \to \Delta s_{t+1} \in \mathbb{R}^{\dim(s)}$$

覆盖所有 state 分量、所有 step。工程量大、目标维度高、且未利用 "loss 只挂在 object 上" 这个 manipulation 先验。作为主 claim 过宽；作为共享基础设施合理。

**方案 C：学 full Jacobian**

$$J_\psi(z_t) \in \mathbb{R}^{\dim(s) \times \dim(a)}$$

在 rope / soft body 场景下 $\dim(s)$ 可到几千，Jacobian 矩阵爆炸；且 backward 每步都要一次矩阵乘，浪费。放弃"完整 Jacobian"，转向 "**object 分量的、conditioned on cotangent 的 VJP**"。

**方案 D：拟合绝对 VJP 值（MSE fit）**

$$\mathcal{L} = \|G_\theta(z, \lambda) - y_{\text{VJP}}\|^2$$

绝对值 fit 对方向偏差不敏感；接触帧 FD 噪声大时，网络容易在方向上偏移。改用方向一致性 loss（见 §7）。

---

## 4. 总体方案

本方案在 forward / backward 上做严格解耦：

$$
\boxed{
\begin{array}{l}
\text{Forward:}\quad s_{t+1} = E(s_t, a_t) \quad \text{(Genesis, 全程)} \\[4pt]
\text{Backward:}\quad
\dfrac{\partial J}{\partial a_t}
=
\begin{cases}
\text{Genesis autograd}, & t \notin \mathcal{W} \\
G_\theta(z_t,\ \lambda^{\text{obj}}_{t+1}), & t \in \mathcal{W}
\end{cases}
\end{array}
}
$$

其中 $\mathcal{W}$ 是**接触窗口**，用 forward-time 的 contact indicator 判定；$G_\theta$ 是**object-loss-conditioned learned VJP 模块**，直接把上游 object-side cotangent 映射到动作梯度。

**两个核心创新点：**

1. **Contact-gated learned VJP**：不学 next-state，不学 full response，只在接触帧学 `λ_obj ↦ g_a` 这一个 linear operator。
2. **Object-centric supervision**：训练标签来自 restored Genesis anchor 上的**方向导数有限差分**，只在 object 分量上计算，不监督全 state。

---

## 5. Contact-Gated Learned VJP 模块

### 5.1 定义

在接触帧的 anchor $(\bar s_t, \bar a_t, \bar s_{t+1}, c_t)$（其中 $c_t$ 是可选的 contact features：contact points, normals, penetration, impulses）上，定义：

**Context encoder**（借用 GradWM §5 的 regime encoder framing）：

$$
z_t = e_\theta(\bar s_t, \bar a_t, \bar s_{t+1}, c_t)
$$

$e_\theta$ 的角色是从后验上下文中判断当前局部接触 regime（未接触 / 刚接触 / 滑动 / stick / 碰撞反弹）。

**VJP head**（本方案核心）：

$$
G_\theta: (z_t, \lambda^{\text{obj}}_{t+1}) \longmapsto g_{a,t} \in \mathbb{R}^{\dim(a)}
$$

物理含义：

$$
G_\theta(z_t, \lambda^{\text{obj}}_{t+1})
\approx
\left(\frac{\partial s_{\text{obj},t+1}}{\partial a_t}\right)^\top
\lambda^{\text{obj}}_{t+1}
$$

### 5.2 架构：矩阵中介 + 线性收缩

不直接从 $(z, \lambda)$ MLP 到 $g_a$，而是引入中间矩阵：

$$
A_\theta(z_t) \in \mathbb{R}^{\dim(s_\text{obj}) \times \dim(a)},
\qquad
G_\theta(z_t, \lambda) = A_\theta(z_t)^\top \lambda
$$

**收益**：

1. 对 $\lambda$ 严格线性：$G_\theta(z, \alpha\lambda + \beta\lambda') = \alpha G_\theta(z, \lambda) + \beta G_\theta(z, \lambda')$，物理正确。
2. 一次预测覆盖任意 $\lambda$，训练时不必显式采样每个 $\lambda$ 方向。
3. $A_\theta$ 可挂物理先验：结构化稀疏（只允许 object-robot contact pair 之间非零）、法向-切向分解、低秩等。
4. Zero-cotangent 硬约束自动满足：$G_\theta(z, 0) = A_\theta(z)^\top \cdot 0 = 0$。

**规模评估**：

- box push：$\dim(s_\text{obj}) \approx 13$（位姿 + 速度），$\dim(a) = 9$，$A_\theta$ 输出 117 个数
- rope grasp：$\dim(s_\text{obj})$ 用 5 个 keypoints × 3 维 = 15，$\dim(a) = 9$，$A_\theta$ 输出 135 个数

MLP 输出维度百量级，前向 <0.1ms，相对 Genesis 一步 250ms 可忽略。

### 5.3 Zero-Cotangent 硬约束

VJP 是一个线性算子，$\lambda = 0$ 时必须 $g_a = 0$。矩阵中介结构 $G_\theta(z, \lambda) = A_\theta(z)^\top \lambda$ 天然满足此约束，无需 baseline subtraction。

**注意**：这个约束不是 GradWM §6 的 $m_\theta(z, 0, 0) = 0$（那是 response head 对 $\Delta s, \Delta a$ 的零约束）。VJP 的零约束在 $\lambda$ 上，不在扰动量上——VJP head 根本不吃扰动量。

### 5.4 接触窗口判定与 Soft Gate

接触窗口 $\mathcal{W}$ 定义为 Genesis forward 报告的 contact indicator：

$$
\mathcal{W} = \{t : \|f_{\text{normal},t}\|_\infty > \tau_f\ \lor\ n_{\text{contact},t} > 0\}
$$

阈值 $\tau_f$ 从数据分位数标定。**这是 forward-time 观测量，不需要 backward audit。**

接触/非接触边界处使用 **soft gate** 而非硬切：

$$
\alpha_t = \sigma\left(w_c \cdot \text{contact\_score}_t + w_d \cdot \text{FD\_consistency}_t + w_o \cdot \text{OOD\_score}_t\right) \in [0,1]
$$

**Backward 混合**：

$$
\frac{\partial J}{\partial a_t}
=
\alpha_t \cdot G_\theta(z_t, \lambda^{\text{obj}}_{t+1})
+
(1 - \alpha_t) \cdot
\left(\frac{\partial s_{t+1}}{\partial a_t}\right)_{\text{Genesis autograd}}^\top \lambda_{t+1}
$$

**Gate 输入来源**：

- `contact_score`：接触强度（normal impulse / penetration）
- `FD_consistency`：在线 FD 探针与 $G_\theta$ 输出的方向余弦（学长 §8.4 的 online probe 简化版）
- `OOD_score`：当前 anchor 与训练分布的距离（ensemble variance 或 kNN 距离）

Soft gate 是**接缝连续性的正确处理**，而不是"零输出硬约束"。低置信度接触帧退回 Genesis 原梯度，是不可靠 learned gradient 的保险丝。

---

## 6. 数据采样管线

### 6.1 Anchor 采样

用当前 policy $\pi_\phi$（冷启动时用随机 / scripted policy）在 Genesis 里跑 rollout，落每一步：

$$
\text{anchor}_t = (\bar s_t, \bar a_t, \bar s_{t+1}, c_t)
$$

用 contact indicator 过滤到接触窗口 $\mathcal{W}$，只保留 $t \in \mathcal{W}$ 的 anchor。

### 6.2 局部扰动采样

对每个 contact anchor：

1. 用 Genesis state restore 恢复到 $\bar s_t$（详见 `notes/LOCAL_ANCHOR_STATE_RESTORE_SURVEY_2026-07-04.md`，two-box 已验证同/跨进程可行）
2. 采 $K$ 个动作扰动方向 $v_k \in \mathbb{R}^{\dim(a)}$（unit-norm，随机或结构化）
3. 采多尺度 $\varepsilon_k \in \{\varepsilon_1, \varepsilon_2, \dots\}$（借用 GradWM §8.2 多尺度理由：硬接触下单一尺度会主要抓求解器数值噪声）
4. 查询仿真器：

$$
s^{+,k}_{\text{obj},t+1} = s_\text{obj}\left(E(\bar s_t, \bar a_t + \varepsilon_k v_k)\right)
$$

$$
s^{-,k}_{\text{obj},t+1} = s_\text{obj}\left(E(\bar s_t, \bar a_t - \varepsilon_k v_k)\right)
$$

5. 中心差分方向导数：

$$
d_k = \frac{s^{+,k}_{\text{obj},t+1} - s^{-,k}_{\text{obj},t+1}}{2\varepsilon_k}
\in \mathbb{R}^{\dim(s_\text{obj})}
$$

**关键**：$d_k$ 是 object 分量的方向导数，不是 full state。这就是 "object-centric" 的具体来源。

### 6.3 状态扰动的物理合法性

Genesis 状态含 quaternion / joint limits / rope segment 约束。对 $\bar s$ 加高斯噪声会产生非法状态。本方案第一版**只扰动 action，不扰动 state**（$\lambda$ 已提供 state-side 的信息），避开此问题。若后续需要 state 扰动，需引入 retraction map $\Pi: X \times \mathbb{R}^n \to X$，参考 manifold optimization 语言（Absil et al.），并做 rejection filter 过滤仿真失败样本。

---

## 7. 训练目标

### 7.1 方向一致性 loss（主）

借用 GradWM §8.3 VJP head 的训练形式，但特化到 object：

$$
\mathcal{L}_{\text{dir}}
=
\mathbb{E}_{t \in \mathcal{W}}\
\mathbb{E}_{(v_k, \varepsilon_k) \sim P_{\text{probe}}}\
\mathbb{E}_{\lambda \sim P_\lambda}
\left\|
v_k^\top\, G_\theta(z_t, \lambda)
-
\lambda^\top d_k
\right\|^2
$$

物理含义：**$G_\theta$ 输出 $g_a$ 在任意方向 $v$ 上的投影，应等于 $\lambda$ 与仿真器 forward 在 $v$ 方向上引起的 object 变化的内积。**

由矩阵中介结构 $G_\theta(z,\lambda) = A_\theta(z)^\top \lambda$，此 loss 等价于对 $A_\theta$ 做多方向 LS 拟合仿真器方向导数：

$$
\mathcal{L}_{\text{dir}} = \mathbb{E}_{t, k}\ \left\| A_\theta(z_t)^\top v_k - d_k \right\|^2 \cdot \|\lambda\|^2 \quad \text{(schematic)}
$$

### 7.2 $\lambda$ 采样策略

训练时 $\lambda \sim P_\lambda$ 需覆盖策略优化时可能出现的 cotangent 分布。可选：

- **随机 unit-norm**：泛化基线
- **任务 loss 梯度方向**：$\lambda = \partial L / \partial s_\text{obj}$ 在 buffered rollout 上采样，贴近真实使用分布
- **混合**：$P_\lambda = w \cdot P_{\text{random}} + (1-w) \cdot P_{\text{task}}$

矩阵中介结构下，$\lambda$ 采样只影响 loss 权重，不影响 $A_\theta$ 的学习目标。理论上单一 $\lambda$ 分布已足以学 $A_\theta$；随机 $\lambda$ 起 regularization 作用。

### 7.3 多尺度一致性 loss（辅）

$$
\mathcal{L}_{\text{multi}}
=
\mathbb{E}_{v}\
\left\|
\frac{d(\varepsilon_1, v) - d(\varepsilon_2, v)}{\|\varepsilon_1 - \varepsilon_2\|}
\right\|^2
$$

惩罚不同 $\varepsilon$ 下方向导数的漂移（若接触未跨 mode，应几乎相同；若跨 mode，此值高，可作为 gate 输入信号）。

### 7.4 可选：Response head 辅助监督

若 VJP head 训练收敛慢或不稳，加辅助 head：

$$
R_\theta(z_t, \Delta a) \to \widehat{\Delta s_\text{obj}}
$$

$$
\mathcal{L}_{\text{resp}}
=
\mathbb{E}\ \left\| R_\theta(z_t, \varepsilon_k v_k) - (s^{+,k}_\text{obj} - \bar s_\text{obj,t+1}) \right\|^2
$$

**用途仅限 shape encoder**：$R_\theta$ 和 $G_\theta$ 共享 $e_\theta$。$R_\theta$ 不参与 forward、不参与 backward，只是训练时多一个 loss 项。若 encoder 足够简单可省。

### 7.5 总目标

$$
\mathcal{L}
=
\mathcal{L}_{\text{dir}}
+
\gamma \mathcal{L}_{\text{multi}}
+
\beta \mathcal{L}_{\text{resp}}
$$

第一版 $\beta = 0$（不用 response head）；$\gamma$ 小值起始（0.1）。

---

## 8. 策略 / 轨迹优化管线

### 8.1 最小可行：显式 action update（推荐第一版）

不接入 custom autograd 图，直接测试 VJP 输出可用性：

$$
a_{t}^{\text{new}} = a_t - \eta\, G_\theta(z_t, \lambda^{\text{obj}}_{t+1})
$$

用 Genesis 重新 rollout 验证真实 task loss 是否下降。**这一步把 "VJP 训对了没" 与 "backward 图接得对不对" 解耦，只测前者。**

### 8.2 短 horizon trajectory optimization

固定初始 state，优化 $[a_0, a_1, \dots, a_{H-1}]$：

$$
\{a_t^*\}
=
\arg\min
\sum_{t=0}^{H-1}
L(s_t, a_t),
\qquad
s_{t+1} = E(s_t, a_t)
$$

Backward 按 §5.4 混合规则：接触帧走 $G_\theta$，非接触帧走 Genesis autograd，soft gate 加权。Adam / short-horizon SHAC-style truncated BPTT。

### 8.3 Policy 优化

$$
\nabla_\phi J
=
\mathbb{E}_\tau\
\sum_t
\left[
\text{(hybrid backward)} \cdot \nabla_\phi a_t
\right]
$$

Short horizon（H=8~32）避免长 BPTT 累积误差。SHAC-style value bootstrap 处理 horizon 尾部。

### 8.4 Online data aggregation

Policy 更新后 anchor 分布漂移，需持续追加：

- 每次 policy update 后用新 policy 采一批 rollout，追加到 anchor buffer
- Fine-tune $G_\theta$ 若干 iter
- Gate 中 OOD score 上升时降低 learned gradient 权重，退回 Genesis 原梯度

---

## 9. 与 GradWM 主线的关系

### 9.1 差异

|  | GradWM (学长) | Contact-VJP-Repair (本方案) |
|---|---|---|
| 主 claim | general local-anchor response 作为 surrogate gradient | contact-gated learned VJP 修补 manipulation policy gradient |
| 主模型 | full-transition response $m_\theta: (z, \Delta s, \Delta a) \to \Delta s'$ | object-VJP $G_\theta: (z, \lambda^\text{obj}) \to g_a$ |
| 覆盖范围 | all steps, all state dims | contact window only, object dims only |
| 监督信号 | Δs response (dim = state) | 方向导数 scalar / low-dim vector |
| Backward 通路 | autograd through $m_\theta$（Jacobian 隐式） | 直接输出 VJP，省一次 autograd |
| 对 reward 结构 | agnostic | 显式利用 "loss 挂在 object 上" |
| 零约束位置 | $m_\theta(z, 0, 0) = 0$（扰动零点） | $G_\theta(z, 0) = 0$（cotangent 零点，由架构保证） |

### 9.2 共享基础设施

两条线可以完全共用：

- Anchor 采样 dataloader
- State restore wrapper（绕开 broken `SimState.serializable()`）
- Perturbation probe API：$(v, \varepsilon) \mapsto E(\bar s, \bar a \pm \varepsilon v)$
- Task 定义、baseline、评估协议
- Contact feature extractor

工程差异只在训练目标（$\mathcal{L}_{\text{local}}$ vs $\mathcal{L}_{\text{dir}}$）和默认 backward 调用策略。

### 9.3 可能的合并版本

$$
\text{GradWM } m_\theta\ \text{+ Contact-VJP head}
$$

$m_\theta$ 覆盖 all steps 作为 general surrogate；$G_\theta$ 只在接触帧激活作为 specialized VJP。在此合并版本中，本方案的 $G_\theta$ 相当于 GradWM §8.3 中 VJP head 的**任务特化 + gated 变体**。

---

## 10. Baseline & Ablation

### 10.1 Baselines

1. Model-free RL：PPO / SAC
2. Genesis full autograd BPTT（现状，已知在接触帧不可靠）
3. Full finite-difference action gradient（贵，作为 upper-bound reference）
4. SPSA / zeroth-order policy gradient
5. GradWM 主线：$m_\theta$ + autograd, all-steps
6. Naive learned dynamics + BPTT：全 $\hat f_\theta$ 替换 $E$
7. **Ours**：contact-gated learned VJP

### 10.2 Ablations

- Gating：hard indicator / soft gate / no gating（VJP everywhere）
- Head 结构：matrix intermediary $A_\theta$ / direct vector head
- $\lambda$ conditioning：conditioned / unconditioned Jacobian head
- 训练目标：$\mathcal{L}_{\text{dir}}$ vs MSE fit vs response-only
- 扰动尺度：single-scale / multi-scale
- Object feature：full state / object-only / contact-only
- Response head 辅助：加 / 不加
- 数据：offline anchors / online aggregation
- Encoder：MLP / GNN（stretch）/ equivariant（stretch）
- 任务：box push / peg insertion / rope grasp

### 10.3 评估指标

- Final task return（Genesis 真实 rollout）
- Policy update success rate（沿 learned VJP 更新后真实 return 是否上升的比例）
- 收敛所需真实 sim step 数
- Gradient direction agreement with FD oracle（sanity only）
- Horizon 长度鲁棒性

---

## 11. 风险与应对

### 11.1 正反不一致

**风险**：forward 走 $E$，backward 走 $G_\theta$，两者是不同函数，标准 BPTT 收敛保证不成立。

**回应**：$G_\theta$ 学的是 **$E$ 在 anchor 处的方向导数**，label 来自真实 Genesis FD。拟合完美时 $G_\theta$ 就是 $E$ 的真实 VJP，函数层面**同源**；剩余偏差只是拟合误差，可用：

- 方向一致性 loss（保证方向对齐）
- Soft gate 截断不可靠 step
- Online aggregation 抑制分布漂移

这与 "另一个独立 world model 当反向" 是本质不同。

### 11.2 Cotangent 分布错配

**风险**：训练 $\lambda \sim P_\lambda$，测试 $\lambda \sim$ task-induced 分布。

**回应**：矩阵中介结构 $G_\theta(z,\lambda) = A_\theta(z)^\top \lambda$ 对 $\lambda$ 严格线性，理论上单一 $\lambda$ 分布已足够学 $A_\theta$。$P_\lambda$ 采样只影响 loss 权重。若观察到方向偏差，混入任务 loss 的真实 $\lambda$ 分布。

### 11.3 接触帧定义歧义

**风险**：什么算接触帧？阈值不同切出不同分布。

**回应**：第一版用 Genesis contact indicator + normal force 阈值（0.99 分位数标定）。边界帧问题由 soft gate 缓解，不用硬切。

### 11.4 State restore 完备性

**风险**：restore 不完整则训练数据脏。

**回应**：`notes/LOCAL_ANCHOR_STATE_RESTORE_SURVEY_2026-07-04.md` 已验证 two-box 同/跨进程 restore 到 qpos/link ~1e-11、qvel ~1e-7 精度。接触帧的 dofs_acc 残差 ~2.56e-4，一步后 contact 恢复。Step 0 收尾需扩到目标任务。

### 11.5 Long horizon 累积

**风险**：learned VJP 误差跨 step 累积，长 horizon 反向爆炸/消失。

**回应**：Short horizon（H=8~32）为 default；SHAC-style value bootstrap 处理尾部；Soft gate 中 OOD score 高时截断该步 backward。

### 11.6 网络结构过于简单

**风险**：MLP + gate 被 reviewer 拍成 engineering combination。

**回应**：novelty 在**问题定义（contact-mediated action gradient repair 作为独立任务）+ 训练目标（方向一致性 + 矩阵中介 + zero-cotangent 硬约束）+ 结构约束**，不在网络堆叠。第一版跑通后视需要加物理先验（$A_\theta$ 半正定 / 法向-切向分解 / equivariance）或 GNN 处理变长接触对。

### 11.7 被质疑 "GradWM 子集"

**风险**：学长可以说 "训好 $m_\theta$ 后 autograd 就是 VJP"。

**回应**：三点独立收益：

1. **信号维度**：VJP 是 $\dim(a)$-向量或方向导数 scalar；response 是 $\dim(s)$-向量。前者信号密度高，样本效率好，尤其 $\dim(s) \gg \dim(a)$ 时。
2. **架构约束**：矩阵中介 + 对 $\lambda$ 线性是 VJP 语义的自然表达；response head 需要 baseline subtraction 才能保证锚点一致。
3. **任务先验利用**：loss 挂在 object 上是 manipulation 的结构事实；$G_\theta$ 只在 object 子空间学，比 full state response 更贴任务。

---

## 12. 实施路线

**Step 0 — State restore 完备性收尾**
在目标任务（box push、simple grasp）上验证 restore 精度，扩自 `scripts/genesis_state_restore_check.py`。

**Step 1 — 单接触帧 FD 诊断**
选定一个明确的接触帧，跑 Genesis autograd vs FD 参考的散点图，量化偏差。确认动机在选定场景成立。

**Step 2 — 单帧最小 VJP**
- 输入：$z_t$（anchor + contact features）、$\lambda^\text{obj}$
- 结构：$A_\theta(z_t)$ MLP，$G_\theta = A_\theta^\top \lambda$
- 数据：Step 0 restore + 32~128 个 $(v, \varepsilon)$ 探针
- Loss：$\mathcal{L}_{\text{dir}}$，多尺度
- 判据：训练 + 留出方向 loss 收敛

**Step 3 — 单帧闭环 sanity（显式 action update）**
$a \leftarrow a - \eta G_\theta(z, \lambda)$，Genesis 重放，看真实 loss 下降。这是"正反不一致会不会毁优化"的第一次实证。

**Step 4 — 短 horizon traj-opt**
接入 custom backward，H=8~32，Adam 优化 action sequence。对比 Genesis full autograd / FD upper-bound。

**Step 5 — Policy 训练**
Short-horizon SHAC，online data aggregation。

**Step 6 — 扩展**
多接触任务、rope grasp、物理先验（$A_\theta$ 结构约束）、equivariance / GNN（stretch）。

---

## 13. 一句话总结

> 不学一个完整可微仿真器，也不学 full-transition 的 general response；只在接触帧上，直接学一个 object-loss-conditioned 的 learned VJP 模块，用方向一致性做监督，矩阵中介保证对 cotangent 线性，soft gate 处理接触/非接触边界。Forward 全程 Genesis，backward 非接触帧走原梯度、接触帧走 learned VJP。目标是把 Genesis 断掉的 `object loss → contact → robot action` 这条策略梯度链专门修补起来。

---

## 参考

- GradWM 主线文档：`/home/ubuntu/下载/GradWM：传统仿真器 Forward + 学习式 Backward.md`
- Genesis 接触梯度实证：`notes/GENESIS_CONTACT_GRADIENT_REPORT.md`
- State restore 调研：`notes/LOCAL_ANCHOR_STATE_RESTORE_SURVEY_2026-07-04.md`
- 项目定位（旧）：`notes/CONTACT_VJP_ROUTE_NOTE_2026-07-03.md`
- 研究方向汇总：`notes/RESEARCH_DIRECTION_SUMMARY_2026-07-06.md`
- 思路评价：`notes/VJP_PATCH_IDEA_CRITIQUE_2026-07-06.md`
