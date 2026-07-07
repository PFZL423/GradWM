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

- 非接触帧 $t \notin \mathcal{W}$：Genesis autograd 可信，两个 VJP 都直接使用。
- 接触帧 $t \in \mathcal{W}$：`∂s_{t+1}/∂a_t` 和 `∂s_{t+1}/∂s_t` 都过接触求解器，都不可靠。

本方案分**两阶段**修补：

- **第一版（action-only VJP）**：只修补 action-side VJP $(\partial s_{t+1}/\partial a_t)^\top \lambda_{t+1}$。适用范围是"$\{a_t\}$ 作为独立决策变量"的场景——single-frame action update（Step 3）、short-horizon trajectory optimization（Step 4，把 action sequence 直接优化，state 传递不进 backward 图）。此时 state-side VJP 不进入优化目标的计算。
- **扩展版（state + action VJP）**：同时修补 state-side 和 action-side VJP，把整条 backward 链在接触帧闭合。适用于 policy BPTT（Step 5），此时 policy 参数 $\phi$ 需要通过未来 state 的 cotangent 反传，$\lambda^s_t$ 不能截断。

两版共享同一套 encoder / 采样 / 训练目标，工程差异只是 head 输出维度和监督 label 是否包含 state 分量（详见 §5.1）。

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

想直接监督完整 VJP 向量需要在每个 anchor 上先构造 label $y_{\text{VJP}} = (\partial s_\text{obj}/\partial a)^\top \lambda$，这至少要 $\dim(a)$ 个探针方向做 LS 拟合出完整 Jacobian 再乘 $\lambda$——**成本高、易被接触帧 FD 噪声污染，且拟合到的完整 Jacobian 大部分维度在优化时用不到**。改用方向一致性 loss（见 §7）：只要求 $G_\theta$ 在采样方向 $v_k$ 上的投影匹配仿真器有限尺度方向导数，样本效率高且直接对齐"梯度下降只用到方向"的实际用途。

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

**VJP head（两版本，共享 encoder）**：

**第一版 · Action-only**（对应 Step 3-4）：

$$
G^{(a)}_\theta: (z_t, \lambda^{\text{obj}}_{t+1}) \longmapsto g_{a,t} \in \mathbb{R}^{\dim(a)}
$$

物理含义：

$$
G^{(a)}_\theta(z_t, \lambda^{\text{obj}}_{t+1})
\approx
\left(\frac{\partial s_{\text{obj},t+1}}{\partial a_t}\right)^\top
\lambda^{\text{obj}}_{t+1}
$$

**扩展版 · Action + State**（对应 Step 5 policy BPTT）：

$$
G^{(s,a)}_\theta: (z_t, \lambda^{\text{obj}}_{t+1}) \longmapsto (g^s_t, g^a_t) \in \mathbb{R}^{\dim(s_\text{obj})} \times \mathbb{R}^{\dim(a)}
$$

其中 $g^s_t \approx (\partial s_{\text{obj},t+1}/\partial s_{\text{obj},t})^\top \lambda^\text{obj}_{t+1}$ 用于把 cotangent 沿时间反传到 $s_t$；$g^a_t$ 与第一版含义相同。

**两版本关系**：扩展版是第一版的严格超集，共享 encoder $e_\theta$ 和采样 / 训练管线，只在 head 输出上增加一个 state-VJP 分支。第一版验证 workable 后再切换到扩展版，工程增量小。除非明确指明，下文默认在讨论第一版 $G^{(a)}_\theta$，简写为 $G_\theta$。

### 5.2 架构：矩阵中介 + 线性收缩

不直接从 $(z, \lambda)$ MLP 到 $g$，而是让网络吐 Jacobian 矩阵，再对 cotangent 做线性收缩。

**第一版（Action-only）**：

$$
A_\theta(z_t) \in \mathbb{R}^{\dim(s_\text{obj}) \times \dim(a)},
\qquad
G^{(a)}_\theta(z_t, \lambda) = A_\theta(z_t)^\top \lambda
$$

**扩展版（+ State）**：额外一个 head 吐 state-Jacobian

$$
B_\theta(z_t) \in \mathbb{R}^{\dim(s_\text{obj}) \times \dim(s_\text{obj})},
\qquad
G^{(s)}_\theta(z_t, \lambda) = B_\theta(z_t)^\top \lambda
$$

$A_\theta$ 对应"仿真器 forward 从 action 到 object 的局部 Jacobian"；$B_\theta$ 对应"从 object state 到 object state 的局部 Jacobian"。两者共享 $e_\theta$，输出分头。

**收益**（$A_\theta, B_\theta$ 通用）：

1. 对 $\lambda$ 严格线性：$G_\theta(z, \alpha\lambda + \beta\lambda') = \alpha G_\theta(z, \lambda) + \beta G_\theta(z, \lambda')$，符合 VJP 作为线性算子的物理性质。
2. 一次预测覆盖任意 $\lambda$，训练时不必对每个 $\lambda$ 方向单独采。
3. 可挂物理先验：结构化稀疏（只在 object-robot contact pair 上非零）、法向/切向分解、contact-pair mask、低秩、SE(3) equivariance 等（$A_\theta$ 是矩形矩阵，无 PSD 约束的余地）。
4. Zero-cotangent 硬约束自动满足：$G_\theta(z, 0) = 0$。
### 5.3 Zero-Cotangent 硬约束

VJP 是一个线性算子，$\lambda = 0$ 时必须 $g_a = 0$。矩阵中介结构 $G_\theta(z, \lambda) = A_\theta(z)^\top \lambda$ 天然满足此约束，无需 baseline subtraction。

**注意**：这个约束不是 GradWM §6 的 $m_\theta(z, 0, 0) = 0$（那是 response head 对 $\Delta s, \Delta a$ 的零约束）。VJP 的零约束在 $\lambda$ 上，不在扰动量上——VJP head 根本不吃扰动量。

### 5.4 接触窗口判定与 Soft Gate

接触窗口 $\mathcal{W}$ 定义为 Genesis forward 报告的 contact indicator：

$$
\mathcal{W} = \{t : \|f_{\text{normal},t}\|_\infty > \tau_f\ \lor\ n_{\text{contact},t} > 0\}
$$

阈值 $\tau_f$ 从数据分位数标定。**这是 forward-time 观测量，不需要 backward audit。**

接触/非接触边界处使用 **soft gate** 而非硬切。定义两个独立门：

$$
\alpha^{\text{contact}}_t = \sigma(w_c \cdot \text{contact\_score}_t) \in [0,1]
$$

$$
\alpha^{\text{trust}}_t = \sigma(w_d \cdot \text{FD\_consistency}_t - w_o \cdot \text{OOD\_score}_t) \in [0,1]
$$

`contact` 门控制"这一步是否在接触帧"，`trust` 门控制"learned VJP 在这一步是否可信"。

**Backward 混合规则**（层级 fallback）：

$$
\left.\frac{\partial J}{\partial a_t}\right|_{\text{used}}
=
\underbrace{(1-\alpha^{\text{contact}}_t) \cdot g_{a,t}^{\text{Genesis}}}_{\text{非接触:\ Genesis}}
+
\underbrace{\alpha^{\text{contact}}_t \cdot \alpha^{\text{trust}}_t \cdot G_\theta(z_t, \lambda_{t+1})}_{\text{接触\ +\ 可信:\ learned VJP}}
+
\underbrace{\alpha^{\text{contact}}_t \cdot (1-\alpha^{\text{trust}}_t) \cdot g^{\text{fallback}}_{a,t}}_{\text{接触\ +\ 不可信:\ 见下}}
$$

**接触帧不可信时的 fallback 层级**（不退回 Genesis 原梯度——那正是要修补的对象）：

1. **降幅度**：$g^{\text{fallback}} = \eta_{\text{shrink}} \cdot G_\theta$，$\eta_{\text{shrink}} \in (0, 1)$，保留方向信息但缩小步长
2. **Clip**：$g^{\text{fallback}} = \text{clip}(G_\theta, \|\cdot\| \le c)$，防止 learned gradient 爆炸带偏 policy
3. **Stop-gradient**：$g^{\text{fallback}} = 0$，直接放弃这一步的 credit assignment
4. **在线 FD-SPSA**（贵，仅在高价值 step 触发）：用小预算探针实时重估 $g_a$

Genesis 原梯度只在**非接触帧**是首选（由 $\alpha^{\text{contact}}_t$ 分支），不作为接触帧的兜底。

**Gate 输入来源**：

- `contact_score`：接触强度（normal impulse / penetration）
- `FD_consistency`：在线 FD 探针与 $G_\theta$ 输出的方向余弦（学长 §8.4 的 online probe 简化版）
- `OOD_score`：当前 anchor 与训练分布的距离（ensemble variance 或 kNN 距离）

Soft gate 是**接缝连续性 + 不可信 learned gradient 隔离**的组合处理。

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

Genesis 状态含 quaternion / joint limits / rope segment 约束。对 $\bar s$ 加高斯噪声会产生非法状态。本方案第一版**只扰动 action，不扰动 state**，避开此问题。扩展版为学 $B_\theta$ 需要 state 扰动时，需引入 retraction map $\Pi: X \times \mathbb{R}^n \to X$，参考 manifold optimization 语言（Absil et al.），并做 rejection filter 过滤仿真失败样本。

### 6.4 Object state 表示（quaternion 处理）

$d_k \in \mathbb{R}^{\dim(s_\text{obj})}$ 是仿真器在 anchor 处的方向导数，若 $s_\text{obj}$ 中含 quaternion 且直接用欧氏差分，会出问题：

1. **双覆盖**：$q$ 和 $-q$ 表示同一姿态，但 $s^+ - s^-$ 可能在两个覆盖分支之间跳跃
2. **球面曲率**：quaternion 在单位球面上，欧氏差不是 tangent 方向
3. **数值跳变**：接近 $\pm 1$ 分量时差分符号翻转

分级处理：

- **box push / peg insertion 等 rigid pose 任务**：object 表示用 $(p, \log R) \in \mathbb{R}^3 \times \mathfrak{so}(3)$（位置 + rotation log-map），$d_k$ 定义在 SE(3) 切空间上，欧氏差分合法
- **rope / cloth 等 deformable 任务**：object 用固定 $K$ 个 keypoint 坐标 $\{p_i\} \in \mathbb{R}^{3K}$（rope 天然对应每 segment 一个点），无旋转，欧氏差分直接合法
- **简化情形**：box push 第一版 loss 仅用 XY 位置和 yaw 时可回避 quaternion（但 sim 内部仍是 quaternion 表示，只在 loss / label 层做投影）

$\lambda^\text{obj}$ 与 $s_\text{obj}$ 必须在**同一坐标图**下定义，train / infer 都用一致表示。

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

**代入矩阵中介结构化简**：由 $G_\theta(z, \lambda) = A_\theta(z)^\top \lambda$，

$$
v_k^\top G_\theta(z_t, \lambda) - \lambda^\top d_k
= v_k^\top A_\theta(z_t)^\top \lambda - \lambda^\top d_k
= \lambda^\top \bigl(A_\theta(z_t)\, v_k - d_k\bigr)
$$

对 $\lambda \sim P_\lambda$（单位球面均匀采样）取期望后：

$$
\mathcal{L}_{\text{dir}} = \mathbb{E}_{t, k}\ \left\| A_\theta(z_t)\, v_k - d_k \right\|^2 \cdot \tfrac{1}{\dim(s_\text{obj})} \quad \text{(schematic)}
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
