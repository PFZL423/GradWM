# Contact-VJP Repair：Genesis Forward + Learned Backward at Contact Frames

## 1. 问题背景

在 contact-rich manipulation 中，任务损失通常定义在 object / rope 状态上，但可优化变量在 robot action 或 policy 参数上。梯度链条需要穿过接触：

$$a_t \rightarrow \text{contact} \rightarrow s_{\text{obj},t+1} \rightarrow L$$

Genesis 的 forward rollout 可以提供较准确的物体运动与接触响应，可以记为：

$$s_{t+1} = E(s_t, a_t)$$

但在使用 BPTT 或 analytic policy gradient 做优化时，需要把 object loss 的梯度从接触后的物体状态传回 robot action：

$$\frac{\partial L}{\partial a_t}=\left(\frac{\partial s_{\text{obj},t+1}}{\partial a_t}\right)^\top \lambda^{\text{obj}}_{t+1},\qquad \lambda^{\text{obj}}_{t+1}=\frac{\partial L}{\partial s_{\text{obj},t+1}}$$

已有 Genesis 实验显示，标准 rigid backward pipeline 在接触传播路径上不可靠。在 two-box 接触实验中，pusher action 对 object state 的 forward sensitivity 非零，但 analytic pusher gradient 为 0。这个 failure mode 正好对应 manipulation 中的关键路径：

$$\text{object loss} \rightarrow \text{contact} \rightarrow \text{robot action}$$

因此，本工作的目标不是替换 Genesis forward，而是在接触窗口修补这条 object-loss-to-action backward path。

## 2. 核心思路

本方案保留 Genesis 作为 forward engine 和数据查询器，只在接触相关窗口引入一个 learned VJP module。整体上，forward 仍由 Genesis 执行，接触相关的 backward signal 由 learned VJP 提供。简化写作：

$$
\boxed{
\begin{aligned}
\text{Forward:}\quad & s_{t+1}=E(s_t,a_t)\quad \text{(Genesis, 全程)}\\
\text{Backward:}\quad & \frac{\partial J}{\partial a_t}=
\begin{cases}
\text{Genesis autograd}, & t \notin \mathcal{W}\\
G_\theta(z_t,\lambda^{\text{obj}}_{t+1}), & t \in \mathcal{W}
\end{cases}
\end{aligned}}
$$

这里 $\mathcal{W}$ 表示接触窗口，形式化定义见 §4。实际使用时不一定硬切，而是通过 §4 中的 soft gate 在 Genesis gradient、learned VJP 和 fallback 之间混合。

对一个接触 anchor：

$$(\bar{s}_t, \bar{a}_t, \bar{s}_{t+1}, c_t)$$

其中 $c_t$ 表示接触上下文，如 contact count、normal、penetration、relative velocity 等。模型先把 anchor context 编码为：

$$z_t = e_\theta(\bar{s}_t, \bar{a}_t, \bar{s}_{t+1}, c_t)$$

第一版采用 action-only VJP，只输出 action 侧梯度：

$$G^{(a)}_\theta(z_t, \lambda^{\text{obj}}_{t+1})\approx \left(\frac{\partial s_{\text{obj},t+1}}{\partial a_t}\right)^\top \lambda^{\text{obj}}_{t+1}=: g^a_t$$

如果后续需要完整 BPTT，可以扩展为 state + action VJP，同时输出 state-side 和 action-side 的反传量：

$$G^{(s,a)}_\theta(z_t, \lambda^{\text{obj}}_{t+1})=(g^s_t,\, g^a_t),\qquad g^s_t\approx \left(\frac{\partial s_{\text{obj},t+1}}{\partial s_{\text{obj},t}}\right)^\top \lambda^{\text{obj}}_{t+1}$$

两种版本的适用范围由 BPTT 递推决定。标准 BPTT 需要沿时间反传 cotangent：

$$
\begin{aligned}
\lambda_t &= \left(\frac{\partial s_{t+1}}{\partial s_t}\right)^\top \lambda_{t+1}+\frac{\partial L}{\partial s_t}\\
\frac{\partial J}{\partial a_t} &= \left(\frac{\partial s_{t+1}}{\partial a_t}\right)^\top \lambda_{t+1}
\end{aligned}
$$

- 第一版只修补 $g^a_t$：适用于 "$\{a_t\}$ 作为独立决策变量" 的情形，包括 single-frame action update 和 short-horizon trajectory optimization。此时 $\lambda_t$ 沿时间的递推可以走 Genesis autograd 或直接截断。
- 扩展版同时修补 $g^s_t$：适用于 policy BPTT，此时 policy 参数 $\phi$ 需要通过 $\lambda_t$ 反传到更早的动作，接触帧的 state-side VJP 不能截断。

两种版本共享 encoder 与训练流程，工程差异主要在 head 输出维度。除非特别指明，下文默认讨论第一版 $G^{(a)}_\theta$，简写为 $G_\theta$。

## 3. 模型结构与监督信号

为了保证 VJP 对上游 cotangent 线性，采用矩阵中介结构。初步版本：

$$A_\theta(z_t) \in \mathbb{R}^{\dim(s_\text{obj}) \times \dim(a)},\qquad G^{(a)}_\theta(z_t, \lambda)=A_\theta(z_t)^\top \lambda$$

扩展版本可以增加一个 state-Jacobian head，并共享 encoder $e_\theta$：

$$B_\theta(z_t) \in \mathbb{R}^{\dim(s_\text{obj}) \times \dim(s_\text{obj})},\qquad G^{(s)}_\theta(z_t, \lambda)=B_\theta(z_t)^\top \lambda$$

这种结构天然满足 zero-cotangent constraint：

$$G_\theta(z_t, 0) = 0$$

$A_\theta, B_\theta$ 都是矩形矩阵，无 PSD 语义；可挂的物理先验包括 contact-pair mask、法向/切向分解、低秩、SE(3) equivariance 等。

训练数据来自 Genesis restored anchor。对每个接触 anchor，采样动作扰动方向 $v_k$ 和尺度 $\varepsilon_k$，再查询 Genesis forward：

$$s^{+,k}_{\text{obj},t+1}=s_\text{obj}\left(E(\bar{s}_t,\bar{a}_t+\varepsilon_k v_k)\right),\qquad s^{-,k}_{\text{obj},t+1}=s_\text{obj}\left(E(\bar{s}_t,\bar{a}_t-\varepsilon_k v_k)\right)$$

由此构造 object-space directional response：

$$d_k=\frac{s^{+,k}_{\text{obj},t+1}-s^{-,k}_{\text{obj},t+1}}{2\varepsilon_k}\in \mathbb{R}^{\dim(s_\text{obj})}$$

主训练目标是方向一致性。我们希望 learned VJP 在动作扰动方向上的投影，匹配 Genesis forward 在同一方向上造成的 object-side loss 变化：

$$\mathcal{L}_{\text{dir}}=\mathbb{E}_{t,k,\lambda}\left\|v_k^\top G_\theta(z_t,\lambda)-\lambda^\top d_k\right\|^2$$

在矩阵结构下，这等价于学习 object response 的局部线性映射：

$$A_\theta(z_t) v_k \approx d_k$$

第一版只扰动 action，不扰动 state，从而避开 quaternion、joint limit、rope constraints 等状态合法性问题。如果扩展版为了学习 $B_\theta$ 需要 state perturbation，则需要引入 retraction map 和 rejection filter。若模型不稳定，可以加入一个辅助 response head：

$$R_\theta(z_t, \Delta a) \rightarrow \Delta s_\text{obj}$$

它只作为 encoder grounding，不参与实际 forward rollout。

Object state 的表示需要单独处理。若直接对含 quaternion 的状态做欧氏差分，会遇到双覆盖、球面曲率和数值跳变问题。可以按任务分级处理：

- rigid pose 任务（box push / peg insert）：用 $(p, \log R) \in \mathbb{R}^3 \times \mathfrak{so}(3)$
- deformable 任务（rope / cloth）：用 $K$ 个 keypoints $\{p_i\} \in \mathbb{R}^{3K}$，rope 天然对应每 segment 一个点
- 简化情形：box push 第一版 loss 可只投影到 XY + yaw

$\lambda^\text{obj}$ 与 $s_\text{obj}$ 必须在同一坐标图下定义。

## 4. 使用方式

最小验证不需要先接入完整 policy training。可以先做 single-frame action update：

$$a_t^{\text{new}}=a_t-\eta G_\theta(z_t,\lambda^{\text{obj}}_{t+1})$$

然后用 Genesis forward 重新 rollout，检查真实 object loss 是否下降。这个实验直接验证 learned VJP 是否给出了有用下降方向。

如果 single-frame update 能稳定降低真实 loss，再扩展到 short-horizon trajectory optimization：

$$\min_{a_0,\dots,a_{H-1}}\sum_{t=0}^{H-1} L(s_t,a_t),\qquad s_{t+1}=E(s_t,a_t)$$

接触窗口可以用 forward-time 的接触信号定义：

$$\mathcal{W}=\{t:\|f_{\text{normal},t}\|_\infty>\tau_f\ \lor\ n_{\text{contact},t}>0\}$$

阈值 $\tau_f$ 从数据分位数标定。这是 forward-time 观测量，不需要额外的 backward audit。

接触窗口内使用 learned VJP，非接触区域使用 Genesis 原生梯度。边界和置信度用两层独立 soft gate 处理：

$$
\begin{aligned}
\alpha^{\text{contact}}_t &= \sigma(w_c \cdot \text{contact\_score}_t)\in[0,1]\\
\alpha^{\text{trust}}_t &= \sigma(w_d \cdot \text{FD\_consistency}_t-w_o \cdot \text{OOD\_score}_t)\in[0,1]
\end{aligned}
$$

`contact gate` 判断这一步是否算接触帧，`trust gate` 判断 learned VJP 在这一步是否可信。混合规则：

$$
\begin{aligned}
g_t={}&(1-\alpha^{\text{contact}}_t)\, g^{\text{Genesis}}_t\\
&+\alpha^{\text{contact}}_t\left[\alpha^{\text{trust}}_t\,G_\theta(z_t,\lambda_{t+1})+(1-\alpha^{\text{trust}}_t)\,g^{\text{fallback}}_t\right]
\end{aligned}
$$

当接触帧上的 learned VJP 不可信时，不直接回退到 Genesis 原梯度，因为那正是要修补的对象。可选 fallback 包括：

1. 降幅度：$g^{\text{fallback}} = \eta_{\text{shrink}} \cdot G_\theta$，保留方向、缩小步长
2. Clip：防止 learned gradient 爆炸带偏 policy
3. Stop-gradient：$g^{\text{fallback}} = 0$，放弃该步的 credit assignment
4. 在线 FD-SPSA：用少量额外查询实时重估，成本较高，只在高价值 step 触发

Genesis 原梯度只在非接触帧（$\alpha^{\text{contact}}$ 低）为首选，不作为接触帧的 fallback。

## 5. 预期贡献与风险

预期贡献是一个 contact/object-centric learned backward module。forward 仍由 Genesis 提供，训练信号来自 Genesis restored anchor 上的 finite-scale perturbation，模型只服务于 object-loss-to-action 这条接触梯度链。

主要风险包括：

1. **FD label 尺度敏感**：硬接触下点态导数可能不稳定，因此目标应表述为 finite-scale useful VJP，而不是严格真实导数。
2. **正反不一致**：forward 来自 Genesis，backward 来自 learned VJP。需要通过单帧 action update 和 short-horizon traj-opt 验证其是否真的降低 Genesis rollout loss。
3. **接触窗口边界效应**：$\mathcal{W}$ 硬阈值可以定义（见 §4），但接触前若干帧的 pre-contact sensitivity 与刚脱离接触的 slip-off frame 是否需要纳入窗口，需通过 FD 查询实测；soft gate 已经在混合层缓冲，但窗口本身宽度仍是超参。
4. **分布漂移**：policy 更新后接触构型会变化，需要 online data aggregation。
5. **完整 BPTT 问题**：只输出 action VJP 是第一版；若要做长 horizon policy optimization，可能需要进一步输出 state-side VJP 或采用截断 horizon。

## 下一步计划

1. **目标场景 state restore 验证**，在 box push / simple grasp 上复测在线状态离线恢复，确认接触 anchor 可重复查询。
2. **单接触帧 FD**：选定接触帧，比较 Genesis autograd 与 FD directional response，量化偏差。
3. **构建最小数据集**：每个 anchor 采样 32-128 个 $(v,\varepsilon)$ probe，生成 $d_k$ 标签。
4. **训练单帧 VJP module**：先用矩阵中介 $A_\theta(z)$，验证留出方向上的 $\mathcal{L}_{\text{dir}}$。
5. **闭环 action update sanity**：用 $a \leftarrow a-\eta G_\theta$ 更新动作，检查 Genesis forward 下 object loss 是否下降。
6. **短 horizon traj-opt**：若单帧有效，再接入 H=8-32 的 action sequence optimization，对比 Genesis autograd、FD/SPSA 和 learned VJP。
