# Phase 1B 总记录 — 2026-06-13

> **下次进 session 看这个文件就够了。** 三段：(1) 我们到了哪儿、(2) 已经否决/确认了什么、(3) 下次该干什么。

---

## 1. Phase 1B 全程 commit 时间线

| # | Commit | 日期 | 内容 |
|---|---|---|---|
| 1B-1 | 2f4ffca | 06-13 | 手写 7-DOF capsule arm + 2-finger gripper，backward 通 |
| 1B-2 | ff2099c | 06-13 | arm + 12-seg rope rigid-rigid 接触 backward 通（**[[rigid-cable-backward-nan]] 后第一次 contact 可微**）|
| 1B-3 | 77ba68d | 06-13 | scripted grasp + grad logger + Figure 1（close-onset grad 跳水 1.7 dec）|
| 1B-3 fix | df1dd76 | 06-13 | review 改 cable 锚点 + finger 方向（grad 跳水变 2 dec）|
| 1B-3 v2 | c06ae1c | 06-13 | 桥型场景（两台子 + cable 横搁，弃竖挂） |
| 1B-3 v3 | 8765b58 | 06-13 | manipulation L-pose 真上提 cable |
| 1B-3 v3 | a5726b7 | 06-13 | 侧面 mp4 视频 + 视觉 polish（cable spacing/arm 全灰）|
| 1B-4 | 6ae9a46 | 06-13 | 启动 rope solver 调研（PBD/FEM/MPM 探针）|
| 1B-4 | d5eca03 | 06-13 | **关键发现**：源码层证实 PBD/FEM 的 backward hooks 全是 `pass` stub |
| 1B-4 | c281c68 | 06-13 | MPM 调参 sweep — 任何 E 都不像绳子 |
| 1B-5 | a8d7fcb | 06-13 | MPM E=200 放 grasp_scene → cable 在 finger 接触下"撕裂" |
| 1B-6 | 50f6756 | 06-13 | rigid cable sweep（N × damping × armature） |
| 1B-7 | db26fbd | 06-13 | 最终 grasp_scene 用 **N=40 + damping=0.001 + armature=0.0001**，pick 视觉成功 |

---

## 2. 已经定死的事实（下次不要再重新探索）

### Genesis 1.1.1 differentiable 体系实情

| Solver | 可微状态 | 证据 |
|---|---|---|
| **rigid（手写资产）** | ✅ 可用 | 1B-1/2/3 实测，backward NaN=0；bundled 资产仍 NaN（[[rigid-cable-backward-nan]]）|
| **rigid（bundled MJCF）** | ❌ NaN | 等 PR #2842 |
| **MPM** | ✅ 可用 | grad=0.41 实测；但 CPIC=True 与 requires_grad=True 互斥（Genesis 显式 raise）|
| **PBD** | ❌ stub-only | 源码 `pbd_solver.py` 6 个 grad hook 全 `pass` |
| **FEM** | ❌ stub-only | `FEMEntity` 缺 `collect_output_grads`，backward AttributeError |

### MPM 不能做绳子（**根本不是 bug，是数学上不行**）

MPM 是体材料（剪切+体积模量），表达不了绳子的"高拉伸 + 低弯曲"组合。三档 E 实测：
- E=3e5（默认）：金属杆
- E=5e3：橡胶棒
- E=200：果冻；放 grasp_scene 接触下**撕裂**（粒子 deformation gradient 奇异化）

业内标准的可弯曲绳子都是 PBD/XPBD 距离约束链（PyBullet/FleX/Houdini），但 Genesis 1.1.1 PBD 不可微。

### 唯一可微 + 物理合理的 rope 路径

**rigid + ball joint chain**，sweep 找出 sweet spot：
- **N=40 段**（更多 fps 暴跌、更少视觉太僵）
- **damping=0.001**（更硬→棒，更软→流体感）
- **armature=0.0001**
- **mass=0.001 kg/段**（默认）
- **总长 0.30m，台心间距 0.10m，cable 居中**（B0 freejoint 必须在 `table_mid - L/2`，否则不对称下滑）
- **fps：4060 上 ~16 step/s；4 卡 4090 上估计 ~350 step/s**

### 硬件状况

**用户实际机器是 4 卡 4090**，不是 4060。所有"显存紧"、"fps 太慢"的旧判断要按 4090 重算（[[user-wants-gpu-with-justification]] 已更新）。

### Genesis API 可微性 cheatsheet（rigid entity）

| API | shape | requires_grad |
|---|---|---|
| `arm.get_dofs_velocity()` | (n_dofs,) | ✅ True |
| `arm.get_dofs_position()` | (n_dofs,) | ❌ False |
| `arm.get_state().pos` | (1, 3) **仅 base** | ✅ True（但只 root 位置）|
| `arm.get_links_pos()` | (n_links, 3) | ❌ False |
| `link.get_pos()` | (3,) | ❌ False |
| `scene.get_state().solvers_state[mpm_idx].pos` | (n_envs, n_particles, 3) | ✅ True（MPM only）|

**含义**：rigid arm 的 differentiable handle 主要是 `get_dofs_velocity()`。要拿 link/cable_tail 位置作 loss，需要自己写 forward kinematics（暂未做）。

---

## 3. 当前 grasp_scene 状态（截至 db26fbd）

**Scene 几何**：
- arm base @ (0,0,0)，arm 7 hinge + 2 prismatic（`scripts/make_arm_mjcf.py`）
- L-pose 初始 qpos = (0, 0.5, 0, 0.94, 0, 1.60, 0)，palm 在 (0.33, 0, 0.178)，fingers 朝下
- 两台子 @ x=0.28 / 0.38（gap 0.10m），台面 z=0.14
- Cable 40 段，total_len=0.30m，居中（起点 x=0.18），rest_z=0.156

**Scripted policy** 5 阶段：settle 30 → approach 15 → close 20 → lift 60 → hold 80 = 205 帧 @ 50fps = 4.1 秒视频

**结果**：lift 末 cable 真被 fingers 提起来 V 字垂下，台面空，**视觉上完整 manipulation pick 成功**

**关键文件**：
- `scripts/make_arm_mjcf.py` — arm MJCF 生成器（**finger axis "0 -1 0"/"0 1 0"，正向 q closes**）
- `scripts/grasp_scene.py` — 主 scene + scripted policy + grad logger
- `scripts/render_grasp_video.py` — 侧面视频
- `scripts/segment_death_line.py::make_cable_mjcf` — 旧 cable 生成器（仍在被 sanity 脚本用）
- `scripts/rigid_cable_sweep.py` — N×damping sweep 脚本（subprocess workers）
- `analysis/grasp_phase1.mp4` — 当前最佳 demo 视频
- `analysis/grad_norm_phase1.png` — Figure 1 grad 跳水曲线
- `analysis/rope_solver_probe/` — PBD/FEM/MPM 对照视频
- `analysis/rigid_cable_sweep/` — N=16/40/50 sweep 视频
- `logs/rigid_cable_sweep.csv` — sweep 数据表

---

## 4. 还可以调（用户问过但没动）的 cable 参数

按收益排：
1. **`stiffness=0.001`**（关节弹性恢复）—— 当前 = 0，cable 无骨头流体感；加上变成柔性电缆/电线
2. **`friction="5 0.005 0.0001"`**（cable + finger geom 上的 contact friction）—— 默认 1.0，加大 fingers 不易滑脱
3. **`mass=0.005`/段**（5× 现值）—— 摆动太久太轻 → 沉甸甸像真绳
4. **`substeps=8`**（当前 4）—— 多接触场景数值更稳
5. ToolEntity 用的 `solref` / `solimp` / `condim` —— 接触响应曲线，一般不动

---

## 5. 下次进 session 直接开干的选项

### 选项 A：先调 cable 物理参数（stiffness/friction/mass）

- 1 小时，目标让视频里 cable 看起来更"沉甸甸"、fingers 抓得更稳
- 改 `scripts/grasp_scene.py::_make_bridge_scene_mjcf` 加 stiffness + friction
- 重跑 render_grasp_video.py 看效果
- 可能要重测 backward NaN（stiffness>0 的 backward 路径未实测）

### 选项 B：写 traj opt 脚本（核心交付物 A 升级）

- 1-2 天，验证 SHAC backward 在 4 卡 4090 上能不能 work
- 新建 `scripts/traj_opt.py`：复用 grasp_scene，60 步 qvel 序列作可学参数，Adam 优化让 cable 抬到目标 z
- Loss 设计：用 `arm.get_dofs_velocity()` reach target qvel（已实证可微）+ 接触力 magnitude 代理 grasp（要测）
- 出 loss-vs-iter 曲线（Figure 2 候选）
- **如果 traj opt 跑通 → 上 SHAC 有把握；跑不通（grad vanish 太狠）→ 直接走 PPO 路线 + 把 vanishing 当 paper 数据**

### 选项 C：把 grasp_scene 包成 Env class，准备 RL 训练

- 1 周，开始 Phase 2 主菜
- `scripts/rope_grasp_env.py`：reset/step/obs/reward 接口
- batch envs（scene.build n_envs=64+），单 episode horizon=100
- Reward dense shaping：approach 距离 + grasp 接触力 + lift 高度
- 先 PPO baseline（forward only，无 backward）跑通

### 选项 D：补 sweep 数据（Figure B failure-mode 地图）

- 0.5-1 天，攒 paper 数据
- N × damping × stiffness × cable_length 多维 sweep，记录每点 backward NaN onset / grad 衰减率
- 这是任务交付物 B 的核心数据

### 我（claude）下次会推荐的顺序

**A → B → C**（先把视觉再调一次，然后做 traj opt 验证可微管线，最后包 Env 进 RL）。或者如果你想直接看"能不能学"，**跳到 B**。**D 单独一条** sweep 数据可以并行收集。

---

## 6. 关键决策（下次直接复用，不要再问用户）

- ✅ 用 rigid + ball joint chain，N=40，damping=0.001，armature=0.0001
- ✅ 机械臂从上方抓（不倒挂），用 manipulation L-pose
- ✅ 两台子 + cable 横搁桥型，cable 居中
- ✅ 视频 fps=50（10× 慢动作回放），horizon 至少 200 帧（看完整 pick + hold）
- ✅ codex 派发用 sandbox=workspace-write、effort=xhigh
- ✅ codex MCP 工具名 `mcp__codex_cli__codex`（下划线）
- ✅ 4 卡 4090 训练机，所有性能预算按这个算

## 7. 还在悬空的问题（下次决策）

- ❓ traj opt 用什么 loss（接触力 vs qvel reach）—— 要写时再决定
- ❓ Phase 2 RL 用 SHAC 还是 PPO baseline 先 —— 看 traj opt 结果
- ❓ Cable 长度变动到多少（30 cm 还是更长）取决于实际 manipulation 任务定义

---

## 关键 memory 引用（最新状态）

- [[project-goal]] — Phase 1B/2/3 路线图
- [[rigid-cable-backward-nan]] — bundled rigid 资产 backward NaN 死区
- [[contact-gradient-vanishing]] — 单接触 grad 100×/step 衰减
- [[phase-1b-grasp-scene]] — 7-DOF arm + cable contact + Figure 1 + Genesis API cheatsheet
- [[genesis-pbd-fem-grad-stubs]] — PBD/FEM stub 证据 + MPM 撕裂 + 唯一可微路径结论
- [[user-wants-gpu-with-justification]] — 4 卡 4090 硬件实情（覆盖原 4060 假设）
- [[dont-trust-bundled-genesis-assets]] — 任何 bundled 资产先 30-step sanity
