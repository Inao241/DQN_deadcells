# DQN_deadcells → Workshop Paper：Project 交接总结

> 更新于 2026-10-25 凌晨 1:55（2x 和 boss HP 已人工确认；第一次试跑只跑了 1 局多，发现并修了两个记录 bug；准备用 `TEST3_pilot.bat` 重跑 3000 步试跑）。作者：Michael Jin（UC Irvine）。仓库：github.com/Inao241/DQN_deadcells，本地 `C:\code\DQN_deadcell\DQN_deadcells`，工作分支 **`paper-experiments`**。

## 1. 目标与背景

- 把自己写的 Dead Cells boss 战 DQN 项目（boss：The Concierge / 看守者）做成一篇 **workshop 短论文**（4–6 页）。
- 合作者：**Pengcheng Xu**（Prof. Jing Zhang 实验室的 PhD 学生）。他负责把文章改成学术写法，并选具体投哪个 workshop。他提到 **AAAI 2027 workshop 截止日期 11/20**（各 workshop 的 DDL 不同，以具体 CFP 为准）。他给的写作参考：lmgame-Bench（ICLR 2026，https://openreview.net/forum?id=qeziG97WUZ）。
- 备选 venue：FDG 2027 Late-Breaking Work（约 2027/3/8）、IEEE CoG 2027 auxiliary paper（约 2027/5/15）、ICLR 2027 workshops（约 2/1）。投稿前都要核实官网日期。

## 2. 论文定位

**主线 = Direction A（实证研究）+ 一个小规模 Direction C（延迟）实验作为亮点。**

- 研究问题：Rainbow 系列组件（Double DQN、PER、Dueling）的结论来自 Atari，而 Atari 环境会等待 agent、能跑上亿帧。在**实时、交互样本很少**的商业游戏里，这些结论还成立吗？
- Direction C（亮点）：DQN 假设动作立刻在 $s_t$ 上执行，但实时游戏里截屏、推理、按键都有延迟，等动作生效时状态已经是 $s_{t+d}$，Markov 性被破坏。盾反、翻滚这类帧敏感的动作应该最先受影响。
  - **做法（2026-10-25 讨论后的建议）**：不要用 1x / 2x 当延迟旋钮（同时改变了反应时间、单位游戏时间的决策数和每小时样本数，而且 `lib/env.py` 的 reset 时间是按 2x 调的）。改为在 `evaluate.py` 里加 action queue，人为延迟 d 步（d ∈ {0,1,2,3}），用训好的 `final.pt` 评估，每个 d 20 局，结合 `damage_events.csv` 看盾反/翻滚相关受伤比例随 d 的变化。零训练成本。可选强版：训练时也加延迟（d=2，3 seeds）。⏳ 代码还没写。
- Direction B（通用 2D 游戏 template / benchmark）留作下一篇。可复现性（要买游戏、Windows、读内存）是它最大的短板。

**审稿人会问的问题（来自 lmgame-Bench 的 OpenReview 意见）**，要提前准备：
1. novelty：要主动讨论哪些结论和 Atari 上的 Rainbow 一致、哪些不一致，以及原因
2. 方差和 seed 数：报告置信区间，不能只给一个数
3. 失败分析要量化：按 boss 招式统计死因和受伤原因（代码已支持）
4. Related Work 紧跟 Introduction；指标单独一节写清楚
5. 写明算力和时间成本

**数据能否支撑论文（2026-10-25 评估）**：
- 3 seeds 是最大的统计弱点（seed 层 bootstrap 只有 10 种组合），4 个 config 的 CI 很可能重叠。建议：seed 0–2 跑完后，时间允许就只给 `dqn` 和 `double_per_dueling` 两端加 seed 3、4。
- 80k 步的依据来自有 bug 的旧版本：试跑后看 learning curve 再定。win rate 区分不开时用 boss HP dealt（%）这种连续指标。
- 只有一个 boss、一个游戏：定位成 case study，写进 Limitations。
- Human baseline（1x、作者本人）只能作为参考上限，不能直接比较。
- 主证据用 learning curve / AUC（sample efficiency），final eval 作补充。

**论文结构**：Intro → Related Work（DQN 系列；游戏 AI / 视觉 RL）→ Environment（MDP 形式化：observation、action、reward 公式，加系统框图）→ Experimental Setup → Results（总表、learning curve、ablation、死因分析）→ Limitations & Conclusion。

**参考文献**：Rainbow（Hessel et al., AAAI 2018）、Revisiting Rainbow（Obando-Ceron & Castro, ICML 2021）、Deep RL that Matters（Henderson et al., AAAI 2018）、Statistical Precipice / rliable（Agarwal et al., NeurIPS 2021）、ViZDoom（Kempka et al., CIG 2016）、Action Space Shaping（Kanervisto et al., CoG 2020）、CS:GO behavioural cloning（Pearce & Zhu, CoG 2022）、Real-Time RL（Ramstedt & Pal, NeurIPS 2019）、lmgame-Bench（ICLR 2026）。

## 3. 代码现状（分支 `paper-experiments`）

原来的 `train.py`、`run.py` 只加了启动时自动设 2x（`GAME_SPEED = 2.0`，设 0 则不动速度），其余没改。新增：

| 文件 | 作用 |
|---|---|
| `ablation_common.py` | 可开关的 Double / PER / Dueling、日志、受伤事件记录 |
| `train_ablation.py` | 训练一个 config × seed，存到独立文件夹（`config.json`、`episodes.csv`、`damage_events.csv`、checkpoint）。文件夹里有没跑完的旧数据时会自动改名成 `<name>_incomplete_<时间>`，不会混在一起 |
| `evaluate.py` | 关闭探索的评估；`--random`、`--human` baseline |
| `run_experiments.py` | 无人值守跑完整队列，已完成的自动跳过，中断后重跑同一命令即可 |
| `analyze.py` | 结果表（含 LaTeX）、learning curve、hierarchical bootstrap CI、Glass's δ、延迟分解、算力成本 |
| `analyze_damage.py` | 受伤时 agent 在做什么、被哪个 boss 招式打中（按伤害值聚类）、死因、战斗阶段；生成每类招式的截图，让人命名 |
| `selftest.py` | 用假游戏测试整个流程，不需要开 Dead Cells。每次运行前会先删掉自己的 `selftest_runs/`、`selftest_analysis/`。假游戏现在也模拟"玩家死后 boss 指针读出 0" |
| `speedhack.py` + `speedhack_dll/dqn_speedhack.lua` | 从 Python 调用 **CE 7.7 本体的 speedhack** 做 2x（见下）；`python speedhack.py --detect` 检查当前游戏是否已加速 |
| `probe_hp_chains.py` | 同时实时读多条 CE pointer chain，筛出稳定的 HP 链；`--dump` 打印 entity 在 +0x100~+0x180 的字段 |
| `SETUP_env.bat`、`requirements.txt`、`check_env.py` | 一键建 conda 环境 `DQN` 并跑 check + selftest |
| `EXPERIMENTS.md`、`HANDOFF.md` | 使用说明、仓库内交接文档 |
| `TEST1_env.bat`、`TEST2_speed.bat` | 测试脚本（会自动激活 `DQN` 环境），双击运行，输出到 `test_logs/` |
| `TEST3_pilot.bat` | 先跑 selftest（结果在 `test_logs/selftest.txt`，没通过就停），再跑 3000 步试跑到 `runs_pilot/pilot3000`。中途停止要按住 `q`，不要直接关窗口（否则什么都不保存） |

`model.py`：增加了 `dueling=True/False` 参数（默认 True，旧 checkpoint 仍能加载）。

**HP 读取（已修好，commit `10dfaaf`）**
- 原因：游戏更新后 `libhl.dll` 的 base 从 `0x48184` 移到 **`0x49184`**，中间的 offset 都没变。旧代码读 `[48184]` 拿到的非零值只是巧合。
- 用 `probe_hp_chains.py` 测了 CE pointer scan 的 15 条候选链（掉血、喝血瓶、进 boss 房、死亡重开）：**B 组（`... → 68 → 12C`，5 层）全程稳定**，就是 `GetHp.py` 原来那条；A 组重开后失效且偶尔读出 0（会被误判死亡），C 组加载关卡时乱跳，都排除。
- **boss 链**：旧 offset + 新 base 可用。boss 房外读不到，进房后有值，打它时下降。**玩家死亡后 boss 链也读出 0**（试跑发现）。
- `lib/GetHp.py`：base 和两条链提成文件顶部常量 `HP_BASE_OFFSET`、`PLAYER_HP_CHAIN`、`BOSS_HP_CHAIN`；读失败或空指针返回 0。
- `ablation_common.py` / `evaluate.py`：**去掉写死的 `PLAYER_MAX_HP` / `BOSS_MAX_HP`**，每局 reset 后读到的 HP 作为该局 max HP，并在每局日志里记录 `player_hp_max` / `boss_hp_max`。
- 试跑读到：玩家满血 **15482**，boss 满血 **205238**。

**2x 加速（已跑通，已目测确认）**
- 历程：v1（改所有模块 import table）注入 1 秒后崩；v2（自写 jmp hook）弃用；v3（注入 CE 的 `speedhack-i386.dll`）发现 **CE 7.7 已经没有这个 DLL**，加速改由 `autorun/SpeedhackV3.lua` 实现。
- 现在的做法（直接用 CE 7.7 原版代码，不改）：
  1. 每次启动游戏后第一次设速度时，`speedhack.py` 最小化、不抢焦点地启动 CE；
  2. `dqn_speedhack.lua`（已复制到 `C:\Program Files\Cheat Engine\autorun\`，只在被 Python 启动时生效，平时手动用 CE 不受影响）打开游戏、调用 `speedhack_setSpeed(2)`，把游戏内 `speedhack_wantedspeed` 这个 float 的地址回报给 Python；
  3. Python **强制**关掉 CE（正常退出会把速度重置为 1）。hook 和 float 都在游戏进程里，加速持续有效；
  4. 地址记在 `%TEMP%\dqn_speedhack_<pid>.txt`，之后改速度（如 human baseline 用 1x）直接写这个 float，不会再启动 CE。
- **同一局游戏重跑训练不用重启**：先读 `%TEMP%` 里保存的地址；没有的话（文件被清掉，或在 CE 窗口里手动开过加速）就从游戏内存里找回 `speedhack_wantedspeed`（CE 两个 hook 都引用、位于一块分配起点的那个全局变量）。`python speedhack.py --detect` 已在真实游戏上验证找对（`0x25580000` = 2.00x）。只有游戏重开后第一次运行才会启动 CE。
- `train.py`、`run.py`、`train_ablation.py`、`evaluate.py`、`run_experiments.py` 启动时都会自动设 2x；`--game-speed 0` / `GAME_SPEED = 0` 表示不动速度。
- hook 撤不干净；万一出现重复 hook（旧 DLL 和 CE 的 hook 叠加等）就**重启游戏**。

**相对原 `train.py` 修复的问题**（论文里描述的是修复后的正确实现）：
- **Double DQN 有 bug**：原代码用 `net(state)` 在当前状态 $s$ 上选 $a'$，正确做法是 $a' = \arg\max_a Q_{online}(s', a)$。已修复，selftest 会数值验证 target 公式。
- move head 在 $s'$ 的 target 改为以"$s'$ 上策略会选的 action"为条件（原来用的是 $s$ 上执行的 action）。
- PER 两个 head 用同一个 priority 公式；第一帧在 reset 之后再截取。
- **glitch 胜利**：`lib/env.py` 只要 boss HP 读到 ≤ 1 就判胜（+100）。现在 `C.is_glitch_win()` 把两种情况记为 `glitch`、不用于学习并强制结束这一局：开局 3 步内的胜利，以及**上一步 boss HP 还高于 5% 就直接"胜利"**（一刀只打 0.2–0.6%，真击杀一定从很低的血量开始，所以这只能是读取失败）。train 和 evaluate 都用这个判断（2026-10-25 新增，⏳ 待 selftest 确认）。
- **死亡时 boss HP 记录**：玩家死后 boss 链读出 0，原来 `episodes.csv` 会把这一局记成 `boss_hp_end = 0`（看起来像 boss 被打死），而 boss 剩余血量是比较各 config 的主要连续指标。现在死亡那一步保留上一步的真实 boss HP（`GameSession.step` 和 human baseline 都改了；reward 不受影响）。（2026-10-25 新增，⏳ 待 selftest 确认）
- 窗口匹配：`getWindowsWithTitle` 是子串匹配，会选到 `...\Dead Cells\deadcells.exe` 窗口，已改为精确匹配。
- **旧结果（100% 通关、350 局赢 56 次）来自有 bug 的版本，不要再引用。**

**实验设计**：
- 配置：`dqn`、`double`、`double_per`、`double_per_dueling`（可选再加 `double_dueling` 做 2×2）
- seed：0、1、2（按 seed 顺序跑，没跑完也有完整的对比）
- 每次训练 **80k 步**。依据：旧 checkpoint 的训练记录显示约 4 万步开始稳定获胜，约 10 万步后趋于平稳（来自有 bug 的旧版本，试跑后再确认）
- 评估：`final.pt`，greedy，每个 seed 20 局；另有 random baseline 和 human baseline（你自己打 20 局，1x 速度）
- 命令：`python run_experiments.py --plan`，然后 `python run_experiments.py --max-steps 80000 --seeds 0 1 2`，最后 `python analyze.py && python analyze_damage.py`

## 4. 电脑环境

- 训练用 conda 环境 **`DQN`**：Python 3.9，torch 2.8.0+cu128（RTX 5080 是 Blackwell 架构，需要 cu128 版本），CUDA 正常。用 `SETUP_env.bat` 建立/修复
- Anaconda base 环境：Python 3.13.9，torch 2.11.0+cu128（不用于本项目）
- 已安装：`dxcam`、`keyboard`、`pygetwindow`、`pywin32` 等（见 `requirements.txt`）
- `tensorboard` 在 base 里已卸载（装它会把 protobuf 升到 7.x，破坏 streamlit、langflow 等包；已降回 protobuf 5.29.3）。代码里是可选的，所有关键数据都在 CSV 里
- Anaconda 的 numpy 和 torch 有 OpenMP 冲突（`OMP: Error #15`），代码和 bat 里已设置 `KMP_DUPLICATE_LIB_OK=TRUE`
- Dead Cells 是 **32 位**进程，Python 是 64 位
- Cheat Engine **7.7**，装在 `C:\Program Files\Cheat Engine`
- 电脑装有 Riot Vanguard，可能干扰 CE 的调试功能（"Find out what writes to this address" 会崩）

## 5. 测试状态

| 项目 | 状态 |
|---|---|
| selftest（GPU） | ✅ 之前通过（一次训练更新约 15.7 ms）。⏳ 10-25 改了 glitch 判断和死亡时 boss HP 记录，新增 2 项检查，需要再跑（`TEST3_pilot.bat` 第一步会跑） |
| 读玩家 HP | ✅ base `0x49184`，B 组链稳定，满血 15482 |
| 读 boss HP | ✅ 人工确认进房后读数正常；训练脚本里读到满血 205238，挨打下降。玩家死后读出 0（已在记录里处理） |
| 2x 加速 | ✅ TEST2 跑通，已目测确认是 2 倍速；`--detect` 能找回已开的加速，重跑不用重启游戏 |
| 窗口匹配 bug | ✅ 已修复 |
| reset 流程 | ✅ 第一次试跑里死亡 → reset → 回 boss 房成功 1 次（第 0 局 → 第 1 局）。⏳ 还需要在完整试跑里连续多局确认 |
| 3000 步试跑 | ⏳ 第一次（`runs_pilot/speed`，10-25 01:37）只跑了第 0 局和第 1 局约 255 步，进程被直接结束，没有 `summary.json` 和 checkpoint；那份数据的 `boss_hp_end` 有上面那个 bug，不用。下一次用 `TEST3_pilot.bat` 跑到 `runs_pilot/pilot3000` |

**第一次试跑的部分数据（第 0 局，ε≈0.86，基本是随机动作）**：468 步、43.7 s、输，打掉 boss 约 3.8%，被打 20 次。每步平均 93 ms（p95 107 ms），其中 env（按键 + 读 HP）70.5 ms、截屏 18.3 ms、训练 3.4 ms、选动作 0.9 ms。reset 里的 sleep 合计约 18 s。粗估一局约 62 s / 470 步，80k 步 ≈ 2.9 h（⏳ 以完整试跑的 `summary.json` 为准；agent 变强后每局更长，reset 占比会下降）。死亡时玩家 HP 停在 **1**（不是 0），致命一击的伤害值总是被截断。

## 6. 下一步（按顺序）

1. **跑完整试跑**：游戏开着、状态和上次试跑开始时一样，双击 `TEST3_pilot.bat`，看到提示后点一下 Dead Cells 窗口。约 1 分钟 selftest + 约 7 分钟试跑。检查：selftest 通过；多局都能正常 reset；每局 `player_hp_max` = 15482、`boss_hp_max` = 205238；输的局 `boss_hp_frac_end` 不是 0；有没有 `glitch`；`summary.json` 的 `wall_time_h` × 80000/3000 ≈ 每次训练小时数
2. （建议）在 `train_ablation.py` 开头加 **preflight check**：读 3 秒 HP，玩家 HP 不合理或 boss HP 异常就直接退出；训练中连续 N 局 1–3 步就结束、或 glitch 太多则自动 abort。防止以后游戏更新导致 offset 再变时白跑一晚
3. **跑完整队列**，数据备份到云盘（上次换电脑丢过数据）。跑的期间写 Direction C 的 delay injection（`evaluate.py` 加 `--action-delay`）
4. 跑 `analyze.py`、`analyze_damage.py`；看 `analysis/damage_clusters/` 里的截图，在 `attack_labels.json` 里给 boss 招式命名后再跑一次
5. **写作**：Intro、Related Work、Environment、Setup 可以边跑实验边写；初稿交给 Pengcheng 改。用 Overleaf + AAAI LaTeX 模板

## 7. Git 状态

- 已 commit：`10dfaaf Fix HP pointer base, read max HP at reset, add chain probe`。**还没 push**，需要自己在终端运行 `git push -u origin paper-experiments`
- **未 commit（应该 commit）**：speedhack 相关（`speedhack.py`、`speedhack_dll/dqn_speedhack.lua`、`train.py`/`run.py` 的 `GAME_SPEED`）、`EXPERIMENTS.md`、`HANDOFF.md`、`README.md`、`selftest.py`、`SETUP_env.bat`、`requirements.txt`、`TEST1_env.bat`、`TEST2_speed.bat`；10-25 新改的 `ablation_common.py`、`train_ablation.py`、`evaluate.py`、`TEST3_pilot.bat`（建议等 selftest 通过后再 commit）
- 不要 commit：`lib/Actions.py`、`lib/SendKey.py`、`run.py`/`train.py` 中只有行尾符（CRLF/LF）的差异；`__pycache__/*.pyc`；`_to_delete/`；`runs_pilot/`
- ⏳ `git status` 显示旧的 `checkpoints/*.pt`、`*.npy`（被 git 跟踪的旧版本模型）已从磁盘删掉，`checkpoints/` 文件夹不存在了。如果是有意删的，可以 commit 这个删除；否则用 `git checkout -- checkpoints` 恢复
- 可清理：`speedhack_dll/` 里旧的 `dqn_speedhack32/64.dll`（已不用）；CE `autorun\` 里一个 0 字节的 `.wtest` 文件；`runs_pilot/speed`（第一次不完整的试跑）
- 从远程 shell 执行 git 会留下 `.git/index.lock`，git 操作最好自己在 Windows 终端做

## 8. 注意事项

- `lib/env.py` 的 reset 时间是按 2x 调的，训练**全程保持 2x**
- 论文要写明游戏速度：2x 下 100 ms 的控制步 = 200 ms 游戏时间，等于改变了 agent 的反应时间
- 死亡时玩家 HP 停在 1，致命一击的伤害值被截断，`analyze_damage.py` 会把这类死因标为 `ambiguous`，需要看 `fatal_blows.png` 人工判断（或按招式聚类用前面非致命的同类伤害推断）
- 所有 config、所有 seed 都要用同一个玩家血量（不要再用 CE 改血量），`player_hp_max` 列可以检查
- seed 只固定了网络初始化和探索，游戏本身不确定
- 游戏更新可能再次改变 `libhl.dll` 的 base：先用 `probe_hp_chains.py` 检查，通常只需改 `HP_BASE_OFFSET`
