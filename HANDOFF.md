# DQN_deadcells → Workshop Paper：Project 交接总结

> 更新于 2026-10-25。作者：Michael Jin（UC Irvine）。仓库：github.com/Inao241/DQN_deadcells，本地 `C:\code\DQN_deadcell\DQN_deadcells`，工作分支 **`paper-experiments`**。

## 1. 目标与背景

- 把自己写的 Dead Cells boss 战 DQN 项目（boss：The Concierge / 看守者）做成一篇 **workshop 短论文**（4–6 页）。
- 合作者：**Pengcheng Xu**（Prof. Jing Zhang 实验室的 PhD 学生）。他负责把文章改成学术写法，并选具体投哪个 workshop。他提到 **AAAI 2027 workshop 截止日期 11/20**（各 workshop 的 DDL 不同，以具体 CFP 为准）。他给的写作参考：lmgame-Bench（ICLR 2026，https://openreview.net/forum?id=qeziG97WUZ）。
- 备选 venue：FDG 2027 Late-Breaking Work（约 2027/3/8）、IEEE CoG 2027 auxiliary paper（约 2027/5/15）、ICLR 2027 workshops（约 2/1）。投稿前都要核实官网日期。

## 2. 论文定位

**主线 = Direction A（实证研究）+ 一个小规模 Direction C（延迟）实验作为亮点。**

- 研究问题：Rainbow 系列组件（Double DQN、PER、Dueling）的结论来自 Atari，而 Atari 环境会等待 agent、能跑上亿帧。在**实时、交互样本很少**的商业游戏里，这些结论还成立吗？
- Direction C（亮点）：DQN 假设动作立刻在 $s_t$ 上执行，但实时游戏里截屏、推理、按键都有延迟，等动作生效时状态已经是 $s_{t+d}$，Markov 性被破坏。游戏速度（1x / 2x）可以作为天然的延迟旋钮。盾反、翻滚这类帧敏感的动作应该最先受影响。
- Direction B（通用 2D 游戏 template / benchmark）留作下一篇。可复现性（要买游戏、Windows、读内存）是它最大的短板。

**审稿人会问的问题（来自 lmgame-Bench 的 OpenReview 意见）**，要提前准备：
1. novelty：要主动讨论哪些结论和 Atari 上的 Rainbow 一致、哪些不一致，以及原因
2. 方差和 seed 数：报告置信区间，不能只给一个数
3. 失败分析要量化：按 boss 招式统计死因和受伤原因（代码已支持）
4. Related Work 紧跟 Introduction；指标单独一节写清楚
5. 写明算力和时间成本

**论文结构**：Intro → Related Work（DQN 系列；游戏 AI / 视觉 RL）→ Environment（MDP 形式化：observation、action、reward 公式，加系统框图）→ Experimental Setup → Results（总表、learning curve、ablation、死因分析）→ Limitations & Conclusion。

**参考文献**：Rainbow（Hessel et al., AAAI 2018）、Revisiting Rainbow（Obando-Ceron & Castro, ICML 2021）、Deep RL that Matters（Henderson et al., AAAI 2018）、Statistical Precipice / rliable（Agarwal et al., NeurIPS 2021）、ViZDoom（Kempka et al., CIG 2016）、Action Space Shaping（Kanervisto et al., CoG 2020）、CS:GO behavioural cloning（Pearce & Zhu, CoG 2022）、Real-Time RL（Ramstedt & Pal, NeurIPS 2019）、lmgame-Bench（ICLR 2026）。

## 3. 代码现状（分支 `paper-experiments`）

原来的 `train.py`、`run.py` 没有改动。新增：

| 文件 | 作用 |
|---|---|
| `ablation_common.py` | 可开关的 Double / PER / Dueling、日志、受伤事件记录 |
| `train_ablation.py` | 训练一个 config × seed，存到独立文件夹（`config.json`、`episodes.csv`、`damage_events.csv`、checkpoint） |
| `evaluate.py` | 关闭探索的评估；`--random`、`--human` baseline |
| `run_experiments.py` | 无人值守跑完整队列，已完成的自动跳过，中断后重跑同一命令即可 |
| `analyze.py` | 结果表（含 LaTeX）、learning curve、hierarchical bootstrap CI、Glass's δ、延迟分解、算力成本 |
| `analyze_damage.py` | 受伤时 agent 在做什么、被哪个 boss 招式打中（按伤害值聚类）、死因、战斗阶段；生成每类招式的截图，让人命名 |
| `selftest.py` | 用假游戏测试整个流程，不需要开 Dead Cells |
| `speedhack.py` + `speedhack_dll/` | 不用 Cheat Engine GUI 的 2x 加速 |
| `EXPERIMENTS.md` | 使用说明 |
| `TEST1_env.bat`、`TEST2_speed.bat` | 测试脚本，双击运行，输出到 `test_logs/` |

`model.py`：增加了 `dueling=True/False` 参数（默认 True，旧 checkpoint 仍能加载）。

**相对原 `train.py` 修复的问题**（论文里描述的是修复后的正确实现）：
- **Double DQN 有 bug**：原代码用 `net(state)` 在当前状态 $s$ 上选 $a'$，正确做法是 $a' = \arg\max_a Q_{online}(s', a)$。已修复，selftest 会数值验证 target 公式。
- move head 在 $s'$ 的 target 改为以"$s'$ 上策略会选的 action"为条件（原来用的是 $s$ 上执行的 action）。
- PER 两个 head 用同一个 priority 公式；第一帧在 reset 之后再截取；开局 3 步内的"胜利"（boss HP 读取失败）记为 `glitch`，不用于学习。
- **旧结果（100% 通关、350 局赢 56 次）来自有 bug 的版本，不要再引用。**

**实验设计**：
- 配置：`dqn`、`double`、`double_per`、`double_per_dueling`（可选再加 `double_dueling` 做 2×2）
- seed：0、1、2（按 seed 顺序跑，没跑完也有完整的对比）
- 每次训练 **80k 步**。依据：旧 checkpoint 的训练记录显示约 4 万步开始稳定获胜，约 10 万步后趋于平稳
- 评估：`final.pt`，greedy，每个 seed 20 局；另有 random baseline 和 human baseline（你自己打 20 局，1x 速度）
- 命令：`python run_experiments.py --plan`，然后 `python run_experiments.py --max-steps 80000 --seeds 0 1 2`，最后 `python analyze.py && python analyze_damage.py`

## 4. 电脑环境

- Anaconda Python 3.13.9，torch 2.11.0+cu128，**RTX 5080**，CUDA 正常
- 已安装：`dxcam`、`keyboard`、`pygetwindow`
- `tensorboard` 已卸载（代码里是可选的，所有关键数据都在 CSV 里）。安装它会把 protobuf 升到 7.x，破坏 Anaconda 里的 streamlit、langflow 等包；已降回 protobuf 5.29.3
- Anaconda 的 numpy 和 torch 有 OpenMP 冲突（`OMP: Error #15`），代码和 bat 里已设置 `KMP_DUPLICATE_LIB_OK=TRUE`
- Dead Cells 是 **32 位**进程，Python 是 64 位（speedhack 已处理这种情况）
- 电脑装有 Riot Vanguard，可能干扰 CE 的调试功能

## 5. 测试状态

| 项目 | 状态 |
|---|---|
| selftest（GPU） | ✅ 全部通过。一次训练更新约 15.7 ms |
| 读 HP | ❌ **玩家和 boss 都读到 0。** `[libhl.dll+48184]` 能读到有效指针（0x50E0000），但 `+4B0` 之后是 0。大概率是游戏更新后中间几层 offset 变了 |
| speedhack v1 | ❌ 注入后约 1 秒游戏崩溃（改写了所有模块的 import table，太激进） |
| speedhack v2 | ⏳ **已编译，未测试**。照搬 CE SpeedhackV3 的做法：只在 `ntdll!RtlQueryPerformanceCounter`、`kernel32!GetTickCount64`、`GetTickCount` 开头写 jmp；替换函数不调用原函数（QPC 用 `NtQueryPerformanceCounter`，tick 读 KUSER_SHARED_DATA） |
| 窗口匹配 bug | ✅ 已修复：`getWindowsWithTitle` 是子串匹配，会选到 `...\Dead Cells\deadcells.exe` 窗口，改为精确匹配（原 `train.py` 也有这个问题） |
| reset 流程、完整训练 | ⏳ 未测试（依赖 HP 读取） |

## 6. 下一步（按顺序）

1. **修 HP 指针**（正在做，`health_address/` 里是 CE 的 pointer scan 结果）
   - pointer scan 设置：max level 6，max offset 0x1000，**"Pointers must end with specific offsets" = `12C`**（旧玩家链和 boss 链最后都是 +12C，是 Entity 类的 HP 字段）
   - 优先选起点是 `libhl.dll+48184` 的路径；多次重开或重启游戏后用 "Rescan memory" 淘汰不稳定的路径
   - **boss HP 要在 boss 战里找**，rescan 时也要每次都进 boss 房
   - "Find out what writes to this address" 一用就崩：换成 VEH Debugger（Settings → Debugger Options），或者直接跳过（12C 大概率没变）
   - 找到后更新 `lib/GetHp.py` 里的 offset；检查满血数值，必要时更新 `ablation_common.py` 里的 `PLAYER_MAX_HP = 15482`、`BOSS_MAX_HP = 234930`
   - 如果 rescan 后还剩很多条：导出结果，写脚本在进 boss 房后自动测试每条链，挑出读数合理的那条
2. **测 speedhack v2**：重启游戏、站在起点，双击 `TEST2_speed.bat`。脚本会"1x 左移 0.6 秒 → 2x 右移 0.6 秒 → 走回去"，截图存到 `test_logs/`。如果 2x 生效，角色最后会在起点**右边**。目测确认游戏确实是 2 倍速。还崩的话就回退到手动开 CE（`--game-speed 0`）
3. **试跑**：`python train_ablation.py --run-dir runs_pilot/speed --max-steps 3000`。确认死亡后 reset 正常，并测速度（`summary.json` 的 `wall_time_h` × 80000/3000 ≈ 每次训练的小时数）
4. **跑完整队列**，数据备份到云盘（上次换电脑丢过数据）
5. 跑 `analyze.py`、`analyze_damage.py`；看 `analysis/damage_clusters/` 里的截图，在 `attack_labels.json` 里给 boss 招式命名后再跑一次
6. **写作**：Intro、Related Work、Environment、Setup 可以边跑实验边写；初稿交给 Pengcheng 改。用 Overleaf + AAAI LaTeX 模板

## 7. 注意事项

- `lib/env.py` 的 reset 时间是按 2x 调的，训练**全程保持 2x**
- 论文要写明游戏速度：2x 下 100 ms 的控制步 = 200 ms 游戏时间，等于改变了 agent 的反应时间
- 如果游戏把 HP 下限截到 0，致命一击的伤害值会被截断，`analyze_damage.py` 会把这类死因标为 `ambiguous`，需要看 `fatal_blows.png` 人工判断
- seed 只固定了网络初始化和探索，游戏本身不确定
- 记得 `git push -u origin paper-experiments` 备份代码
- 2026-10-25 的改动（tensorboard 可选、OpenMP、窗口匹配、speedhack v2、测试脚本、本文件）都已提交到 `paper-experiments`
