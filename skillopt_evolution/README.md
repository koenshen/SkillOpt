# SkillOpt 二次进化

第一阶段的一个已结束的运行目录是唯一的 skill 来源。第二阶段只读取对应 benchmark 的固定 val 划分及源目录中的 dev 答卷、skill、编辑和历史。每个配置单独建种群。已有的 TriviaQA、HotpotQA、SearchQA 使用 SkillOpt 的 QA adapter；BIRD 使用原生 SQL 执行评分。

## TriviaQA 复现命令

在 CPU 的 SkillOpt 仓库根目录执行。以下只读检查会验证逐题文件、原难题顺序和早停批次，打印初始个体与第一代 A/B，不调用模型、不创建结果目录。

```bash
cd /mnt/sdd3/xuancheng/code/github/SkillOpt
source050=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050
source075=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
.venv/bin/python -B -m skillopt_evolution.run --source "$source050" --inspect
.venv/bin/python -B -m skillopt_evolution.run --source "$source075" --inspect
```

正式运行使用已配置的 OpenAI-compatible 服务，优化模型与答题模型沿用该来源归档中的设置。下面的两条命令各自创建独立实验；需要并行时在两个终端或 tmux 中分别执行。请提前在运行环境中设置 `OPENAI_COMPATIBLE_BASE_URL` 和 `OPENAI_COMPATIBLE_API_KEY`。

```bash
.venv/bin/python -u -m skillopt_evolution.run --source "$source050" --strategy matrix
.venv/bin/python -u -m skillopt_evolution.run --source "$source075" --strategy matrix
```

结果和源运行放在同一层。命名示例为 `outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_<北京时间戳>_evo2_rho050/`。对照实验可用同一 `--source`、同一 `--seed`，将 `--strategy` 换成 `random`。断点恢复使用 `--resume <已有二次进化目录>`，程序检查输入和实现指纹，恢复已完成的生成、验证及其计量记录。

## 每一代的行为

- 先按文本哈希去重，在完整 dev 答卷中选分数最高的最多五个。出现同分时优先保留原生 best，然后按归档的稳定顺序选。初始、最终及各步完整版本均可入选。
- 固定本代父本、best 和难题顺序。矩阵模式要求父本 X 的某道错题有至少两个不同落选文本实际答对。优先使用能修复全局 best 弱点的 A，以及和 A 同父、实际做过相同题但答错的 B；优先尝试尚未使用的组合。缺少有效对照时回退到从种群随机抽不同文本的 A、B；供体不足会记录跳过原因。
- LLM 读取 A、B 的差异，提取变异，把它融入当代 best 的处理方式，再用局部编辑与 X 交叉。日志保留引用的原文、索引、差异解释、预期修复和退步。程序校验引用文本和 patch 的实际落点。无效输出有上限地重试，全部失败会留在对应位置的记录里。
- 每个种群位置每代最多产生一个参加验证的新 skill。所有文本在验证前冻结。不同位置串行评分，单个批次最多同时答 4 道题。每个子代以自己父本为门槛，完整分数严格更高才替换，打平保留父本。相同文本在同一实验内可使用已有逐题答卷；拿到了更低的父本门槛时，会重新计算上界，仍有胜算就补答。排序历史只追加新观测。
- 默认最多 3 代、连续 2 代无人被替换时结束，可用 `--generations`、`--patience` 调整。参数、缺额和跳过原因随本次运行保存，失败的模型调用与正常验证淘汰分别记录。

## 查看结果与成本

| 文件 | 可以核对的内容 |
|---|---|
| `manifest.json` | 源运行绝对路径、模型、种子、生成参数、Git 提交、dev 划分和源轨迹指纹 |
| `initial_population.json` | 完整答卷数量、去重数量、入选和未入选名单、排序、缺额、被 5 的上限截掉的个数 |
| `archive_audit.json`、`matrix.tsv` | 源轨迹缺少新答卷的步骤；矩阵列为固定 dev ID，1/0 表示正常评估，None 表示没有记录，ERR 表示执行失败 |
| `generations/NNN/frozen.json` | 整代的父本、全局 best、dev 顺序和每个位置选择 A/B 的原因、支持题及反例 |
| `slot_XX/generation_attempts/`、`proposal.json`、`child_skill.md` | 每次发给模型的提示、原始输出、引用及 patch 校验、最终实际采用的编辑和子代全文 |
| `slot_XX/evaluation_inputs.json`、`evaluation_attempts/`、`decision.json` | 实评题 ID、父本门槛、缓存来源、逐批上界、失败与重试、修复和退步题（含生成器已见例题之外的变化） |
| `slot_XX/token_events/`、每代 `summary.json` | 单个生成请求或 dev 批次的起止时间与 token，以及按位置、按代汇总 |
| 根目录 `summary.json`、`best_skill.md` | 每代替换数、最终 best、相对初始 best 多答对几题、停止原因、总消耗；异常退出时另存 `last_failure.json` |

`summary.json → incremental_usage` 将 **generation** 和 **validation** 分开，每项都有 `input_tokens`、`output_tokens`、`total_tokens` 和 `calls`。其中 `calls` 统计后端返回并记账的调用；重试成功产生的用量亦纳入，对同一份缓存重复读取不会新增用量。底层供应商未返回的用量不能从日志还原：进程被强制中断的事件保留为 `unfinished_events`，没有账面 token 的调用事件列入 `zero_reported_token_events`。账单核对需考虑这些条目。

`native_step_usage` 单列第一阶段 `history.json` 各训练步已有的输入/输出 token；该口径不含步外的初始化或 epoch 末工作。只比较本次二次进化额外花了多少，用 `incremental_usage.total`。

`native_steps_plus_evolution_usage` 给出上述第一阶段逐步成本与本次二次进化的合计，分别列出输入、输出和总 token。

## 用旧的父本和供体对照两版生成提示词

矩阵模式给生成器一个虚构的维修记录示例，演示如何从多条编辑中找出能解释作答差异的那一条，并用反例限定它的适用范围。每个实际展示的 target/risk 题都要留下判断：新编辑会改变哪个选择，或者这条规则解释不了该题。解释不了任何 target 时允许返回 `skip_pair`。程序核对原文引用、编辑位置和声明的答案方向；候选的实际成绩由独立的答题调用给出。

本轮固定 TriviaQA 首次 `p=0.75` 二次进化的 10 个位置。每个位置使用原父本、全局 best、A/B、题目和原来的 dev 排序。同一代先前位置的变异说明也取自旧请求。每个位置独立对照，第一代新生成的子代不进入第二代的父本。这里只使用 dev。

在 CPU 的 SkillOpt 仓库根目录，使用相同的 API 环境运行：

```bash
prior=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_rho075
.venv/bin/python -B -u -m skillopt_evolution.replay_prompt --prior-run "$prior"
```

比较分母固定为旧子代实际做过的 15 次 target 和 5 次 risk（按“题 × 子代”计）。若新子代在其中某题之前提前淘汰，独立补测该题，仅用于成对比较，补测分数不进入 dev 选优。主动跳过和无效输出分别核对；沿用父本的答案不作为新子代的实测结果。运行被打断时用 `--resume <本次回放目录>` 续跑。
