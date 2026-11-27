# SkillOpt 二次进化

第一阶段的一个已结束的运行目录是唯一的 skill 来源。第二阶段用固定 train 检查回退，用固定 val 选优，从源运行读取 dev 答卷、skill、编辑和历史。每个配置单独建种群。已有的 TriviaQA、HotpotQA、SearchQA 使用 SkillOpt 的 QA adapter；BIRD 使用原生 SQL 执行评分。

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
.venv/bin/python -u -m scripts.evolution_test.pipeline --source "$source050" --strategy matrix
.venv/bin/python -u -m scripts.evolution_test.pipeline --source "$source075" --strategy matrix
```

结果和源运行放在同一层。命名示例为 `outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_<北京时间戳>_evo2_rho050/`。对照实验可用同一 `--source`、同一 `--seed`，将 `--strategy` 换成 `random`。断点恢复使用 `--resume <已有二次进化目录>`，程序检查输入和实现指纹，恢复已完成的生成、验证及其计量记录。

## 每一代的行为

- 先按文本哈希去重，在完整 dev 答卷中选分数最高的最多五个。出现同分时优先保留原生 best，然后按归档的稳定顺序选。初始、最终及各步完整版本均可入选。
- 矩阵策略在一代开始时冻结父本、全局 best、历史答卷和难题顺序。针对父本 X，A、B 必须是两个不同的历史落选文本，且都在同一道 X 答错的题上实际答对。优先考虑 best 也答错的题。没有合适的成功供体时不拿失败版本充当 B。
- 生成器比较实际送给答题模型的原始题面和文段、可见回复、EM 和 skill 文本。它需要引用 A、B 各自体现同一做法的规则，引用真实输入中的证据，再指出 X 缺少这条规则，或有哪条现有指令和它冲突。归档只含最终回答时不补造内部推理。
- 每个矩阵提案先归因为“共同规则已在 X”“没有证据”“缺少一步”或“X 中两句指令冲突”。已有且没有冲突、无证据的组合要换对。若判为冲突，必须在同一目标题上指出两句不同且互不重叠的 X 指令，给出两种指令各引出的回答及依据，只替换引起错误的那一句。缺少步骤时仅在一个精确定位的位置补规则。诊断及引用记录在每次供体尝试的 `proposal.json`。
- 子代进入 dev 前，在独立于 dev 划分的训练题上与 X 配对作答。选 4 道与目标相近的题，另选 4 道与保护条件相近、4 道分散的题，后两组须由 X 实际答对。训练题根据修复/保护线索和题干词汇重合选取；剔除与 dev 同 ID、同规范化题干及与展示例题共享参考答案的题。每次出现父对、子错，就对这些题用新调用分别复答父子；同一题两轮均退步则放弃该编辑，换供体补位。这个小样本检查只用于筛掉有重复退步的编辑。每次检验的题目、逐题作答和结果保存在 `donor_attempts/NN/train_check/`，相关费用单列为 `train_check`。
- 每个父本要生成一个实际不同的新 skill。供体组合解释不了证据、补丁无效或文本重复，就在该位置换新的供体组合。同一位置重选时排除已经用过的无序供体组合。若所有可用组合耗尽仍未填满，整代记为未完成并报错，不能计为停滞或触发 test。当前 TriviaQA 两组的种群均为 5。
- 全代子代文本确定后才进入 dev 评估。每批 4 题，用自己的父本作为严格接受门槛，选优时打平保留父本。每代结束统一更新种群和全局 best。默认最多 3 代；连续 2 个**完成**的代无人替换时提前结束。使用上述 pipeline 命令时，只有进化成功结束才运行 1,000 题 test，成本写入该运行的 `test/`。

## 查看结果与成本

| 文件 | 可以核对的内容 |
|---|---|
| `manifest.json` | 源运行绝对路径、模型、种子、生成参数、Git 提交、dev 划分和源轨迹指纹 |
| `initial_population.json` | 完整答卷数量、去重数量、入选和未入选名单、排序、缺额、被 5 的上限截掉的个数 |
| `archive_audit.json`、`matrix.tsv` | 源轨迹缺少新答卷的步骤；矩阵列为固定 dev ID，1/0 表示正常评估，None 表示没有记录，ERR 表示执行失败 |
| `generations/NNN/generation_inputs.json`、`frozen.json` | 各位置最初的供体计划，以及填满全部子代后实际采用的供体、父本、best 和 dev 题序 |
| `slot_XX/donor_attempts/NN/`、`proposal.json`、`child_skill.md` | 每次更换的 A/B、实际输入文段和可见回答、原始模型输出、共同规则和补丁；最终采用的编辑与子代全文 |
| `slot_XX/evaluation_inputs.json`、`evaluation_attempts/`、`decision.json` | 实评题 ID、父本门槛、缓存来源、逐批上界、失败与重试、修复和退步题（含生成器已见例题之外的变化） |
| 各供体尝试和验证的 `token_events/`、每代 `summary.json` | 单个生成请求或 dev 批次的起止时间与 token，以及按位置、按代汇总 |
| 根目录 `summary.json`、`best_skill.md` | 每代替换数、最终 best、相对初始 best 多答对几题、停止原因、总消耗；异常退出时另存 `last_failure.json` |

`summary.json → incremental_usage` 将 **generation**、**train_check** 和 **validation** 分开，每项都有 `input_tokens`、`output_tokens`、`total_tokens` 和 `calls`。其中 `calls` 统计后端返回并记账的调用；重试成功产生的用量亦纳入，对同一份缓存重复读取不会新增用量。底层供应商未返回的用量不能从日志还原：进程被强制中断的事件保留为 `unfinished_events`，没有账面 token 的调用事件列入 `zero_reported_token_events`。账单核对需考虑这些条目。

`native_step_usage` 单列第一阶段 `history.json` 各训练步已有的输入/输出 token；该口径不含步外的初始化或 epoch 末工作。只比较本次二次进化额外花了多少，用 `incremental_usage.total`。

`native_steps_plus_evolution_usage` 给出上述第一阶段逐步成本与本次二次进化的合计，分别列出输入、输出和总 token。

## 历史实验：固定旧的一对一错供体对照（2026-11-26）

该次历史对照的提示词给生成器一个虚构的维修记录示例，演示如何从多条编辑中找出能解释作答差异的那一条，并用反例限定它的适用范围。每个实际展示的 target/risk 题都要留下判断：新编辑会改变哪个选择，或者这条规则解释不了该题。解释不了任何 target 时允许返回 `skip_pair`。程序核对原文引用、编辑位置和声明的答案方向；候选的实际成绩由独立的答题调用给出。

该实验固定 TriviaQA 首次 `p=0.75` 二次进化的 10 个位置。每个位置使用原父本、全局 best、A/B、题目和原来的 dev 排序。同一代先前位置的变异说明也取自旧请求。每个位置独立对照，第一代新生成的子代不进入第二代的父本。这里只使用 dev。

在 CPU 的 SkillOpt 仓库根目录，使用相同的 API 环境运行：

```bash
prior=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_rho075
.venv/bin/python -B -u -m skillopt_evolution.replay_prompt --prior-run "$prior"
```

比较分母固定为旧子代实际做过的 15 次 target 和 5 次 risk（按“题 × 子代”计）。若新子代在其中某题之前提前淘汰，独立补测该题，仅用于成对比较，补测分数不进入 dev 选优。主动跳过和无效输出分别核对；沿用父本的答案不作为新子代的实测结果。运行被打断时用 `--resume <本次回放目录>` 续跑。
