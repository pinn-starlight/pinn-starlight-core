# pinn-starlight-core — 给 DSH 的工作区约定

本文件是 DeepSeek Harness 在本仓库的工作指令，**优先级高于任何自动生成的计划**。
这是一个处于论文/答辩阶段的研究仓库：不追求"更现代"，也不允许为了跑通或好看而改动实验协议。

---

## 一、铁律

### 1. 只有一条工作分支 `stage2_thu`，禁止碰 `main`

- `main` 是基线（当前 `e8eb520`）。**所有改动都在 `stage2_thu` 上做，不新建任何分支。**
- 历史上按任务开 `dsh/*` 分支导致分支堆积、进度错位，已废弃这个做法；`dsh/` 前缀不再使用。
- 动手前先 `git branch --show-current`；不是 `stage2_thu` 就先切过去，**不要拉新分支**。
- 需要保留某次尝试时，用 commit 或 tag，不要用分支。
- 改完停在工作区并报告：改动文件清单 / `git diff main --stat` / 验证结果。
- 代码留在工作区：不 stash、不切回 `main`、不用 git worktree。

### 2. 禁止 AI 合并、推送、改写历史

禁止执行：`git merge`、`git rebase`、`git cherry-pick`、`git push`、`git reset --hard`、`git clean -fd`、强推、任何改 `main` 引用的操作。

合并由用户在 PyCharm 或命令行完成。用户说"弄好了"意思是**看 diff**，不是"帮我合了"；有歧义就只汇报，不动 git 历史。

### 3. 科学协议优先于代码整洁

- test 只在 Global Configuration Lock 冻结之后打开；选择/调参只用 train / validation。
- 不改随机种子（20260728 / 20260729 / 20260730）、执行预算、候选空间、split 规则来"改善"结果。
- 不改 `experiments/outputs/`、`paper_figures/` 里已冻结的数字与图；要重算必须走脚本。
- 重构必须数值等价：方法代码改了就跑 smoke test；涉及指标的改动要说明差多少、为什么。

### 4. 不碰生成物与本地状态

`.venv/`、`.idea/`、`.agents/`、`.dsh*/`、`__pycache__/`、`*.lock`（依赖升级需用户明确要求）、`artifacts/`、`collections/`、`test/`、`experiments/outputs/`。

不做顺手重构、不整仓格式化、不改无关文件。报告里不要把这些当成"顺手清理"的成果。

---

## 二、环境与验证（本机没有 GPU）

- Python 3.11，用仓库内 `.venv/bin/python`；系统 `python` 是 3.14 且没装依赖。
- 本机 `nvidia-smi` 不存在，`torch.cuda.is_available() == False`：Stage-A / Stage-B / CV5 / E4 的正式路径**跑不了**。
  不要为了"本地能跑"删掉 `_formal_device()` 的 CUDA 检查，也不要悄悄换成 CPU 预算。
- 验证分级：
  1. `.venv/bin/python -m tools.smoke_test`：17 项，CPU 约 25 秒，**退出码 0 才算过**。
  2. 数据缺失时：`--only imports,laplacian,generators,inventory`。
  3. `.venv/bin/ruff check .` 和 `.venv/bin/ruff format --check .`（line-length 100）。
  4. 仓库**没有 pytest 用例**，`test/` 是数据目录不是测试目录；也没有 CI。
- 数据缺失跑 `bash tools/fetch_smoke_data.sh`（从 `artifacts/` 解出约 382 MB）；不要伪造数据绕过检查。
- 只读沙箱下 matplotlib 会因没有可写缓存目录报 `OSError`，把 `MPLCONFIGDIR` 指到可写目录即可。

---

## 三、目录与新增代码放哪

| 路径 | 职责 |
| --- | --- |
| `src/pinn_starlight_core/` | 可安装核心包：MLP、损失、α / forcing、图像 IO；**不依赖 experiments** |
| `experiments/configs/` | 配置唯一来源：方法登记、种子、候选空间、预算、split |
| `experiments/common/{baselines,data,utils}/` | 四种方法、数据与背景、指标与实验基础设施 |
| `experiments/scripts/<stage>/` | 各阶段入口，经 `protocol/run.py` 或 `run_stage2.py` 编排 |
| `experiments/scripts/diagnostics/` | 非正式机制探针；只读 validation，输出到 `experiments/outputs/diagnostics/` |
| `experiments/cli.py`、`experiments/common/pipeline.py` | `pinn-starlight` 演示/发布 CLI；**不接入正式入口** |
| `scripts/figures/` | 论文图 3/4/6 的收集与渲染（只读冻结结果） |
| `tools/` | smoke test、单图演示、服务器归档恢复/重建/回填 |
| `research_diagnostics/` | 冻结的探索性诊断，不属于正式主流程 |
| `collections/`、`test/` | 数据根（git 忽略） |
| `artifacts/` | 本地大文件归档，只有 README 入库 |

依赖方向单向：`run_stage2.py → experiments/scripts → experiments/{configs,common} → src/pinn_starlight_core`。

新增代码的位置约定：

- 新方法 → `experiments/common/baselines/`，并在 `experiments/configs/formal_experiment.py` 登记；
- 新指标 → `experiments/common/utils/metrics.py` 或 `scores.py`；
- 新阶段 → `experiments/scripts/<stage>/`，通过 `protocol/run.py` 或 `run_stage2.py` 编排；
- 机制探针 → `experiments/scripts/diagnostics/`，只读 validation，不得改动正式 config/lock；
- CLI / 演示 → `experiments/cli.py` + `experiments/common/pipeline.py`，不要塞进 `run_stage2.py`；
- 一次性脚本 / 出图 → `scripts/figures/`；搬运与复现 → `tools/`；别往仓库根目录堆；
- 常量只在 `experiments/configs/`，不要在脚本里复制第二份。

---

## 四、改代码的流程

1. `git status` 确认干净；有未提交改动先问用户，不自己提交或丢弃。
2. 确认在 `stage2_thu` 上；**不要新建分支**，直接在当前 HEAD 上改。
3. 动手前先给影响面：协议、数据、GPU、输出格式、已有数字会不会变；大功能先出方案再写。
4. 小步改，跑 smoke test 相关子集（能全量就全量）。
5. 中文 conventional commit：`feat(scope): ...` / `fix` / `refactor` / `docs` / `style` / `chore`，只提交本任务相关文件。
   **每个提交必须标明作者 AI 的模型**：在提交信息末尾空一行，加 `Model: <模型名>`（如 `Model: claude-fable-5-2`）；模型名取自系统提示里的真实标识，不确定就写 `Model: unknown`，不要猜。多个 AI 先后改同一分支时，各自的提交各自标注。
6. 汇报分支、文件清单、验证结果、风险；合并留给用户。

---

## 五、回答风格

- 用中文；**先给结论和实际改动**，再补说明；默认写短。
- 需求不明确时最多先问一个问题，问完就动手。
- 不复述推理过程，不为显而易见的事反复读文件。
- 不编数字；没验证的事说"没验证"。
- 不用 `present` 展示源码文件，只在交付文档 / 图片 / 表格时用。

---

## 六、已知边界（扩展功能前先看）

- **本机无 GPU**：只能验证 CPU 路径（smoke test、`tools/run_one_image.py`），正式指标无法本地复现。
- **E4 原始图像不在本地**：只有 16 张输出归档，E4 不能重跑。
- **表 VII 机制实验脚本未入库**：只有归档 CSV；`research_diagnostics/pinn/pinn_preservation_frozen.py` 是同一条研究线的实现。
- **`docs/` 被 `.gitignore` 忽略**：`CODE_MAP.md`、`CODE_CHECK_PLAYBOOK.md` 只在本地。
- **数据体积大**：正式合成数据约 10.85 GB，`artifacts/` 约 9.6 GB，都不入库。
- **文档计数以代码为准**：smoke 现在是 17 项、论文方法 4 个（`coordinate_pinn`、`sgc_pinn`、`fft_gaussian`、`cosine_ls`），启动器只有 `run_stage2.sh`；改工具后顺手更新 README / PROJECT_STRUCTURE。

---

## 七、merge 流程（用户执行，AI 只讲解）

1. `git diff main stage2_thu --stat`，再逐行看；确认没有意外删除和夹带文件。
2. `git checkout main`——合并永远是"把 X 合进我现在所在的分支"。
3. `git merge stage2_thu`；用 Merge，不用 Rebase。
4. `git ls-tree -r --name-only HEAD -- src/ experiments/` 确认文件真的在。

冲突：手工删掉 `<<<<<<<` / `=======` / `>>>>>>>` 标记留想要的一边，`git add` 后 `git commit`；PyCharm 三栏窗口更直观。
