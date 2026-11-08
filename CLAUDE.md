# Claude 工作规则（本项目）

## 服务器操作规则（必须遵守）

- **不准直接修改服务器上的任何内容**：不得在服务器上新建、编辑、上传文件（包括 scp、heredoc 写文件、/tmp 临时脚本、~/ 下的配置文件）。
- 所有代码改动必须在本地 Mac 的仓库（`/Users/xuancheng/pycharm_code_261106/SkillOpt`）中完成，流程：
  1. 先 `git fetch`，确保本地与 `origin/main` 同步、是最新代码；
  2. 本地修改并 `git commit`（保证 git 有记录可查），`git push origin main`；
  3. 再到服务器上 `git pull --ff-only` 更新。
- 所有远程操作都在本地 Mac 上通过 `ssh cpu ...` / `ssh gpu ...` 执行。只读分析脚本通过标准输入传给远程解释器（如 `ssh cpu 'python3 - <<EOF ... EOF'`），不要上传文件。
- 不在服务器上存放 API key，不擅自启动实验；启动、停止作业或删除结果前先征得用户同意。

## 服务器信息

- 代码与结果：`ssh cpu`，路径 `/mnt/sdd3/xuancheng/code/github/SkillOpt`，实验结果在 `outputs/`。
- 忽略 `outputs/261008-0400-backup/` 下的旧结果，只看 `outputs/` 根目录下的 run。
