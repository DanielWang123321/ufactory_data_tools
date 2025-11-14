# LeRobot 0.4.1 兼容性开发计划

## 背景与目标
- 现有 `ufactory_data_tools` 管线只针对 LeRobot 0.3.x / v2.1 数据集格式设计，文档中也明确要求固定版本，并警告不要使用 0.3.4（`README.md:14`, `README.md:139`, `TRAINING_GUIDE.md:1`）。
- LeRobot 0.4.1 将底层数据集规范升级为 v3.0（`C:\Users\72863\miniconda3\Lib\site-packages\lerobot\datasets\lerobot_dataset.py:79`），同时 `LeRobotDataset.add_frame` 签名发生变化，要求通过 frame 字典携带 task 信息（`...lerobot_dataset.py:1074`）。
- 目标是在保持数据捕获 → HDF5 → LeRobot → 训练整体流程的前提下，完成对 0.4.1 的兼容，确保新增/既有数据都能顺利训练，并补齐配套文档与验证步骤。

## 现状评估

### 主要兼容性缺口
1. **依赖/文档版本落后**  
   - 多处硬编码 0.3.3（`README.md:14`, `TRAINING_GUIDE.md:10`, `README.md:60`、`README.md:150`, `README.md:139`, `verify_dataset.py:6`），`requirements.txt:13` 仅声明 `lerobot>=0.3.0`，与 0.4.1 实际要求不符。
2. **`hdf5_to_lerobot.py` 转换逻辑仍按 v2.0/v2.1 假设实现**  
   - 文件头部仍写着 “convert ... to ... v2.0 format”（`hdf5_to_lerobot.py:2`），`populate_dataset` 依旧调用旧签名 `dataset.add_frame(frame, task)` 且 frame 中没有 `task` 键（`hdf5_to_lerobot.py:218-231`），在 0.4.1 上会直接抛出 TypeError。
   - 输出路径/说明仍宣称为 “LeRobot v2.1 converter”（`README.md:60`, `README.md:150`），与 0.4.1 的 v3.0 结构不符。
3. **已有数据迁移指引缺失**  
   - 当前仓库没有为历史 v2.1 数据提供升级方案，而 0.4.1 官方脚本 `lerobot/datasets/v30/convert_dataset_v21_to_v30.py` 可用（`C:\Users\72863\miniconda3\Lib\site-packages\lerobot\datasets\v30\convert_dataset_v21_to_v30.py:1`）。需要在项目中明确说明如何处理旧数据。
4. **验证/训练脚本引用过时**  
   - `verify_dataset.py` 的提示与报错仍指向 “pip install lerobot>=0.3.0”（`verify_dataset.py:87`），并未对 v3.0 结构做检查。
   - 文档中的训练命令仍为 `python -m lerobot.scripts.train`（`README.md:77`, `TRAINING_GUIDE.md:22`），但 0.4.1 中已经改为 `lerobot.scripts.lerobot_train`（`C:\Users\72863\miniconda3\Lib\site-packages\lerobot\scripts\lerobot_train.py:1`），旧命令会失败。

## 开发计划

### 1. 依赖与文档对齐
1. **依赖调整**  
   - 将 `requirements.txt` 中的 `lerobot>=0.3.0` 改为 `lerobot==0.4.1` 或 `>=0.4.1,<0.5`，并注明需要 Python 3.10+、`datasets/pyarrow` 依赖。（必要时补充 `pip install --upgrade pip` 步骤，避免旧版本缺少 Wheels）。
2. **README / TRAINING_GUIDE 刷新**  
   - 更新安装与 pipeline 描述，改写为 “输出 v3.0 数据集，可与 LeRobot 0.4.1+ 训练脚本直接兼容”。  
   - 训练命令替换为 `python -m lerobot.scripts.lerobot_train ...`，同时补充 0.4.1 新增的配置字段（如 `cfg.wandb`, `cfg.policy` 字段）与推荐超参。  
   - 在 FAQ/故障排查部分新增 “如果检测到 `ForwardCompatibilityError` 需要运行 v30 转换脚本”的说明，指向本项目新增的小节或官方脚本路径。

### 2. `hdf5_to_lerobot.py` 改造
1. **代码层面改动**  
   - 引入 `frame["task"] = task`，并将 `dataset.add_frame(frame, task)` 改为 `dataset.add_frame(frame)`，从而匹配新签名（参考 `...lerobot_dataset.py:1074`）。
   - 为清晰起见，将文件开头/日志中的版本描述更新为 v3.0/LeRobot 0.4.1，避免误导。
2. **结构与配置**  
   - 在 `DatasetConfig` 内增加对 `batch_encoding_size` 的透传选项，以利用 0.4.1 新增的批量视频编码能力（`...lerobot_dataset.py:1185` 附近）。  
   - 检查 `create_empty_dataset` 中 `dtype="video"` 的使用，在 0.4.1 中仍然合法，但需要确认 `use_videos=False` 时自动写 PNG；必要时提供 CLI 参数允许用户切换。
3. **日志 & 错误处理**  
   - 0.4.1 在写入失败时会留下部分 chunk；建议在 `populate_dataset` 的异常路径中打印 episode 索引，并提示重跑或使用 `dataset.clear_episode_buffer()`，防止未完成的 parquet 影响下次运行。

### 3. 旧数据迁移与向后兼容
1. **脚本引用**  
   - 文档中新增 “旧数据升级” 段落，附带命令示例 `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --root <path>`，并说明输出目录结构差异（参考官方脚本注释 `...convert_dataset_v21_to_v30.py:33-122`）。
2. **项目内辅助工具（可选）**  
   - 如果需要在仓库内直接调用，可提供一个 `tools/upgrade_v21_dataset.py`，包装官方脚本并加上仓库自定义默认参数（如 `--root ./lerobot_data`）。该工具不必复制源码，只需做 CLI 代理。

### 4. 验证与自动化测试
1. **`verify_dataset.py` 强化**  
   - 更新提示语到 0.4.1，加载成功后检查 `ds.meta.info["codebase_version"] == "v3.0"` 并在失败时给出清晰错误。  
   - 增加一个针对首帧的结构校验（例如确认 `task_index`, `episode_index`, `timestamp` 存在），从而尽早发现转换遗漏。
2. **集成测试建议**  
   - 新增一个最小 HDF5 fixture（可由 `data_to_hdf5` 生成 1~2 帧）并在 CI 或手动脚本中运行 `hdf5_to_lerobot.py` + `verify_dataset.py`，确保流程在 0.4.1 下跑通。  
   - 在手动验证指南中记录：运行 `python -m lerobot.scripts.lerobot_train --dataset.repo_id=...` 100 步，确认训练脚本能够扫描到新数据。

### 5. 训练链路与用户指南
1. **命令更新**  
   - 将所有 `python -m lerobot.scripts.train` 替换为 `python -m lerobot.scripts.lerobot_train`，并验证新的配置路径（`lerobot.configs.train.TrainPipelineConfig`）在指南中的参数名字仍然正确。  
   - 根据 0.4.1 默认配置，检查 `--dataset.repo_id`、`--dataset.root` 是否需要调整（0.4.1 开始推荐通过配置文件/`cfg.dataset.repo_id`，可在文档中给出 YAML/CLI 两种写法）。
2. **新增注意事项**  
   - 记录 0.4.1 引入的 `CODEBASE_VERSION` 校验：提醒用户当 `LeRobotDataset` 抛出 `ForwardCompatibilityError` 时，需要升级数据或使用新版本脚本。

## 验证策略
- **功能验证**：使用一段真实 HDF5 数据运行新的转换脚本，确保 `lerobot_data/meta/info.json` 中 `codebase_version` 为 v3.0 且 `verify_dataset.py` 通过。
- **兼容性验证**：在已安装 `lerobot==0.4.1` 的环境执行 README 中的训练命令（1）快速 100 步 smoke test；（2）如有必要，再跑一次全量训练以确认无 runtime error。
- **文档验证**：交叉检查 README / TRAINING_GUIDE / FAQ 与实际命令是否一致，特别是安装和训练部分。

## 风险与待确认
- 是否需要继续支持 0.3.x？若需要，则需在代码中检测版本并提供双路径逻辑，否则可以直接声明“最低版本 0.4.1”并去除旧兼容分支。
- `LeRobotDataset` 在 0.4.1 中默认写入 chunked parquet/视频，可能比旧版更占磁盘；需要在文档中说明磁盘预估值以及 `chunks_size` 可配置项。
- 旧数据若只保留 HDF5，直接用新脚本重新导出即可；若用户只有 v2.1 LeRobot 数据集，则必须执行 v30 官方转换脚本，此过程需写权限和额外空间，需在文档中提示。
