# parameter-golf（社区 fork）

本仓库是 [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) 挑战代码的**社区 fork**，在保留上游训练与 `records/` 结构的前提下，为参赛者补充**本地研发工具**：无需先完整训练即可跑通 checkpoint / 生成 demo、自动记录实验结果、简易分析与时间粗算等。

**说明：本仓库不代表 OpenAI。** 赛题规则、算力资助、Discord 与正式提交要求请以 [openai/parameter-golf](https://github.com/openai/parameter-golf) 为准。

## 这个仓库适合谁

适合已经了解 Parameter Golf（16MB 产物内训练、FineWeb 验证集上以 bits per byte 计分等）的选手：在官方仓库能看到的赛事说明、排行榜与 FAQ 这里不再重复；这里只写**本 fork 多提供的便利**。

## 相对上游，本 fork 提供什么

| 能力 | 说明 |
|------|------|
| [`scripts/download_cli_demo_checkpoint.py`](scripts/download_cli_demo_checkpoint.py) | 准备与 `train_gpt.py` 兼容的 **FP16 demo checkpoint**（随机初始化，仅作接线/脚本测试，不表示质量）。优先使用仓库内 [`demo_assets/baseline_demo_fp16.pt.lzma`](demo_assets/baseline_demo_fp16.pt.lzma)，也可通过 URL 或环境变量 `PARAMETER_GOLF_DEMO_CKPT_URL` / `PARAMETER_GOLF_DEMO_RAW_BASE` 指定。 |
| [`scripts/generate_demo.py`](scripts/generate_demo.py) | 从 checkpoint **流式生成文本**（支持 `final_model.pt`、int8 往返、DDP `module.*` 等）。依赖本 fork 在 [`train_gpt.py`](train_gpt.py) 中增加的 **`GPT.forward_logits()`**。可选：配合 `--data-path` 与 tokenizer 预览 FineWeb 验证集解码。 |
| `train_gpt.py` → **`results.tsv` 自动追加** | 仅在 **rank 0**：正常结束写 **COMPLETE**（含 round-trip **`val_bpb`**、峰值显存、git 短 hash 等）；未捕获异常写 **CRASH**。环境变量：`EXPERIMENT_DESC`、`RESULTS_TSV_PATH`、`DISABLE_RESULTS_TSV=1`。该文件 **gitignore**，仅作本地实验台账。 |
| [`analysis.ipynb`](analysis.ipynb) | 读取 **`results.tsv`**，汇总 / 可视化多次实验（如 `val_bpb` 等）。 |
| [`scripts/h100_time_guess.py`](scripts/h100_time_guess.py) | 与 **8×H100、10 分钟**训练上限相关的粗算或对照 `run.log` 的步骤级 sanity check（仍以真实硬件为准）。 |
| [`agent.md`](agent.md) | 本地研发流程说明：与 upstream 对齐、环境变量、日志与实验循环等。 |

**提交 `records/` 时：** 仍须遵守赛方对自包含 `train_gpt.py` 等要求；本 fork 的日志与 demo 工具面向**日常开发**，若追求与极简记录完全一致，提交前请自行裁剪或说明额外改动。

## 快速上手（clone 本 fork 以使用自带 `demo_assets/`）

1. **仅准备 demo checkpoint（不训练）**

```bash
python scripts/download_cli_demo_checkpoint.py -o cli_demo_checkpoint.pt
```

2. **流式生成（需 SentencePiece tokenizer，见下方数据）**

```bash
python scripts/generate_demo.py \
  --checkpoint cli_demo_checkpoint.pt \
  --tokenizer ./data/tokenizers/fineweb_1024_bpe.model \
  --no-show-sample
```

可选：增加 `--data-path ./data/datasets/fineweb10B_sp1024/` 以打印一小段 FineWeb **验证集**解码再生成。

3. **训练并写入本地 `results.tsv`**

```bash
EXPERIMENT_DESC="实验简述" torchrun --standalone --nproc_per_node=1 train_gpt.py
```

之后用 [`analysis.ipynb`](analysis.ipynb) 查看汇总。

4. **粗算与日志对照**

```bash
python scripts/h100_time_guess.py
python scripts/h100_time_guess.py check run.log
```

## 数据与环境

- 数据集与 tokenizer 下载、目录结构：见 [`data/README.md`](data/README.md)。
- CUDA 训练依赖等与上游一致时，可参考官方仓库的 `requirements.txt`；MLX 本地路径见 `train_gpt_mlx.py` 与上游文档。

## 与 OpenAI 上游对齐（可选）

```bash
git remote add upstream https://github.com/openai/parameter-golf.git   # 若尚未添加
git fetch upstream
git log upstream/main..HEAD --oneline   # 查看本 fork 相对上游的提交
```

更细的日常工作流见 [`agent.md`](agent.md)。

## 致谢

本挑战与代码源自 OpenAI 与各 `records/` 贡献者；第三方说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
