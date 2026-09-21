# 金融文本智能分析系统 - 模型文件

## DeepSeek-R1-Distill-Qwen-1.5B 模型（`models/deepseek/`）

| 项目 | 详情 |
| --- | --- |
| 架构 | Qwen2 |
| 参数量 | 1.5B（28 层，hidden=1536，12 个注意力头） |
| 支持精度 | FP16 / INT8 / AWQ |
| 显存占用 | 3.2GB (FP16) / 1.8GB (INT8) / 1.2GB (AWQ) |
| 加载方式 | `AutoModelForCausalLM.from_pretrained('models/deepseek')` |

> `models/deep/` 为同一模型的权重副本。
