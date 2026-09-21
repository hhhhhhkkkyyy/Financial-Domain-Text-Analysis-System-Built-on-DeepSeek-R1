# 📈 金融文本智能分析系统

## 基于 DeepSeek-R1 的金融领域文本分析平台（前后端分离架构）

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.36+-orange.svg)](https://huggingface.co/transformers)
[![Streamlit](https://img.shields.io/badge/Web-Streamlit-ff69b4.svg)](https://streamlit.io)
[![DeepSeek](https://img.shields.io/badge/Model-DeepSeek--R1--Distill--Qwen--1.5B-6366f1.svg)](https://huggingface.co/deepseek-ai)

> **自然语言处理课程大作业 + 实训项目**  
> 题目：基于 DeepSeek-R1 的金融领域文本分析系统

---

## 项目简介

本系统面向金融领域的非结构化文本数据（上市公司公告、业绩说明会纪要、监管文件、财经新闻、研报摘要、投资者问答记录等），基于轻量级大语言模型 **DeepSeek-R1-Distill-Qwen-1.5B**，自动从文本中抽取关键的金融事件信息（如盈利预测调整、风险提示、经营指标变动、并购动态、监管处罚、评级变动等），并以结构化的 JSON 格式输出分析结果。

系统采用 **「关键词规则引擎 + 轻量级大模型」混合架构**，在保证可用性的基础上，通过领域适配优化策略与提示工程技术，在有限算力环境下达到专业级的金融语义理解水平，完整覆盖从数据导入、模型部署、提示优化、结构化抽取、批量处理、性能评估到可视化展示的全链路流程。

---

## 前后端架构设计

系统采用清晰的 **前端（展示层）与后端（业务层 / 服务层）分离** 设计：

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        前端 · 展示层 (Frontend)                         │
│                                                                     │
│   Streamlit Web 界面 (src/web/streamlit_app.py)                        │
│   ┌──────────┬──────────┬──────────┬──────────┬──────────┐           │
│   │ 单文件分析 │ 长文档处理 │ 批量处理  │ 实验对比  │ 数据总览  │           │
│   └──────────┴──────────┴──────────┴──────────┴──────────┘           │
│   文件上传 / 文本粘贴 / 参数选择 / 结果展示 / 图表可视化 / 导出下载         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 直接调用后端模块（同一 Python 进程）
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        后端 · 业务层 (Backend)                         │
│                                                                     │
│   CLI 入口 (app.py)：analyze / batch / experiment / demo              │
│                                                                     │
│   FinancialAnalysisPipeline (src/analysis_pipeline.py)               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ 文档预处理 → 长文档分段 → 金融事件抽取 → 热点分析 → 摘要生成     │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────┬────────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│      服务层 (Service)          │  │        基础设施层 (Infra)           │
│                              │  │                                  │
│ ModelService                 │  │ PromptManager   (提示模板)        │
│  DeepSeek-R1-Distill-Qwen    │  │ ResultExporter  (JSON/CSV 导出)   │
│   (Transformers 本地推理)      │  │ ChartGenerator  (可视化图表)      │
│                              │  │ InferenceLogger (监控日志)        │
│                              │  │ ResultEvaluator (结果评估)        │
│                              │  │ HotEventAnalyzer (热点识别)       │
└──────────────────────────────┘  └──────────────────────────────────┘
```

### 前后端职责划分

| 层 | 位置 | 职责 |
| --- | --- | --- |
| 前端（展示层） | `src/web/streamlit_app.py` | 页面布局、交互控件、文件上传、结果渲染、图表展示、导出下载 |
| 后端（业务层） | `app.py` + `src/analysis_pipeline.py` | 业务流程编排、命令行入口、分析管道 |
| 后端（服务层） | `src/utils/model_service.py` | DeepSeek-R1 模型加载、推理、量化、资源管理 |
| 后端（基础设施层） | `src/export/`、`src/visualization/`、`src/financial_extraction/` 等 | 事件抽取、热点分析、结果评估、导出、可视化、日志 |

前端（Streamlit）与后端通过**直接导入 Python 模块**进行通信，无需额外的 HTTP 服务：前端页面调用 `FinancialEventExtractor`、`ModelService`、`PromptManager` 等后端组件，完成「上传 → 抽取 → 展示 → 导出」的完整闭环。同时，后端核心逻辑也通过 `app.py` 提供**命令行接口（CLI）**，可脱离 Web 界面独立运行。

---

## 核心功能

### 大作业功能（基础系统）

| 模块 | 说明 |
| --- | --- |
| 🚀 模型服务部署 | 本地部署 DeepSeek-R1-Distill-Qwen-1.5B，提供统一推理接口，支持同步与批量调用 |
| 🧠 提示工程 | 3 套提示方案（结构化 / 思维链 / 少样本）对比实验与优化 |
| 📋 结构化事件抽取 | 抽取事件类型、方向、影响指标、时间范围、原因/依据等字段 |
| 📄 数据导入与预处理 | 支持 TXT / Markdown / CSV / PDF / DOCX 多格式，批量处理文档集 |
| 📦 结果导出 | 抽取结果导出为 JSON、CSV 格式 |
| 🧪 对比实验 | 提示模板对比、量化部署对比、批次处理对比 |

### 实训扩展功能（专业化模块）

| 模块 | 说明 | 技术方案 |
| --- | --- | --- |
| 🏷️ 多类型事件抽取 | 6 类金融事件标签体系，分类识别并结构化输出 | 关键词规则 + LLM |
| 📚 长文档分段处理 | 面向年报、招股说明书等超长文档的智能分段与分块抽取 | 章节 / 段落 / 句子三级策略 |
| 📦 批量文档处理 | 一批金融文档批量处理，记录状态与耗时，生成批处理报告 | 批量循环 + 错误隔离 |
| 📊 结果对比评估 | 模型抽取结果与人工标注对比，计算 Precision / Recall / F1，错误分类分析 | ResultEvaluator |
| 🔥 热点事件识别 | 批量文档统计高频事件主题与趋势，生成可视化图表 | HotEventAnalyzer |
| 🖥️ 系统监控日志 | 记录每次调用的输入、输出及推理耗时，支持日志查询与审计 | InferenceLogger |
| 🌐 Web 交互界面 | 文档上传、实时抽取、批量处理、结果导出与可视化展示 | Streamlit |

---

## 金融事件类型体系

系统定义了 6 类金融事件类型标签：

| 事件类型 | 说明 |
| --- | --- |
| 盈利预测调整 | 公司盈利预测、业绩指引的调整 |
| 风险提示 | 各类金融风险的警示与提示 |
| 并购动态 | 收购、合并、重组等事件 |
| 监管处罚 | 监管机构的处罚与整改 |
| 评级变动 | 券商评级、目标价的调整 |
| 经营指标变动 | 营收、利润、毛利率等经营指标的变动 |

每条事件包含以下结构化字段：

- **方向（direction）**：正面 / 负面 / 中性
- **影响指标（impact_metrics）**：营收、毛利率、净利率、现金流、资产负债率、ROE、EPS、市盈率、市净率等
- **时间范围（time_range）**：如 Q3、下半年、2024 财年等
- **原因 / 依据（reason）**：事件发生的具体原因或依据
- **原文引用（source_text）**：抽取来源片段
- **置信度（confidence）**

---

## 技术栈

| 组件 | 技术选型 | 说明 |
| --- | --- | --- |
| 基础模型 | DeepSeek-R1-Distill-Qwen-1.5B | 轻量级大语言模型，金融语义分析 |
| 推理框架 | HuggingFace Transformers | 本地推理（FP16 / INT8 / AWQ 量化） |
| 规则引擎 | Python 正则 + 关键词词典 | 默认快速抽取方案 |
| 前端框架 | Streamlit | 交互式 Web 界面 |
| 文档处理 | PyPDF2 / pdfplumber / python-docx | 多格式解析 |
| 可视化 | Matplotlib / Plotly | 事件分布、趋势图表 |
| 数据存储 | JSON / CSV | 结果导出与数据集 |
| API 客户端 | OpenAI SDK | 可选远程 API 调用 |

> 说明：题目要求基于 vLLM 进行服务化部署，当前实现采用 HuggingFace Transformers 本地推理（配置中已预留量化选项，可平滑迁移至 vLLM 服务化部署）。

---

## 快速开始

### 环境要求

- Python 3.9+
- 内存 ≥ 8GB
- GPU 可选（规则引擎模式无需 GPU；DeepSeek 模型需 ≥4GB 显存）

### 安装与启动

```bash
# 1. 进入项目目录
cd legal-contract-analysis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Web 前端界面
streamlit run src/web/streamlit_app.py
```

浏览器打开 `http://localhost:8501`，上传金融文本或粘贴文本即可开始分析。

### 命令行使用（后端 CLI）

```bash
# 分析单个文件（规则引擎模式，无需模型）
python app.py analyze text.txt

# 使用本地模型深度分析
python app.py analyze text.txt --use-model --model-path D:\\model

# 批量处理目录下所有金融文档
python app.py batch ./data/contracts/

# 运行对比实验
python app.py experiment ./data/ --type all
python app.py experiment ./data/ --type prompt
python app.py experiment ./data/ --type quantization
python app.py experiment ./data/ --type batch_size
python app.py experiment ./data/ --type multi_type

# 生成 500 条金融文本样本数据
python app.py demo --count 500
```

---

## 项目结构

```text
legal-contract-analysis/
├── app.py                              # 后端 CLI 入口（analyze / batch / experiment / demo）
├── requirements.txt                    # Python 依赖
├── configs/
│   └── config.yaml                     # 系统配置（模型 / 金融分析 / 实验参数）
├── src/                                # 后端源码
│   ├── analysis_pipeline.py            # 核心分析管道（FinancialAnalysisPipeline）
│   ├── data_generator_financial.py     # 金融文本数据生成器（500+ 样本，含标注）
│   ├── financial_extraction/           # 金融事件抽取（实训核心）
│   │   ├── financial_event_extractor.py #   金融事件结构化抽取
│   │   ├── hot_event_analyzer.py        #   热点事件识别与趋势分析
│   │   ├── long_document_processor.py   #   长文档分段处理
│   │   ├── monitor_logger.py            #   系统监控与日志
│   │   └── result_evaluator.py          #   抽取结果对比评估
│   ├── preprocessing/
│   │   └── document_processor.py       # 文档预处理（TXT/MD/CSV/PDF/DOCX）
│   ├── export/
│   │   └── exporter.py                 # 结果导出（JSON/CSV）
│   ├── visualization/
│   │   └── chart_generator.py          # 可视化图表生成（Matplotlib）
│   ├── experiments/
│   │   └── experiment_runner.py        # 对比实验运行器
│   ├── batch/
│   │   └── batch_processor.py          # 批量处理引擎
│   ├── web/                            # 前端（展示层）
│   │   └── streamlit_app.py            # Streamlit Web 界面
│   └── utils/
│       ├── model_service.py            # 模型服务（DeepSeek-R1 + Transformers）
│       └── prompt_templates.py         # 提示模板管理（3+ 套方案）
├── models/
│   ├── deepseek/                       # DeepSeek-R1-Distill-Qwen-1.5B 权重
│   └── deep/                           # DeepSeek 模型权重副本
├── data/
│   ├── contracts/                      # 金融文本数据集（financial_0001~0500.txt + 标注）
│   └── test_samples/                   # 多类型测试样本（并购/监管/评级/研报等）
├── charts/                             # 可视化图表输出
├── experiments_results/                # 实验运行结果（JSON）
├── output/                             # 分析输出目录
└── logs/                               # 推理日志
```

---

## 模型与部署

| 项目 | 详情 |
| --- | --- |
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B |
| 架构 | Qwen2（1.5B，28 层，hidden=1536，12 个注意力头） |
| 推理框架 | HuggingFace Transformers |
| 支持精度 | FP16 / INT8 / AWQ（bitsandbytes / auto-gptq） |
| 显存占用 | 约 3.2GB (FP16) / 1.8GB (INT8) / 1.2GB (AWQ) |
| 生成参数 | temperature=0.3, top_p=0.9, repetition_penalty=1.1, max_new_tokens=2048 |

模型权重已就绪于 `models/deepseek/`，可通过 `ModelService` 本地加载，或通过 `--model-path` 指定自定义路径。

---

## 数据集

金融文本样本由 `src/data_generator_financial.py` 自动生成，覆盖 5 种金融文档类型：

- 业绩说明会纪要（performance_briefing）
- 上市公司公告（announcement）
- 财经新闻（financial_news）
- 研报摘要（research_report）
- 风险提示（risk_warning）

共生成 500 条金融文本样本（`data/contracts/financial_0001~0500.txt`），每条样本同时生成人工标注结果，用于模型性能评估。

`data/test_samples/` 提供多类型测试样本：并购公告、财经新闻、风险提示、监管处罚、评级变动、研报摘要、业绩说明会、业绩预告等。

---

## 对比实验

| 实验 | 内容 |
| --- | --- |
| 提示模板对比 | 3 套提示方案的结构一致性、信息抽取准确率对比 |
| 量化部署对比 | FP16 vs INT8/AWQ 在显存占用、推理延迟、输出质量的差异 |
| 批次处理对比 | batch size（1/4/8/16）下的吞吐量与平均延迟 |
| 多类型抽取评估 | 分事件类型统计 Precision / Recall / F1 |

实验运行结果保存于 `experiments_results/`，可视化图表生成于 `charts/`。

---

## 输出与可视化

- **结构化结果**：每次分析输出 JSON 分析报告 + CSV 事件列表；
- **图表**：事件类型分布饼图、方向分布图、批次效率图、提示方案对比图、量化对比图、分类型性能图；
- **日志**：`logs/inference_*.jsonl` 记录每次调用的输入、输出与耗时。

---

## 常见问题

**Q1：没有 GPU 可以运行吗？**

可以。系统默认使用关键词规则引擎，CPU 即可流畅运行；DeepSeek 深度分析为可选组件，需要 GPU。

**Q2：模型文件在哪里？**

DeepSeek-R1-Distill-Qwen-1.5B 权重位于 `models/deepseek/`，可通过 `--model-path` 指定，或直接由 `ModelService` 加载。

**Q3：如何生成金融样本数据集？**

```bash
python app.py demo --count 500
```

**Q4：数据是真实的吗？**

金融文本样本由 `data_generator_financial.py` 按真实金融文本结构生成，保证事件类型与方向分布可控；实验中的事件数/覆盖率由 `ExperimentRunner` 实际运行得到，precision/recall/F1 值部分为基于抽检的估算值。

---

## 免责声明

本系统的分析结果仅供学习与研究参考，不构成任何投资建议或专业金融意见。本项目仅用于教育学习目的。

---

**Natural Language Processing Course Project | 2025-2026**
