"""
提示工程模块 - 金融事件抽取的多套提示模板
Prompt Engineering Module - Multiple Prompt Templates for Financial Event Extraction
"""

from typing import List, Dict, Optional

# ============================================================
# 提示模板方案1: 结构化提示 (Structured Prompt)
# ============================================================

PROMPT_V1_STRUCTURED = {
    "name": "结构化提示方案",
    "description": "使用结构化格式明确指定金融事件抽取任务和输出要求",

    "financial_extraction": """你是一位专业的金融文本分析专家。请仔细分析以下金融文本，抽取其中所有关键金融事件信息。

## 金融文本
{contract_text}

## 抽取要求
请从以下维度进行结构化信息抽取：

1. **事件类型**：判断属于以下哪类金融事件
   - 盈利预测调整
   - 风险提示
   - 经营指标变动
   - 并购动态
   - 监管处罚
   - 评级变动

2. **方向**：正面 / 负面 / 中性

3. **影响指标**：如毛利率、营收、利润、现金流、市场份额等

4. **时间范围**：如Q3、下半年、2024财年、本月等

5. **原因/依据描述**：事件发生的具体原因或依据

## 输出格式
请以JSON格式输出：

```json
{{
  "events": [
    {{
      "event_type": "事件类型",
      "direction": "正面/负面/中性",
      "impact_metrics": ["影响指标1", "影响指标2"],
      "time_range": "时间范围",
      "reason": "原因/依据描述",
      "source_text": "原文引用"
    }}
  ]
}}
```""",

    "risk_detection": """你是一位金融风险分析专家。请仔细审查以下金融文本，检测潜在风险点。

## 金融文本
{contract_text}

## 风险检测要求
请从以下维度检测风险：
1. 市场风险
2. 信用风险
3. 流动性风险
4. 操作风险
5. 合规风险
6. 声誉风险

## 输出格式
请以JSON格式输出风险类别、等级、描述和建议。""",

    "summary_generation": """你是一位金融文本分析专家。请为以下金融文本生成结构化摘要。

## 金融文本
{contract_text}

## 摘要要求
请生成包含以下内容的结构化摘要：
1. 文本类型识别
2. 核心事件概述
3. 关键财务指标
4. 主要风险提示
5. 趋势判断

## 输出格式
请以JSON格式输出结构化摘要。"""
}


# ============================================================
# 提示模板方案2: 分步骤思维链提示 (Chain-of-Thought)
# ============================================================

PROMPT_V2_COT = {
    "name": "思维链提示方案",
    "description": "引导模型分步骤推理，逐步分析金融文本",

    "financial_extraction": """你是一位专业的金融文本分析专家。请按照以下步骤抽取金融事件信息。

## 金融文本
{contract_text}

## 分析步骤
请严格按照以下步骤进行分析：

**步骤1：文本类型识别**
先判断文本类型（业绩说明会纪要/上市公司公告/财经新闻/监管文件/投资者问答）。

**步骤2：关键信息定位**
定位文本中涉及的具体金额、时间、指标变化等关键数据。

**步骤3：事件分类**
根据步骤2的信息判断事件类型：
- 盈利预测调整：涉及盈利预期、收入预测的调整
- 风险提示：涉及各类风险的警示
- 经营指标变动：涉及营收、利润、毛利率等指标变化
- 并购动态：涉及收购、合并、重组等
- 监管处罚：涉及监管机构的处罚
- 评级变动：涉及信用评级、目标价调整

**步骤4：方向判断**
判断事件对公司的整体影响方向（正面/负面/中性）。

**步骤5：综合输出**
汇总所有抽取结果，以JSON格式输出包含事件类型、方向、影响指标、时间范围和原因。""",

    "risk_detection": """你是一位金融风险分析专家。请按照以下步骤检测金融文本中的风险。

## 金融文本
{contract_text}

## 分析步骤
**步骤1：识别风险信号**
**步骤2：分类风险类型**
**步骤3：评估风险等级**
**步骤4：提出应对建议**

请在步骤4中以JSON格式输出完整结果。""",

    "summary_generation": """你是一位金融文本分析专家。请按照以下步骤生成摘要。

## 金融文本
{contract_text}

## 分析步骤
**步骤1：识别文本类型**
**步骤2：提取核心事件**
**步骤3：梳理财务指标**
**步骤4：评估风险要点**
**步骤5：判断趋势方向**

请在步骤5中以JSON格式输出完整的结构化摘要。"""
}


# ============================================================
# 提示模板方案3: 少量样本提示 (Few-Shot Prompt)
# ============================================================

PROMPT_V3_FEWSHOT = {
    "name": "少量样本提示方案",
    "description": "提供金融事件抽取示例，引导模型学习输出模式",

    "financial_extraction": """你是一位专业的金融文本分析专家。请参考以下示例，分析给定的金融文本。

## 示例分析

示例金融文本：
"公司预计2024年全年营收将达到45-48亿元，同比增长25%-30%，主要得益于新产品线的快速放量。毛利率有望从去年的32%提升至35%左右。"

示例抽取结果：
```json
{{
  "events": [
    {{
      "event_type": "经营指标变动",
      "direction": "正面",
      "impact_metrics": ["营收", "毛利率"],
      "time_range": "2024年全年",
      "reason": "新产品线快速放量带动收入增长，规模效应提升毛利率",
      "source_text": "公司预计2024年全年营收将达到45-48亿元，同比增长25%-30%"
    }}
  ]
}}
```

## 待分析金融文本
{contract_text}

请参照上述示例的风格和格式，抽取所有金融事件信息，以JSON格式输出。""",

    "risk_detection": """你是一位金融风险分析专家。请参考以下示例，分析金融文本风险。

## 示例分析

示例文本："公司当前资产负债率已达到78%，短期借款占总负债的60%以上。"
示例风险分析：
```json
{{
  "risks": [
    {{
      "risk_category": "流动性风险",
      "risk_level": "高",
      "risk_content": "资产负债率78%，短期借款占比超60%",
      "risk_description": "高负债率叠加短期债务集中，存在流动性压力",
      "suggestion": "建议优化债务结构，增加长期融资比例"
    }}
  ]
}}
```

## 待分析金融文本
{contract_text}

请参照上述示例的风格和深度，检测所有潜在风险点，以JSON格式输出。""",

    "summary_generation": """你是一位金融文本分析专家。请参考以下示例格式生成摘要。

## 示例摘要
```json
{{
  "document_type": "业绩说明会纪要",
  "core_events": [
    "公司2024年Q3营收同比增长20%",
    "毛利率提升至35%"
  ],
  "key_metrics": [
    "营收: 45-48亿元",
    "毛利率: 35%",
    "净利率: 12%"
  ],
  "risk_alerts": [
    "行业竞争加剧可能影响利润率"
  ],
  "trend": "正面"
}}
```

## 待分析金融文本
{contract_text}

请参照上述示例格式，生成该文本的结构化摘要，以JSON格式输出。"""
}


# ============================================================
# 实训模块提示模板
# ============================================================

PRACTICAL_PROMPTS = {
    "multi_event_extraction": """你是一位金融文本分析专家。请识别以下金融文本中的所有事件类型。

## 金融文本
{contract_text}

## 事件类型标签体系
- 盈利预测调整
- 风险提示
- 并购动态
- 监管处罚
- 评级变动
- 经营指标变动

每个事件需输出：事件类型、方向、影响指标、时间范围、原因。""",

    "long_document_analysis": """你是一位金融文档分析专家。请分析以下长文档片段。

## 文档片段
{contract_text}

## 分析要求
请抽取本片段中的关键金融事件信息，以JSON格式输出。""",

    "hot_event_analysis": """你是一位金融舆情分析专家。请分析以下文本集合中的热点事件。

## 文本集合
{contract_text}

请识别高频提及的金融事件主题，分析趋势，以JSON格式输出。""",
}


# ============================================================
# 提示模板管理器
# ============================================================

class PromptManager:
    """提示模板管理器"""

    def __init__(self):
        self.templates = {
            "v1_structured": PROMPT_V1_STRUCTURED,
            "v2_cot": PROMPT_V2_COT,
            "v3_fewshot": PROMPT_V3_FEWSHOT,
            "practical": PRACTICAL_PROMPTS,
        }
        self.current_version = "v1_structured"

    def get_template(self, task: str, version: str = None) -> str:
        """获取指定任务和版本的提示模板"""
        version = version or self.current_version
        template_set = self.templates.get(version)
        if not template_set:
            raise ValueError(f"未知的提示方案版本: {version}. 可用版本: {list(self.templates.keys())}")
        template = template_set.get(task)
        if not template:
            raise ValueError(f"未知的任务类型: {task}. 可用任务: {list(template_set.keys())}")
        return template

    def format_prompt(self, task: str, text: str, version: str = None,
                      **kwargs) -> str:
        """格式化提示模板"""
        template = self.get_template(task, version)
        formatted = template.format(contract_text=text, **kwargs)
        return formatted

    def list_versions(self) -> List[str]:
        return list(self.templates.keys())

    def list_tasks(self, version: str = None) -> List[str]:
        version = version or self.current_version
        template_set = self.templates.get(version, {})
        return [k for k in template_set.keys() if k != "name" and k != "description"]

    def get_version_info(self, version: str) -> Dict[str, str]:
        template_set = self.templates.get(version, {})
        return {
            "version": version,
            "name": template_set.get("name", ""),
            "description": template_set.get("description", ""),
            "tasks": self.list_tasks(version)
        }
