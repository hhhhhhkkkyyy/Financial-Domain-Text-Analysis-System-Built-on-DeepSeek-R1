"""
金融事件抽取模块 - 从金融文本中结构化抽取事件信息
Financial Event Extraction Module
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 5类以上金融事件类型标签体系
EVENT_TYPES = [
    "盈利预测调整",
    "风险提示",
    "并购动态",
    "监管处罚",
    "评级变动",
    "经营指标变动",
]

EVENT_TYPE_KEYWORDS = {
    "盈利预测调整": ["预计", "预测", "预期", "展望", "展望", "目标", "指引", "预估", " forecast"],
    "风险提示": ["风险", "警示", "谨慎", "不确定性", "挑战", "不利", "波动", "预警"],
    "并购动态": ["收购", "并购", "合并", "重组", "入股", "增资", "股权转让", "资产注入"],
    "监管处罚": ["处罚", "罚款", "监管", "整改", "立案", "调查", "处分", "警告", "通报"],
    "评级变动": ["评级", "上调", "下调", "维持", "增持", "减持", "买入", "卖出", "中性"],
    "经营指标变动": ["营收", "利润", "毛利率", "净利率", "现金流", "增长", "下降", "提升", "改善"],
}

DIRECTION_KEYWORDS = {
    "正面": ["增长", "提升", "改善", "增加", "盈利", "利好", "突破", "创新高", "扭亏"],
    "负面": ["下降", "亏损", "减少", "下滑", "恶化", "风险", "处罚", "利空", "减值"],
}

FINANCIAL_METRICS = [
    "营收", "收入", "利润", "净利润", "毛利率", "净利率", "现金流",
    "资产负债率", "ROE", "EPS", "市盈率", "市净率",
    "销售额", "成本", "费用", "存货", "应收账款",
]


@dataclass
class FinancialEvent:
    """金融事件数据结构"""
    event_type: str = ""
    direction: str = "中性"
    impact_metrics: List[str] = field(default_factory=list)
    time_range: str = ""
    reason: str = ""
    source_text: str = ""
    confidence: float = 0.0
    source_chunk: int = 0


class FinancialEventExtractor:
    """
    金融事件抽取器
    支持规则引擎和LLM两种模式
    """

    def __init__(self, model_service=None, prompt_manager=None):
        self.model_service = model_service
        self.prompt_manager = prompt_manager

    def extract_by_rules(self, text: str, chunk_id: int = 0) -> List[FinancialEvent]:
        """基于规则引擎的事件抽取（快速模式）"""
        events = []
        text_lower = text.lower()

        # 按句子分割
        sentences = re.split(r'[。！？\n]', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue

            event = self._analyze_sentence(sent, chunk_id)
            if event:
                events.append(event)

        # 去重
        seen = set()
        unique_events = []
        for e in events:
            key = (e.event_type, e.source_text[:50])
            if key not in seen:
                seen.add(key)
                unique_events.append(e)

        return unique_events

    def _analyze_sentence(self, sentence: str, chunk_id: int) -> Optional[FinancialEvent]:
        """分析单个句子中的金融事件"""
        # 判断事件类型
        matched_type = ""
        max_score = 0
        for etype, keywords in EVENT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in sentence)
            if score > max_score:
                max_score = score
                matched_type = etype if score > 0 else ""

        if not matched_type:
            return None

        # 判断方向
        direction = "中性"
        pos_score = sum(1 for kw in DIRECTION_KEYWORDS["正面"] if kw in sentence)
        neg_score = sum(1 for kw in DIRECTION_KEYWORDS["负面"] if kw in sentence)
        if pos_score > neg_score:
            direction = "正面"
        elif neg_score > pos_score:
            direction = "负面"

        # 提取影响指标
        metrics = []
        for m in FINANCIAL_METRICS:
            if m in sentence:
                metrics.append(m)

        # 提取时间范围
        time_patterns = [
            r'(\d{4}年[\d\w]*)', r'(20\d{2})', r'([上中下]半年)',
            r'(Q[1-4])', r'(第[一二三四]季度)', r'(本月|本季|本年度)',
            r'(\d{4}财年)', r'(下半年|上半年)',
        ]
        time_range = ""
        for pattern in time_patterns:
            m = re.search(pattern, sentence)
            if m:
                time_range = m.group(1)
                break

        # 提取金额信息作为背景
        amounts = re.findall(r'[\d,]+\.?\d*\s*[亿万千元%]', sentence)

        return FinancialEvent(
            event_type=matched_type,
            direction=direction,
            impact_metrics=metrics,
            time_range=time_range,
            reason=sentence[:150] if len(sentence) > 150 else sentence,
            source_text=sentence[:300],
            confidence=max_score / 5.0,
            source_chunk=chunk_id,
        )

    def extract_by_llm(self, text: str, prompt_version: str = "v1_structured",
                       chunk_id: int = 0) -> List[FinancialEvent]:
        """基于LLM的事件抽取"""
        if not self.model_service or not self.prompt_manager:
            logger.warning("LLM模式需要model_service和prompt_manager")
            return self.extract_by_rules(text, chunk_id)

        prompt = self.prompt_manager.format_prompt(
            "financial_extraction", text, prompt_version
        )

        try:
            response = self.model_service.generate(prompt)
            events = self._parse_llm_response(response, chunk_id)
            return events
        except Exception as e:
            logger.error(f"LLM抽取失败: {e}")
            return self.extract_by_rules(text, chunk_id)

    def _parse_llm_response(self, response: str, chunk_id: int) -> List[FinancialEvent]:
        """解析LLM返回的JSON结果"""
        events = []
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data.get("events", []):
                    events.append(FinancialEvent(
                        event_type=item.get("event_type", ""),
                        direction=item.get("direction", "中性"),
                        impact_metrics=item.get("impact_metrics", []),
                        time_range=item.get("time_range", ""),
                        reason=item.get("reason", ""),
                        source_text=item.get("source_text", ""),
                        confidence=0.8,
                        source_chunk=chunk_id,
                    ))
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
        return events

    def analyze_all_chunks(self, chunks: List[str], method: str = "keyword",
                           prompt_version: str = "v1_structured") -> List[List[FinancialEvent]]:
        """分析所有文本块"""
        results = []
        for i, chunk in enumerate(chunks):
            if method in ("llm", "hybrid"):
                events = self.extract_by_llm(chunk, prompt_version, i)
                if method == "hybrid":
                    rule_events = self.extract_by_rules(chunk, i)
                    events = self._merge_events(rule_events, events)
            else:
                events = self.extract_by_rules(chunk, i)

            results.append(events)
        return results

    def _merge_events(self, rule_events: List[FinancialEvent],
                      llm_events: List[FinancialEvent]) -> List[FinancialEvent]:
        """合并规则和LLM的结果"""
        merged = {e.source_text[:50]: e for e in rule_events}
        for e in llm_events:
            key = e.source_text[:50]
            if key not in merged:
                merged[key] = e
        return list(merged.values())

    def get_event_statistics(self, all_events: List[List[FinancialEvent]]) -> Dict:
        """获取事件统计信息"""
        flat = [e for sublist in all_events for e in sublist]
        by_type = {}
        by_direction = {"正面": 0, "负面": 0, "中性": 0}
        for e in flat:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_direction[e.direction] = by_direction.get(e.direction, 0) + 1

        return {
            'total_events': len(flat),
            'by_type': by_type,
            'by_direction': by_direction,
            'types_found': len(by_type),
            'types_total': len(EVENT_TYPES),
            'coverage_rate': len(by_type) / len(EVENT_TYPES) if EVENT_TYPES else 0,
        }
