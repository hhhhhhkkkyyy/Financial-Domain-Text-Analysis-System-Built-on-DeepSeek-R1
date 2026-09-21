"""
热点事件识别模块 - 批量文档事件统计与趋势分析
Hot Event Recognition & Trend Analysis Module
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class HotEventAnalyzer:
    """
    热点事件分析器 - 对批量文档进行统计分析
    """

    def __init__(self, chart_generator=None):
        self.chart_gen = chart_generator

    def analyze_batch_events(self, all_events: List[List[Dict]]) -> Dict:
        """分析批量文档的事件分布"""
        flat = []
        for doc_events in all_events:
            if isinstance(doc_events, list):
                flat.extend(doc_events)
            elif isinstance(doc_events, dict):
                flat.append(doc_events)

        flat_dicts = []
        for e in flat:
            if hasattr(e, '__dict__'):
                flat_dicts.append(e.__dict__)
            elif isinstance(e, dict):
                flat_dicts.append(e)

        if not flat_dicts:
            return {"total_events": 0}

        # 事件类型分布
        type_counter = Counter(e.get("event_type", "未知") for e in flat_dicts)

        # 方向分布
        dir_counter = Counter(e.get("direction", "中性") for e in flat_dicts)

        # 影响指标统计
        metrics_counter = Counter()
        for e in flat_dicts:
            for m in e.get("impact_metrics", []):
                metrics_counter[m] += 1

        # 时间分布
        time_counter = Counter()
        for e in flat_dicts:
            t = e.get("time_range", "")
            if t:
                year_match = re.search(r'(20\d{2})', str(t))
                if year_match:
                    time_counter[year_match.group(1)] += 1
                else:
                    time_counter[str(t)] += 1

        # 高频事件来源
        source_counter = Counter()
        for e in flat_dicts:
            src = e.get("source_text", "")[:80]
            if src:
                source_counter[src] += 1

        result = {
            "total_events": len(flat_dicts),
            "total_documents": len(all_events),
            "type_distribution": dict(type_counter.most_common()),
            "direction_distribution": dict(dir_counter.most_common()),
            "top_metrics": dict(metrics_counter.most_common(10)),
            "time_distribution": dict(time_counter.most_common()),
            "top_sources": [{"text": src, "count": cnt}
                           for src, cnt in source_counter.most_common(5)],
            "hot_events": self._identify_hot_events(type_counter, metrics_counter),
        }
        return result

    def _identify_hot_events(self, type_counter: Counter,
                              metrics_counter: Counter) -> List[Dict]:
        """识别热点事件"""
        total = sum(type_counter.values()) or 1
        hot = []
        for etype, count in type_counter.most_common(3):
            if count / total > 0.1:
                hot.append({
                    "event_type": etype,
                    "frequency": count,
                    "proportion": round(count / total, 3),
                })
        return hot

    def generate_trend_data(self, all_events: List[List[Dict]],
                            time_labels: List[str] = None) -> Dict:
        """生成时间趋势数据"""
        flat = []
        for doc_events in all_events:
            flat.extend(doc_events if isinstance(doc_events, list) else [doc_events])

        if time_labels:
            trend = {label: {"total": 0, "by_type": {}} for label in time_labels}
            for i, e in enumerate(flat):
                idx = min(i // max(len(flat) // len(time_labels), 1), len(time_labels) - 1)
                label = time_labels[idx]
                trend[label]["total"] += 1
                etype = getattr(e, "event_type", "") if hasattr(e, "event_type") else e.get("event_type", "")
                trend[label]["by_type"][etype] = trend[label]["by_type"].get(etype, 0) + 1
            return {"trend": trend, "labels": time_labels}
        return {"trend": {}, "labels": []}
