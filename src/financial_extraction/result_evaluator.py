"""
抽取结果对比分析模块 - 计算准确率、召回率、F1值
Result Comparison & Evaluation Module
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """评估结果"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    total_predicted: int = 0
    total_ground_truth: int = 0
    correct: int = 0
    by_type: Dict[str, Dict] = field(default_factory=dict)
    error_analysis: Dict = field(default_factory=dict)


class ResultEvaluator:
    """
    结果评估器 - 对比模型抽取结果与人工标注
    """

    def evaluate(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationResult:
        """评估抽取结果"""
        result = EvaluationResult()
        result.total_predicted = len(predictions)
        result.total_ground_truth = len(ground_truth)

        by_type = {}
        correct = 0

        for gt in ground_truth:
            gt_type = gt.get("event_type", "")
            if gt_type not in by_type:
                by_type[gt_type] = {"tp": 0, "fp": 0, "fn": 0, "predicted": 0, "ground_truth": 0}
            by_type[gt_type]["ground_truth"] += 1

            # 查找匹配的预测
            matched = False
            for pred in predictions:
                if self._match_event(pred, gt):
                    matched = True
                    correct += 1
                    by_type[gt_type]["tp"] += 1
                    break
            if not matched:
                by_type[gt_type]["fn"] += 1

        for pred in predictions:
            ptype = pred.get("event_type", "")
            if ptype not in by_type:
                by_type[ptype] = {"tp": 0, "fp": 0, "fn": 0, "predicted": 0, "ground_truth": 0}
            by_type[ptype]["predicted"] += 1

            matched = False
            for gt in ground_truth:
                if self._match_event(pred, gt):
                    matched = True
                    break
            if not matched:
                by_type[ptype]["fp"] += 1

        result.correct = correct
        result.precision = correct / max(result.total_predicted, 1)
        result.recall = correct / max(result.total_ground_truth, 1)
        result.f1_score = 2 * result.precision * result.recall / max(result.precision + result.recall, 1e-10)

        per_type = {}
        for etype, counts in by_type.items():
            p = counts["tp"] / max(counts["predicted"], 1)
            r = counts["tp"] / max(counts["ground_truth"], 1)
            f1 = 2 * p * r / max(p + r, 1e-10)
            per_type[etype] = {
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f1, 3),
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            }
        result.by_type = per_type

        # 错误分析
        errors = {"type_mismatch": 0, "direction_mismatch": 0, "missing": 0}
        for pred in predictions:
            for gt in ground_truth:
                if pred.get("event_type") != gt.get("event_type") and \
                   self._text_overlap(pred.get("source_text", ""), gt.get("source_text", "")):
                    errors["type_mismatch"] += 1
                    break
            if pred.get("direction", "") != "":
                for gt in ground_truth:
                    if self._text_overlap(pred.get("source_text", ""), gt.get("source_text", "")) and \
                       pred.get("direction") != gt.get("direction"):
                        errors["direction_mismatch"] += 1
                        break
        result.error_analysis = errors

        return result

    def _match_event(self, pred: Dict, gt: Dict) -> bool:
        """判断两个事件是否匹配"""
        if pred.get("event_type") != gt.get("event_type"):
            return False
        pred_text = pred.get("source_text", "")[:100]
        gt_text = gt.get("source_text", "")[:100]
        return self._text_overlap(pred_text, gt_text)

    def _text_overlap(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """判断两段文本是否有足够重叠"""
        if not text1 or not text2:
            return False
        s1, s2 = set(text1), set(text2)
        intersection = s1 & s2
        return len(intersection) / max(len(s1 | s2), 1) > threshold

    def compare_results(self, results1: List[Dict], results2: List[Dict]) -> Dict:
        """对比两组抽取结果的差异"""
        comparison = {
            "total_1": len(results1),
            "total_2": len(results2),
            "common": 0,
            "only_in_1": 0,
            "only_in_2": 0,
        }
        set1 = {(e.get("event_type", ""), e.get("source_text", "")[:100]) for e in results1}
        set2 = {(e.get("event_type", ""), e.get("source_text", "")[:100]) for e in results2}
        comparison["common"] = len(set1 & set2)
        comparison["only_in_1"] = len(set1 - set2)
        comparison["only_in_2"] = len(set2 - set1)
        return comparison
