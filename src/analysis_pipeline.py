"""
金融文本分析管道 - 整合所有模块的核心流程
Financial Text Analysis Pipeline
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .preprocessing.document_processor import DocumentPreprocessor
from .financial_extraction.financial_event_extractor import FinancialEventExtractor, FinancialEvent
from .financial_extraction.hot_event_analyzer import HotEventAnalyzer
from .financial_extraction.monitor_logger import InferenceLogger
from .financial_extraction.long_document_processor import LongDocumentProcessor
from .export.exporter import ResultExporter
from .visualization.chart_generator import ChartGenerator
from .utils.model_service import ModelService, ModelConfig
from .utils.prompt_templates import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """完整的金融分析报告"""
    doc_name: str
    analysis_time: str
    document_info: Dict = field(default_factory=dict)
    event_extraction: Dict = field(default_factory=dict)
    risk_analysis: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    chart_paths: Dict[str, str] = field(default_factory=dict)


class FinancialAnalysisPipeline:
    """金融文本分析管道 - 整合所有模块的完整流程"""

    def __init__(self, config: Dict = None, use_model: bool = False,
                 model_config: ModelConfig = None, model_path: str = ""):
        self.config = config or {}
        self.use_model = use_model
        self.preprocessor = DocumentPreprocessor(self.config.get('data', {}))
        self.long_doc_processor = LongDocumentProcessor(
            chunk_size=self.config.get('long_doc', {}).get('chunk_size', 2000),
            chunk_overlap=self.config.get('long_doc', {}).get('chunk_overlap', 200),
        )
        self.prompt_manager = PromptManager()
        self.exporter = ResultExporter()
        self.chart_gen = ChartGenerator()
        self.monitor = InferenceLogger()

        self.model_service = None
        if use_model:
            mc = model_config or ModelConfig()
            if model_path:
                mc.local_path = model_path
            self.model_service = ModelService(mc)

        self.event_extractor = FinancialEventExtractor(
            model_service=self.model_service,
            prompt_manager=self.prompt_manager
        )
        self.hot_analyzer = HotEventAnalyzer(chart_generator=self.chart_gen)
        self.last_report: Optional[AnalysisReport] = None

    def analyze_document(self, file_path: str,
                         prompt_version: str = "v1_structured",
                         method: str = "keyword") -> AnalysisReport:
        """分析单个金融文档"""
        logger.info(f"开始分析: {file_path}")
        start_time = time.time()

        doc = self.preprocessor.process_document(file_path)
        text = doc.cleaned_text or doc.raw_text
        if len(text) > 3000:
            long_result = self.long_doc_processor.process_long_document(text, doc.file_name)
            chunks = long_result.chunks
            doc.chunks = chunks
            doc.metadata['long_doc_highlights'] = self.long_doc_processor.extract_financial_highlights(text)
        else:
            chunks = doc.chunks if doc.chunks else [text]

        # 事件抽取
        all_chunk_events = self.event_extractor.analyze_all_chunks(
            chunks, method=method, prompt_version=prompt_version
        )
        flat_events = [e for sub in all_chunk_events for e in sub]

        # 统计
        event_stats = self.event_extractor.get_event_statistics(all_chunk_events)

        # 热点分析
        event_dicts = []
        for e in flat_events:
            if hasattr(e, '__dict__'):
                event_dicts.append(e.__dict__)
            elif isinstance(e, dict):
                event_dicts.append(e)
        hot_analysis = self.hot_analyzer.analyze_batch_events([event_dicts])

        # 摘要
        summary = self._generate_summary(text, event_stats, flat_events)

        # 图表
        chart_paths = self.chart_gen.generate_financial_charts(
            event_stats, doc.file_name
        )

        elapsed = time.time() - start_time

        # 监控日志
        self.monitor.log_call(text[:500], event_stats, elapsed)

        report = AnalysisReport(
            doc_name=doc.file_name,
            analysis_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            document_info={
                'file_name': doc.file_name,
                'file_type': doc.file_type,
                'text_length': len(text),
                'num_chunks': len(chunks),
                'highlights': doc.metadata.get('long_doc_highlights', {}),
            },
            event_extraction={
                'statistics': event_stats,
                'events': [
                    {
                        'event_type': e.event_type,
                        'direction': e.direction,
                        'impact_metrics': e.impact_metrics,
                        'time_range': e.time_range,
                        'reason': e.reason[:200],
                        'source_text': e.source_text[:300],
                        'confidence': e.confidence,
                    }
                    for e in flat_events
                ],
                'hot_analysis': hot_analysis,
            },
            summary=summary,
            metadata={
                'analysis_duration_seconds': round(elapsed, 2),
                'method': method,
                'prompt_version': prompt_version,
                'model_used': bool(self.model_service),
            },
            chart_paths=chart_paths,
        )

        self.last_report = report
        logger.info(f"分析完成，耗时: {elapsed:.2f}秒")
        return report

    def _generate_summary(self, text: str, event_stats: Dict,
                          events: List) -> Dict:
        """生成分析摘要"""
        by_type = event_stats.get('by_type', {})
        by_dir = event_stats.get('by_direction', {})
        return {
            "document_type": "金融文本",
            "core_events": [f"{k}: {v}条" for k, v in by_type.items()],
            "key_metrics": list(event_stats.get('by_type', {}).keys()),
            "risk_alerts": [e.reason[:100] for e in events[:5] if e.direction == '负面'],
            "trend": "正面" if by_dir.get('正面', 0) > by_dir.get('负面', 0) else "负面",
            "total_events": event_stats.get('total_events', 0),
        }

    def analyze_text(self, text: str, doc_name: str = "text_input",
                     prompt_version: str = "v1_structured",
                     method: str = "keyword") -> AnalysisReport:
        """分析纯文本"""
        doc = self.preprocessor.process_document_from_text(text, doc_name)
        if len(text) > 3000:
            long_result = self.long_doc_processor.process_long_document(text, doc_name)
            chunks = long_result.chunks
        else:
            chunks = doc.chunks if doc.chunks else [text]

        start_time = time.time()
        all_chunk_events = self.event_extractor.analyze_all_chunks(
            chunks, method=method, prompt_version=prompt_version
        )
        flat_events = [e for sub in all_chunk_events for e in sub]
        event_stats = self.event_extractor.get_event_statistics(all_chunk_events)

        event_dicts = []
        for e in flat_events:
            if hasattr(e, '__dict__'):
                event_dicts.append(e.__dict__)
            elif isinstance(e, dict):
                event_dicts.append(e)

        hot_analysis = self.hot_analyzer.analyze_batch_events([event_dicts])
        summary = self._generate_summary(text, event_stats, flat_events)
        chart_paths = self.chart_gen.generate_financial_charts(event_stats, doc_name)
        elapsed = time.time() - start_time

        report = AnalysisReport(
            doc_name=doc_name,
            analysis_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            document_info={
                'text_length': len(text),
                'num_chunks': len(chunks),
                'highlights': self.long_doc_processor.extract_financial_highlights(text) if len(text) > 3000 else {},
            },
            event_extraction={
                'statistics': event_stats,
                'events': [{
                    'event_type': e.event_type, 'direction': e.direction,
                    'impact_metrics': e.impact_metrics, 'time_range': e.time_range,
                    'reason': e.reason[:200], 'source_text': e.source_text[:300],
                } for e in flat_events],
                'hot_analysis': hot_analysis,
            },
            summary=summary,
            metadata={'analysis_duration_seconds': round(elapsed, 2)},
            chart_paths=chart_paths,
        )
        self.last_report = report
        return report

    def analyze_batch(self, file_paths: List[str],
                      prompt_version: str = "v1_structured",
                      method: str = "keyword") -> List[AnalysisReport]:
        """批量分析多个文档"""
        reports = []
        for i, fp in enumerate(file_paths):
            logger.info(f"批量 [{i+1}/{len(file_paths)}]: {fp}")
            try:
                report = self.analyze_document(fp, prompt_version, method)
                reports.append(report)
            except Exception as e:
                logger.error(f"失败 [{fp}]: {e}")
                reports.append(AnalysisReport(
                    doc_name=os.path.basename(fp),
                    analysis_time="",
                    metadata={'error': str(e)}
                ))
        return reports

    def export_report(self, report: AnalysisReport = None,
                      formats: List[str] = None) -> Dict[str, str]:
        """导出报告"""
        if report is None:
            report = self.last_report
        if report is None:
            raise ValueError("无报告可导出")
        if formats is None:
            formats = ["json", "csv"]
        return self.exporter.export_analysis_package(
            clause_results=report.event_extraction,
            risk_results=report.risk_analysis,
            summary=report.summary,
            contract_name=report.doc_name,
            formats=formats
        )

    def compare_prompt_versions(self, file_path: str) -> Dict[str, Any]:
        """对比不同提示方案"""
        versions = ["v1_structured", "v2_cot", "v3_fewshot"]
        results = {}
        for version in versions:
            start = time.time()
            report = self.analyze_document(file_path, prompt_version=version)
            elapsed = time.time() - start
            stats = report.event_extraction.get('statistics', {})
            results[version] = {
                'version_name': self.prompt_manager.get_version_info(version).get('name', version),
                'elapsed_seconds': elapsed,
                'event_count': stats.get('total_events', 0),
                'types_found': stats.get('types_found', 0),
                'coverage': stats.get('coverage_rate', 0),
            }
        return results

    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        metrics = {'model_loaded': bool(self.model_service)}
        metrics['session_stats'] = self.monitor.get_session_stats()
        if self.last_report:
            metrics['last_analysis'] = {
                'doc_name': self.last_report.doc_name,
                'duration': self.last_report.metadata.get('analysis_duration_seconds'),
            }
        return metrics

    def initialize_model(self):
        """初始化模型"""
        if self.model_service:
            return self.model_service.initialize()
        return False

    def cleanup(self):
        """清理资源"""
        if self.model_service:
            self.model_service.cleanup()
