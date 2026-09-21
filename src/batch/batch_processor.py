"""
批量金融文档处理模块 - 批量审查和分析多份金融文档
Batch Contract Processing Module
实训部分模块
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """单个批处理任务"""
    file_path: str
    file_name: str
    status: str = "pending"  # pending/processing/completed/failed
    start_time: float = 0
    end_time: float = 0
    elapsed_seconds: float = 0
    result: Any = None
    error: str = ""


@dataclass
class BatchReport:
    """批处理报告"""
    batch_id: str
    start_time: str
    end_time: str
    total_files: int
    successful: int
    failed: int
    total_elapsed: float
    avg_elapsed: float
    tasks: List[BatchTask]
    summary: Dict = field(default_factory=dict)


class BatchProcessor:
    """
    批量金融文档处理器
    支持并行批量处理多个金融文档，记录处理状态和耗时
    """

    def __init__(self, max_workers: int = 4, output_dir: str = "./output/batch"):
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: List[BatchTask] = []

    def process_batch(self, file_paths: List[str],
                      process_func: Callable,
                      progress_callback: Callable = None,
                      **kwargs) -> BatchReport:
        """
        批量处理文件

        Args:
            file_paths: 文件路径列表
            process_func: 处理函数，签名为 func(file_path, **kwargs) -> Any
            progress_callback: 进度回调 func(current, total, task)
            **kwargs: 传递给process_func的额外参数
        """
        batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        self.tasks = [
            BatchTask(file_path=fp, file_name=os.path.basename(fp))
            for fp in file_paths
        ]

        total = len(self.tasks)

        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single, task, process_func, **kwargs): task
                for task in self.tasks
            }

            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    task.status = "completed"
                    task.result = result
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"任务失败 [{task.file_name}]: {e}")

                task.end_time = time.time()
                task.elapsed_seconds = task.end_time - task.start_time
                completed += 1

                if progress_callback:
                    progress_callback(completed, total, task)

        end_time = datetime.now()
        successful = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        elapsed_times = [t.elapsed_seconds for t in self.tasks if t.elapsed_seconds > 0]

        report = BatchReport(
            batch_id=batch_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_files=total,
            successful=successful,
            failed=failed,
            total_elapsed=(end_time - start_time).total_seconds(),
            avg_elapsed=sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0,
            tasks=self.tasks,
            summary=self._build_summary(self.tasks)
        )

        # 保存报告
        self._save_report(report)

        return report

    def _process_single(self, task: BatchTask, process_func: Callable,
                        **kwargs) -> Any:
        """处理单个文件"""
        task.status = "processing"
        task.start_time = time.time()
        logger.info(f"开始处理: {task.file_name}")
        result = process_func(task.file_path, **kwargs)
        logger.info(f"处理完成: {task.file_name}")
        return result

    def process_contract_batch(self, file_paths: List[str],
                                analyzer_func: Callable,
                                progress_callback: Callable = None) -> BatchReport:
        """
        批量分析金融文档（便捷方法）
        analyzer_func: 金融文档分析函数，返回分析结果字典
        """
        return self.process_batch(
            file_paths, analyzer_func,
            progress_callback=progress_callback
        )

    def get_task_status(self, file_name: str = None) -> List[Dict]:
        """获取任务状态"""
        if file_name:
            tasks = [t for t in self.tasks if t.file_name == file_name]
        else:
            tasks = self.tasks

        return [
            {
                'file_name': t.file_name,
                'status': t.status,
                'elapsed': f"{t.elapsed_seconds:.2f}s",
                'error': t.error
            }
            for t in tasks
        ]

    def _build_summary(self, tasks: List[BatchTask]) -> Dict:
        """构建批处理摘要"""
        status_count = {'completed': 0, 'failed': 0, 'pending': 0}
        for t in tasks:
            status_count[t.status] = status_count.get(t.status, 0) + 1

        return {
            'completion_rate': status_count['completed'] / max(len(tasks), 1),
            'failure_rate': status_count['failed'] / max(len(tasks), 1),
            'status_distribution': status_count,
        }

    def _save_report(self, report: BatchReport):
        """保存批处理报告"""
        report_path = self.output_dir / f"{report.batch_id}_report.json"
        report_data = {
            'batch_id': report.batch_id,
            'start_time': report.start_time,
            'end_time': report.end_time,
            'total_files': report.total_files,
            'successful': report.successful,
            'failed': report.failed,
            'total_elapsed': f"{report.total_elapsed:.2f}s",
            'avg_elapsed': f"{report.avg_elapsed:.2f}s",
            'summary': report.summary,
            'tasks': [
                {
                    'file_name': t.file_name,
                    'status': t.status,
                    'elapsed': f"{t.elapsed_seconds:.2f}s",
                    'error': t.error,
                }
                for t in report.tasks
            ]
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"批处理报告已保存: {report_path}")

    def generate_batch_summary_markdown(self, report: BatchReport) -> str:
        """生成批处理摘要(Markdown)"""
        md = f"""# 批量金融文档处理报告

## 基本信息
- **批次ID**: {report.batch_id}
- **开始时间**: {report.start_time}
- **结束时间**: {report.end_time}
- **总耗时**: {report.total_elapsed:.2f}秒

## 处理统计
| 指标 | 数值 |
|------|------|
| 文件总数 | {report.total_files} |
| 成功处理 | {report.successful} ✅ |
| 处理失败 | {report.failed} ❌ |
| 成功率 | {report.summary['completion_rate']:.1%} |
| 平均耗时 | {report.avg_elapsed:.2f}秒/文件 |

## 处理详情
| 文件 | 状态 | 耗时 |
|------|------|------|
"""
        for t in report.tasks:
            status_icon = "✅" if t.status == "completed" else "❌"
            md += f"| {t.file_name} | {status_icon} {t.status} | {t.elapsed_seconds:.2f}s |\n"

        # 错误汇总
        failed_tasks = [t for t in report.tasks if t.status == "failed"]
        if failed_tasks:
            md += "\n## 错误汇总\n\n"
            for t in failed_tasks:
                md += f"- **{t.file_name}**: {t.error}\n"

        return md
