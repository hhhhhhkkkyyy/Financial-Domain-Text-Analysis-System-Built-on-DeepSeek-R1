"""
系统监控与日志模块 - 记录每次调用的输入、输出及耗时
System Monitoring & Logging Module
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class InferenceLogger:
    """推理日志记录器 - 记录每次调用的完整信息"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = []

    def log_call(self, input_text: str, output: Any, elapsed: float,
                 model_name: str = "DeepSeek-R1-1.5B", status: str = "success",
                 extra: Optional[Dict] = None):
        """记录一次调用"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input_length": len(input_text),
            "output": str(output)[:1000] if output else "",
            "elapsed_seconds": round(elapsed, 3),
            "status": status,
            "extra": extra or {},
        }
        self.session_log.append(record)

        log_file = self.log_dir / f"inference_{self.current_session}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_session_stats(self) -> Dict:
        """获取当前会话统计"""
        if not self.session_log:
            return {"total_calls": 0}

        total = len(self.session_log)
        success = sum(1 for r in self.session_log if r["status"] == "success")
        total_time = sum(r["elapsed_seconds"] for r in self.session_log)
        total_input = sum(r["input_length"] for r in self.session_log)

        return {
            "total_calls": total,
            "success_calls": success,
            "failed_calls": total - success,
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_call": round(total_time / total, 3) if total > 0 else 0,
            "total_input_chars": total_input,
            "avg_input_length": round(total_input / total, 1) if total > 0 else 0,
        }

    def get_log_by_time(self, start_time: str, end_time: str) -> List[Dict]:
        """按时间范围查询日志"""
        result = []
        for record in self.session_log:
            if start_time <= record["timestamp"] <= end_time:
                result.append(record)
        return result

    def export_logs(self, format: str = "json") -> str:
        """导出日志"""
        filepath = self.log_dir / f"session_{self.current_session}.{format}"
        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.session_log, f, ensure_ascii=False, indent=2)
        return str(filepath)
