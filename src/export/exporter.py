"""
结果导出模块 - 支持JSON、CSV、Markdown、HTML格式输出
Result Export Module (Supports JSON, CSV, Markdown, HTML)
"""

import json
import os
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ResultExporter:
    """分析结果导出器"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self, data: Any, filename: str = None,
                indent: int = 2, ensure_ascii: bool = False) -> str:
        """导出为JSON"""
        if filename is None:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return str(filepath)

    def to_csv(self, data: List[Dict], filename: str = None) -> str:
        """导出为CSV"""
        if filename is None:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.output_dir / filename

        if isinstance(data, dict):
            data = [data]
        if data:
            flattened = []
            for item in data:
                flat = {}
                for k, v in item.items():
                    if isinstance(v, (dict, list)):
                        flat[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        flat[k] = v
                flattened.append(flat)
            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                if flattened:
                    writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened)
        return str(filepath)

    def to_markdown(self, analysis_result: Dict, filename: str = None) -> str:
        """导出为Markdown"""
        if filename is None:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_dir / filename
        md = self._build_markdown_report(analysis_result)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        return str(filepath)

    def to_html(self, analysis_result: Dict, filename: str = None) -> str:
        """导出为HTML"""
        if filename is None:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        md_content = self._build_markdown_report(analysis_result)

        try:
            import markdown
            html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        except ImportError:
            html_body = f"<pre>{md_content}</pre>"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>金融文本分析报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
            max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.8;
            color: #333; background: #f8f9fa;
        }}
        .container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; margin-top: 30px; }}
        h3 {{ color: #2b6cb0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: left; }}
        th {{ background: #edf2f7; }}
        .positive {{ color: #38a169; font-weight: bold; }}
        .negative {{ color: #e53e3e; font-weight: bold; }}
        .neutral {{ color: #4299e1; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
        <hr>
        <footer>
            <p><em>本报告由金融文本智能分析系统自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </footer>
    </div>
</body>
</html>"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        return str(filepath)

    def export_analysis_package(self, clause_results: Dict,
                                 risk_results: List,
                                 summary: Any,
                                 contract_name: str = "",
                                 formats: List[str] = None) -> Dict[str, str]:
        """导出完整分析包"""
        if formats is None:
            formats = ["json", "csv"]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = contract_name.replace('/', '_').replace('\\', '_')[:50]

        summary_dict = summary
        if hasattr(summary, 'to_dict'):
            summary_dict = summary.to_dict()
        elif not isinstance(summary, dict):
            summary_dict = {"raw": str(summary)}

        analysis_data = {
            "doc_name": safe_name,
            "analysis_time": datetime.now().isoformat(),
            "event_extraction": clause_results,
            "risk_analysis": risk_results,
            "summary": summary_dict,
        }

        exported = {}
        for fmt in formats:
            fmt_lower = fmt.lower()
            if fmt_lower == "json":
                path = self.to_json(analysis_data, f"{safe_name}_analysis_{timestamp}.json")
            elif fmt_lower == "csv":
                events = clause_results.get('events', []) if isinstance(clause_results, dict) else []
                if events:
                    path = self.to_csv(events, f"{safe_name}_events_{timestamp}.csv")
                else:
                    continue
            elif fmt_lower == "markdown":
                path = self.to_markdown(analysis_data, f"{safe_name}_analysis_{timestamp}.md")
            elif fmt_lower == "html":
                path = self.to_html(analysis_data, f"{safe_name}_analysis_{timestamp}.html")
            else:
                continue
            exported[fmt_lower] = path

        return exported

    def _build_markdown_report(self, data: Dict) -> str:
        """构建Markdown报告"""
        md = f"""# 金融文本智能分析报告

**文档名称**: {data.get('doc_name', '未指定')}
**分析时间**: {data.get('analysis_time', datetime.now().isoformat())}

---

## 一、事件抽取结果

"""
        events = data.get('event_extraction', {})
        if isinstance(events, dict):
            stats = events.get('statistics', {})
            md += f"**抽取事件总数**: {stats.get('total_events', 0)}\n\n"
            by_type = stats.get('by_type', {})
            if by_type:
                md += "| 事件类型 | 数量 |\n|---------|------|\n"
                for t, c in by_type.items():
                    md += f"| {t} | {c} |\n"

            event_list = events.get('events', [])
            if event_list:
                md += "\n### 事件详情\n\n"
                for i, e in enumerate(event_list[:10]):
                    direction_icon = {'正面': '🟢', '负面': '🔴', '中性': '🔵'}.get(e.get('direction', ''), '⚪')
                    md += f"""#### {i+1}. {direction_icon} {e.get('event_type', '')}

- **方向**: {e.get('direction', '')}
- **影响指标**: {', '.join(e.get('impact_metrics', [])) or '无'}
- **时间范围**: {e.get('time_range', '') or '未指定'}
- **原因**: {e.get('reason', '')[:200]}
- **原文**: {e.get('source_text', '')[:200]}

"""

        md += "\n---\n\n## 二、风险分析\n\n"
        risks = data.get('risk_analysis', [])
        if risks:
            for r in risks[:5]:
                md += f"- **{r.get('risk_category', '')}** [{r.get('risk_level', '')}]: {r.get('risk_description', '')[:200]}\n"

        md += "\n---\n\n## 三、摘要\n\n"
        summary = data.get('summary', {})
        if isinstance(summary, dict):
            md += f"- **事件总数**: {summary.get('total_events', 0)}\n"
            md += f"- **趋势**: {summary.get('trend', '未知')}\n"
            for evt in summary.get('core_events', []):
                md += f"- {evt}\n"

        md += "\n---\n\n*本报告由金融文本智能分析系统基于DeepSeek-R1生成*"
        return md
