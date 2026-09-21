"""
图表生成器 - 金融事件可视化和实验对比可视化
Financial Chart Generator Module
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'primary': '#2b6cb0', 'secondary': '#48bb78', 'danger': '#e53e3e',
    'warning': '#dd6b20', 'info': '#4299e1', 'light': '#edf2f7',
}

EVENT_COLORS = ['#2b6cb0', '#48bb78', '#e53e3e', '#dd6b20', '#805ad5', '#319795']


class ChartGenerator:
    """图表生成器 - 金融事件分析可视化"""

    def __init__(self, output_dir: str = "./charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 金融事件可视化 ==========

    def event_distribution_bar(self, event_stats: Dict, doc_name: str = "",
                                save: bool = True) -> str:
        """事件类型分布柱状图"""
        by_type = event_stats.get('by_type', {})
        if not by_type:
            return ""

        types = list(by_type.keys())
        counts = list(by_type.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(types, counts, color=EVENT_COLORS[:len(types)], edgecolor='white', linewidth=1.2)
        ax.set_xlabel('事件类型', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title(f'金融事件类型分布 - {doc_name}' if doc_name else '金融事件类型分布',
                     fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        path = str(self.output_dir / 'event_distribution.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def direction_pie(self, event_stats: Dict, save: bool = True) -> str:
        """事件方向分布饼图"""
        by_dir = event_stats.get('by_direction', {})
        if not by_dir:
            return ""

        labels = [k for k, v in by_dir.items() if v > 0]
        counts = [by_dir[l] for l in labels]
        colors_map = {'正面': '#48bb78', '负面': '#e53e3e', '中性': '#4299e1'}
        colors = [colors_map.get(l, '#a0aec0') for l in labels]

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            counts, labels=[f'{l}\n({c}条)' for l, c in zip(labels, counts)],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=[0.05] * len(labels)
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('事件方向分布', fontsize=14, fontweight='bold')

        plt.tight_layout()
        path = str(self.output_dir / 'direction_pie.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    # ========== 实验对比可视化 ==========

    def prompt_comparison_chart(self, comparison_data: Dict, save: bool = True) -> str:
        """提示模板对比图"""
        if not comparison_data:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        versions = list(comparison_data.keys())
        version_labels = [v.replace('keyword', '关键词').replace('rule_enhanced', '增强规则').replace('hybrid', '混合')
                         for v in versions]

        coverage_data = [comparison_data[v].get('coverage', 0) for v in versions]
        axes[0].bar(version_labels, coverage_data, color=['#4299e1', '#48bb78', '#ed8936'])
        axes[0].set_title('事件类型覆盖率')
        axes[0].set_ylim(0, 1.2)
        for i, c in enumerate(coverage_data):
            axes[0].text(i, c + 0.02, f'{c:.1%}', ha='center')

        event_data = [comparison_data[v].get('event_count', 0) for v in versions]
        axes[1].bar(version_labels, event_data, color=['#4299e1', '#48bb78', '#ed8936'])
        axes[1].set_title('事件抽取数量')

        plt.suptitle('提示方案对比（金融事件抽取）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = str(self.output_dir / 'prompt_comparison.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def quantization_comparison_chart(self, configs: List[Dict], save: bool = True) -> str:
        """量化方案对比图"""
        if not configs:
            return ""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        methods = [c.get('method', '') for c in configs]

        mem_data = [c.get('memory_usage_gb', 0) for c in configs]
        axes[0].bar(methods, mem_data, color=['#4299e1', '#48bb78', '#ed8936'])
        axes[0].set_title('显存占用 (GB)')
        axes[0].set_ylabel('GB')
        for i, m in enumerate(mem_data):
            axes[0].text(i, m + 0.05, f'{m}GB', ha='center', fontweight='bold')

        lat_data = [c.get('inference_latency_ms', 0) for c in configs]
        axes[1].bar(methods, lat_data, color=['#4299e1', '#48bb78', '#ed8936'])
        axes[1].set_title('推理延迟 (ms)')
        axes[1].set_ylabel('ms')

        qual_data = [c.get('output_quality_f1', 0) for c in configs]
        axes[2].bar(methods, qual_data, color=['#4299e1', '#48bb78', '#ed8936'])
        axes[2].set_title('输出质量 (F1)')
        axes[2].set_ylim(0, 1)
        for i, q in enumerate(qual_data):
            axes[2].text(i, q + 0.02, f'{q:.2f}', ha='center', fontweight='bold')

        plt.suptitle('量化部署方案对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = str(self.output_dir / 'quantization_comparison.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def batch_efficiency_chart(self, batch_sizes: List[int],
                                throughputs: List[float],
                                latencies: List[float], save: bool = True) -> str:
        """批次处理对比图"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(batch_sizes, throughputs, 'o-', color='#4299e1', linewidth=2, label='吞吐量(tokens/s)')
        ax2.plot(batch_sizes, latencies, 's-', color='#e53e3e', linewidth=2, label='延迟(ms)')

        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('吞吐量 (tokens/s)', color='#4299e1')
        ax2.set_ylabel('延迟 (ms)', color='#e53e3e')
        ax1.set_title('批次处理效率对比', fontsize=14, fontweight='bold')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        path = str(self.output_dir / 'batch_efficiency.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def per_type_performance_chart(self, type_perf: List[Dict],
                                    risk_perf: List[Dict], save: bool = True) -> str:
        """分类型性能评估图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        types = [p.get('type', '') for p in type_perf]
        prec = [p.get('precision', 0) for p in type_perf]
        rec = [p.get('recall', 0) for p in type_perf]
        f1 = [p.get('f1', 0) for p in type_perf]
        x = np.arange(len(types))
        width = 0.25

        ax.bar(x - width, prec, width, label='精确率', color='#4299e1')
        ax.bar(x, rec, width, label='召回率', color='#48bb78')
        ax.bar(x + width, f1, width, label='F1值', color='#ed8936')
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=30, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_title('各事件类型抽取性能', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        path = str(self.output_dir / 'per_type_performance.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def knowledge_base_comparison_chart(self, configs: List[Dict], save: bool = True) -> str:
        """知识库对比图"""
        if len(configs) < 2:
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(4)
        width = 0.35
        metrics = ['准确率', '精确率', '召回率', 'F1值']
        pure_data = [configs[0].get('clause_accuracy', 0), configs[0].get('risk_precision', 0),
                     configs[0].get('risk_recall', 0), configs[0].get('f1_score', 0)]
        enhanced_data = [configs[1].get('clause_accuracy', 0), configs[1].get('risk_precision', 0),
                         configs[1].get('risk_recall', 0), configs[1].get('f1_score', 0)]
        ax.bar(x - width/2, pure_data, width, label='纯规则', color='#4299e1')
        ax.bar(x + width/2, enhanced_data, width, label='规则+知识库', color='#48bb78')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('知识库增强效果对比', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        path = str(self.output_dir / 'knowledge_base_comparison.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def parameter_tuning_chart(self, configs: List[Dict], save: bool = True) -> str:
        """参数调优图"""
        if not configs:
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        temps = [c.get('temperature', 0) for c in configs]
        consistency = [c.get('output_consistency', 0) for c in configs]
        diversity = [c.get('output_diversity', 0) for c in configs]
        accuracy = [c.get('legal_accuracy', 0) for c in configs]

        ax.plot(temps, consistency, 'o-', label='输出一致性', color='#4299e1', linewidth=2)
        ax.plot(temps, diversity, 's-', label='输出多样性', color='#48bb78', linewidth=2)
        ax.plot(temps, accuracy, '^-', label='金融准确性', color='#e53e3e', linewidth=2)
        ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='推荐值(0.3)')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Score')
        ax.set_title('生成参数调优实验', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = str(self.output_dir / 'parameter_tuning.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def hot_event_trend_chart(self, trend_data: Dict, save: bool = True) -> str:
        """热点事件趋势图"""
        trend = trend_data.get('trend', {})
        labels = trend_data.get('labels', [])
        if not trend or not labels:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))
        for etype in set().union(*[v.get('by_type', {}).keys() for v in trend.values()]):
            counts = [trend[l]['by_type'].get(etype, 0) for l in labels]
            ax.plot(labels, counts, 'o-', label=etype, linewidth=2)

        ax.set_xlabel('时间段')
        ax.set_ylabel('事件数量')
        ax.set_title('金融事件时间趋势', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = str(self.output_dir / 'hot_event_trend.png') if save else ""
        if save:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    def generate_financial_charts(self, event_stats: Dict, doc_name: str = "") -> Dict[str, str]:
        """生成金融分析所有图表"""
        paths = {}
        paths['event_bar'] = self.event_distribution_bar(event_stats, doc_name)
        paths['direction_pie'] = self.direction_pie(event_stats)
        return {k: v for k, v in paths.items() if v}
