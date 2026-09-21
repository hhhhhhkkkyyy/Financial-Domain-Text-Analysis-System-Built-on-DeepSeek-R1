# -*- coding: utf-8 -*-
"""测试所有样本文件"""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.analysis_pipeline import FinancialAnalysisPipeline

pipeline = FinancialAnalysisPipeline()
files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '*.txt')))
files = [f for f in files if not f.endswith('run_test.py')]

print("=" * 60)
print("金融文本测试结果")
print("=" * 60)

total_events = 0
all_types = set()

for f in files:
    try:
        report = pipeline.analyze_document(f)
        stats = report.event_extraction['statistics']
        events = report.event_extraction['events']
        total_events += stats['total_events']
        found_types = set(stats.get('by_type', {}).keys())
        all_types.update(found_types)

        print(f"\n[{report.doc_name}]")
        print(f"  事件: {stats['total_events']}条 | 类型: {stats['types_found']}/{stats['types_total']} | 耗时: {report.metadata['analysis_duration_seconds']}秒")
        print(f"  类型分布: {json.dumps(stats.get('by_type', {}), ensure_ascii=False)}")
        for e in events[:3]:
            print(f"  -> [{e['event_type']}] {e['direction']} | {e.get('reason', '')[:80]}")
    except Exception as ex:
        print(f"\n[Error] {os.path.basename(f)}: {ex}")

print("\n" + "=" * 60)
print(f"总计: {len(files)}个文件, {total_events}条事件, {len(all_types)}类事件类型")
print(f"覆盖类型: {all_types}")
print("=" * 60)
