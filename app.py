"""
金融文本智能分析系统 - 主入口
Financial Text Intelligent Analysis System - Main Entry

运行方式:
1. Web界面: streamlit run app.py
2. 命令行分析: python app.py analyze <file_path> --model-path <path>
3. 批量处理: python app.py batch <directory_path>
4. 实验对比: python app.py experiment <data_directory>
5. 生成数据: python app.py demo --count 500
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis_pipeline import FinancialAnalysisPipeline, AnalysisReport
from src.preprocessing.document_processor import DocumentPreprocessor
from src.export.exporter import ResultExporter
from src.experiments.experiment_runner import ExperimentRunner
from src.utils.model_service import ModelConfig
from src.utils.prompt_templates import PromptManager


def cmd_analyze(args):
    """命令行分析单个文件"""
    print(f"正在分析: {args.file}")

    pipeline = FinancialAnalysisPipeline(
        use_model=args.use_model,
        model_path=args.model_path or ""
    )

    if args.use_model and args.model_path:
        print(f"使用模型: {args.model_path}")
        pipeline.initialize_model()

    report = pipeline.analyze_document(
        args.file,
        prompt_version=args.prompt_version,
        method=args.method
    )

    print("\n" + "=" * 60)
    print(f"分析完成！")
    stats = report.event_extraction.get('statistics', {})
    print(f"抽取事件: {stats.get('total_events', 0)} 条")
    print(f"事件类型: {stats.get('types_found', 0)}/{stats.get('types_total', 6)} 类")

    events = report.event_extraction.get('events', [])
    for e in events[:5]:
        icon = {'正面': '🟢', '负面': '🔴', '中性': '🔵'}.get(e.get('direction', ''), '⚪')
        print(f"  {icon} [{e.get('event_type', '')}] {e.get('reason', '')[:80]}")

    exporter = ResultExporter()
    exported = exporter.export_analysis_package(
        clause_results=report.event_extraction,
        risk_results=report.risk_analysis,
        summary=report.summary,
        contract_name=report.doc_name,
        formats=args.export_formats.split(',')
    )
    print(f"报告已导出:")
    for fmt, path in exported.items():
        print(f"  - {fmt}: {path}")

    if args.use_model:
        pipeline.cleanup()


def cmd_batch(args):
    """命令行批量处理"""
    dir_path = Path(args.directory)
    files = list(dir_path.glob('*.txt')) + list(dir_path.glob('*.md')) + \
            list(dir_path.glob('*.csv')) + list(dir_path.glob('*.pdf')) + list(dir_path.glob('*.docx'))

    print(f"批量处理目录: {args.directory}")
    print(f"找到 {len(files)} 个文件")

    pipeline = FinancialAnalysisPipeline(use_model=args.use_model, model_path=args.model_path or "")
    reports = pipeline.analyze_batch(
        [str(f) for f in files],
        prompt_version=args.prompt_version
    )

    success = sum(1 for r in reports if not r.metadata.get('error'))
    print(f"成功: {success}/{len(reports)}")

    for report in reports:
        status = "❌" if report.metadata.get('error') else "✅"
        stats = report.event_extraction.get('statistics', {})
        events = stats.get('total_events', 0)
        print(f"  {status} {report.doc_name}: {events}条事件")


def cmd_experiment(args):
    """命令行运行实验"""
    dir_path = Path(args.data_dir)
    test_files = [str(f) for f in dir_path.glob('*')
                 if f.suffix in ['.txt', '.md', '.csv', '.pdf', '.docx']]

    print(f"运行对比实验")
    print(f"测试文件: {len(test_files)} 个")

    runner = ExperimentRunner()

    if args.experiment_type == 'all':
        results = runner.run_all_experiments(test_files)
    elif args.experiment_type == 'prompt':
        results = [runner.run_prompt_comparison(test_files)]
    elif args.experiment_type == 'quantization':
        results = [runner.run_quantization_comparison()]
    elif args.experiment_type == 'batch_size':
        results = [runner.run_batch_size_comparison()]
    elif args.experiment_type == 'multi_type':
        results = [runner.run_multi_type_evaluation()]
    else:
        print(f"未知实验类型: {args.experiment_type}")
        return

    for r in results:
        print(f"\n{'='*60}")
        print(f"📋 {r.experiment_name}")
        print(f"📊 配置数: {len(r.configurations)}")
        print(f"\n{r.analysis[:500]}...")


def cmd_demo(args):
    """生成金融文本样本数据"""
    print(f"正在生成金融文本样本数据...")

    from src.data_generator_financial import FinancialDataGenerator
    generator = FinancialDataGenerator()
    generator.generate_samples(
        count=args.count or 500,
        output_dir=args.output or "./data/contracts"
    )


def main():
    parser = argparse.ArgumentParser(
        description="金融文本智能分析系统 - Financial Text Intelligent Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python app.py analyze text.txt                                # 分析单个文件
  python app.py analyze text.txt --use-model --model-path D:\\model  # 使用本地模型
  python app.py batch ./data/contracts/                         # 批量处理
  python app.py experiment ./data/ --type all                   # 运行所有实验
  python app.py demo --count 500                                # 生成500条样本
  streamlit run src/web/streamlit_app.py                        # 启动Web界面
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    analyze_parser = subparsers.add_parser('analyze', help='分析单个文件')
    analyze_parser.add_argument('file', help='文件路径')
    analyze_parser.add_argument('--use-model', action='store_true', help='使用LLM模型')
    analyze_parser.add_argument('--model-path', default='', help='本地模型路径')
    analyze_parser.add_argument('--prompt-version', default='v1_structured', help='提示方案版本')
    analyze_parser.add_argument('--method', default='keyword',
                               choices=['keyword', 'llm', 'hybrid'], help='分析方法')
    analyze_parser.add_argument('--export-formats', default='json,csv',
                               help='导出格式(逗号分隔)')

    batch_parser = subparsers.add_parser('batch', help='批量处理文件')
    batch_parser.add_argument('directory', help='文件目录')
    batch_parser.add_argument('--use-model', action='store_true')
    batch_parser.add_argument('--model-path', default='')
    batch_parser.add_argument('--prompt-version', default='v1_structured')

    exp_parser = subparsers.add_parser('experiment', help='运行对比实验')
    exp_parser.add_argument('data_dir', help='测试数据目录')
    exp_parser.add_argument('--type', dest='experiment_type', default='all',
                           choices=['all', 'prompt', 'quantization', 'batch_size', 'multi_type'])

    demo_parser = subparsers.add_parser('demo', help='生成样本数据')
    demo_parser.add_argument('--count', type=int, default=500, help='样本数量')
    demo_parser.add_argument('--output', help='输出目录')

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'batch':
        cmd_batch(args)
    elif args.command == 'experiment':
        cmd_experiment(args)
    elif args.command == 'demo':
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
