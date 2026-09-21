"""
金融文本数据生成器 - 生成500+金融文本样本（含人工标注）
Financial Text Data Generator
"""

import os
import json
import random
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# 公司名称
COMPANIES = [
    "华泰科技", "龙芯半导体", "鼎盛医疗", "智云软件", "绿能新能源",
    "东方通信", "南方航空", "招商银行", "平安保险", "万科地产",
    "贵州茅台", "五粮液", "格力电器", "美的集团", "比亚迪",
    "宁德时代", "药明康德", "海康威视", "中芯国际", "腾讯控股",
]

# 行业
INDUSTRIES = [
    "科技", "半导体", "医疗", "软件", "新能源",
    "通信", "航空", "银行", "保险", "地产",
    "白酒", "家电", "汽车", "电池", "生物医药",
    "安防", "芯片", "互联网",
]

# 金融文本模板 - 5种类型
TEMPLATES = {
    "performance_briefing": [  # 业绩说明会纪要
        "公司{company}今日召开{year}年{quarter}业绩说明会。{ceo}表示，{quarter}公司实现营收{revenue}亿元，同比{revenue_change}{revenue_pct}%；净利润{profit}亿元，同比{profit_change}{profit_pct}%。{reason}。展望{next_period}，公司预计{outlook}。毛利率达到{gross_margin}%，较去年同期{gross_margin_change}。",
        "在{year}年{quarter}业绩说明会上，{company}管理层披露了最新经营情况。季度内公司实现营业收入{revenue}亿元，同比增长{revenue_pct}%。归属于上市公司股东的净利润{profit}亿元，{profit_change}{profit_pct}%。{reason}。公司表示{outlook}。",
    ],
    "announcement": [  # 上市公司公告
        "{company}（股票代码：{code}）发布{year}年度业绩预告。预计{year}年归属于上市公司股东的净利润为{profit}亿元，同比{profit_change}{profit_pct}%。{reason}。投资者请注意相关风险。",
        "{company}公告称，公司拟以自有资金{amount}亿元收购{target}公司{share}%股权。本次交易完成后，{target}将成为公司全资子公司。{reason}。",
        "{company}关于收到{regulator}《行政处罚决定书》的公告。公司因{reason}被处以{penalty}万元罚款。公司表示{response}。",
    ],
    "financial_news": [  # 财经新闻
        "{company}{year}年{quarter}财报出炉：营收{revenue}亿元{revenue_change}{revenue_pct}%，净利{profit}亿元同比{profit_change}{profit_pct}%。分析师认为，{reason}。多家机构发布研报{rating}评级。",
        "据行业分析师报告，受{reason}影响，{industry}板块{trend}。其中{company}表现亮眼，{metric}{change}{pct}%。预计{outlook}。",
    ],
    "research_report": [  # 研报摘要
        "我们维持对{company}的{rating}评级，目标价调整为{target_price}元。{reason}。预计公司{year}年EPS为{eps}元，对应PE为{pe}倍。",
        "首次覆盖{company}，给予{rating}评级。我们认为{reason}。公司核心产品{product}市占率持续提升，{metric}指标向好。",
    ],
    "risk_warning": [  # 风险提示
        "{company}发布风险提示公告。公司注意到{risk_factor}可能对经营产生影响。当前公司资产负债率为{debt_ratio}%，{risk_detail}。投资者请注意投资风险。",
        "鉴于{reason}，{company}调整了对{year}年的业绩预期。预计全年营收增速将从之前的{old_pct}%下调至{new_pct}%。{outlook}",
    ],
}

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
REVENUE_RANGE = (1.5, 500)
PROFIT_RANGE = (0.1, 100)
PCT_CHANGE = (5, 80)

# 人工标注结果模板
def generate_ground_truth(text: str) -> List[Dict]:
    """基于模板生成标注结果"""
    events = []

    # 提取营收信息
    import re
    rev_match = re.search(r'营收[约达]?(\d+[\.\d]*)亿', text)
    if rev_match:
        direction = "正面" if any(kw in text for kw in ["增长", "提升", "增加"]) else "负面"
        events.append({
            "event_type": "经营指标变动",
            "direction": direction,
            "impact_metrics": ["营收"],
            "time_range": "全年",
            "reason": f"营收为{rev_match.group(1)}亿元",
            "source_text": rev_match.group(0),
        })

    profit_match = re.search(r'净[利润]?[约为达]?(\d+[\.\d]*)亿', text)
    if profit_match:
        direction = "正面" if any(kw in text for kw in ["增长", "提升", "增加"]) else "负面"
        events.append({
            "event_type": "经营指标变动",
            "direction": direction,
            "impact_metrics": ["净利润"],
            "time_range": "本期",
            "reason": f"净利润为{profit_match.group(1)}亿元",
            "source_text": profit_match.group(0),
        })

    if "收购" in text or "并购" in text:
        events.append({
            "event_type": "并购动态",
            "direction": "中性" if "完成" in text else "正面",
            "impact_metrics": [],
            "time_range": "",
            "reason": "公司进行收购/并购活动",
            "source_text": text[text.find("收购" if "收购" in text else "并购")-20:text.find("收购" if "收购" in text else "并购")+40] if text.find("收购") >= 0 or text.find("并购") >= 0 else "收购相关",
        })

    if "风险" in text or "处罚" in text or "罚款" in text:
        events.append({
            "event_type": "风险提示",
            "direction": "负面",
            "impact_metrics": [],
            "time_range": "",
            "reason": text[:100],
            "source_text": text[:150],
        })

    if "评级" in text or "目标价" in text:
        events.append({
            "event_type": "评级变动",
            "direction": "正面" if "买入" in text or "增持" in text else "中性",
            "impact_metrics": [],
            "time_range": "",
            "reason": "机构发布研报评级",
            "source_text": text[max(0, text.find("评级")-30):text.find("评级")+30] if "评级" in text else text[:100],
        })

    if not events:
        events.append({
            "event_type": "经营指标变动",
            "direction": "中性",
            "impact_metrics": [],
            "time_range": "",
            "reason": "常规经营信息披露",
            "source_text": text[:100],
        })

    return events


class FinancialDataGenerator:
    """金融文本数据生成器"""

    def __init__(self):
        self.doc_types = {
            "performance_briefing": "业绩说明会纪要",
            "announcement": "上市公司公告",
            "financial_news": "财经新闻",
            "research_report": "研报摘要",
            "risk_warning": "风险提示",
        }

    def generate_sample(self, doc_type: str = None) -> Dict:
        """生成单条样本"""
        if doc_type is None:
            doc_type = random.choice(list(TEMPLATES.keys()))

        company = random.choice(COMPANIES)
        year = random.choice([2023, 2024, 2025, 2026])
        quarter = random.choice(QUARTERS)
        revenue = round(random.uniform(*REVENUE_RANGE), 1)
        profit = round(random.uniform(*PROFIT_RANGE), 1)
        rev_change = random.choice(["增长", "下降"])
        profit_change = random.choice(["增长", "下降"])
        rev_pct = round(random.uniform(5, 80), 1)
        profit_pct = round(random.uniform(5, 80), 1)

        reasons = [
            "受益于行业需求旺盛",
            "受宏观经济环境影响",
            "得益于产品结构优化",
            "受原材料价格上涨影响",
            "核心产品市场份额提升",
            "新产品线实现量产",
            "成本控制效果显著",
            "受汇率波动影响",
            "行业竞争加剧",
            "政策利好推动",
        ]
        reason = random.choice(reasons)

        template = random.choice(TEMPLATES[doc_type])
        text = template.format(
            company=company,
            year=year,
            quarter=quarter,
            next_period=f"{year+1}年",
            revenue=revenue,
            profit=profit,
            revenue_change=rev_change,
            profit_change=profit_change,
            revenue_pct=rev_pct,
            profit_pct=profit_pct,
            reason=reason,
            gross_margin=round(random.uniform(20, 60), 1),
            gross_margin_change=random.choice(["提升", "下降"]) + str(round(random.uniform(1, 10), 1)) + "个百分点",
            outlook=random.choice([
                "保持稳健增长态势",
                "有望实现扭亏为盈",
                "将持续加大研发投入",
                "积极拓展海外市场",
                "维持全年经营目标不变",
            ]),
            code=str(random.randint(600000, 609999)),
            amount=round(random.uniform(0.5, 50), 2),
            target=random.choice(["华芯半导体", "瑞康医疗", "蓝云科技", "远景能源"]),
            share=random.randint(51, 100),
            regulator=random.choice(["证监会", "上交所", "深交所", "国家市场监督管理总局"]),
            penalty=round(random.uniform(10, 5000), 1),
            response=random.choice([
                "将积极配合整改",
                "对处罚决定无异议",
                "将提起行政复议",
            ]),
            industry=random.choice(INDUSTRIES),
            trend=random.choice(["整体向好", "面临压力", "稳步复苏"]),
            metric=random.choice(["营收", "利润", "毛利率", "市占率"]),
            change=random.choice(["增长", "下降", "提升", "下滑"]),
            pct=round(random.uniform(5, 50), 1),
            rating=random.choice(["买入", "增持", "中性", "减持"]),
            target_price=round(random.uniform(10, 500), 2),
            eps=round(random.uniform(0.5, 20), 2),
            pe=round(random.uniform(10, 100), 1),
            product=random.choice(["AI芯片", "新能源电池", "创新药", "工业软件", "智能传感器"]),
            risk_factor=random.choice(["原材料价格波动", "下游需求减弱", "汇率波动", "政策调整"]),
            debt_ratio=round(random.uniform(30, 85), 1),
            risk_detail=random.choice([
                "短期偿债压力较大",
                "应收账款周转放缓",
                "存货跌价风险增加",
            ]),
            old_pct=round(random.uniform(20, 50), 1),
            new_pct=round(random.uniform(5, 20), 1),
            ceo=random.choice(["张明", "李强", "王磊", "陈华"]),
        )

        ground_truth = generate_ground_truth(text)

        return {
            "text": text,
            "doc_type": doc_type,
            "doc_type_name": self.doc_types.get(doc_type, ""),
            "source": random.choice(["公司官网", "交易所", "财经媒体", "券商平台"]),
            "ground_truth": ground_truth,
        }

    def generate_samples(self, count: int = 500, output_dir: str = "./data/contracts") -> List[Dict]:
        """批量生成样本数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 确保覆盖多种文档类型
        samples = []
        doc_types = list(TEMPLATES.keys())
        per_type = count // len(doc_types)

        for dt in doc_types:
            for _ in range(per_type):
                samples.append(self.generate_sample(doc_type=dt))

        # 补足余数
        remaining = count - len(samples)
        for _ in range(remaining):
            samples.append(self.generate_sample())

        random.shuffle(samples)

        # 写入文件
        dataset = []
        for i, sample in enumerate(samples):
            filename = f"financial_{i+1:04d}.txt"
            filepath = output_path / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(sample["text"])

            dataset.append({
                "id": i + 1,
                "filename": filename,
                "doc_type": sample["doc_type"],
                "doc_type_name": sample["doc_type_name"],
                "source": sample["source"],
                "text": sample["text"],
                "ground_truth": sample["ground_truth"],
            })

        # 保存标注文件
        annotation_path = output_path / "annotations.json"
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"[OK] 已生成 {len(samples)} 条金融文本样本")
        print(f"[OK] 保存路径: {output_path}")
        print(f"[OK] 标注文件: {annotation_path}")
        print(f"[OK] 类型分布:")
        from collections import Counter
        type_counter = Counter(s["doc_type_name"] for s in samples)
        for t, c in type_counter.most_common():
            print(f"   - {t}: {c}条")

        return dataset
