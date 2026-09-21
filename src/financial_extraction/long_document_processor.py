"""
长文档分段处理模块 - 针对上市公司年报、招股说明书等超长文档
Long Document Chunk Processor for Financial Documents
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 年报/招股书常见章节标题
ANNUAL_REPORT_SECTIONS = [
    "公司简介", "会计数据", "业务概要", "经营情况", "公司治理",
    "环境与社会责任", "重要事项", "股份变动", "财务报告",
    "董事会报告", "管理层讨论", "风险因素", "募集资金使用",
]

PROSPECTUS_SECTIONS = [
    "招股说明书", "风险因素", "发行人基本情况", "业务与技术",
    "公司治理", "财务会计信息", "管理层分析", "募集资金运用",
    "未来发展规划", "本次发行情况",
]


@dataclass
class ChunkResult:
    """文档分块结果"""
    chunks: List[str] = field(default_factory=list)
    chunk_metadata: List[Dict] = field(default_factory=list)
    section_map: Dict[str, List[int]] = field(default_factory=dict)
    total_chunks: int = 0


class LongDocumentProcessor:
    """
    长文档处理器 - 专为上市公司年报/招股书等超长文档设计
    支持按章节/段落/句子三级分段策略
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_long_document(self, text: str, doc_name: str = "") -> ChunkResult:
        """处理超长文档的主入口"""
        if len(text) <= self.chunk_size:
            return ChunkResult(chunks=[text], total_chunks=1)

        # 策略1: 按文档类型章节标题分段
        chunks_by_section = self._split_by_sections(text)
        if len(chunks_by_section) > 1:
            return self._build_result(chunks_by_section, "section")

        # 策略2: 按段落分段
        chunks_by_para = self._split_by_paragraphs(text)
        if len(chunks_by_para) > 1:
            return self._build_result(chunks_by_para, "paragraph")

        # 策略3: 按句子分段（兜底）
        chunks_by_sent = self._split_by_sentences(text)
        return self._build_result(chunks_by_sent, "sentence")

    def _split_by_sections(self, text: str) -> List[str]:
        """按章节标题分段 - 识别年报/招股书的结构"""
        section_patterns = []
        for section in ANNUAL_REPORT_SECTIONS + PROSPECTUS_SECTIONS:
            section_patterns.append(section)

        pattern = r'(第[一二三四五六七八九十]+[章节节]\s*[^\n]+|' + \
                  '|'.join(section_patterns) + r')'
        matches = list(re.finditer(pattern, text))

        if len(matches) <= 1:
            return [text]

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if len(section_text) > 0:
                if len(section_text) > self.chunk_size:
                    sub_chunks = self._split_by_paragraphs(section_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(section_text)

        return chunks if chunks else [text]

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分段"""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) < self.chunk_size:
                current += para + "\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_by_sentences(para)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = para + "\n"

        if current.strip():
            chunks.append(current.strip())

        return self._add_overlap(chunks) if len(chunks) > 1 else chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分段（兜底策略）"""
        sentences = re.split(r'(?<=[。！？；])\s*', text)
        chunks = []
        current = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) < self.chunk_size:
                current += sent
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent

        if current.strip():
            chunks.append(current.strip())

        return self._add_overlap(chunks) if len(chunks) > 1 else chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为相邻块添加重叠内容"""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = chunks[i - 1][-self.chunk_overlap:] if len(chunks[i - 1]) > self.chunk_overlap else chunks[i - 1]
            result.append(overlap + "\n...\n" + chunks[i])
        return result

    def _build_result(self, chunks: List[str], strategy: str) -> ChunkResult:
        """构建分块结果"""
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "chunk_index": i,
                "chunk_length": len(chunk),
                "strategy": strategy,
            })

        return ChunkResult(
            chunks=chunks,
            chunk_metadata=metadata,
            total_chunks=len(chunks),
        )

    def extract_financial_highlights(self, text: str) -> Dict:
        """从长文档中提取财务摘要（年报重点数据）"""
        highlights = {}

        # 提取营业收入
        rev_pattern = r'(营业[收总]入|营业收入)[约为达]?[\d,]+\.?\d*\s*[亿万元]'
        rev_match = re.search(rev_pattern, text)
        if rev_match:
            highlights["revenue"] = rev_match.group(0)

        # 提取净利润
        profit_pattern = r'(净利润|归[属母]?[公?]?净[利润]?)[约为达]?[\d,]+\.?\d*\s*[亿万元]'
        profit_match = re.search(profit_pattern, text)
        if profit_match:
            highlights["net_profit"] = profit_match.group(0)

        # 提取毛利率
        margin_pattern = r'(毛利率|销售毛利率)[约为达]?[\d,]+\.?\d*\s*%'
        margin_match = re.search(margin_pattern, text)
        if margin_match:
            highlights["gross_margin"] = margin_match.group(0)

        # 提取每股收益
        eps_pattern = r'(每股收益|基本每股收益)[约为达]?[\d,]+\.?\d*'
        eps_match = re.search(eps_pattern, text)
        if eps_match:
            highlights["eps"] = eps_match.group(0)

        return highlights

    def get_document_structure(self, text: str) -> List[Dict]:
        """分析文档的章节结构"""
        sections = []
        patterns = [
            r'(第[一二三四五六七八九十]+[章节节]\s*[^\n]+)',
            r'([一二三四五六七八九十]+[、．\.]\s*[^\n]+)',
            r'(\d+[、．\.]\s*[^\n]+)',
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 3:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else min(start + 500, len(text))
                    sections.append({
                        "title": match.group().strip()[:60],
                        "offset": start,
                        "length": end - start,
                    })
                break

        return sections
