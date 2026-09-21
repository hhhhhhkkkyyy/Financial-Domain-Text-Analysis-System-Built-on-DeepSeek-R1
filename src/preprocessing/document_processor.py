"""
文档预处理模块 - 支持PDF、Word、TXT等格式的金融文档读取与清洗
Document Preprocessing Module
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据结构"""
    file_name: str
    file_type: str  # pdf, docx, txt, md
    raw_text: str = ""
    cleaned_text: str = ""
    chunks: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class DocumentPreprocessor:
    """
    金融文档预处理器
    支持多种格式读取、文本清洗、分段处理
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_length = self.config.get("max_document_length", 100000)
        self.chunk_size = self.config.get("chunk_size", 2000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)

    # ========== 文档读取 ==========

    def read_document(self, file_path: str) -> Document:
        """读取文档，自动识别格式"""
        ext = os.path.splitext(file_path)[1].lower()
        doc = Document(
            file_name=os.path.basename(file_path),
            file_type=ext.lstrip('.')
        )

        readers = {
            '.pdf': self._read_pdf,
            '.docx': self._read_docx,
            '.doc': self._read_docx,
            '.txt': self._read_txt,
            '.md': self._read_txt,
            '.markdown': self._read_txt,
        }

        reader = readers.get(ext)
        if reader is None:
            raise ValueError(f"不支持的文件格式: {ext}。支持的格式: {list(readers.keys())}")

        doc.raw_text = reader(file_path)
        doc.metadata['original_length'] = len(doc.raw_text)
        return doc

    def _read_pdf(self, file_path: str) -> str:
        """读取PDF文档"""
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        if not text.strip():
            logger.warning(f"PDF文件 {file_path} 未能提取到文本内容")
        return text

    def _read_docx(self, file_path: str) -> str:
        """读取Word文档"""
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        # 也读取表格中的文本
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    paragraphs.append(" | ".join(row_text))
        return "\n".join(paragraphs)

    def _read_txt(self, file_path: str) -> str:
        """读取纯文本/Markdown文档"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法识别文件编码: {file_path}")

    # ========== 文本清洗 ==========

    def clean_text(self, doc: Document) -> Document:
        """清洗文本：去除噪音、规范化格式"""
        text = doc.raw_text

        # 1. 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 2. 统一中文标点
        replacements = {
            '︰': '：', '﹖': '？', '﹗': '！', '﹒': '。', '﹔': '；',
            '﹦': '＝', '﹡': '*', '﹟': '#', '﹠': '&', '﹨': '\\',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 3. 保留金融文档相关格式特征
        # 保留条款编号模式 (第X条, 第X款, etc.)
        # 保留金额模式
        # 保留日期模式

        # 4. 规范化换行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. 统一数字格式
        text = re.sub(r'[零一二三四五六七八九十百千万亿]+', lambda m: m.group(), text)

        doc.cleaned_text = text.strip()
        doc.metadata['cleaned_length'] = len(doc.cleaned_text)
        return doc

    # ========== 长文档分段 ==========

    def split_document(self, doc: Document, chunk_size: int = None,
                       chunk_overlap: int = None) -> Document:
        """分段处理长文档，用于超长文档的完整解析"""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        text = doc.cleaned_text or doc.raw_text

        if len(text) <= chunk_size:
            doc.chunks = [text]
        else:
            chunks = []
            # 优先按段落边界分段
            paragraphs = text.split('\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + '\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # 如果单个段落太长，按句子分割
                    if len(para) > chunk_size:
                        sentences = re.split(r'([。！？])', para)
                        sub_chunk = ""
                        for i in range(0, len(sentences), 2):
                            sent = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                            if len(sub_chunk) + len(sent) <= chunk_size:
                                sub_chunk += sent
                            else:
                                if sub_chunk:
                                    chunks.append(sub_chunk.strip())
                                sub_chunk = sent
                        if sub_chunk:
                            current_chunk = sub_chunk + '\n'
                        else:
                            current_chunk = ""
                    else:
                        current_chunk = para + '\n'
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # 添加重叠
            if chunk_overlap > 0 and len(chunks) > 1:
                overlapped_chunks = [chunks[0]]
                for i in range(1, len(chunks)):
                    overlap_text = chunks[i-1][-chunk_overlap:] if len(chunks[i-1]) > chunk_overlap else chunks[i-1]
                    overlapped_chunks.append(overlap_text + '\n...\n' + chunks[i])
                chunks = overlapped_chunks

            doc.chunks = chunks

        doc.metadata['num_chunks'] = len(doc.chunks)
        logger.info(f"文档 {doc.file_name} 已分为 {len(doc.chunks)} 个段落块")
        return doc

    # ========== 文本直接处理 ==========

    def process_document_from_text(self, text: str, file_name: str = "text_input",
                                    chunk: bool = True) -> Document:
        """从文本字符串直接创建Document对象（无需文件）"""
        doc = Document(
            file_name=file_name,
            file_type="txt",
            raw_text=text,
        )
        doc = self.clean_text(doc)
        if chunk:
            doc = self.split_document(doc)
        return doc

    # ========== 文件处理 ==========

    def process_document(self, file_path: str, chunk: bool = True) -> Document:
        """完整的文档处理流程"""
        doc = self.read_document(file_path)
        doc = self.clean_text(doc)
        if chunk:
            doc = self.split_document(doc)
        return doc

    def batch_process(self, file_paths: List[str], chunk: bool = True) -> List[Document]:
        """批量处理多个文档"""
        results = []
        for fp in file_paths:
            try:
                doc = self.process_document(fp, chunk=chunk)
                results.append(doc)
                logger.info(f"已处理: {fp}")
            except Exception as e:
                logger.error(f"处理文件 {fp} 失败: {e}")
                results.append(Document(
                    file_name=os.path.basename(fp),
                    file_type=os.path.splitext(fp)[1].lstrip('.'),
                    metadata={'error': str(e)}
                ))
        return results

    # ========== 工具方法 ==========

    def extract_sections(self, doc: Document) -> List[Dict]:
        """提取文档的章节结构"""
        text = doc.cleaned_text or doc.raw_text
        sections = []

        # 匹配常见文档章节标题模式
        patterns = [
            r'(第[一二三四五六七八九十百千万\d]+条[.\s\w]*)',
            r'(第[一二三四五六七八九十百千万\d]+章[.\s\w]*)',
            r'(第[一二三四五六七八九十百千万\d]+节[.\s\w]*)',
            r'([一二三四五六七八九十]+、[^\n]+)',
            r'(\d+\.\s*[^\n]+)',
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i+1].start() if i+1 < len(matches) else len(text)
                    sections.append({
                        'title': match.group().strip(),
                        'content': text[start:end].strip()[:500],
                        'position': (start, end)
                    })
                break

        return sections

    def get_statistics(self, doc: Document) -> Dict:
        """获取文档统计信息"""
        text = doc.cleaned_text or doc.raw_text
        return {
            'file_name': doc.file_name,
            'file_type': doc.file_type,
            'original_length': len(doc.raw_text),
            'cleaned_length': len(doc.cleaned_text),
            'num_chunks': len(doc.chunks),
            'estimated_clauses': text.count('条') + text.count('款'),
            'has_amounts': bool(re.search(r'\d+[\d,.]*\s*[元万元]', text)),
            'has_dates': bool(re.search(r'\d{4}年\d{1,2}月\d{1,2}日', text)),
        }
