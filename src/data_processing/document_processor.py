import os
import docx
import docx2txt
from typing import List, Dict
import jieba
import re

class LegalDocumentProcessor:
    """法律文档处理器"""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.documents = []
        
    def load_documents(self) -> List[Dict[str, str]]:
        """
        加载所有法律文档
        
        Returns:
            List[Dict[str, str]]: 包含文档内容和元数据的列表
        """
        documents = []
        
        # 遍历数据库路径下的所有.docx文件
        for filename in os.listdir(self.database_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(self.database_path, filename)
                try:
                    # 提取文档内容
                    content = docx2txt.process(file_path)
                    
                    # 提取文档元数据
                    law_name = self._extract_law_name(filename)
                    law_date = self._extract_law_date(filename)
                    
                    documents.append({
                        'content': content,
                        'law_name': law_name,
                        'date': law_date,
                        'source': filename
                    })
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
        
        self.documents = documents
        return documents
    
    def _extract_law_name(self, filename: str) -> str:
        """
        从文件名提取法律名称
        
        Args:
            filename (str): 文件名
            
        Returns:
            str: 法律名称
        """
        # 移除日期部分和扩展名
        name = re.sub(r'_\d{8}\.docx$', '', filename)
        name = re.sub(r'_\d{6}\.docx$', '', name)
        return name
    
    def _extract_law_date(self, filename: str) -> str:
        """
        从文件名提取法律日期
        
        Args:
            filename (str): 文件名
            
        Returns:
            str: 法律日期
        """
        # 提取日期部分
        date_match = re.search(r'_(\d{8}|\d{6})\.docx$', filename)
        if date_match:
            return date_match.group(1)
        return "未知日期"
    
    def split_document(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        将文档分割成较小的块
        
        Args:
            content (str): 文档内容
            chunk_size (int): 块大小
            chunk_overlap (int): 重叠大小
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = min(start + chunk_size, content_length)
            chunk = content[start:end]
            chunks.append(chunk)
            
            # 移动起始位置
            start = end - chunk_overlap
            if start >= content_length:
                break
                
        return chunks
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        提取文本关键词
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[str]: 关键词列表
        """
        # 使用jieba进行分词
        words = jieba.lcut(text)
        
        # 过滤掉长度小于2的词和停用词
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        
        return list(set(keywords))  # 去重

# 使用示例
if __name__ == "__main__":
    processor = LegalDocumentProcessor("../Database/")
    docs = processor.load_documents()
    print(f"成功加载 {len(docs)} 个法律文档")
    
    # 显示第一个文档的信息
    if docs:
        first_doc = docs[0]
        print(f"文档名称: {first_doc['law_name']}")
        print(f"发布日期: {first_doc['date']}")
        print(f"文件来源: {first_doc['source']}")
        print(f"内容长度: {len(first_doc['content'])} 字符")