import re
from typing import List, Dict, Tuple

class LegalAnalyzer:
    """法律问题分析器"""
    
    def __init__(self):
        # 定义法律领域关键词
        self.legal_domains = {
            '劳动法': ['劳动合同', '工资', '加班', '解雇', '工伤', '社保', '年假', '试用期'],
            '合同法': ['合同', '违约', '履行', '解除', '变更', '效力', '条款'],
            '刑法': ['犯罪', '刑罚', '有期徒', '拘役', '罚金', '死刑', '无期徒刑'],
            '民法典': ['民事', '物权', '债权', '侵权', '婚姻', '继承', '人格权'],
            '公司法': ['公司', '股东', '董事会', '注册资本', '股权', '分红'],
            '知识产权法': ['专利', '商标', '著作权', '知识产权', '侵权']
        }
        
        # 复杂问题关键词
        self.complex_keywords = [
            '怎么办', '如何处理', '怎么解决', '涉及多个', '复杂情况', 
            '争议', '纠纷', '诉讼', '起诉', '赔偿', '法律责任'
        ]
        
    def identify_legal_domain(self, question: str) -> List[str]:
        """
        识别法律问题所属领域
        
        Args:
            question (str): 用户问题
            
        Returns:
            List[str]: 匹配的法律领域列表
        """
        matched_domains = []
        
        for domain, keywords in self.legal_domains.items():
            # 检查问题中是否包含该领域的关键词
            for keyword in keywords:
                if keyword in question:
                    matched_domains.append(domain)
                    break  # 找到匹配就跳出内层循环
                    
        return list(set(matched_domains))  # 去重
    
    def extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        提取问题中的实体信息
        
        Args:
            question (str): 用户问题
            
        Returns:
            Dict[str, List[str]]: 提取的实体字典
        """
        entities = {
            'persons': [],      # 人员
            'organizations': [], # 组织机构
            'amounts': [],      # 金额
            'dates': [],        # 日期
            'locations': []     # 地点
        }
        
        # 提取金额（如：1000元、5万元）
        amounts = re.findall(r'\d+(?:\.\d+)?[万]?(?:元|美元|欧元)', question)
        entities['amounts'] = amounts
        
        # 提取日期（如：2023年、2023年1月、2023年1月1日）
        dates = re.findall(r'\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?', question)
        entities['dates'] = dates
        
        return entities
    
    def is_complex_question(self, question: str) -> bool:
        """
        判断是否为复杂问题
        
        Args:
            question (str): 用户问题
            
        Returns:
            bool: 是否为复杂问题
        """
        # 检查是否包含复杂问题关键词
        for keyword in self.complex_keywords:
            if keyword in question:
                return True
                
        # 检查问题长度（较长的问题可能更复杂）
        if len(question) > 50:
            return True
            
        return False
    
    def generate_lawyer_recommendation(self, domains: List[str]) -> str:
        """
        生成律师推荐信息
        
        Args:
            domains (List[str]): 法律领域列表
            
        Returns:
            str: 律师推荐信息
        """
        if not domains:
            return "建议您咨询专业律师获得更准确的法律意见。"
            
        domain_text = "、".join(domains)
        return f"这是一个涉及{domain_text}的复杂法律问题，建议您尽快联系专业的{domain_text}律师进行详细咨询。"
    
    def analyze_question(self, question: str) -> Dict[str, any]:
        """
        全面分析法律问题
        
        Args:
            question (str): 用户问题
            
        Returns:
            Dict[str, any]: 分析结果
        """
        result = {
            'question': question,
            'domains': self.identify_legal_domain(question),
            'entities': self.extract_entities(question),
            'is_complex': self.is_complex_question(question),
            'recommendation': ""
        }
        
        if result['is_complex']:
            result['recommendation'] = self.generate_lawyer_recommendation(result['domains'])
            
        return result

# 使用示例
if __name__ == "__main__":
    analyzer = LegalAnalyzer()
    
    # 测试问题分析
    test_questions = [
        "我在公司工作三年了，被无故解雇，应该如何维权？",
        "签订的房屋租赁合同有争议怎么办？",
        "交通事故造成的人身损害赔偿标准是什么？"
    ]
    
    for question in test_questions:
        result = analyzer.analyze_question(question)
        print(f"问题: {question}")
        print(f"涉及领域: {result['domains']}")
        print(f"是否复杂: {result['is_complex']}")
        if result['recommendation']:
            print(f"推荐建议: {result['recommendation']}")
        print("-" * 50)