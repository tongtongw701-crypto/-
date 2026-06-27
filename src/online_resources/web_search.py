import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from datetime import datetime

class LegalWebSearch:
    """法律网络搜索工具 (智能切片增强版)"""
    
    def __init__(self, llm_caller=None, logger=None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'https://www.baidu.com/'
        }
        # 垃圾内容关键词
        self.bad_keywords = ['推广', '下载', '注册', '登录', 'ICP备', '关注我们', '点击查看', '免费试用', '立即体验']
        
        # 广告标题关键词
        self.ad_title_keywords = ['厂家', '价格', '批发', '加盟', '报价', '哪家好', '排行榜', '多少钱', '费用', 
                                '不收费', '免费咨询', '律师在线', '胜诉率', '电话咨询', '法行宝', 'AI律师', 'AI法','千千律约']
        
        # 优质内容关键词
        self.good_keywords = ['罚款', '元', '规定', '条例', '第', '条', '实施', '执行', '禁止', '允许', '办法', '通知']
        
        # LLM 回调函数（用于智能切片）
        self.llm_caller = llm_caller
        self.logger = logger if logger else print

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        执行搜索并返回结构化数据（用于交叉验证）
        返回格式: [{"title": "...", "url": "...", "content": "...", "credibility": 7, "date": "2024-07-01"}, ...]
        """
        # 【时间感知】如果用户问题包含时间敏感词，自动添加年份
        time_keywords = ["最新", "现在", "什么时候", "何时", "实施", "生效", "执行"]
        current_year = datetime.now().year
        
        if any(kw in query for kw in time_keywords):
            # 加入当前年份和近期年份，提高时效性
            search_query = f"{query} {current_year} {current_year-1}"
        else:
            search_query = f"{query} 规定"
        
        self.logger(f"   [网搜] 正在百度搜索: {search_query} ...")
        
        try:
            url = f"https://www.baidu.com/s?wd={urllib.parse.quote(search_query)}"
            response = requests.get(url, headers=self.headers, timeout=8)
            
            if response.status_code != 200:
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.find_all('div', class_=['result', 'c-container'])
            
            candidates = []
            for item in items: 
                text_content = item.get_text()
                
                # 广告检测
                if "广告" in text_content or "保障" in text_content: 
                    if 'tuiguang' in str(item) or text_content.strip().endswith("广告"):
                        continue
                
                title_tag = item.find('h3')
                if not title_tag: continue
                title = title_tag.get_text().strip()
                
                if any(ad_word in title for ad_word in self.ad_title_keywords):
                    continue

                link_tag = item.find('a')
                if not link_tag: continue
                href = link_tag['href']

                snippet = ""
                abstract_tag = item.find('div', class_='c-abstract')
                if not abstract_tag:
                    abstract_tag = item.find('div', class_='c-content-right')
                
                if abstract_tag:
                    snippet = abstract_tag.get_text().strip()
                else:
                    full_div_text = item.get_text().strip()
                    title_text = title_tag.get_text().strip()
                    snippet = full_div_text.replace(title_text, "")[:200]

                candidates.append({
                    'title': title,
                    'href': href,
                    'snippet': snippet
                })
                
                if len(candidates) >= max_results:
                    break
            
            if not candidates:
                return []
            
            self.logger(f"   [深度] 正在并行阅读 {len(candidates)} 个网页...")
            results = []
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self._fetch_and_analyze, c, query): c for c in candidates}
                
                for future in futures:
                    candidate = futures[future]
                    result = future.result()
                    
                    if result:
                        results.append(result)
                    else:
                        # 兜底
                        results.append({
                            'title': candidate['title'],
                            'url': candidate['href'],
                            'content': candidate['snippet'],
                            'credibility': 3,  # 摘要可信度低
                            'date': None,
                            'source_type': '摘要'
                        })
            
            # 【时间优先排序】按发布日期降序（最新的在前）
            results.sort(key=lambda x: x.get('date') or '2000-01-01', reverse=True)
            
            return results

        except Exception as e:
            self.logger(f"   [X] 搜索异常: {e}")
            return []

    def _fetch_and_analyze(self, candidate: dict, user_query: str) -> Dict:
        """访问链接 -> 提取正文 -> 智能切片 -> 评分"""
        url = candidate['href']
        title = candidate['title']
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=4, allow_redirects=True, verify=False)
            
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type: return None
            if resp.status_code != 200: return None
            
            if "wappass.baidu.com" in resp.url or "验证" in resp.text[:500]:
                return None

            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript', 'aside']):
                tag.decompose()
            
            # 提取所有文本块
            text_blocks = []
            for tag in soup.find_all(['p', 'div', 'article', 'section', 'li', 'td']):
                if len(tag.find_all('a')) > 5: continue
                txt = tag.get_text().strip()
                
                is_valid = False
                if len(txt) > 20: is_valid = True
                elif len(txt) > 5 and any(k in txt for k in self.good_keywords): is_valid = True
                
                if is_valid and not any(bad in txt for bad in self.bad_keywords):
                    text_blocks.append(txt)
            
            text_blocks = list(dict.fromkeys(text_blocks)) 
            full_text = "\n".join(text_blocks)
            
            if len(full_text) < 50: return None
            
            # 【智能切片】如果文本太长，用 LLM 提取相关段落
            if len(full_text) > 3000 and self.llm_caller:
                self.logger(f"      -> [切片] 正在智能提取相关段落...")
                relevant_content = self._smart_slice(full_text, user_query, text_blocks)
            else:
                relevant_content = full_text[:2000]
            
            # 【可信度评分】
            credibility = self._score_credibility(url)
            
            # 【时间提取增强】提取发布日期和生效日期
            date_info = self._extract_date_enhanced(full_text, resp.text[:1000])
            
            return {
                'title': title,
                'url': url,
                'content': relevant_content,
                'credibility': credibility,
                'date': date_info['publish_date'],
                'effective_date': date_info.get('effective_date'),
                'source_type': '深度阅读'
            }
            
        except Exception:
            return None

    def _smart_slice(self, full_text: str, user_query: str, text_blocks: List[str]) -> str:
        """智能切片：让 LLM 从长文中挑选最相关的段落"""
        if not self.llm_caller:
            return full_text[:2000]
        
        # 只取前 30 个段落（避免 Token 爆炸）
        sample_blocks = text_blocks[:30]
        
        # 给每个段落编号
        numbered_text = ""
        for i, block in enumerate(sample_blocks, 1):
            numbered_text += f"[{i}] {block[:100]}...\n"
        
        prompt = f"""
以下是一篇网页的内容（已分段并编号），请从中挑选出与用户问题最相关的 3-5 个段落。

用户问题: {user_query}

段落列表:
{numbered_text}

请只输出段落编号，用逗号分隔，例如：1,3,7,12
"""
        
        try:
            response = self.llm_caller(prompt)
            # 解析编号
            selected_ids = []
            for item in response.replace('，', ',').split(','):
                try:
                    num = int(item.strip())
                    if 1 <= num <= len(sample_blocks):
                        selected_ids.append(num - 1)  # 转为索引
                except:
                    continue
            
            if selected_ids:
                return '\n\n'.join([sample_blocks[i] for i in selected_ids])
            else:
                return full_text[:2000]
        except:
            return full_text[:2000]

    def _score_credibility(self, url: str) -> int:
        """根据域名判断可信度 (1-10分)"""
        if 'gov.cn' in url or 'www.gov.cn' in url:
            return 10  # 政府官网
        if 'edu.cn' in url:
            return 8   # 教育机构
        if any(x in url for x in ['xinhuanet.com', 'people.com.cn', 'chinanews.com']):
            return 9   # 官方媒体
        if any(x in url for x in ['baidu.com', 'zhihu.com', 'sohu.com']):
            return 6   # 知名平台
        return 4  # 普通网站

    def _extract_date(self, text: str) -> str:
        """从网页中提取发布日期（兼容旧方法）"""
        result = self._extract_date_enhanced(text, text)
        return result['publish_date']

    def _extract_date_enhanced(self, content: str, html_header: str = "") -> Dict:
        """增强版时间提取：提取发布日期和生效日期"""
        result = {
            'publish_date': None,
            'effective_date': None
        }
        
        # 日期正则
        patterns = [
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{4})/(\d{2})/(\d{2})',
        ]
        
        # 1. 提取发布日期（优先从 HTML 头部的 meta 标签或开头找）
        for pattern in patterns:
            match = re.search(pattern, html_header[:800])
            if match:
                groups = match.groups()
                result['publish_date'] = f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
                break
        
        # 如果头部没找到，从正文开头找
        if not result['publish_date']:
            for pattern in patterns:
                match = re.search(pattern, content[:500])
                if match:
                    groups = match.groups()
                    result['publish_date'] = f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
                    break
        
        # 2. 提取生效/实施日期（关键词：自...起施行、生效时间）
        effective_patterns = [
            r'自[^\d]*(\d{4})年(\d{1,2})月(\d{1,2})日[^\d]*起[施实生效]',
            r'[施实生效][行用]{0,2}时间[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'(\d{4})年(\d{1,2})月(\d{1,2})日[^\d]*起[施实生效]',
        ]
        
        for pattern in effective_patterns:
            match = re.search(pattern, content[:1000])
            if match:
                groups = match.groups()
                result['effective_date'] = f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
                break
        
        return result

    def format_results(self, results: List[Dict]) -> str:
        """将结构化数据格式化为文本（用于 LLM 阅读）"""
        output = []
        current_year = datetime.now().year
        
        for i, item in enumerate(results, 1):
            credibility_stars = "⭐" * min(item['credibility'], 10)
            
            # 【时间信息增强】
            date_info = ""
            if item.get('date'):
                publish_year = int(item['date'][:4]) if item['date'] else 2000
                # 标注过期信息
                if current_year - publish_year > 4:
                    date_info = f" ⚠️ (发布于: {item['date']}, 信息可能已过期)"
                else:
                    date_info = f" (发布于: {item['date']})"
            
            # 生效日期
            if item.get('effective_date'):
                date_info += f" | 📅 生效时间: {item['effective_date']}"
            
            formatted = f"""[来源 {i}: {item['title']}] {item['source_type']}
链接: {item['url']}
可信度: {credibility_stars} ({item['credibility']}/10){date_info}
【内容】:
{item['content'][:800]}...
"""
            output.append(formatted)
        
        return "\n\n".join(output)

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    s = LegalWebSearch()
    results = s.search("南京市电动车管理条例")
    self.logger(s.format_results(results))
