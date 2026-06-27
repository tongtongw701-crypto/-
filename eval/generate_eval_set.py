import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import Config
from src.data_processing.document_processor import LegalDocumentProcessor


_ARTICLE_SPLIT_RE = re.compile(r"\n(?=第[一二三四五六七八九十百千万0-9]+条)")


def _iter_articles(doc_text: str) -> Iterable[str]:
    """
    将法律文本按"第X条"切分成条文块（粗切分，足够用于评估集生成）。
    """
    if not doc_text:
        return []
    parts = _ARTICLE_SPLIT_RE.split(doc_text)
    # 过滤掉不含"第X条"的段落
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        if p.startswith("第") and "条" in p[:20]:
            yield p


def _extract_article_title(article_block: str) -> str:
    """
    从条文块提取"第X条"标题行。
    """
    first_line = article_block.splitlines()[0].strip()
    # 只取到条号+标题（如果有）
    return first_line[:50]


def _compact_text(text: str, max_chars: int = 600) -> str:
    """
    截断条文块，避免 ground_truth 过长影响评估稳定性。
    """
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


def build_eval_set(
    config: Config,
    per_law: int = 20,
    seed: int = 42,
    max_truth_chars: int = 600,
) -> List[Dict]:
    """
    从本地 Database/*.docx 自动生成评估题集（无需额外 LLM）：
    - 问题：围绕具体条文"第X条主要规定什么"
    - 标准答案：截断后的条文块文本
    """
    random.seed(seed)
    processor = LegalDocumentProcessor(config.DATABASE_PATH)
    docs = processor.load_documents()

    eval_items: List[Dict] = []
    for d in docs:
        law_name = d.get("law_name") or d.get("source") or "未知法律"
        articles = list(_iter_articles(d.get("content", "")))
        if not articles:
            continue

        # 条文过短/明显噪声过滤
        articles = [a for a in articles if len(a) >= 60]
        if not articles:
            continue

        sample_n = min(per_law, len(articles))
        sampled = random.sample(articles, sample_n)

        for a in sampled:
            title = _extract_article_title(a)
            question = f"《{law_name}》{title}主要规定了什么？请概括要点。"
            ground_truth = _compact_text(a, max_chars=max_truth_chars)
            eval_items.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "law_name": law_name,
                    "article_title": title,
                    "source": d.get("source"),
                }
            )

    return eval_items


def save_jsonl(items: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cfg = Config()
    per_law = int(os.environ.get("EVAL_PER_LAW", "20"))
    out_path = os.environ.get("EVAL_OUT", "eval/eval_set.jsonl")
    max_truth_chars = int(os.environ.get("EVAL_MAX_TRUTH_CHARS", "600"))

    items = build_eval_set(cfg, per_law=per_law, max_truth_chars=max_truth_chars)
    save_jsonl(items, out_path)

    print(f"[OK] 已生成评估集: {len(items)} 条 -> {out_path}")

