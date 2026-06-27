"""
Prompt 模板管理器
从 prompts/ 目录加载 .txt 模板文件，支持 {变量} 占位符替换
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.resolve()
_cache: dict[str, str] = {}


def load_prompt(name: str, **kwargs) -> str:
    """加载 prompt 模板并填入变量

    Args:
        name: 模板文件名（不含 .txt 后缀），如 "pre_enhancement"
        **kwargs: 模板中的变量，如 original_query="xxx"

    Returns:
        填充变量后的 prompt 字符串
    """
    if name not in _cache:
        path = _PROMPTS_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt 模板不存在: {path}")
        _cache[name] = path.read_text(encoding="utf-8")

    template = _cache[name]
    if kwargs:
        return template.format(**kwargs)
    return template
