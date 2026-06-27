import os
import json
from typing import Any

def save_json(data: Any, filepath: str) -> None:
    """
    保存数据为JSON文件
    
    Args:
        data (Any): 要保存的数据
        filepath (str): 文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Any:
    """
    从JSON文件加载数据
    
    Args:
        filepath (str): 文件路径
        
    Returns:
        Any: 加载的数据
    """
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir_exists(dir_path: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path (str): 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)