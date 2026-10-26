import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # src 的上一级 = 项目根
path = BASE_DIR / "config.yaml"

load_dotenv(override=False)

def load_config(path:str=path):
    with open(path,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_secrets() -> dict:

    return  {
        'llm1_api_key': os.getenv('DEEPSEEK_API_KEY'),
        'llm1_base_url': os.getenv('DEEPSEEK_BASE_URL'),
        'llm2_api_key': os.getenv('CLOSEAI_API_KEY'),
        'llm2_base_url': os.getenv('CLOSEAI_BASE_URL'),
        'llm3_api_key': os.getenv('SILICONFLOW_API_KEY'),
        'llm3_base_url': os.getenv('SILICONFLOW_BASE_URL'),
    }