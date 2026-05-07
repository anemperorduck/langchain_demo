import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": "https://api.deepseek.com",
}

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": "https://api.openai.com/v1",
}

LLM_CONFIG = {
    "model_name": "qwen3.5-plus",
    "api_key": DASHSCOPE_CONFIG["api_key"],
    "base_url": DASHSCOPE_CONFIG["base_url"],
    "temperature": 0.7,
    "max_tokens": 1000,
}

IMG_PATH = "images/huangshan.png"