from openai import OpenAI
import numpy as np
from src.config import load_config, get_secrets
from src.embed_model import EmbeddingClient

cfg = load_config()
secret = get_secrets()

texts = [
    "基础版支持多少个正式成员？",
    "2. 基础版 - 成员人数上限：10 人",
    "这里的“7 个自然日内”从支付成功时间开始计算",
]

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# 方式 1：openai SDK 直调
client = OpenAI(api_key=secret["llm3_api_key"], base_url=secret["llm3_base_url"])
resp = client.embeddings.create(model="BAAI/bge-m3", input=texts)
vecs_sdk = [d.embedding for d in resp.data]

print("=== SDK 直调 ===")
print("维度:", len(vecs_sdk[0]))
print("前 5 位:", vecs_sdk[0][:5])
print("相关相似度:", cos(vecs_sdk[0], vecs_sdk[1]))
print("不相关相似度:", cos(vecs_sdk[0], vecs_sdk[2]))

# 方式 2：LangChain init_embeddings
embed = EmbeddingClient(cfg, secret)
vecs_lc = embed.embed_documents_batched(texts)

print("\n=== LangChain 封装 ===")
print("维度:", len(vecs_lc[0]))
print("前 5 位:", vecs_lc[0][:5])
print("相关相似度:", cos(vecs_lc[0], vecs_lc[1]))
print("不相关相似度:", cos(vecs_lc[0], vecs_lc[2]))