from src.config import load_config, get_secrets
from src.embed_model import EmbeddingClient

cfg = load_config()
secret = get_secrets()
print("config embed.model:", cfg["embed"]["model"])
print("base_url:", secret["llm3_base_url"])

embed = EmbeddingClient(cfg, secret)

# 1. 维度
q_vec = embed.embed_query("基础版支持多少个正式成员？")
print("\nquery 向量维度:", len(q_vec))

# 2. 手动算相似度
import numpy as np

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

related = "2. 基础版 - 价格：99 元 / 用户 / 月 - 成员人数上限：10 人"
unrelated = "这里的“7 个自然日内”从支付成功时间开始计算，到第 7 日的 23:59:59 截止"

related_vec = embed.embed_documents_batched([related])[0]
unrelated_vec = embed.embed_documents_batched([unrelated])[0]

print("\n相似度（应该 0.6+）:", cos(q_vec, related_vec))
print("相似度（应该 0.2 左右）:", cos(q_vec, unrelated_vec))