from src.config import load_config, get_secrets
from src.embed_model import EmbeddingClient
import numpy as np

cfg = load_config()
secret = get_secrets()
embed = EmbeddingClient(cfg, secret)

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

q = embed.embed_query("基础版支持多少个正式成员？")
related = embed.embed_documents_batched(["2. 基础版 - 成员人数上限：10 人"])[0]
unrelated = embed.embed_documents_batched(["这里的 7 个自然日内从支付成功时间开始计算"])[0]

print("相关相似度:", cos(q, related))      # 期望 0.7+
print("不相关相似度:", cos(q, unrelated))  # 期望 0.2-0.4