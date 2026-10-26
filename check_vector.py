
from src.config import load_config, get_secrets
from src.vector_store import MilvusStore
from src.embed_model import EmbeddingClient
import numpy as np

cfg = load_config()
secret = get_secrets()
store = MilvusStore(cfg)
embed = EmbeddingClient(cfg, secret)

print("persist_dir:", store.persist_dir)
print("count:", store.count())

# 1. 查库里 chunk 0 的向量前 5 位
res = store.collection.get(ids=["0"], include=["embeddings", "documents"])
vec_db = res["embeddings"][0]
print("\n=== 库里 chunk 0 向量 ===")
print("前 5 位:", vec_db[:5])
print("模长:", float(np.linalg.norm(vec_db)))

# 2. 用新 embed_query 算一次，看和库里的一致不一致
q_vec = embed.embed_query(res["documents"][0][:100])
print("\n=== 现场 embed_query 同一段文本 ===")
print("前 5 位:", q_vec[:5])

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print("\n库里向量 vs 现场向量 相似度:", cos(vec_db, q_vec))

# 3. 用新 query 检索，看 chunk 1 排第几
q = "基础版支持多少个正式成员？"
q_vec2 = embed.embed_query(q)
hits = store.search(q_vec2, 10)
print("\n=== 新 query 的 top-10 ===")
for i, h in enumerate(hits, 1):
    print(f"{i}. chunk_id={h['entity']['chunk_id']}  score={h['distance']:.4f}")