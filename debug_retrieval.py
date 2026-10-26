from src.config import load_config, get_secrets
from src.vector_store import MilvusStore
from src.embed_model import EmbeddingClient

cfg = load_config()
secret = get_secrets()
store = MilvusStore(cfg)
embed = EmbeddingClient(cfg, secret)

# 1. 库里有多少条
print("collection count:", store.count())

# 2. 用第 1 题检索
q = "基础版支持多少个正式成员？"
q_vec = embed.embed_query(q)
hits = store.search(q_vec, 5)

print("\n=== hits ===")
for h in hits:
    print(h)

print("\n=== 逐字段类型 ===")
if hits:
    first = hits[0]["entity"]
    print("chunk_id:", first["chunk_id"], "type:", type(first["chunk_id"]))
    print("source:", first["source"])
    print("text[:50]:", first["text"][:50])
else:
    print("hits 为空！")
