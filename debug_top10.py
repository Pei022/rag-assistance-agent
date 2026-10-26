from src.config import load_config, get_secrets
from src.vector_store import MilvusStore
from src.embed_model import EmbeddingClient

cfg = load_config()
secret = get_secrets()
store = MilvusStore(cfg)
embed = EmbeddingClient(cfg, secret)

q = "基础版支持多少个正式成员？"
q_vec = embed.embed_query(q)
hits = store.search(q_vec, 10)   # top-10

print(f"query: {q}\n")
for i, h in enumerate(hits, 1):
    print(f"{i}. chunk_id={h['entity']['chunk_id']}  score={h['distance']:.4f}")
    print(f"   {h['entity']['text'][:80]}")
    print()