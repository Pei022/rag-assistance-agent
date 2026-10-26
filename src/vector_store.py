# import logging
# from pymilvus import MilvusClient
#
# logger = logging.getLogger(__name__)
#
#
# class MilvusStore:
#     def __init__(self, cfg: dict):
#         m = cfg["milvus"]
#         self.db_name = m["db_name"]
#         self.collection = m["collection_name"]
#         self.metric_type = m["metric_type"]
#         self.dim = cfg["embed"]["dim"]
#
#         self._ensure_db()
#         self._ensure_collection()
#         self.client = MilvusClient(uri=m["uri"])
#
#     def _ensure_db(self):
#         if self.db_name not in self.client.list_databases():
#             self.client.create_database(db_name=self.db_name)
#         self.client.use_database(self.db_name)
#
#     def _ensure_collection(self):
#         if self.client.has_collection(self.collection):
#             logger.info("collection exists, skip create")
#         else:
#             self.client.create_collection(
#                 collection_name=self.collection,
#                 dimension=self.dim,
#                 metric_type=self.metric_type,
#             )
#             logger.info(f"collection {self.collection} created")
#
#     def upsert_batched(self, data: list[dict], batch_size: int = 100):
#         for i in range(0, len(data), batch_size):
#             batch = data[i:i + batch_size]
#             try:
#                 self.client.upsert(collection_name=self.collection, data=batch)
#             except Exception as e:
#                 logger.error(f"upsert batch {i} failed: {e}")
#                 raise
#         self.client.flush(collection_name=self.collection)
#         logger.info(f"upserted {len(data)} rows")
#
#     def search(self, query_vector: list[float], k: int) -> list[dict]:
#         """检索时必须返回 source，保证引用可追溯。"""
#         results = self.client.search(
#             collection_name=self.collection,
#             data=[query_vector],
#             limit=k,
#             output_fields=["text", "source", "chunk_id"],
#         )
#         return results[0] if results else []
#
#
#     def count(self) -> int:
#         stats = self.client.get_collection_stats(collection_name=self.collection)
#         return int(stats.get("row_count", 0))


# region
import logging
import chromadb

logger = logging.getLogger(__name__)


class MilvusStore:
    def __init__(self, cfg: dict):
        m = cfg["milvus"]
        self.collection_name = m["collection_name"]
        self.persist_dir = "./chroma_db"
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # get_or_create：存在就复用
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"chroma collection {self.collection_name} ready")

    def upsert_batched(self, data: list[dict], batch_size: int = 100):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.collection.upsert(
                ids=[str(d["id"]) for d in batch],
                embeddings=[d["vector"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[
                    {"source": d["source"], "chunk_id": d["chunk_id"]}
                    for d in batch
                ],
            )
        logger.info(f"upserted {len(data)} rows")

    def search(self, query_vector: list[float], k: int) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        if res and res["ids"] and res["ids"][0]:
            for idx in range(len(res["ids"][0])):
                hits.append({
                    "entity": {
                        "text": res["documents"][0][idx],
                        "source": res["metadatas"][0][idx].get("source", "unknown"),
                        "chunk_id": res["metadatas"][0][idx].get("chunk_id", -1),
                    },
                    # Chroma cosine distance = 1 - similarity，转回相似度
                    "distance": 1 - res["distances"][0][idx],
                })
        return hits

    def count(self) -> int:
        return self.collection.count()

# endregion