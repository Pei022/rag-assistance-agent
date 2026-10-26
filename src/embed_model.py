# import logging
# import time
#
# from langchain.embeddings import init_embeddings
# from langchain_openai import OpenAIEmbeddings
#
# logger = logging.getLogger(__name__)
#
#
# class EmbeddingClient:
#     def __init__(self, cfg: dict,secret: dict):
#         self.cfg = cfg["embed"]
#         self.secret = secret
#         self.batch_size = self.cfg.get("batch_size", 32)
#         self.model = OpenAIEmbeddings(
#             model=self.cfg['model'],
#             openai_api_key=self.secret['llm3_api_key'],
#             openai_api_base=self.secret["llm3_base_url"],
#             check_embedding_ctx_length=False,
#         )
#
#     def embed_documents_batched(self, doc_texts):
#         vectors = []
#         # i每次跳 batch_size 个数循环
#         for i in range(0, len(doc_texts), self.batch_size):
#             batch = doc_texts[i:i + self.batch_size]
#             vecs = None
#             for attempt in range(3):
#                 try:
#                     vecs = self.model.embed_documents(batch)
#                     if len(vecs) != len(batch):
#                         raise ValueError(
#                             f"embedding count mismatch: got {len(vecs)}, expected {len(batch)}"
#                         )
#                     break
#                 except Exception as e:
#                     if attempt == 2:
#                         raise
#                     logger.warning(f"batch {i} attempt {attempt} failed: {e}, retry")
#                     time.sleep(2 ** attempt)
#             vectors.extend(vecs)
#         return vectors
#
#     def embed_query(self, text: str) -> list[float]:
#         return self.model.embed_query(text)


import logging
import time
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, cfg: dict, secret: dict):
        self.cfg = cfg["embed"]
        self.batch_size = self.cfg.get("batch_size", 32)
        self.client = OpenAI(
            api_key=secret["llm3_api_key"],
            base_url=secret["llm3_base_url"],
        )
        self.model_name = self.cfg["model"]
        logger.info(f"EmbeddingClient model={self.model_name}, base_url={secret['llm3_base_url']}")

    def embed_documents_batched(self, doc_texts):
        vectors = []
        for i in range(0, len(doc_texts), self.batch_size):
            batch = doc_texts[i:i + self.batch_size]
            vecs = None
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                    )
                    vecs = [d.embedding for d in resp.data]
                    if len(vecs) != len(batch):
                        raise ValueError(
                            f"embedding count mismatch: got {len(vecs)}, expected {len(batch)}"
                        )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"batch {i} attempt {attempt} failed: {e}, retry")
                    time.sleep(2 ** attempt)
            vectors.extend(vecs)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return resp.data[0].embedding