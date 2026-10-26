# import logging
# from src.rag_pipeline import RAGPipeline
#
# logging.basicConfig(level=logging.INFO)
#
# if __name__ == "__main__":
#     pipeline = RAGPipeline()
#     result = pipeline.ingest("data/docs1.txt")
#     print("ingest done:", result)

import csv
import logging
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    pipeline = RAGPipeline()
    result = pipeline.ingest("data/docs1.txt")
    print("ingest done:", result)

    # 重新切一次，导出 chunk 对照表（切分是确定性的，结果和 ingest 时一致）
    docs = TextLoader("data/docs1.txt", encoding="utf-8").load()
    chunks = pipeline.splitter.split(docs)

    with open(RESULTS_DIR / "chunk_map.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "source", "text_preview"])
        for c in chunks:
            writer.writerow([
                c.metadata["chunk_id"],
                c.metadata.get("source", ""),
                c.page_content[:80].replace("\n", " "),
            ])
    print(f"chunk_map saved: {RESULTS_DIR / 'chunk_map.csv'}, {len(chunks)} chunks")