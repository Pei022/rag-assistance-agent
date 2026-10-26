from langchain_text_splitters import RecursiveCharacterTextSplitter


class Splitter:
    def __init__(self, cfg: dict):
        c = cfg["chunking"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=c["chunk_size"],
            chunk_overlap=c["chunk_overlap"],
            separators=c["separators"],
        )

    def split(self, docs):
        chunks = self.splitter.split_documents(docs)
        # 给每个 chunk 补 chunk_id，后面写入和引用都用得上
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata.setdefault("source", "docs1")
        return chunks
