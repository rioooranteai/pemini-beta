from app.services.ingestion.models import Chunk

class TextChunker:

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, doc_id: str, content: str, metadata: dict) -> list[Chunk]:
        words = content.split()
        chunks = []
        start = 0
        index = 0

        while start < len(words):
            end = start + self._chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{index}",
                content=chunk_text,
                metadata={**metadata, "doc_id": doc_id, "chunk_index": index}
            ))

            start += self._chunk_size - self._chunk_overlap
            index += 1

        return chunks