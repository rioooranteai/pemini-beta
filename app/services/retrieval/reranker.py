from rank_bm25 import BM25Okapi
from app.services.retrieval.base.retrieval_base import RetrievedChunk


class BM25Reranker:

    def __init__(self, bm25_weight: float = 0.4, vector_weight: float = 0.6) -> None:
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight

    def rerank(
        self,
        query: str,
        vector_results: list[RetrievedChunk],
        all_chunks: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        if not vector_results:
            return []

        # Build BM25 index from all chunks
        tokenized_corpus = [chunk.content.lower().split() for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        # Score query against corpus
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Map chunk_id -> bm25 score
        bm25_score_map = {
            chunk.chunk_id: bm25_scores[i]
            for i, chunk in enumerate(all_chunks)
        }

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_score_map.values()) or 1.0
        normalized_bm25 = {
            k: v / max_bm25 for k, v in bm25_score_map.items()
        }

        # Combine vector score + BM25 score
        reranked = []
        for chunk in vector_results:
            bm25_score = normalized_bm25.get(chunk.chunk_id, 0.0)
            hybrid_score = (
                self._vector_weight * chunk.score +
                self._bm25_weight * bm25_score
            )
            reranked.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=hybrid_score
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]