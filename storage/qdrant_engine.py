from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from schema import CompositeEmbedding
from typing import List, Dict
import numpy as np

class ADSEVectorStore:
    def __init__(self, collection_name: str = "scientific_papers", host: str = ":memory:"):
        self.client = QdrantClient(host)
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        # Configure named vectors for each disentangled aspect
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "topic": VectorParams(size=256, distance=Distance.COSINE),
                "method": VectorParams(size=256, distance=Distance.COSINE),
                "citation": VectorParams(size=256, distance=Distance.COSINE),
                "rigor": VectorParams(size=128, distance=Distance.COSINE),
            }
        )

    def upsert_paper(self, composite_emb: CompositeEmbedding, payload: Dict):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=abs(hash(composite_emb.paper_id)) % (10**12), # Unique Int ID
                    vector={
                        "topic": composite_emb.topic_vector,
                        "method": composite_emb.method_vector,
                        "citation": composite_emb.citation_vector,
                        "rigor": composite_emb.rigor_vector,
                    },
                    payload={"paper_id": composite_emb.paper_id, **payload}
                )
            ]
        )

    def dynamic_aspect_search(
        self, 
        query_emb: CompositeEmbedding, 
        weights: Dict[str, float], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Executes a dynamic weighted similarity search across the disentangled sub-spaces.
        """
        # Normalize weights
        total_w = sum(weights.values())
        norm_weights = {k: v / total_w for k, v in weights.items()}

        # Multi-vector query execution
        results_by_aspect = {}
        for aspect in ["topic", "method", "citation", "rigor"]:
            if norm_weights.get(aspect, 0.0) > 0.0:
                vector_val = getattr(query_emb, f"{aspect}_vector")
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector_val,
                    using=aspect,
                    limit=top_k * 3,
                    with_payload=True
                ).points
                results_by_aspect[aspect] = {hit.payload["paper_id"]: hit.score for hit in res}

        # Aggregate weighted scores
        all_paper_ids = set()
        for aspect_hits in results_by_aspect.values():
            all_paper_ids.update(aspect_hits.keys())

        final_scores = []
        for pid in all_paper_ids:
            score = 0.0
            for aspect, w in norm_weights.items():
                if w > 0.0:
                    aspect_score = results_by_aspect[aspect].get(pid, 0.0)
                    score += w * aspect_score
            final_scores.append({"paper_id": pid, "composite_score": score})

        # Sort by final aggregated score
        final_scores.sort(key=lambda x: x["composite_score"], reverse=True)
        return final_scores[:top_k]