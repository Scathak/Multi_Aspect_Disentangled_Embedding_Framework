import numpy as np
from typing import Dict, Tuple
from pydantic import BaseModel
from schema import CompositeEmbedding, PaperMetadata

class ComparisonReport(BaseModel):
    paper_id_a: str
    paper_id_b: str
    similarity_vector: Dict[str, float]
    dai_score: float
    relationship_label: str
    rigor_delta: float
    recommendation: str

class ArticleComparator:
    def __init__(self, dai_threshold: float = 0.30):
        self.dai_threshold = dai_threshold

    @staticmethod
    def _cosine_similarity(v1: list, v2: list) -> float:
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compare(self, emb_a: CompositeEmbedding, emb_b: CompositeEmbedding) -> ComparisonReport:
        # 1. Compute element-wise aspect similarities
        s_topic = self._cosine_similarity(emb_a.topic_vector, emb_b.topic_vector)
        s_method = self._cosine_similarity(emb_a.method_vector, emb_b.method_vector)
        s_citation = self._cosine_similarity(emb_a.citation_vector, emb_b.citation_vector)
        s_rigor = self._cosine_similarity(emb_a.rigor_vector, emb_b.rigor_vector)

        similarity_vector = {
            "topic": round(s_topic, 4),
            "method": round(s_method, 4),
            "citation": round(s_citation, 4),
            "rigor": round(s_rigor, 4)
        }

        # 2. Differential Alignment Index (DAI)
        dai = s_method - s_topic

        # 3. Rigor Delta
        rigor_mag_a = np.linalg.norm(emb_a.rigor_vector)
        rigor_mag_b = np.linalg.norm(emb_b.rigor_vector)
        rigor_delta = float(rigor_mag_a - rigor_mag_b)

        # 4. Classify Scientific Relationship Taxonomy
        relationship, recommendation = self._classify_relationship(
            s_topic, s_method, s_citation, dai, rigor_delta
        )

        return ComparisonReport(
            paper_id_a=emb_a.paper_id,
            paper_id_b=emb_b.paper_id,
            similarity_vector=similarity_vector,
            dai_score=round(dai, 4),
            relationship_label=relationship,
            rigor_delta=round(rigor_delta, 4),
            recommendation=recommendation
        )

    def _classify_relationship(self, s_topic: float, s_method: float, 
                               s_citation: float, dai: float, rigor_delta: float) -> Tuple[str, str]:
        if s_topic > 0.80 and s_method > 0.80:
            if s_citation > 0.70:
                label = "Direct Lineage / Near-Duplicate"
                rec = "Flag for potential prior-art overlap or direct continuation."
            else:
                label = "Parallel Independent Work / Direct Competitor"
                rec = "Compare benchmark results closely; published concurrently without cross-citation."
        
        elif dai > self.dai_threshold:
            label = "Cross-Domain Method Transfer"
            rec = "High potential for interdisciplinary innovation. Method applied to a novel target domain."
            
        elif dai < -self.dai_threshold and s_topic > 0.60:
            label = "Alternative Paradigm / Competing Method"
            rec = "Tackles the same research question using entirely different technical machinery."
            
        elif s_topic < 0.30 and s_method < 0.30:
            label = "Unrelated Research"
            rec = "Orthogonal domain and methods. No action required."
            
        else:
            label = "Weak Contextual Overlap"
            rec = "Shares peripheral citations or superficial domain terms."

        if abs(rigor_delta) > 0.40:
            label += " (Rigor Mismatch Detected)"

        return label, rec