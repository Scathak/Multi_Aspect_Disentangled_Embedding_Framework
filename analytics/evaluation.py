import numpy as np
from schema import CompositeEmbedding
from typing import List

class ScientificEvaluator:
    @staticmethod
    def calculate_novelty(target_paper: CompositeEmbedding, neighborhood_papers: List[CompositeEmbedding]) -> float:
        """
        Calculates Methodological Novelty: High novelty means the method is distinct
        relative to papers in the same topic space.
        """
        if not neighborhood_papers:
            return 0.0

        target_method = np.array(target_paper.method_vector)
        sims = []
        for neighbor in neighborhood_papers:
            neighbor_method = np.array(neighbor.method_vector)
            cosine_sim = np.dot(target_method, neighbor_method) / (
                np.linalg.norm(target_method) * np.linalg.norm(neighbor_method) + 1e-9
            )
            sims.append(cosine_sim)

        # Novelty is inversely proportional to average method similarity
        return float(1.0 - np.mean(sims))

    @staticmethod
    def calculate_rigor_to_impact_ratio(paper_emb: CompositeEmbedding) -> float:
        """
        Determines whether a paper's popularity matches its methodological rigor.
        R > 1.0: Hidden Gem (High Rigor, Lower Citations/Impact)
        R < 0.5: Potential Hype (Low Rigor, High Citation Presence)
        """
        rigor_norm = np.linalg.norm(paper_emb.rigor_vector)
        citation_norm = np.linalg.norm(paper_emb.citation_vector) + 1e-9
        return float(rigor_norm / citation_norm)