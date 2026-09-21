import torch
from schema import PaperMetadata, CompositeEmbedding
from encoders.topic_encoder import TopicEncoder
from encoders.method_encoder import MethodEncoder
from encoders.citation_encoder import CitationGNN
from encoders.rigor_encoder import RigorFeatureExtractor, RigorMLP

class ADSEPipeline:
    def __init__(self):
        print("Initializing ADSE Pipeline encoders...")
        self.topic_enc = TopicEncoder()
        self.method_enc = MethodEncoder()
        self.citation_gnn = CitationGNN()
        self.rigor_extractor = RigorFeatureExtractor()
        self.rigor_mlp = RigorMLP()
        self.eval()

    def eval(self):
        self.topic_enc.eval()
        self.method_enc.eval()
        self.citation_gnn.eval()
        self.rigor_mlp.eval()

    def process_paper(self, paper: PaperMetadata, graph_context_vector: torch.Tensor = None) -> CompositeEmbedding:
        """
        Processes a single paper input into a composite 4-aspect embedding.
        """
        # 1. Topic Sub-embedding (256-D)
        e_topic = self.topic_enc(paper.title, paper.abstract)
        
        # 2. Method Sub-embedding (256-D)
        e_method = self.method_enc(paper.methods_text)
        
        # 3. Citation Sub-embedding (256-D)
        # If Graph neural network inference is omitted for simple execution, fall back to Topic-aware initializer
        if graph_context_vector is None:
            e_citation = e_topic.clone()
        else:
            e_citation = graph_context_vector

        # 4. Rigor Sub-embedding (128-D)
        rigor_feats = self.rigor_extractor.extract(paper)
        e_rigor = self.rigor_mlp(rigor_feats)

        return CompositeEmbedding(
            paper_id=paper.paper_id,
            topic_vector=e_topic.tolist(),
            method_vector=e_method.tolist(),
            citation_vector=e_citation.tolist(),
            rigor_vector=e_rigor.tolist()
        )