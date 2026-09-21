import json
from schema import PaperMetadata
from pipeline import ADSEPipeline
from storage.qdrant_engine import ADSEVectorStore

def load_and_index_papers(json_path: str):
    pipeline = ADSEPipeline()
    vector_store = ADSEVectorStore()

    # Load custom JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        raw_papers = json.load(f)

    indexed_embeddings = {}

    for paper_dict in raw_papers:
        # Validate through Pydantic Schema
        paper = PaperMetadata(**paper_dict)
        
        # Process through ADSE Encoders
        composite_emb = pipeline.process_paper(paper)
        indexed_embeddings[paper.paper_id] = composite_emb
        
        # Upsert into Qdrant Storage
        vector_store.upsert_paper(
            composite_emb=composite_emb,
            payload={
                "title": paper.title,
                "venue": paper.venue,
                "year": paper.publication_year
            }
        )
        print(f"✅ Indexed paper: [{paper.paper_id}] - {paper.title}")

    return vector_store, indexed_embeddings

if __name__ == "__main__":
    store, embs = load_and_index_papers("my_papers.json")