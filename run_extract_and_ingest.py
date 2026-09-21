#!/usr/bin/env python
"""
Run this script to:
1. Walk /PdfSet1/,
2. Convert every PDF into a PaperMetadata instance,
3. Encode it with ADSEPipeline,
4. Persist the resulting composite embedding in Qdrant.
"""

import os
from pathlib import Path

# -------------  Imports from your repo -----------------
from extract_pdf import parse_pdf_to_metadata   # <-- the helper you posted
from schema import PaperMetadata
from pipeline import ADSEPipeline
from storage.qdrant_engine import ADSEVectorStore

def walk_pdfs(root: str) -> list[Path]:
    """Return a sorted list of all .pdf files under `root`."""
    return sorted(Path(root).rglob("*.pdf"))
def print_vector_store_contents(vector_store: ADSEVectorStore):
    """
    Retrieves and prints all records currently stored in Qdrant.
    """
    print("\n" + "="*50)
    print("🗄️  CONTENTS OF VECTOR STORE")
    print("="*50)

    # Assuming ADSEVectorStore exposes the underlying QdrantClient and collection_name
    # If your attributes are named differently, change 'client' and 'collection_name' accordingly.
    client = vector_store.client 
    collection_name = getattr(vector_store, "collection_name", "papers") # Fallback to "papers"

    # Use scroll to get all points in the collection
    records, point_id = client.scroll(
        collection_name=collection_name,
        limit=100,             # Adjust if you have more than 100 papers
        with_payload=True,     # Fetch metadata
        with_vectors=False     # Set to True if you want to see the giant arrays of numbers
    )

    if not records:
        print("Vector store is empty.")
        return

    for record in records:
        print(f"ID: {record.id}")
        print(f"Payload: {record.payload}")
        if record.vector:
            print(f"Vector length: {len(record.vector)}")
        print("-" * 30)
        
def main():
    pdf_folder = "PdfSet1"          # <- change if your folder is elsewhere
    pdf_files  = walk_pdfs(pdf_folder)

    print(f"🔍 Found {len(pdf_files)} PDF(s) in '{pdf_folder}'.")
    pipeline   = ADSEPipeline()
    vector_store = ADSEVectorStore()

    for idx, pdf_path in enumerate(pdf_files, start=1):
        paper_id = f"paper_{idx:03d}"  # deterministic ID

        # 1️⃣ Turn PDF into metadata
        paper_meta = parse_pdf_to_metadata(str(pdf_path), paper_id)

        # 2️⃣ Validate & encode (pydantic + torch)
        composite_emb = pipeline.process_paper(paper_meta)

        # 3️⃣ Persist to Qdrant
        vector_store.upsert_paper(
            composite_emb=composite_emb,
            payload={
                "title":   paper_meta.title,
                "venue":   paper_meta.venue,
                "year":    paper_meta.publication_year
            }
        )
        print(f"✅ Indexed [{paper_id}] – {paper_meta.title}")

    print("\n🚀 All done!  Vector store is ready for queries.")
    # ------------------------------------------------------------------
    #  Print the content of the vector store
    # ------------------------------------------------------------------

    print_vector_store_contents(vector_store)

if __name__ == "__main__":
    main()
