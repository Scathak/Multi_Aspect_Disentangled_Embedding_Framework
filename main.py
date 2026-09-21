if __name__ == "__main__":
    from schema import PaperMetadata
    from pipeline import ADSEPipeline
    from storage.qdrant_engine import ADSEVectorStore
    from analytics.evaluation import ScientificEvaluator

    # 1. Instantiate Core Engine
    pipeline = ADSEPipeline()
    vector_store = ADSEVectorStore()

    # 2. Mock Dataset: Quantum Machine Learning Papers
    papers = [
        PaperMetadata(
            paper_id="paper_001",
            title="Language Models are Few-Shot Learners",
            abstract="We present a VQE approach applied to small molecule electronic structure calculations.",
            methods_text="We optimize parameterized quantum circuits using Adam gradient descent. Rigorous ablation of ansatz depth was performed.",
            results_text="Extensive experimental validation achieved chemical accuracy with p < 0.01 across 100 runs.",
            code_url="https://arxiv.org/abs/2005.14165",
            citations=["paper_002"],
            authors=["Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei"],
            venue="Quantum Science and Technology",
            publication_year=2023
        ),
        PaperMetadata(
            paper_id="paper_002",
            title="Quantum Convolutional Neural Networks for Image Classification",
            abstract="A novel QCNN framework for quantum state classification tasks.",
            methods_text="We implement quantum convolution layers using unitary transformations.",
            results_text="Evaluated on synthetic data, showing faster convergence compared to standard classical neural nets.",
            code_url="", # Missing code
            citations=[],
            authors=["Bob Jones"],
            venue="Preprint Server",
            publication_year=2022
        )
    ]

    # 3. Process and Index Papers
    embeddings = {}
    for p in papers:
        emb = pipeline.process_paper(p)
        embeddings[p.paper_id] = emb
        vector_store.upsert_paper(
            composite_emb=emb, 
            payload={"title": p.title, "year": p.publication_year}
        )
        print(f" Successfully processed and indexed: {p.paper_id}")

    # 4. Perform Task-Specific Dynamic Dynamic Querying
    query_paper = embeddings["paper_001"]

    print("\n--- [QUERY 1: Literature Review Search (Focus: Topic + Citation)] ---")
    lit_weights = {"topic": 0.6, "method": 0.1, "citation": 0.3, "rigor": 0.0}
    lit_results = vector_store.dynamic_aspect_search(query_paper, weights=lit_weights, top_k=2)
    print(lit_results)

    print("\n--- [QUERY 2: Method Borrowing / Novelty Search (Focus: Method)] ---")
    method_weights = {"topic": 0.1, "method": 0.8, "citation": 0.0, "rigor": 0.1}
    method_results = vector_store.dynamic_aspect_search(query_paper, weights=method_weights, top_k=2)
    print(method_results)

    # 5. Scientific Evaluation
    novelty = ScientificEvaluator.calculate_novelty(
        target_paper=embeddings["paper_001"],
        neighborhood_papers=[embeddings["paper_002"]]
    )
    rigor_ratio = ScientificEvaluator.calculate_rigor_to_impact_ratio(embeddings["paper_001"])

    print("\n--- [EVALUATION METRICS] ---")
    print(f"Paper ID: paper_001")
    print(f"Methodological Novelty Score: {novelty:.4f}")
    print(f"Rigor-to-Impact Ratio:         {rigor_ratio:.4f}")