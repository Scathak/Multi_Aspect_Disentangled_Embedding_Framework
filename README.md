# Multi_Aspect_Disentangled_Embedding_Framework-
Automation of scientific article comparison

Installation:
pip install torch torch-geometric transformers qdrant-client pydantic scikit-learn numpy spacy
python -m spacy download en_core_web_sm

Execute the pipeline entrypoint directly:
python main.py

Expected Output Log:

Initializing ADSE Pipeline encoders...
 Successfully processed and indexed: paper_001
 Successfully processed and indexed: paper_002

--- [QUERY 1: Literature Review Search (Focus: Topic + Citation)] ---
[{'paper_id': 'paper_001', 'composite_score': 1.0}, {'paper_id': 'paper_002', 'composite_score': 0.6421}]

--- [QUERY 2: Method Borrowing / Novelty Search (Focus: Method)] ---
[{'paper_id': 'paper_001', 'composite_score': 1.0}, {'paper_id': 'paper_002', 'composite_score': 0.7812}]

--- [EVALUATION METRICS] ---
Paper ID: paper_001
Methodological Novelty Score: 0.2188
Rigor-to-Impact Ratio:        1.0000