from schema import PaperMetadata
from pipeline import ADSEPipeline
from analytics.comparator import ArticleComparator
from analytics.evaluation import ScientificEvaluator
from analytics.visualization import plot_aspect_radar

# 1. Initialize System
pipeline = ADSEPipeline()
comparator = ArticleComparator()

# 2. Define Your Paper (Paper A)
# my_paper = PaperMetadata(
    # paper_id="MY_RESEARCH_2024",
    # title="Transformer Models for Exoplanet Transit Detection",
    # abstract="We apply self-attention architectures to Kepler light-curve time-series data to detect exoplanets.",
    # methods_text="We used dynamic multi-head self-attention with positional encoding. Code and models released at github.com/lab/exo-transformer.",
    # results_text="Achieved 98.4% accuracy with p < 0.001. Comprehensive ablation study conducted.",
    # code_url="https://github.com/lab/exo-transformer",
    # citations=[],
    # authors=["You"],
    # venue="Astronomy & AI",
    # publication_year=2024
# )

my_paper = PaperMetadata(
    paper_id="MY_RESEARCH_2024",
    title="Transformer Models for Exoplanet Transit Detection",
    abstract="We apply self-attention architectures to Kepler light-curve time-series data to detect exoplanets.",
    methods_text="We used dynamic multi-head self-attention with positional encoding. Code and models released at github.com/lab/exo-transformer.",
    results_text="Achieved 98.4% accuracy with p < 0.001. Comprehensive ablation study conducted.",
    code_url="https://github.com/lab/exo-transformer",
    citations=[],
    authors=["You"],
    venue="Astronomy & AI",
    publication_year=2024
)


# 3. Define Baseline/Competitor Paper (Paper B)
existing_paper = PaperMetadata(
    paper_id="NLP_BERT_2019",
    title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    abstract="We introduce BERT for language understanding tasks using bidirectional transformers.",
    methods_text="Masked language modeling with multi-head self-attention.",
    results_text="State-of-the-art results across 11 NLP tasks.",
    code_url="https://github.com/google-research/bert",
    citations=[],
    authors=["Devlin et al."],
    venue="NAACL",
    publication_year=2019
)

# 4. Process Embeddings
emb_a = pipeline.process_paper(my_paper)
emb_b = pipeline.process_paper(existing_paper)

# 5. Execute Comparative Diagnostic
report = comparator.compare(emb_a, emb_b)
novelty = ScientificEvaluator.calculate_novelty(emb_a, [emb_b])
rigor_ratio = ScientificEvaluator.calculate_rigor_to_impact_ratio(emb_a)

# 6. Print Interpretable Diagnostics
print("\n" + "="*60)
print(f"       ADSE RESEARCH EVALUATION REPORT FOR: [{emb_a.paper_id}]")
print("="*60)
print(f"Compared Against Paper:   {emb_b.paper_id}")
print(f"Similarity Breakdown:     {report.similarity_vector}")
print(f"DAI Score:                {report.dai_score}")
print(f"Taxonomy Relationship:    {report.relationship_label}")
print(f"Actionable Strategy:      {report.recommendation}")
print("-" * 60)
print(f"Methodological Novelty:   {novelty:.4f}  (High > 0.70)")
print(f"Rigor-to-Impact Ratio:    {rigor_ratio:.4f}  (Hidden Gem > 1.2 | Hype < 0.5)")
print("="*60 + "\n")

# 7. Generate Visual Radar Chart
plot_aspect_radar(emb_a, emb_b, save_path="my_paper_radar.png")
print("📊 Radar Chart saved as 'my_paper_radar.png'")