import numpy as np
import matplotlib.pyplot as plt
from schema import CompositeEmbedding
from analytics.comparator import ArticleComparator

def plot_aspect_radar(emb_a: CompositeEmbedding, emb_b: CompositeEmbedding, save_path: str = None):
    comparator = ArticleComparator()
    report = comparator.compare(emb_a, emb_b)
    
    categories = ['Topic\nSimilarity', 'Method\nSimilarity', 'Citation Graph\nSimilarity', 'Rigor Profile\nSimilarity']
    values = [
        report.similarity_vector['topic'],
        report.similarity_vector['method'],
        report.similarity_vector['citation'],
        report.similarity_vector['rigor']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, color='#1f77b4', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    plt.title(f"Aspect Comparison: {emb_a.paper_id} vs {emb_b.paper_id}\nLabel: {report.relationship_label}", 
              size=11, color='black', y=1.1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()