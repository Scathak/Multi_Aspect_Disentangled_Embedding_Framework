import re
import math
import torch
import torch.nn as nn
from schema import PaperMetadata, RigorFeatures

class RigorFeatureExtractor:
    @staticmethod
    def extract(paper: PaperMetadata) -> RigorFeatures:
        text_to_check = f"{paper.methods_text} {paper.results_text}".lower()
        
        # Binary signal checks
        has_code = 1.0 if paper.code_url and len(paper.code_url) > 0 else 0.0
        has_dataset = 1.0 if "huggingface.co/datasets" in text_to_check or "github.com" in text_to_check or "zenodo" in text_to_check else 0.0
        
        # Regex checks for methodological rigor
        p_values = len(re.findall(r'p\s*[\&lt;\=\&gt;]\s*0\.\d+', text_to_check))
        
        # Sample size heuristic: matches "n = XYZ" or "sample size of XYZ"
        sample_matches = re.findall(r'(?:n\s*=\s*|sample size of\s*)(\d+)', text_to_check)
        sample_sizes = [int(m) for m in sample_matches if int(m) > 0]
        max_sample = max(sample_sizes) if sample_sizes else 1.0
        
        ablation = 1.0 if "ablation study" in text_to_check or "ablation analysis" in text_to_check else 0.0
        baselines = len(re.findall(r'baseline|state-of-the-art|sota', text_to_check))

        return RigorFeatures(
            has_code=has_code,
            has_dataset_link=has_dataset,
            p_value_count=min(float(p_values), 10.0) / 10.0, # Scaled [0, 1]
            sample_size_log=float(math.log10(max_sample + 1.0)) / 6.0, # Scaled log size
            ablation_mentioned=ablation,
            baseline_count=min(float(baselines), 15.0) / 15.0
        )

class RigorMLP(nn.Module):
    def __init__(self, input_dim: int = 6, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, features: RigorFeatures) -> torch.Tensor:
        feat_tensor = torch.tensor([
            features.has_code,
            features.has_dataset_link,
            features.p_value_count,
            features.sample_size_log,
            features.ablation_mentioned,
            features.baseline_count
        ], dtype=torch.float32).unsqueeze(0)
        
        out = self.net(feat_tensor)
        return torch.nn.functional.normalize(out, p=2, dim=1).squeeze(0)