import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MethodEncoder(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", output_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, methods_text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            methods_text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Masked Mean Pooling over token sequence
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
        projected = self.projection(mean_pooled)
        return torch.nn.functional.normalize(projected, p=2, dim=1).squeeze(0)