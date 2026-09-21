import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TopicEncoder(nn.Module):
    def __init__(self, model_name: str = "allenai/specter2_base", output_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        # Projection head to reduce dimensionality to 256-D
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)
        
    def forward(self, title: str, abstract: str) -> torch.Tensor:
        text = f"{title} {self.tokenizer.sep_token} {abstract}"
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # CLS token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
        projected = self.projection(cls_embedding)
        return torch.nn.functional.normalize(projected, p=2, dim=1).squeeze(0)