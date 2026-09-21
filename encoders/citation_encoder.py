import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from typing import Dict

class CitationGNN(nn.Module):
    def __init__(self, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        # Heterogeneous Graph Convolution handling relations: paper-cites-paper, author-wrote-paper
        self.conv1 = HeteroConv({
            ('paper', 'cites', 'paper'): SAGEConv(output_dim, hidden_dim),
            ('author', 'wrote', 'paper'): SAGEConv(output_dim, hidden_dim),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, output_dim),
            ('author', 'wrote', 'paper'): SAGEConv(hidden_dim, output_dim),
        }, aggr='mean')

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.normalize(x, p=2, dim=-1) for key, x in x_dict.items()}
        return x_dict