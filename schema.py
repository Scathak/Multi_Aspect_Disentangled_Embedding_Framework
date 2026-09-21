from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict
import numpy as np

class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    abstract: str
    methods_text: str
    results_text: str
    code_url: Optional[str] = None
    citations: List[str] = Field(default_factory=list) # List of paper_ids cited
    authors: List[str] = Field(default_factory=list)
    venue: str = ""
    publication_year: int

class RigorFeatures(BaseModel):
    has_code: float
    has_dataset_link: float
    p_value_count: float
    sample_size_log: float
    ablation_mentioned: float
    baseline_count: float

class CompositeEmbedding(BaseModel):
    paper_id: str
    topic_vector: List[float]    # 256-D
    method_vector: List[float]   # 256-D
    citation_vector: List[float] # 256-D
    rigor_vector: List[float]    # 128-D

    class Config:
        arbitrary_types_allowed = True