"""Generate grounded answers with Qwen3-0.6B."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .indexer import read_text
from .models import MinimalSource


MODEL_NAME = "Qwen/Qwen3-0.6B"


class Generator:
    """Load Qwen3-0.6B once and generate answers."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and the model on CPU."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.model.eval()