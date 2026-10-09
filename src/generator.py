"""Generate grounded answers with Qwen3-0.6B."""

from typing import Any, Dict, List, Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .indexer import read_text
from .models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 12000

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the "
    "vLLM codebase. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say that you "
    "do not know. Be concise."
)


def build_context(
    sources: List[MinimalSource],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Read each source from disk and join them within a size budget."""
    parts: List[str] = []
    total = 0
    cache: Dict[str, Optional[str]] = {}
    for i, src in enumerate(sources, start=1):
        if src.file_path not in cache:
            cache[src.file_path] = read_text(src.file_path)
        text = cache[src.file_path]
        if text is None:
            continue
        snippet = text[
            src.first_character_index:src.last_character_index
        ].strip()
        if not snippet:
            continue
        if total + len(snippet) > max_chars:
            break
        parts.append(f"[{i}] ({src.file_path})\n{snippet}")
        total += len(snippet)
    return "\n\n".join(parts)


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    """Build the chat messages: instructions plus context and question."""
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    """Load Qwen3-0.6B once and generate answers."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and the model on CPU."""
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        cast(torch.nn.Module, self.model).eval()

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an answer grounded in the given context."""
        if not question.strip():
            return ""
        if not context.strip():
            return "I do not know: no relevant context was found."
        messages = build_messages(question, context)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        n = inputs["input_ids"].shape[1]
        text: str = self.tokenizer.decode(out[0][n:], skip_special_tokens=True)
        return text.strip()
