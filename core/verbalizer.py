"""Activation verbalizer module.
Translates internal residual-stream activation vectors into human-readable natural language
using Logit Lens vocabulary projection and semantic concept synthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenPrediction:
    token: str
    token_id: int
    probability: float
    logit: float


@dataclass
class VerbalizationResult:
    layer_index: int
    token_position: int
    input_token: str
    top_predictions: List[TokenPrediction]
    natural_language_summary: str
    entropy: float
    l2_norm: float


class ActivationVerbalizer:
    """Decodes residual stream vectors into natural language concepts via vocabulary projection."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        unembedding_head: torch.nn.Module,
        final_norm: Optional[torch.nn.Module] = None,
        top_k: int = 10,
    ):
        self.tokenizer = tokenizer
        self.unembedding_head = unembedding_head
        self.final_norm = final_norm
        self.top_k = top_k
        self.device = next(unembedding_head.parameters()).device

    @torch.no_grad()
    def project_to_vocabulary(self, activation_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects a residual stream vector h_l onto the vocabulary space: logits = W_U * norm(h_l)."""
        vec = activation_vector.to(self.device)
        
        # Ensure 2D tensor of shape (1, hidden_size)
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)

        # Apply final norm if available (vital for RMSNorm/LayerNorm architectures like Granite)
        if self.final_norm is not None:
            # Cast norm input to match weight dtype
            norm_dtype = next(self.final_norm.parameters()).dtype
            vec = self.final_norm(vec.to(norm_dtype))

        head_dtype = next(self.unembedding_head.parameters()).dtype
        logits = self.unembedding_head(vec.to(head_dtype))  # shape: (1, vocab_size)
        probs = F.softmax(logits.float(), dim=-1)[0]
        return logits[0], probs

    def verbalize(
        self,
        activation_vector: torch.Tensor,
        layer_index: int,
        token_position: int,
        input_token: str = "",
    ) -> VerbalizationResult:
        """Produces a structured natural language explanation of what thoughts/concepts are encoded in the vector."""
        logits, probs = self.project_to_vocabulary(activation_vector)
        
        # Calculate Shannon entropy (lower = sharper commitment to specific concept)
        valid_probs = probs[probs > 0]
        entropy = float(-(valid_probs * torch.log2(valid_probs)).sum().item())
        l2_norm = float(torch.linalg.vector_norm(activation_vector).item())

        top_probs, top_indices = torch.topk(probs, k=self.top_k)
        
        predictions: List[TokenPrediction] = []
        clean_tokens: List[str] = []
        
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            raw_token = self.tokenizer.decode([idx])
            token_str = raw_token.strip()
            # If the token is empty (e.g. pure space/newline), show repr
            display_str = token_str if token_str else repr(raw_token)
            predictions.append(
                TokenPrediction(
                    token=display_str,
                    token_id=idx,
                    probability=float(p),
                    logit=float(logits[idx].item()),
                )
            )
            if token_str and token_str not in clean_tokens and len(token_str) > 1:
                clean_tokens.append(token_str)

        # Generate human-readable verbalization text
        top1 = predictions[0]
        top3_tokens = [p.token for p in predictions[:3]]
        
        if top1.probability > 0.4:
            certainty = "elevata certezza"
            focus_desc = f"è fortemente polarizzato verso il concetto '{top1.token}' ({top1.probability:.1%})"
        elif top1.probability > 0.15:
            certainty = "moderata certezza"
            focus_desc = f"esplora attivamente concetti affini come {', '.join(repr(t) for t in top3_tokens)}"
        else:
            certainty = "fase di elaborazione diffusa (bassa convergenza)"
            focus_desc = f"mantiene un ventaglio semantico aperto ({', '.join(repr(t) for t in top3_tokens)})"

        summary = (
            f"[Layer {layer_index} | Posizione token: '{input_token}'] "
            f"L'attivazione interna codifica pensieri con {certainty}. "
            f"La proiezione semantica nel vocabolario {focus_desc}. "
            f"(Norma L2: {l2_norm:.2f}, Entropia: {entropy:.2f} bit)."
        )

        return VerbalizationResult(
            layer_index=layer_index,
            token_position=token_position,
            input_token=input_token,
            top_predictions=predictions,
            natural_language_summary=summary,
            entropy=entropy,
            l2_norm=l2_norm,
        )
