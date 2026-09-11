"""Model loader module for IBM Granite models.
Supports loading Granite 4.2 3B (unquantized in bfloat16/float16) and Granite 8B (4-bit or float16)
with native Apple Silicon (MPS) acceleration and memory management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """Determine the fastest available device (Apple Silicon MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """Return appropriate floating point precision."""
    if device.type == "mps":
        # Apple Silicon MPS has robust native support for bfloat16 and float16
        return torch.bfloat16
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.type != "cpu" else torch.float32


class GraniteEngine:
    """Manages loading, tokenization, forward passes, and generation for IBM Granite models."""

    def __init__(
        self,
        model_id: str = "ibm-granite/granite-4.2-3b",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        
        # Device resolution
        if device is None or device == "auto":
            self.device = get_optimal_device()
        else:
            self.device = torch.device(device)
            
        # Dtype resolution
        if torch_dtype is None or torch_dtype == "auto":
            self.dtype = get_optimal_dtype(self.device)
        elif torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.config: Optional[AutoConfig] = None
        self.num_layers: int = 0
        self.hidden_size: int = 0

    def load(self) -> GraniteEngine:
        """Load tokenizer and model weights into memory."""
        logger.info(f"Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model {self.model_id} on {self.device} with dtype={self.dtype}...")
        
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.dtype,
        }
        
        if self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            # Native direct placement onto MPS/CUDA/CPU
            load_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        
        if not self.load_in_4bit:
            self.model.to(self.device)
            
        self.model.eval()
        self.config = self.model.config

        # Extract architecture specs
        self.num_layers = getattr(
            self.config, "num_hidden_layers", getattr(self.config, "n_layer", 0)
        )
        self.hidden_size = getattr(
            self.config, "hidden_size", getattr(self.config, "n_embd", 0)
        )

        logger.info(
            f"Successfully loaded {self.model_id} ({self.num_layers} layers, hidden_size={self.hidden_size})"
        )
        return self

    def get_layer_modules(self) -> List[torch.nn.Module]:
        """Returns the list of transformer decoder layer modules."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        
        # Standard Hugging Face Granite / Llama architecture
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        else:
            raise AttributeError("Unable to locate transformer layers in model hierarchy.")

    def get_final_norm(self) -> Optional[torch.nn.Module]:
        """Returns the final LayerNorm / RMSNorm module prior to the lm_head."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        return None

    def get_unembedding_head(self) -> torch.nn.Module:
        """Returns the unembedding linear projection (lm_head)."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        raise AttributeError("No lm_head found on model.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generates text from prompt, returning prompt tokens, generated tokens, and full text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_length = input_ids.shape[1]

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        output_ids = self.model.generate(**generate_kwargs)
        generated_ids = output_ids[0, prompt_length:]

        prompt_tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids.tolist()]
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        return {
            "prompt": prompt,
            "full_text": full_text,
            "output_text": output_text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "input_ids": input_ids,
            "output_ids": output_ids,
        }

    @torch.no_grad()
    def generate_and_inspect(
        self,
        prompt: str,
        hook_manager: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generates text and then runs a forward pass on the full output sequence with hook_manager attached."""
        gen_result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Clone output_ids to avoid inference tensor issues and run forward pass to capture all tokens
        seq_tensor = gen_result["output_ids"].clone().to(self.device)
        with hook_manager.capture():
            self.model(seq_tensor)

        return gen_result
