"""Model Interpretability Framework Core Module.
Provides model loading, residual stream activation hooks, Logit Lens, and Anthropic NLA integration.
"""

from .model_loader import GraniteEngine
from .activation_hooks import ResidualStreamHookManager
from .verbalizer import ActivationVerbalizer
from .anthropic_nla import AnthropicNLAClient, NLACritic

__all__ = [
    "GraniteEngine",
    "ResidualStreamHookManager",
    "ActivationVerbalizer",
    "AnthropicNLAClient",
    "NLACritic",
]
