"""Unit tests for the Model Interpretability Framework components.
Tests hook management, logit lens projection, and Anthropic NLA interfaces using lightweight synthetic layers.
"""

import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from core.activation_hooks import ResidualStreamHookManager
from core.verbalizer import ActivationVerbalizer
from core.anthropic_nla import AnthropicNLAClient, NLACritic, NLAConfig


class DummyDecoderLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (self.linear(x), None)


class TestInterpretabilityFramework(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.vocab_size = 100
        self.num_layers = 4
        self.layers = [DummyDecoderLayer(self.d_model) for _ in range(self.num_layers)]

    def test_hook_manager_capture(self):
        mgr = ResidualStreamHookManager(self.layers)
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, self.d_model)

        with mgr.capture():
            current = x
            for layer in self.layers:
                current = layer(current)[0]

        # Verify all layers captured
        self.assertEqual(len(mgr.captured_activations), self.num_layers)
        
        # Test activation extraction
        vec = mgr.get_activation(layer_idx=1, token_pos=-1)
        self.assertEqual(vec.shape, (self.d_model,))

        # Test layer norms and similarities
        norms = mgr.compute_layer_norms(token_pos=0)
        self.assertEqual(len(norms), self.num_layers)

        sims = mgr.compute_layer_cosine_similarities(token_pos=0)
        self.assertEqual(len(sims), self.num_layers - 1)

    def test_anthropic_nla_scaling_and_critic(self):
        cfg = NLAConfig(
            d_model=self.d_model,
            injection_char="㈎",
            injection_scale=150.0,
            prompt_template="Explain: {injection_char}",
        )
        client = AnthropicNLAClient(cfg)
        critic = NLACritic(self.d_model)

        v = torch.randn(self.d_model)
        scaled_v = client.scale_activation(v)
        
        # Check scaled L2 norm is 150.0
        self.assertAlmostEqual(torch.linalg.vector_norm(scaled_v).item(), 150.0, places=3)

        # Check critic MSE
        res = critic.compute_mse_loss(v, v)
        self.assertAlmostEqual(res["mse"], 0.0, places=5)
        self.assertAlmostEqual(res["cosine_similarity"], 1.0, places=4)
        self.assertAlmostEqual(res["fidelity_score"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
