import torch
from pathlib import Path
from core.anthropic_nla import AnthropicNLAClient, NLAConfig
from core.model_loader import GraniteEngine
from core.activation_hooks import ResidualStreamHookManager

from core.verbalizer import ActivationVerbalizer

print("=== Caricamento Modello Base Granite 4.2 ===")
engine = GraniteEngine(model_id="ibm-granite/granite-4.2-3b")
engine.load()

layers = engine.get_layer_modules()
hook_mgr = ResidualStreamHookManager(layers)
verbalizer = ActivationVerbalizer(
    tokenizer=engine.tokenizer,
    unembedding_head=engine.get_unembedding_head(),
    final_norm=engine.get_final_norm(),
    top_k=5,
)

ckpt_20 = Path("checkpoints/granite-nla-actor-layer20")
ckpt_28 = Path("checkpoints/granite-nla-actor-layer28")

test_cases = [
    {
        "domain": "Letteratura Italiana (Dante Alighieri - Divina Commedia)",
        "prompt": "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura ché la diritta via era smarrita",
        "target_word": "oscura",
    },
    {
        "domain": "Filosofia Morale / Esistenziale (Nietzsche / Coscienza)",
        "prompt": "He who has a why to live can bear almost any how, facing the abyss of human existence",
        "target_word": "abyss",
    },
    {
        "domain": "Poesia Romantica / Emozione (Leopardi - L'Infinito)",
        "prompt": "Sempre caro mi fu quest'ermo colle, e questa siepe, che da tanta parte dell'ultimo orizzonte il guardo esclude",
        "target_word": "orizzonte",
    }
]

for tc in test_cases:
    print(f"\n" + "="*70)
    print(f"📖 DOMINIO: {tc['domain']}")
    print(f"Testo: '{tc['prompt']}'")
    print("="*70)
    
    gen_res = engine.generate_and_inspect(
        prompt=tc["prompt"],
        hook_manager=hook_mgr,
        max_new_tokens=1,
        temperature=0.0,
    )
    tokens = gen_res["prompt_tokens"]
    
    # Trova il token target
    target_idx = len(tokens) - 1
    for i, t in enumerate(tokens):
        if tc["target_word"].lower() in t.lower():
            target_idx = i
            break
            
    print(f"Token ispezionato: [{target_idx}] '{tokens[target_idx]}'")
    
    # 1. Logit Lens (Vocab Projection) al Layer 20 e 28
    act_20 = hook_mgr.get_activation(layer_idx=20, token_pos=target_idx)
    v_res_20 = verbalizer.verbalize(act_20, layer_index=20, token_position=target_idx, input_token=tokens[target_idx])
    print(f"\n🔍 [Logit Lens - Layer 20]:\n   -> Top Token: {', '.join([f'{p.token} ({p.probability*100:.1f}%)' for p in v_res_20.top_predictions[:3]])}")

    # 2. NLA Layer 20 (Mid-level conceptual representation)
    if ckpt_20.exists() and (ckpt_20 / "model.safetensors").exists():
        client_20 = AnthropicNLAClient(NLAConfig(d_model=2560, injection_char="*", injection_scale=150.0, prompt_template=""))
        client_20.load_local_actor(ckpt_20, device=engine.device)
        res_20 = client_20.generate_explanation(act_20, context_hint="Layer 20")
        print(f"\n🧠 [NLA Layer 20 - Semantica Astratta]:\n   -> \"{res_20['explanation']}\"")
        del client_20
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        print("\n🧠 [NLA Layer 20]: Checkpoint non ancora scaricato in checkpoints/granite-nla-actor-layer20/")

    # 3. Logit Lens al Layer 28
    act_28 = hook_mgr.get_activation(layer_idx=28, token_pos=target_idx)
    v_res_28 = verbalizer.verbalize(act_28, layer_index=28, token_position=target_idx, input_token=tokens[target_idx])
    print(f"\n🔍 [Logit Lens - Layer 28]:\n   -> Top Token: {', '.join([f'{p.token} ({p.probability*100:.1f}%)' for p in v_res_28.top_predictions[:3]])}")

    # 4. NLA Layer 28 (Anthropic 2/3 Depth semantic core)
    if ckpt_28.exists() and (ckpt_28 / "model.safetensors").exists():
        client_28 = AnthropicNLAClient(NLAConfig(d_model=2560, injection_char="*", injection_scale=150.0, prompt_template=""))
        client_28.load_local_actor(ckpt_28, device=engine.device)
        res_28 = client_28.generate_explanation(act_28, context_hint="Layer 28")
        print(f"\n🎯 [NLA Layer 28 - Core Astratto 2/3 Depth]:\n   -> \"{res_28['explanation']}\"")
        del client_28
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        print("\n🎯 [NLA Layer 28]: Checkpoint non ancora scaricato in checkpoints/granite-nla-actor-layer28/")

print("\n" + "="*70)
print("Test comparativo completato!")
