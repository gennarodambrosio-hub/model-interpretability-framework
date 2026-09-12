"""Streamlit Web Interface for Model Interpretability Framework.
Inspects IBM Granite 4.2 activations and translates them into natural language concepts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from core.activation_hooks import ResidualStreamHookManager
from core.anthropic_nla import AnthropicNLAClient, NLAConfig
from core.model_loader import GraniteEngine, get_optimal_device
from core.verbalizer import ActivationVerbalizer

st.set_page_config(
    page_title="Model Interpretability Framework | Granite 4.2 & NLA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Caricamento del modello base in memoria...")
def load_granite_engine(model_id: str, load_in_4bit: bool):
    engine = GraniteEngine(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
    )
    engine.load()
    return engine


_CURRENT_LOADED_ACTOR: tuple[str, AnthropicNLAClient] | None = None

def load_nla_actor_client(checkpoint_path: str):
    """Loads NLA actor client ensuring previous models are freed from unified memory."""
    global _CURRENT_LOADED_ACTOR
    
    if _CURRENT_LOADED_ACTOR is not None and _CURRENT_LOADED_ACTOR[0] == checkpoint_path:
        return _CURRENT_LOADED_ACTOR[1]

    # Free previous actor from VRAM/MPS
    if _CURRENT_LOADED_ACTOR is not None:
        del _CURRENT_LOADED_ACTOR
        _CURRENT_LOADED_ACTOR = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = NLAConfig(
        d_model=2560,
        injection_char="*",
        injection_scale=150.0,
        prompt_template="Explain the concept represented by this activation: {injection_char} Explanation: ",
        actor_model_path=checkpoint_path,
    )
    client = AnthropicNLAClient(config=config)
    p = Path(checkpoint_path)
    if p.exists():
        client.load_local_actor(p)
        _CURRENT_LOADED_ACTOR = (checkpoint_path, client)
    return client


def main():
    st.title("🧠 Model Interpretability Framework")
    st.caption("Esplorazione e traduzione delle attivazioni interne in linguaggio naturale per la famiglia IBM Granite")

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Configurazione Modello")
        
        model_choice = st.selectbox(
            "Seleziona Modello",
            options=[
                ("ibm-granite/granite-4.2-3b", "Granite 4.2 3B (Non quantizzato - Consigliato)", False),
                ("ibm-granite/granite-3.2-2b-instruct", "Granite 3.2 2B Instruct (Leggero)", False),
                ("ibm-granite/granite-4.2-8b", "Granite 4.2 8B (4-bit Quantizzato)", True),
            ],
            format_func=lambda x: x[1],
            index=0,
        )
        model_id, _, load_in_4bit = model_choice

        device = get_optimal_device()
        st.info(f"Dispositivo attivo: **{device.type.upper()}** (Apple Silicon Neural/GPU Accelerator)")

        st.subheader("Parametri Generazione")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
        max_new_tokens = st.slider("Max New Tokens", min_value=16, max_value=512, value=128, step=16)
        top_k_display = st.slider("Top-K Concetti (Logit Lens)", min_value=3, max_value=20, value=8, step=1)

        st.markdown("---")
        st.markdown(
            """
            **Riferimento Metodologico:**
            - Anthropic Research: [*Natural Language Autoencoders*](https://www.anthropic.com/research/natural-language-autoencoders)
            - Paper: [*Turning Claude's thoughts into text*](https://transformer-circuits.pub/2026/nla/index.html)
            - Repository: `kitft/natural_language_autoencoders`
            """
        )

    # --- INITIALIZE MODEL ---
    try:
        engine = load_granite_engine(model_id, load_in_4bit)
    except Exception as e:
        st.error(f"Errore durante il caricamento di {model_id}: {str(e)}")
        st.stop()

    layers = engine.get_layer_modules()
    final_norm = engine.get_final_norm()
    lm_head = engine.get_unembedding_head()
    verbalizer = ActivationVerbalizer(
        tokenizer=engine.tokenizer,
        unembedding_head=lm_head,
        final_norm=final_norm,
        top_k=top_k_display,
    )

    # --- PROMPT INPUT SECTION ---
    st.subheader("1. Inserisci Prompt")
    
    col_preset, _ = st.columns([2, 1])
    preset_choice = col_preset.selectbox(
        "Oppure scegli un esempio di test:",
        options=[
            "Personalizzato",
            "Completa la rima: Nel mezzo del cammin di nostra vita, mi ritrovai per una selva",
            "Ragionamento logico: Se tutti i gatti miagolano e Felix è un gatto, cosa fa Felix?",
            "Pianificazione interna: Scrivi una poesia di due versi su un orologio fermo.",
        ],
    )

    if preset_choice != "Personalizzato":
        default_prompt = preset_choice.split(": ", 1)[-1]
    else:
        default_prompt = "Nel mezzo del cammin di nostra vita, mi ritrovai per una selva"

    prompt = st.text_area("Prompt di input:", value=default_prompt, height=90)

    if st.button("🚀 Invia Input & Ispeziona Attivazioni", type="primary"):
        with st.spinner("Esecuzione forward pass ed estrazione attivazioni..."):
            hook_mgr = ResidualStreamHookManager(layers)
            gen_result = engine.generate_and_inspect(
                prompt=prompt,
                hook_manager=hook_mgr,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            st.session_state["gen_result"] = gen_result
            st.session_state["hook_mgr"] = hook_mgr

    # --- DISPLAY RESULTS ---
    if "gen_result" in st.session_state:
        gen = st.session_state["gen_result"]
        hook_mgr = st.session_state["hook_mgr"]

        st.markdown("---")
        st.subheader("2. Output Generato")
        
        # Display thinking block if Granite uses reasoning mode
        output_text = gen["output_text"]
        if "<think>" in output_text and "</think>" in output_text:
            think_part = output_text.split("<think>")[1].split("</think>")[0]
            answer_part = output_text.split("</think>")[1]
            with st.expander("💭 Processo di Ragionamento Interno (<think>)", expanded=True):
                st.write(think_part.strip())
            st.success(answer_part.strip())
        else:
            st.success(output_text if output_text.strip() else "(Nessun token generato oltre il prompt)")

        st.markdown("---")
        st.subheader("3. Mappa dei Token e Selezione Attivazione")

        all_tokens = gen["prompt_tokens"] + gen["generated_tokens"]
        st.write("Clicca su un token o seleziona l'indice per ispezionare il suo vettore di stato interno:")

        selected_token_idx = st.slider(
            "Indice Token (0 = primo token del prompt, valore massimo = ultimo token generato):",
            min_value=0,
            max_value=len(all_tokens) - 1,
            value=len(gen["prompt_tokens"]) - 1,
        )
        selected_token = all_tokens[selected_token_idx]
        is_prompt_token = selected_token_idx < len(gen["prompt_tokens"])
        
        st.markdown(
            f"Token selezionato: **`{repr(selected_token)}`** "
            f"({'Prompt' if is_prompt_token else 'Generato'}, Posizione #{selected_token_idx})"
        )

        st.markdown("---")
        st.subheader("4. Traduzione dell'Attivazione in Linguaggio Naturale")

        tab_lens, tab_trajectory, tab_nla_info = st.tabs([
            "🔍 Traduzione Semantica (Logit Lens)",
            "📈 Traiettoria tra Layer (Norma & Similarità)",
            "🔬 Architettura Anthropic NLA",
        ])

        with tab_lens:
            col_ctrl, col_res = st.columns([1, 2])
            
            with col_ctrl:
                selected_layer = st.slider(
                    "Seleziona Layer di Ispezione:",
                    min_value=0,
                    max_value=engine.num_layers - 1,
                    value=engine.num_layers // 2,
                )
                
                # Fetch activation
                try:
                    act_vec = hook_mgr.get_activation(
                        layer_idx=selected_layer,
                        token_pos=selected_token_idx,
                    )
                    v_res = verbalizer.verbalize(
                        activation_vector=act_vec,
                        layer_index=selected_layer,
                        token_position=selected_token_idx,
                        input_token=selected_token,
                    )
                    st.metric("Norma L2 del Vettore", f"{v_res.l2_norm:.2f}")
                    st.metric("Entropia Semantica", f"{v_res.entropy:.2f} bit")
                except Exception as ex:
                    st.error(f"Impossibile estrarre attivazione: {ex}")
                    v_res = None

            with col_res:
                if v_res is not None:
                    st.info(f"🗣️ **Logit Lens (Proiezione Vocabolario):**\n\n{v_res.natural_language_summary}")

                    # --- ANTHROPIC NLA SECTION ---
                    # Automatically select actor checkpoint based on the selected layer
                    if selected_layer == 35:
                        nla_ckpt = PROJECT_ROOT / "checkpoints" / "granite-nla-actor-layer35"
                        target_layer_str = "Layer 35 (Deep Output Stage)"
                    else:
                        nla_ckpt = PROJECT_ROOT / "checkpoints" / "granite-nla-actor-layer20"
                        target_layer_str = f"Layer {selected_layer}"

                    if nla_ckpt.exists() and (nla_ckpt / "model.safetensors").exists():
                        st.divider()
                        st.markdown("### 🎯 Anthropic Natural Language Autoencoder (Actor)")
                        st.caption(f"Pesi addestrati su 4x A100 (Continuous Vector Injection - {target_layer_str})")
                        
                        btn_nla = st.button("✨ Genera Spiegazione con Actor NLA", key=f"btn_nla_{selected_token_idx}_{selected_layer}")
                        if btn_nla:
                            try:
                                with st.spinner(f"Iniezione continua del vettore al Layer {selected_layer} e generazione spiegazione concettuale..."):
                                    nla_client = load_nla_actor_client(str(nla_ckpt))
                                    nla_res = nla_client.generate_explanation(
                                        activation_vector=act_vec,
                                        context_hint=f"Token: '{selected_token}' al Layer {selected_layer}",
                                    )
                                    st.session_state[f"nla_cache_{selected_layer}_{selected_token_idx}"] = nla_res
                            except Exception as e_nla:
                                st.error(f"Inferenza NLA: {e_nla}")
                                st.session_state[f"nla_cache_{selected_layer}_{selected_token_idx}"] = None

                        cached_res = st.session_state.get(f"nla_cache_{selected_layer}_{selected_token_idx}")
                        if cached_res:
                            st.success(f"**Concetto Spiegato in Linguaggio Naturale dall'Actor:**\n\n> *\"{cached_res['explanation']}\"*")
                            col_n1, col_n2 = st.columns(2)
                            col_n1.metric("Norma L2 Scalata (Target)", f"{cached_res['scaled_l2_norm']:.1f}")
                            col_n2.metric("Stato Modello", f"Addestrato per Layer {selected_layer}")

                    st.divider()
                    # Bar chart of top tokens
                    df_preds = pd.DataFrame([
                        {"Token": p.token, "Probabilità": p.probability, "Logit": p.logit}
                        for p in v_res.top_predictions
                    ])
                    fig = px.bar(
                        df_preds,
                        x="Probabilità",
                        y="Token",
                        orientation="h",
                        title=f"Top Concetti Proiettati dal Logit Lens al Layer {selected_layer}",
                        color="Probabilità",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=320)
                    st.plotly_chart(fig, use_container_width=True)

        with tab_trajectory:
            st.write("Evoluzione della norma L2 attraverso tutti i layer del modello per il token selezionato:")
            try:
                norms = hook_mgr.compute_layer_norms(token_pos=selected_token_idx)
                df_norms = pd.DataFrame([
                    {"Layer": l, "Norma L2": val} for l, val in norms.items()
                ])
                fig_norm = px.line(
                    df_norms,
                    x="Layer",
                    y="Norma L2",
                    markers=True,
                    title=f"Traiettoria della Norma L2 lungo i Layer (Token: '{selected_token}')",
                )
                st.plotly_chart(fig_norm, use_container_width=True)

                sims = hook_mgr.compute_layer_cosine_similarities(token_pos=selected_token_idx)
                df_sims = pd.DataFrame([
                    {"Transizione": k, "Cosine Similarity": v} for k, v in sims.items()
                ])
                fig_sim = px.bar(
                    df_sims,
                    x="Transizione",
                    y="Cosine Similarity",
                    title="Similarità Coseno tra Layer Consecutivi (Stabilità delle Attivazioni)",
                    range_y=[0, 1],
                )
                st.plotly_chart(fig_sim, use_container_width=True)
            except Exception as e_traj:
                st.warning(f"Calcolo traiettoria: {e_traj}")

        with tab_nla_info:
            st.markdown(
                r"""
                ### Come funziona la tecnica Natural Language Autoencoder (Anthropic)
                Nel paper di Anthropic (*Turning Claude's thoughts into text*):
                1. **Target Model (Frozen):** Il modello target (es. Granite) riceve l'input ed estrae il vettore $h_l \in \mathbb{R}^{d_{model}}$.
                2. **Activation Verbalizer (AV / Actor):** Un modello modificato riceve $h_l$ iniettato come embedding al posto del carattere marcatore `㈎` e genera una spiegazione testuale del pensiero interno.
                3. **Activation Reconstructor (AR / Critic):** Un secondo modello legge la spiegazione testuale e tenta di ricostruire il vettore originale $\hat{h}_l$.
                4. **Reward di Addestramento (RL/GRPO):** Viene calcolato l'errore quadratico medio $\text{MSE}(h_l, \hat{h}_l)$. Più basso è l'MSE, più la spiegazione verbale è fedele allo stato numerico.
                
                > **Stato dei Checkpoint:**
                > I checkpoint NLA pre-addestrati rilasciati dalla community (`kitft/nla-models`) coprono architetture come Gemma 3, Qwen 2.5 e Llama 3.
                > Per la famiglia Granite, il framework include già la predisposizione del client NLA (`core/anthropic_nla.py`) e offre la verbalizzazione diretta tramite Logit Lens integrata nel tab principale.
                """
            )


if __name__ == "__main__":
    main()
