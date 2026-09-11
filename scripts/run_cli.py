#!/usr/bin/env python3
"""CLI utility to test prompt execution and activation verbalization from terminal.
Usage:
    python scripts/run_cli.py --prompt "Nel mezzo del cammin di nostra vita" --layer 16
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.model_loader import GraniteEngine
from core.activation_hooks import ResidualStreamHookManager
from core.verbalizer import ActivationVerbalizer

console = Console()


def run_cli(prompt: str, model_id: str, target_layer: int, max_tokens: int):
    console.print(Panel.fit(f"[bold cyan]Interpretabilità Granite 4.2:[/bold cyan] [yellow]{model_id}[/yellow]"))

    console.print("[dim]Caricamento modello su GPU/MPS...[/dim]")
    engine = GraniteEngine(model_id=model_id)
    engine.load()

    hook_mgr = ResidualStreamHookManager(engine.get_layer_modules())
    verbalizer = ActivationVerbalizer(
        tokenizer=engine.tokenizer,
        unembedding_head=engine.get_unembedding_head(),
        final_norm=engine.get_final_norm(),
        top_k=8,
    )

    console.print(f"\n[bold green]Input Prompt:[/bold green] {prompt}")
    console.print("[dim]Esecuzione forward pass con cattura del residual stream...[/dim]")

    gen_result = engine.generate_and_inspect(
        prompt=prompt,
        hook_manager=hook_mgr,
        max_new_tokens=max_tokens,
    )

    output_text = gen_result["output_text"]
    console.print(f"[bold green]Output Generato:[/bold green] {output_text.strip()}\n")

    # Inspect the last prompt token
    last_prompt_idx = len(gen_result["prompt_tokens"]) - 1
    inspected_token = gen_result["prompt_tokens"][last_prompt_idx]

    layer_to_inspect = min(target_layer, engine.num_layers - 1)
    act_vec = hook_mgr.get_activation(layer_idx=layer_to_inspect, token_pos=last_prompt_idx)
    v_res = verbalizer.verbalize(
        activation_vector=act_vec,
        layer_index=layer_to_inspect,
        token_position=last_prompt_idx,
        input_token=inspected_token,
    )

    console.print(Panel(v_res.natural_language_summary, title=f"🧠 Spiegazione Attivazione (Layer {layer_to_inspect})", border_style="cyan"))

    table = Table(title=f"Top Concetti Proiettati nel Vocabolario (Layer {layer_to_inspect})")
    table.add_column("Token", style="bold yellow")
    table.add_column("Probabilità", style="green")
    table.add_column("Logit", style="cyan")

    for p in v_res.top_predictions:
        table.add_row(p.token, f"{p.probability:.2%}", f"{p.logit:.2f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI di interpretabilità per IBM Granite")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Nel mezzo del cammin di nostra vita, mi ritrovai per una selva",
        help="Prompt da inviare al modello",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-granite/granite-4.2-3b",
        help="ID del modello (default: ibm-granite/granite-4.2-3b)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Indice del layer da ispezionare",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Numero massimo di token generati",
    )
    args = parser.parse_args()
    run_cli(args.prompt, args.model, args.layer, args.max_tokens)
