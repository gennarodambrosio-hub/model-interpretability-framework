#!/usr/bin/env python3
"""Script to verify and download IBM Granite models for interpretability.
Usage:
    python scripts/download_model.py --model ibm-granite/granite-4.2-3b
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoConfig, AutoTokenizer
from rich.console import Console
from rich.panel import Panel

console = Console()


def verify_or_download(model_id: str, download_weights: bool = False):
    console.print(Panel.fit(f"[bold cyan]Verifica Modello:[/bold cyan] [yellow]{model_id}[/yellow]"))

    try:
        console.print("[dim]Download configurazione e tokenizer da Hugging Face...[/dim]")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", "N/A"))
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", "N/A"))
        vocab_size = getattr(config, "vocab_size", len(tokenizer))

        console.print(f"✅ [bold green]Configurazione caricata con successo![/bold green]")
        console.print(f"   • Numero di layer: [bold]{num_layers}[/bold]")
        console.print(f"   • Dimensione nascosta (hidden_size): [bold]{hidden_size}[/bold]")
        console.print(f"   • Dimensione vocabolario: [bold]{vocab_size}[/bold]")

        if download_weights:
            console.print("[dim]Download completo dei pesi in corso (richiede qualche minuto)...[/dim]")
            from core.model_loader import GraniteEngine
            engine = GraniteEngine(model_id=model_id)
            engine.load()
            console.print("🎉 [bold green]Modello scaricato e caricato in memoria con successo![/bold green]")
        else:
            console.print("[cyan]Nota: I pesi del modello verranno scaricati automaticamente alla prima esecuzione dell'app o della CLI.[/cyan]")

    except Exception as e:
        console.print(f"❌ [bold red]Errore durante la verifica di {model_id}:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica e download del modello IBM Granite")
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-granite/granite-4.2-3b",
        help="ID del modello Hugging Face (default: ibm-granite/granite-4.2-3b)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Scarica immediatamente tutti i pesi del modello in locale",
    )
    args = parser.parse_args()
    verify_or_download(args.model, download_weights=args.full)
