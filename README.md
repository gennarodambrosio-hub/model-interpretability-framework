# Model Interpretability Framework: IBM Granite & Natural Language Autoencoders

Framework avanzato per la **Mechanistic Interpretability** e la **traduzione in linguaggio naturale delle attivazioni interne (residual stream)** dei modelli della famiglia **IBM Granite (4.2 3B, 4.2 8B, 3.2 2B)**, ispirato alla ricerca di Anthropic: [*Natural Language Autoencoders: Turning Claude's thoughts into text*](https://www.anthropic.com/research/natural-language-autoencoders) (Maggio 2026).

---

## 📌 Indice dei Contenuti
1. [Obiettivo Scientifico](#obiettivo-scientifico)
2. [Analisi di Fattibilità & Hardware (Apple Silicon M5)](#analisi-di-fattibilità--hardware)
3. [La Tecnica NLA (Natural Language Autoencoders)](#la-tecnica-nla)
4. [Architettura del Framework](#architettura-del-framework)
5. [Installazione e Setup](#installazione-e-setup)
6. [Utilizzo dell'Interfaccia Web (Streamlit)](#utilizzo-dellinterfaccia-web)
7. [Utilizzo da Riga di Comando (CLI)](#utilizzo-da-riga-di-comando)
8. [Test di Unità](#test-di-unità)

---

## 🎯 Obiettivo Scientifico
Nei modelli di linguaggio transformer, l'elaborazione interna avviene sotto forma di vettori continui nello spazio ad alta dimensionalità del *residual stream* ($h_l \in \mathbb{R}^{d_{model}}$). Comprendere cosa rappresentano questi numeri ("pensieri latenti") prima che il modello emetta l'output testuale è uno dei problemi cardine dell'allineamento e della mechanistic interpretability.

Questo framework:
- Cattura in tempo reale i vettori di attivazione attraverso registrazioni di **PyTorch Forward Hooks** sui blocchi del decoder transformer.
- Traduce i vettori interni in **concetti comprensibili ed espliciti in linguaggio naturale**.
- Integra la specifica architetturale degli **Anthropic Natural Language Autoencoders (AV/AR)**.

---

## 💻 Analisi di Fattibilità & Hardware

### Specifiche di Riferimento (Mac M5, 16 GB Unified RAM)
1. **Granite 4.2 3B (Scelta Primaria Consigliata - Non Quantizzato):**
   - Dimensione: 3.2 miliardi di parametri.
   - Occupazione in memoria: **~6.2 GB in bfloat16 non quantizzato**.
   - **Vantaggio:** Gira con precisione totale e latenza minima sulla memoria unificata Apple Silicon, mantenendo inalterata la geometria del residual stream senza alcuna distorsione dovuta a quantizzazione.
2. **Granite 4.2 8B (Alternativa con Quantizzazione a 4-bit):**
   - Dimensione: 8.2 miliardi di parametri.
   - Occupazione in memoria: ~16 GB in FP16 (saturazione RAM) oppure **~5.2 GB in quantizzazione 4-bit (bitsandbytes / GGUF)**.
   - Disponibile nel selettore dell'app per test di comparazione.

---

## 🔬 La Tecnica NLA (Anthropic, Maggio 2026)

Anthropic ha proposto di superare i limiti dei tradizionali Sparse Autoencoders (SAE) addestrando un autoencoder che utilizza il **linguaggio naturale** come spazio latente:

```
                          ┌─────────────────────────────┐
                          │   Target Model (Granite)    │
                          │   Estrarre Attivazione h_l  │
                          └──────────────┬──────────────┘
                                         │ Vettore h_l
                                         ▼
                          ┌─────────────────────────────┐
                          │ Activation Verbalizer (AV)  │
                          │   Inietta h_l come token    │
                          │     e genera spiegazione    │
                          └──────────────┬──────────────┘
                                         │ Testo <explanation>...</explanation>
                                         ▼
                          ┌─────────────────────────────┐
                          │ Activation Reconstructor    │
                          │       (AR / Critic)         │
                          │   Ricostruisce vettore ĥ_l  │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                             MSE Loss ||h_l - ĥ_l||²
                             (Reward per GRPO / RL)
```

### Dettagli Tecnici Chiave:
- **Injection del Vettore:** L'attivazione $h_l$ viene riscalata alla norma canonica (es. 150.0) e iniettata al posto dell'embedding di un carattere speciale (es. `㈎`) in un prompt prestabilito.
- **Disponibilità Pesi:** Anthropic e `kitft/natural_language_autoencoders` hanno reso pubblici pesi addestrati per Gemma 3, Qwen 2.5 e Llama 3. **Non esistono checkpoint NLA pre-addestrati rilasciati da IBM o Anthropic per Granite**. Addestrare da zero un NLA per Granite richiede un cluster con GPU server (H100) per eseguire milioni di rollout in RL distribuito (GRPO).
- **Soluzione Implementata:** 
  1. Il framework include il client completo compatibile con le specifiche Anthropic/kitft (`core/anthropic_nla.py`).
  2. Fornisce contemporaneamente una modalità di **Verbalizzazione Semantica Immediata (Logit Lens)** per Granite, che proietta istantaneamente le attivazioni nello spazio del vocabolario tramite $W_U$ e l'unembedding head, traducendole in linguaggio naturale senza necessità di pesi proprietari esterni.

---

## 🏗️ Architettura del Framework

```
model-interpretability-framework/
├── README.md                      # Documentazione completa e scientifica
├── requirements.txt              # Dipendenze Python
├── config/
│   └── default_config.yaml       # Configurazione parametri di default
├── core/
│   ├── __init__.py               # Esportazione API
│   ├── model_loader.py          # Caricatore Granite con gestione MPS/Device/Dtype
│   ├── activation_hooks.py      # Hook PyTorch per residual stream e metriche L2/Coseno
│   ├── verbalizer.py            # Logit Lens e sintesi semantica in linguaggio naturale
│   └── anthropic_nla.py         # Specifica NLA Anthropic (Client, Scaling, Critic MSE)
├── app/
│   └── web_ui.py                # Interfaccia interattiva Streamlit
├── scripts/
│   ├── download_model.py        # Script di verifica/download modello da Hugging Face
│   └── run_cli.py               # Esecuzione rapida da terminale
└── tests/
    └── test_components.py       # Test di unità automatizzati
```

---

## 🚀 Installazione e Guida Rapida per Docenti e Ricercatori

Per provare il sistema su qualsiasi macchina (Mac Apple Silicon, Linux con GPU NVIDIA o Windows):

### 1. Clonazione del Repository
```bash
git clone https://github.com/gennarodambrosio-hub/model-interpretability-framework.git
cd model-interpretability-framework
```

### 2. Creazione dell'Ambiente Virtuale (Python 3.10, 3.11 o 3.12)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(Su sistemi Windows: `.venv\Scripts\activate`)*

### 3. Installazione delle Dipendenze
```bash
pip install -r requirements.txt
```

### 4. Avvio dell'Interfaccia Grafica Web (Streamlit)
```bash
streamlit run app/web_ui.py
```
> **Nota:** Al primissimo avvio, Hugging Face scaricherà automaticamente i pesi del modello Granite 4.2 3B nella cache locale (`~/.cache/huggingface/`). Tutti i successivi avvii saranno istantanei. L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`.

---

## 🔬 Esperimento Guidato: Come "Leggere" il Pensiero Interno del Modello

Per toccare con mano la decodifica delle attivazioni interne lungo il **Residual Stream** ($h_0 \to \dots \to h_{39}$):

1. **Modello attivo:** Lasciare selezionato **Granite 4.2 3B** (non quantizzato, precisione bfloat16 preservata al 100%).
2. **Prompt di Test:** Selezionare dal menu a tendina l'esempio di Dante:
   > *"Nel mezzo del cammin di nostra vita, mi ritrovai per una selva"*
   e cliccare su **`🚀 Invia Input & Ispeziona Attivazioni`**.
3. **Mappa dei Token:** Nello slider al punto 3, selezionare il token finale del prompt: **`'va'`** (fine di *selva*).
4. **Viaggio tra i Layer (Punto 4):**
   * **Ai Layer Iniziali (Layer 04 - 12):** La proiezione Logit Lens mostra la comprensione letterale della parola: il modello attiva concetti come **`jungle`** (88.8%) e **`forest`** (63.0%).
   * **Ai Layer Intermedi (Layer 20 - 28):** Si registra un picco di entropia (~10 bit); il modello apre la rete associativa e attiva attributi cromatici e ambientali (**`verde`**, **`libre`**, **`env`**).
   * **Ai Layer Alti di Pianificazione (Layer 36):** L'entropia crolla e, **ancora prima di emettere le parole successive in output**, il modello attiva concetti semantici astratti legati allo smarrimento e all'oscurità:
     - **`unfamiliar` (47.5%)** *(sconosciuta / perduta)*
     - **`mysterious` (22.4%)** *(misteriosa)*
     - **`unknown` (10.6%)** *(ignota)*
   * **Al Layer Finale (Layer 39):** L'energia (Norma L2) sale a 99.00 e la testa di proiezione collassa sulla preposizione di continuità sintattica italiana **`"di"`** (54.2%), portando alla generazione del verso successivo.

---

## ⌨️ Utilizzo da Riga di Comando (CLI)

Per test rapidi o ambienti headless senza interfaccia grafica:

```bash
python scripts/run_cli.py \
    --model ibm-granite/granite-4.2-3b \
    --prompt "Nel mezzo del cammin di nostra vita, mi ritrovai per una selva" \
    --layer 16 \
    --max_tokens 32
```

---

## 🧪 Test di Unità

Esegui la suite di test per verificare il corretto funzionamento degli hook e dei componenti NLA:
```bash
python -m unittest discover -s tests
```

---

## 🔗 Repository GitHub
Repository remoto sincronizzato:
**[gennarodambrosio-hub/model-interpretability-framework](https://github.com/gennarodambrosio-hub/model-interpretability-framework.git)**
