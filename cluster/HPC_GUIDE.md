# Guida al Training NLA su Cluster HPC UNISA (4x NVIDIA A100-80GB)

Questa guida illustra la procedura passo-passo per addestrare l'Actor e il Critic di **Natural Language Autoencoders (NLA)** per **IBM Granite** sul cluster HPC UNISA, sfruttando un nodo con **4x A100 80GB (320 GB VRAM)** sulla partizione `gpuq`.

---

## 1. Connessione al Cluster (Login Node)
Accedi al cluster dal tuo terminale:
```bash
ssh g.dambrosio65@lnode02.hpc.unisa.it
```

---

## 2. Clonazione del Progetto
Sul cluster, posizionati nella tua home e clona il repository:
```bash
cd ~
git clone https://github.com/gennarodambrosio-hub/model-interpretability-framework.git
cd model-interpretability-framework
```

---

## 3. Setup Ambiente Virtuale (Python 3 + CUDA 12 + PyTorch)
Esegui lo script di setup che crea la cartella log, cache e il virtualenv dedicato:
```bash
bash cluster/setup_env.sh
```
*Lo script configurerà l'ambiente in `~/venvs/nla-training-cu12` con supporto nativo CUDA e bfloat16.*

---

## 4. Lancio del Job Slurm con 4x A100 (Partizione `gpuq`)
Il file [`cluster/run_nla_4xa100.sbatch`](run_nla_4xa100.sbatch) è già preconfigurato con:
* Partizione: **`gpuq`**
* Account: **`-A did_tesi_nlp_330`**
* Risorse: **4 GPU A100-80GB**, 32 CPU core
* Limite di tempo: **`06:00:00`** (rispetta il limite massimo di 7 ore della partizione `gpuq`)

Lancia il job con:
```bash
sbatch cluster/run_nla_4xa100.sbatch
```

---

## 5. Monitoraggio del Job
1. **Verifica stato nella coda:**
   ```bash
   squeue -u $USER
   ```
2. **Visualizza i log in tempo reale:**
   ```bash
   tail -f ~/nla_logs/nla-*.out
   ```
   *(Oppure per gli errori: `tail -f ~/nla_logs/nla-*.err`)*

3. **Verifica utilizzo GPU:**
   Una volta assegnato il nodo (es. `gnode04`):
   ```bash
   srun --jobid=<JOB_ID> nvidia-smi
   ```

---

## 6. Risultati e Checkpoint
Al termine del job, i pesi addestrati saranno salvati in:
`~/nla_checkpoints/checkpoint_epoch_3/`

Per scaricare i pesi sul tuo Mac locale per l'interfaccia Streamlit:
```bash
# Esegui dal tuo Mac:
scp -r g.dambrosio65@lnode02.hpc.unisa.it:~/nla_checkpoints/checkpoint_epoch_3 /Users/gennaro/Desktop/model-interpretability-framework/checkpoints/
```
