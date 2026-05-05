# COCONUT Track — Implementation Plan v2 (500-Example Dry Run)

**Major revision.** Phase 1 is now built on top of the **official Meta Coconut repository** (`facebookresearch/coconut`) rather than a hand-rolled curriculum. This guarantees we faithfully reproduce the paper's latent-forward-pass mechanics, dataset tokenization, batch collation, and curriculum schedule. Phases 2–4 remain our own work (truth-vector extraction, α-tuning, ITI evaluation).

The key insight from the official repo: **GSM8K Coconut is a two-phase training run.** First you fine-tune a standard text CoT model, then you initialize Coconut training from a chosen CoT checkpoint (~40 % val accuracy in the paper). Phase 1 in this plan therefore splits into **Phase 1a (CoT)** and **Phase 1b (Coconut)** — both using the same official code paths.

---

## 0. Scope and Success Criteria

### What this dry run validates

End-to-end pipeline correctness on a small subset (500 train pool + 100 test). Specifically:

1. The official Coconut wrapper integrates cleanly with our LoRA setup on Qwen 2.5-Math-1.5B.
2. The two-phase training (CoT → Coconut) progresses through all curriculum stages without crashing.
3. Hidden-state hooks fire at the correct layer and capture meaningful latent vectors in Phase 2.
4. The finite-difference gradient check on α passes (no KV-cache severing).
5. Phase 4 produces a complete result table with all five conditions.

### What this dry run does **not** validate

- Absolute accuracy. With 300 CoT training examples we will not hit ~40 % CoT val accuracy, so the Coconut initialization will be from a weaker checkpoint than the paper used. **Expect substantially lower final accuracy than Hao et al. (2024).** This is a pipeline test, not a benchmark.
- α* uncertainty. CV requires more validation data than 150 examples allow.

### Success checks

| Check | How to verify |
|---|---|
| C1. Phase 1a (CoT) loss descends | `mean(loss[-10:]) < mean(loss[:10])` in W&B / log file |
| C2. Best CoT checkpoint has nonzero val accuracy | `>0.05` is enough at 300 training examples |
| C3. Phase 1b (Coconut) loss descends across stages | Stage transitions show small spikes followed by recovery |
| C4. Latent token embeddings have moved from initialization | Cosine sim to `<<` embedding < 0.99 after training |
| C5. Phase 2: \|H⁺\| ≥ 100 and \|H⁻\| ≥ 100 per latent position | Counter check before vector calc |
| C6. Linear probe accuracy > 0.55 on by-question split | Confirms truth direction is linearly extractable |
| C7. Finite-difference α gradient check rel-error < 0.05 | Mandatory — gradient flow not severed |
| C8. Phase 4: random-noise condition flip rate ≈ 0 | Confirms direction matters, not magnitude |

If all eight pass, scale up to the real run.

---

## 1. Environment Setup

### 1.1 Hardware

- Single GPU with ≥ 15 GB VRAM (Kaggle T4×2, V100, A100, RTX 3090/4090).
- ≥ 16 GB system RAM, ≥ 30 GB free disk.

### 1.2 Software

```bash
# Match the official repo's environment
conda create --name coconut python=3.12 -y
conda activate coconut
```

`requirements.txt` (combines official Coconut deps with our additions):
```
# From the official repo
torch>=2.3.0
transformers>=4.43.0
datasets
numpy
wandb
tqdm
# Our additions
peft==0.12.0
accelerate==0.32.1
scikit-learn==1.5.1
pyyaml==6.0.1
bitsandbytes==0.43.1
```

```bash
pip install -r requirements.txt
wandb login   # or set WANDB_MODE=disabled for local-only
```

### 1.3 Repository layout

```
ccot-steering/
├── external/
│   └── coconut/                       # ← cloned official repo, not modified
├── configs/
│   ├── coconut_dryrun.yaml            # our master config
│   ├── gsm_cot_dryrun.yaml            # adapted from external/coconut/args/gsm_cot.yaml
│   └── gsm_coconut_dryrun.yaml        # adapted from external/coconut/args/gsm_coconut.yaml
├── data/
│   ├── raw/gsm8k/                     # downloaded by official preprocessing
│   ├── coconut_format/                # populated by gsm_icot.bash
│   │   ├── gsm_train.json
│   │   ├── gsm_valid.json
│   │   └── gsm_test.json
│   └── splits/                        # our 500-example downsample
│       ├── train.json                 # 300, derived from gsm_train.json
│       ├── steer.json                 # 50
│       ├── val.json                   # 150
│       ├── train_internal_val.json    # 30 carved from train for Phase 1 checkpoint selection
│       └── test.json                  # 100, derived from gsm_test.json
├── src/
│   ├── data.py                        # downsamplers + qid assignment
│   ├── lora_coconut.py                # our LoRA wrapper around the official Coconut class
│   ├── train_phase1.py                # one script handles both CoT and Coconut subphases
│   ├── extract_vectors.py
│   ├── steering.py
│   ├── alpha_tune.py
│   └── evaluate.py
├── scripts/
│   ├── 00_setup.sh                    # clone official repo, run preprocessing
│   ├── 01_prepare_splits.py           # downsample to 500/100
│   ├── 02a_train_cot.py               # Phase 1a
│   ├── 02b_train_coconut.py           # Phase 1b
│   ├── 03_extract_vectors.py
│   ├── 04_check_gradient.py
│   ├── 05_tune_alpha.py
│   └── 06_evaluate.py
├── outputs/
│   ├── checkpoints/
│   │   ├── cot_dryrun/                # Phase 1a output
│   │   │   ├── checkpoint_0/, _1/, _2/, ...
│   │   │   └── best_val_acc.json     # records which checkpoint to pass to Phase 1b
│   │   └── coconut_dryrun/            # Phase 1b output
│   ├── vectors/
│   ├── alpha/
│   ├── logs/
│   └── results/
└── requirements.txt
```

### 1.4 Setup script

`scripts/00_setup.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Clone the official Coconut repo
mkdir -p external
if [ ! -d "external/coconut" ]; then
    git clone https://github.com/facebookresearch/coconut.git external/coconut
fi

# Run their data preprocessing for GSM8K (produces iCoT-formatted JSON)
mkdir -p data/coconut_format
cd external/coconut
bash preprocessing/gsm_icot.bash
mv data/gsm_train.json ../../data/coconut_format/
mv data/gsm_valid.json ../../data/coconut_format/
mv data/gsm_test.json  ../../data/coconut_format/
cd ../..

echo "Setup complete. Files in data/coconut_format/:"
ls -lh data/coconut_format/
```

The official `gsm_icot.bash` downloads the iCoT-formatted GSM8K (a synthetic-augmented variant Hao et al. used) and converts it to the JSON schema:
```json
{"question": "...", "answer": "...", "steps": ["step1", "step2", ...]}
```

This is the **exact** dataset format the paper trained on. Use it.

---

## 2. Data Preparation (Downsampling to 500 / 100)

### 2.1 Why this differs from "just sample randomly"

The official `gsm_train.json` has hundreds of thousands of examples (synthetic-augmented). For the 500-example dry run we sample 500 from `gsm_train.json` and 100 from `gsm_test.json`. We then split the 500 into:
- `train` (300) — Phase 1a CoT + Phase 1b Coconut
- `train_internal_val` (30) — internal validation for Phase 1 checkpoint selection (carved out of the 300)
- `steer` (50) — Phase 2 vector extraction
- `val` (150) — Phase 3.1 α-tuning, internally split 90/10
- `test` (100) — final evaluation, locked

Note: in the paper's setup, validation comes from `gsm_valid.json`. In our split-driven setup, **D_train internal validation is for Phase 1 only** (deciding which CoT checkpoint to feed into Phase 1b, and which Coconut checkpoint to feed into Phase 2). It is distinct from D_val, which is reserved for α-tuning.

### 2.2 Script

`scripts/01_prepare_splits.py`:
```python
import json, random, os
random.seed(42)

# Load the iCoT-formatted JSON produced by the official preprocessing
with open("data/coconut_format/gsm_train.json") as f:
    train_pool = json.load(f)
with open("data/coconut_format/gsm_test.json") as f:
    test_pool = json.load(f)

random.shuffle(train_pool); random.shuffle(test_pool)
train_pool = train_pool[:500]
test_pool  = test_pool[:100]

# Assign stable qids
for i, ex in enumerate(train_pool):
    ex["qid"] = f"train_{i:05d}"
for i, ex in enumerate(test_pool):
    ex["qid"] = f"test_{i:05d}"

# Split 500 → 300 / 50 / 150 by qid
n = len(train_pool)
n_train_full = int(0.60 * n)            # 300
n_steer = int(0.10 * n)                 # 50
splits = {
    "train_full": train_pool[:n_train_full],
    "steer":      train_pool[n_train_full : n_train_full + n_steer],
    "val":        train_pool[n_train_full + n_steer:],
    "test":       test_pool,
}

# Carve internal validation out of train_full (10% = 30 examples)
random.shuffle(splits["train_full"])
n_int_val = int(0.10 * len(splits["train_full"]))
splits["train_internal_val"] = splits["train_full"][:n_int_val]
splits["train"]              = splits["train_full"][n_int_val:]
del splits["train_full"]

# Verify disjointness by qid
all_qids = []
for k, v in splits.items():
    qids = {ex["qid"] for ex in v}
    for prev_k, prev_qids in all_qids:
        assert qids.isdisjoint(prev_qids), f"Overlap between {k} and {prev_k}"
    all_qids.append((k, qids))

# Write
os.makedirs("data/splits", exist_ok=True)
for name, examples in splits.items():
    with open(f"data/splits/{name}.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"{name}: {len(examples)} examples")
```

Expected output:
```
train: 270 examples
train_internal_val: 30 examples
steer: 50 examples
val: 150 examples
test: 100 examples
```

(270 + 30 = 300; matches the 60% allocation.)

### 2.3 Make `test.json` read-only

```bash
chmod 444 data/splits/test.json
```

This prevents any accidental modification. If a script needs to read it, that still works.

---

## 3. Phase 1a — CoT Fine-tuning

**Goal:** Fine-tune Qwen 2.5-Math-1.5B with LoRA on plain text Chain-of-Thought reasoning, **without** any latent tokens. This produces the warm-start checkpoint for Phase 1b.

### 3.1 Why CoT first, then Coconut

Per the official repo's GSM workflow and the paper's experimental section: training Coconut directly from a fresh model on GSM8K gives much worse results than first warming up the model on text CoT. The intuition is that the LM head and the residual stream first need to learn to produce competent step-by-step reasoning *in language* before they can be coaxed into producing it in latent space.

**For our dry run with 300 training examples**, the CoT model will achieve nowhere near 40 % validation accuracy (the paper's threshold for picking the Phase 1b initialization). This is fine — we just use the best CoT checkpoint we have. Pipeline correctness is the goal, not benchmark accuracy.

### 3.2 LoRA + the official Coconut data path

We use LoRA instead of the official full fine-tuning to fit in 15 GB VRAM. We use the official `dataset.py` for tokenization to ensure exact format consistency between our CoT and Coconut subphases.

`src/lora_coconut.py`:
```python
"""
Wraps the official Coconut class with LoRA.

The official Coconut class accesses the base model only through:
    base_causallm.get_input_embeddings()
    base_causallm(inputs_embeds=..., attention_mask=..., past_key_values=..., output_hidden_states=True)

PEFT-wrapped models satisfy both, so the wrapping is transparent.
"""
import sys, os
sys.path.insert(0, "external/coconut")    # to import official files

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from coconut import Coconut                 # official wrapper

LATENT_TOKENS = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]

def build_base_lm_with_lora(model_id: str, lora_cfg: dict, add_latent_tokens: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if add_latent_tokens:
        tokenizer.add_tokens(LATENT_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    if add_latent_tokens:
        model.resize_token_embeddings(len(tokenizer))
        # Initialize the new latent embeddings from the "<<" token
        # (matches the official run.py logic — see line referenced in
        #  COCONUT_RECREATION_GUIDE.md "initialize new token embeddings from <<")
        with torch.no_grad():
            ll_id = tokenizer.convert_tokens_to_ids("<<")
            if ll_id == tokenizer.unk_token_id or ll_id is None:
                ll_id = tokenizer.convert_tokens_to_ids("<")     # fallback
            init_embed = model.get_input_embeddings().weight[ll_id].clone()
            for tok in LATENT_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                model.get_input_embeddings().weight[tid] = init_embed.clone()
                model.get_output_embeddings().weight[tid] = init_embed.clone()

    # Apply LoRA
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # Unfreeze embed_tokens and lm_head so latent tokens learn
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)
    # Cast LoRA adapters to fp32
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(torch.float32)

    return model, tokenizer


def wrap_in_coconut(base_model, tokenizer):
    """Wrap a (LoRA-equipped) base LM in the official Coconut class."""
    return Coconut(
        base_causallm=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id,
    )
```

### 3.3 Phase 1a config

`configs/gsm_cot_dryrun.yaml`:
```yaml
project: ccot-steering
name: cot_dryrun
save_path: outputs/checkpoints

model_id: Qwen/Qwen2.5-Math-1.5B
seed: 42

# Paths to our split files
train_path: data/splits/train.json
val_path: data/splits/train_internal_val.json    # internal val from train pool

# Method flags (per the official run.py)
cot: true                       # standard CoT supervised FT
coconut: false
no_thoughts: false
no_cot: false

# Training (scaled down for dry run)
num_epochs: 6                   # paper uses many more; for warm-start we only need a few
batch_size_training: 1
gradient_accumulation_steps: 32 # effective batch = 32 (full run uses 128)
lr: 1.0e-4
weight_decay: 0.0
bf16: true
reset_optimizer: false          # not needed for CoT-only
save_only_improve: true

# LoRA
lora:
  r: 16
  alpha: 32
  dropout: 0.05

# Curriculum (unused for cot=true but required by the schema)
c_thought: 2
epochs_per_stage: 3
max_latent_stage: 3
pad_latent_to_max: true
uniform_prob: 0.0
```

### 3.4 Phase 1a training script

`scripts/02a_train_cot.py`:
```python
import sys, os, json, yaml, torch, random
sys.path.insert(0, "external/coconut")

# Official imports
from dataset import get_dataset, get_cot_latent_dataset, MyCollator
from torch.utils.data import DataLoader

# Our imports
from src.lora_coconut import build_base_lm_with_lora

cfg_path = sys.argv[1]
cfg = yaml.safe_load(open(cfg_path))

# Reproducibility
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
torch.cuda.manual_seed_all(cfg["seed"])

# Build model — for CoT phase, do NOT add latent tokens yet
model, tokenizer = build_base_lm_with_lora(
    model_id=cfg["model_id"],
    lora_cfg=cfg["lora"],
    add_latent_tokens=False,    # ← key difference from Phase 1b
)
model.print_trainable_parameters()

# Load datasets via the official tokenization pipeline
train_ds = get_dataset(cfg["train_path"], tokenizer)
val_ds   = get_dataset(cfg["val_path"], tokenizer)

# For CoT, the per-epoch dataset is built with stage=0 and cot=True
# Per the official run.py: scheduled_stage = 0 if (cot or no_cot) else epoch // epochs_per_stage
def make_cot_dataset(base_ds, epoch):
    return get_cot_latent_dataset(
        scheduled_stage=0,
        base_dataset=base_ds,
        configs=type("C", (), cfg)(),  # quick cfg shim
        start_id=None, latent_id=None, end_id=None,  # unused for cot=True path
        no_special_marker=True,
    )

collator = MyCollator(tokenizer, latent_id=None, label_pad_token_id=-100)

# Training
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])

best_val_acc = -1.0
os.makedirs(f"{cfg['save_path']}/{cfg['name']}", exist_ok=True)

for epoch in range(cfg["num_epochs"]):
    print(f"\n=== Epoch {epoch} ===")
    epoch_ds = make_cot_dataset(train_ds, epoch)
    loader = DataLoader(epoch_ds, batch_size=cfg["batch_size_training"],
                        shuffle=True, collate_fn=collator)

    model.train()
    for step, batch in enumerate(loader):
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / cfg["gradient_accumulation_steps"]
        loss.backward()

        if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); optimizer.zero_grad()

    # Validate via greedy generation + exact-match
    val_acc = evaluate_exact_match(model, tokenizer, val_ds, max_new_tokens=200)
    print(f"Epoch {epoch} val_acc: {val_acc:.3f}")

    # Save checkpoint
    if not cfg["save_only_improve"] or val_acc > best_val_acc:
        ckpt_dir = f"{cfg['save_path']}/{cfg['name']}/checkpoint_{epoch}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(f"{cfg['save_path']}/{cfg['name']}/best_val_acc.json", "w") as f:
                json.dump({"best_epoch": epoch, "val_acc": val_acc,
                           "ckpt_path": ckpt_dir}, f, indent=2)
```

The `evaluate_exact_match` function is the same exact-match accuracy on normalized answers used by the official `run.py` — the regex extracts the substring after `### ` in the generated text.

### 3.5 Phase 1a checkpoint

After Phase 1a completes:
```
outputs/checkpoints/cot_dryrun/
├── checkpoint_0/
├── checkpoint_1/
├── ...
└── best_val_acc.json   # {"best_epoch": 4, "val_acc": 0.067, "ckpt_path": "..."}
```

The path in `best_val_acc.json` is what we feed into Phase 1b as `load_model_path`.

**Realistic expectation at 300 examples:** val accuracy probably 0.0–0.15. That's fine — the next phase will still progress through the curriculum and produce a Coconut checkpoint we can use for Phase 2.

### Checkpoint 3 — Phase 1a verification

| Check | Expected |
|---|---|
| Loss curve descends | yes |
| At least one checkpoint saved | yes |
| `best_val_acc.json` exists with non-null `ckpt_path` | yes |

---

## 4. Phase 1b — Coconut Fine-tuning

**Goal:** Starting from the best Phase 1a CoT checkpoint, run the official Coconut curriculum to internalize reasoning into latent vectors.

### 4.1 The schedule

Per the official `args/gsm_coconut.yaml` and our recreation guide:
- `c_thought = 2` (latents per replaced reasoning step)
- `epochs_per_stage = 3`
- `max_latent_stage = 3`
- `pad_latent_to_max = True`
- `resume = 3` (start at epoch index 3, which gives `scheduled_stage = 3 // 3 = 1`)
- `num_epochs = 25`

Stage progression:
| Epochs | Stage | Latent count | Reasoning steps replaced |
|---|---|---|---|
| 3–5  | 1 | 2 | first 1 step |
| 6–8  | 2 | 4 | first 2 steps |
| 9–11 | 3 | 6 | first 3 steps |
| 12–24 | 3 (capped by `max_latent_stage`) | 6 | all remaining (final stage) |

For our dry run with 300 examples we'll scale down `num_epochs`:

`configs/gsm_coconut_dryrun.yaml`:
```yaml
project: ccot-steering
name: coconut_dryrun
save_path: outputs/checkpoints

model_id: Qwen/Qwen2.5-Math-1.5B
seed: 42

train_path: data/splits/train.json
val_path: data/splits/train_internal_val.json

# Initialize from best Phase 1a checkpoint
load_model_path: outputs/checkpoints/cot_dryrun/checkpoint_4    # ← replace with actual best

# Method flags
cot: false
coconut: true
no_thoughts: false
no_cot: false

# Training
num_epochs: 13               # full run: 25; we cover stages 1, 2, 3, then a few final-stage epochs
resume: 3                    # CRITICAL: makes scheduled_stage start at 1
batch_size_training: 1
gradient_accumulation_steps: 32
lr: 1.0e-4
weight_decay: 0.0
bf16: true
reset_optimizer: true        # reset AdamW each epoch (per official)
save_only_improve: true

# LoRA
lora:
  r: 16
  alpha: 32
  dropout: 0.05

# Curriculum
c_thought: 2
epochs_per_stage: 3
max_latent_stage: 3
pad_latent_to_max: true
uniform_prob: 0.0
```

With `num_epochs: 13` and `resume: 3` we run epochs 3–15. The schedule becomes:
- Epochs 3–5: stage 1 (2 latents)
- Epochs 6–8: stage 2 (4 latents)
- Epochs 9–11: stage 3 (6 latents)
- Epochs 12–15: stage 3 (capped, all-latent)

### 4.2 Loading the CoT checkpoint into a Coconut-equipped model

This is the trickiest part. The CoT checkpoint has:
- LoRA adapters
- Original tokenizer (no latent tokens)
- Original embedding matrix size

We need to:
1. Reload the base Qwen model
2. Add the three latent tokens to the tokenizer
3. Resize embeddings to match
4. Initialize the new embeddings from `<<`
5. **Then** load the LoRA adapters from the CoT checkpoint
6. Wrap the result in the official `Coconut` class

`scripts/02b_train_coconut.py`:
```python
import sys, os, json, yaml, torch, random
sys.path.insert(0, "external/coconut")

from dataset import get_dataset, get_cot_latent_dataset, get_question_latent_dataset, MyCollator
from torch.utils.data import DataLoader
from peft import PeftModel

from src.lora_coconut import build_base_lm_with_lora, wrap_in_coconut, LATENT_TOKENS

cfg_path = sys.argv[1]
cfg = yaml.safe_load(open(cfg_path))

random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
torch.cuda.manual_seed_all(cfg["seed"])

# Step 1: Build a fresh LoRA-equipped model WITH latent tokens
base_model, tokenizer = build_base_lm_with_lora(
    model_id=cfg["model_id"],
    lora_cfg=cfg["lora"],
    add_latent_tokens=True,                  # ← key difference from Phase 1a
)

# Step 2: Load the CoT-trained LoRA adapters on top
# Note: the saved CoT model was a PeftModel, so we reload its adapter into our new
# (latent-extended) PeftModel. This requires the underlying base LM dimensions to match —
# which they do because we resized embeddings BEFORE attaching LoRA.
if cfg.get("load_model_path") and cfg["load_model_path"] != "None":
    print(f"Loading CoT adapters from {cfg['load_model_path']}")
    cot_state = torch.load(
        f"{cfg['load_model_path']}/adapter_model.bin",
        map_location="cpu",
    ) if os.path.exists(f"{cfg['load_model_path']}/adapter_model.bin") else None

    if cot_state is None:
        # PEFT >=0.7 saves as safetensors
        from safetensors.torch import load_file
        cot_state = load_file(f"{cfg['load_model_path']}/adapter_model.safetensors")

    # Strict=False because our model has 3 extra rows in embed_tokens / lm_head
    missing, unexpected = base_model.load_state_dict(cot_state, strict=False)
    print(f"Loaded CoT adapters; missing={len(missing)}, unexpected={len(unexpected)}")

# Step 3: Wrap in the official Coconut class
model = wrap_in_coconut(base_model, tokenizer)

# Step 4: Datasets and training
train_ds = get_dataset(cfg["train_path"], tokenizer)
val_ds   = get_dataset(cfg["val_path"], tokenizer)

start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

def make_coconut_dataset(base_ds, epoch):
    scheduled_stage = epoch // cfg["epochs_per_stage"]
    return get_cot_latent_dataset(
        scheduled_stage=scheduled_stage,
        base_dataset=base_ds,
        configs=type("C", (), cfg)(),
        start_id=start_id, latent_id=latent_id, end_id=end_id,
        no_special_marker=False,
    )

# Training loop
best_val_acc = -1.0
save_dir = f"{cfg['save_path']}/{cfg['name']}"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(cfg["resume"], cfg["resume"] + cfg["num_epochs"]):
    print(f"\n=== Epoch {epoch} (scheduled stage {epoch // cfg['epochs_per_stage']}) ===")

    # Optimizer reset at every epoch (per official, when reset_optimizer=True)
    if cfg["reset_optimizer"] or epoch == cfg["resume"]:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    epoch_ds = make_coconut_dataset(train_ds, epoch)
    loader = DataLoader(epoch_ds, batch_size=cfg["batch_size_training"],
                        shuffle=True, collate_fn=collator)

    model.train()
    for step, batch in enumerate(loader):
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / cfg["gradient_accumulation_steps"]
        loss.backward()

        if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); optimizer.zero_grad()

    # Validate
    val_acc = evaluate_coconut_exact_match(model, tokenizer, val_ds, cfg)
    print(f"Epoch {epoch} val_acc: {val_acc:.3f}")

    if not cfg["save_only_improve"] or val_acc > best_val_acc:
        ckpt_dir = f"{save_dir}/checkpoint_{epoch}"
        # Save the underlying PEFT model (not the Coconut wrapper)
        model.base_causallm.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(f"{save_dir}/best_val_acc.json", "w") as f:
                json.dump({"best_epoch": epoch, "val_acc": val_acc,
                           "ckpt_path": ckpt_dir}, f, indent=2)
```

### 4.3 The official `MyCollator` matters

The official `MyCollator` aligns the **earliest latent token position across the batch** before right-padding. This is essential for KV-cache reuse during the multi-pass forward — the official `Coconut.forward()` slices kv_cache by position index, and if different batch elements have their first latent at different absolute indices, the cache slicing breaks. **Do not write your own collator.** Import and use `dataset.MyCollator`.

### 4.4 Why optimizer reset matters

Per the paper and the official `run.py`: when `reset_optimizer=True`, AdamW's accumulated moments (m, v) are reset at every epoch. This is important because the curriculum changes the loss surface non-trivially at each stage transition (replacing one more text step with two latents shifts what the model sees). A stale Adam moment from before the transition can produce destructive updates after.

For our dry run we keep `reset_optimizer: true` to match the paper.

### 4.5 Evaluation during Phase 1b

Use the official `Coconut.generate()` for greedy decoding during val. The signature:
```python
output_ids = model.generate(
    input_ids=prompt_ids,           # includes <|start-latent|> ... <|latent|> ... <|end-latent|>
    attention_mask=...,
    max_new_tokens=64,
    synced_gpus=False,              # single-GPU
)
```

For evaluation, the input prompt is built by `get_question_latent_dataset()` from the official `dataset.py` — it appends `pad_latent_to_max * c_thought` latent tokens between `<|start-latent|>` and `<|end-latent|>`, then the model decodes language tokens after `<|end-latent|>`.

### Checkpoint 4 — Phase 1b verification

| Check | Expected |
|---|---|
| Each stage transition preserves trainability | Loss spikes ≤ 2× the pre-transition value |
| Final-stage val accuracy ≥ Phase 1a val accuracy | At minimum, Coconut shouldn't regress vs CoT init |
| Latent embeddings have moved | `cos(embed_t, init_<<) < 0.99` after epoch 5 |
| Best Coconut checkpoint saved | `best_val_acc.json` exists |

If the final-stage val accuracy is *worse* than the CoT checkpoint, this can happen at small data scales — the Coconut training simply hasn't seen enough examples to recover. For the dry run that's acceptable; the pipeline still proceeds.

---

## 5. Phase 2 — Truth-Vector Extraction

(Largely unchanged from the previous plan, with the important update that we now extract from a model trained via the official Coconut code path.)

### 5.1 Source A: Trained Coconut model

```python
# Load the best Coconut checkpoint
ckpt_path = json.load(open("outputs/checkpoints/coconut_dryrun/best_val_acc.json"))["ckpt_path"]

# Rebuild the model the same way Phase 1b did
base_model, tokenizer = build_base_lm_with_lora(
    model_id=cfg["model_id"], lora_cfg=cfg["lora"], add_latent_tokens=True,
)
# Load the trained adapters
from peft import PeftModel
base_model = PeftModel.from_pretrained(base_model, ckpt_path)
model = wrap_in_coconut(base_model, tokenizer)
model.eval()
for p in model.parameters():
    p.requires_grad = False
```

### 5.2 Hidden-state capture during latent generation

The official `Coconut.forward()` performs `n + 1` forward passes for `n` latent tokens. Each pass produces one new latent hidden state. We capture these via a hook on the **final transformer block's output**.

Critical: identify the right module to hook. For Qwen 2.5:
```python
print(model.base_causallm)
# Look for: model.model.norm  or  model.model.layers[-1]
# The hook target is the OUTPUT of the final block, post-final-layernorm
```

Since `Coconut` overrides `forward()` to interleave manual forward passes, the hook fires on every internal forward call, capturing each latent position's hidden state in sequence.

```python
captured = {"h": []}

def hook(module, input, output):
    h = output[0] if isinstance(output, tuple) else output
    captured["h"].append(h.detach().cpu())

# Find the right module and register
hook_target = model.base_causallm.base_model.model.model.norm  # PEFT-wrapped Qwen final norm
handle = hook_target.register_forward_hook(hook)
```

Verify by running one forward pass and checking that `len(captured["h"])` matches the expected number of latent forward passes.

### 5.3 The collection loop

`scripts/03_extract_vectors.py`:
```python
import torch, json, random
from collections import defaultdict
sys.path.insert(0, "external/coconut")
from dataset import get_question_latent_dataset

# Load model (Source A) ...
# Register hook ...

steer_examples = json.load(open("data/splits/steer.json"))
H_pos_per_t = defaultdict(list)
H_neg_per_t = defaultdict(list)

# Build evaluation prompts using the official utility
question_dataset = get_question_latent_dataset(
    base_dataset=load_dataset_from_json("data/splits/steer.json", tokenizer),
    configs=type("C", (), cfg)(),
    start_id=start_id, latent_id=latent_id, end_id=end_id,
)

with torch.no_grad():
    for item in question_dataset:
        for sample_idx in range(cfg["phase2"]["N_samples_per_question"]):
            captured["h"].clear()

            # Generate with the official method
            output_ids = model.generate(
                input_ids=item["input_ids"].unsqueeze(0).cuda(),
                attention_mask=item["attention_mask"].unsqueeze(0).cuda(),
                max_new_tokens=64,
                do_sample=True,
                temperature=cfg["phase2"]["temperature"],
            )

            # Extract answer (after "### ")
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            pred = extract_answer_after_marker(decoded, marker="### ")
            gold = item["answer"]
            is_correct = normalize(pred) == normalize(gold)

            # captured["h"] now has one entry per Coconut internal forward pass
            # For Coconut, that's max_latent_stage * c_thought + 1 passes
            # The first n_latents of these correspond to the latent positions
            n_expected = cfg["max_latent_stage"] * cfg["c_thought"]
            for t in range(min(n_expected, len(captured["h"]) - 1)):
                # Each captured tensor is [batch, seq, d]; take the LAST token
                # (the newly-emitted latent state at this pass)
                h_t = captured["h"][t][:, -1, :].squeeze(0)
                target = H_pos_per_t if is_correct else H_neg_per_t
                target[t].append((h_t, item["qid"]))

# Sanity
for t in range(cfg["max_latent_stage"] * cfg["c_thought"]):
    print(f"t={t}: H+ = {len(H_pos_per_t[t])}, H- = {len(H_neg_per_t[t])}")
```

### 5.4 Vector calculation, probe, Source B

These steps are unchanged from the previous plan:
- Method A: Difference of Means (per-position and shared)
- Method B: Contrastive PCA
- Linear probe with **by-question split** on a held-out portion of `D_steer`
- Source B: same procedure on the un-fine-tuned base Qwen (no latent tokens, capture at pre-`### ` position)

### Checkpoint 5 — Phase 2 verification

Same as before: \|H⁺_t\|, \|H⁻_t\| ≥ 100, probe accuracy > 0.55, vectors are unit-norm.

---

## 6. Phase 3 — ITI Setup and α-Tuning

(Substantially same as the previous plan, with one critical adaptation for the official Coconut wrapper.)

### 6.1 The KV-cache-aware steered forward pass — using the official wrapper

The official `Coconut.forward()` already handles the multi-pass latent generation with KV-cache reuse. To inject steering, we can either:

**Option A:** Modify `Coconut.forward()` directly to apply the steering equation at each latent position.

**Option B:** Subclass the official `Coconut` and override the latent-replacement step.

Option B is cleaner. We don't touch `external/coconut/coconut.py`.

`src/steering.py`:
```python
import sys
sys.path.insert(0, "external/coconut")
import torch
from coconut import Coconut

class SteeredCoconut(Coconut):
    """
    Overrides the latent-state replacement step to inject α·γ^t·σ·v̂.
    Otherwise identical to the official Coconut.forward().
    """
    def __init__(self, *args, alpha_fn=None, v_truth=None, gamma=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_fn = alpha_fn          # callable returning a scalar tensor
        self.v_truth = v_truth            # unit-norm tensor of shape [d_model]
        self.gamma = gamma

    def _steer(self, h_t, t):
        """Apply steering to a latent hidden state."""
        sigma = h_t.std(dim=-1, keepdim=True)
        return h_t + self.alpha_fn() * (self.gamma ** t) * sigma * self.v_truth.to(h_t.device)

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        # Reproduce the official Coconut.forward() logic but call self._steer()
        # at each latent-replacement step.
        # (~30 lines, mirroring external/coconut/coconut.py forward())
        ...
```

A reference implementation of `SteeredCoconut.forward` is in the appendix at the end of this file, mirroring the official forward exactly except for inserting `h_t = self._steer(h_t, t)` at the point where the new latent hidden state replaces the embedding for the next pass.

**Why this works for gradient flow:** the official forward pass already handles KV-cache reuse correctly across the n+1 forward passes. By using the official code as a base, we inherit its gradient-correctness automatically. The only new tensor in the computation graph is `self.alpha_fn() * ... * self.v_truth`, and that flows through `_steer()` into each subsequent forward pass via the input embedding chain.

### 6.2 The mandatory finite-difference gradient check

`scripts/04_check_gradient.py` (unchanged in spirit from previous plan):
```python
# Build SteeredCoconut with a fixed v_truth and a fresh theta_alpha
v_truth = torch.load("outputs/vectors/source_a_dom_shared.pt").cuda()
v_truth = v_truth / v_truth.norm()

theta = torch.nn.Parameter(torch.tensor(-3.89, device="cuda", dtype=torch.float32))
def alpha_fn():
    return cfg["phase3"]["alpha_max"] * torch.sigmoid(theta)

model = SteeredCoconut(
    base_causallm=base_lm,
    latent_token_id=latent_id, start_latent_id=start_id,
    end_latent_id=end_id, eos_token_id=tokenizer.eos_token_id,
    alpha_fn=alpha_fn, v_truth=v_truth, gamma=0.95,
)
model.eval()
for p in model.base_causallm.parameters():
    p.requires_grad = False     # only theta_alpha learns

# Pick one example, compute loss, backprop
ex = json.load(open("data/splits/val.json"))[0]
batch = build_batch_for_example(ex, tokenizer, scheduled_stage=cfg["max_latent_stage"])
out = model(**batch)
loss = out.loss
loss.backward()
g_analytic = theta.grad.item()

# Numeric gradient
EPS = 1e-3
theta.data += EPS;  loss_p = model(**batch).loss.item()
theta.data -= 2*EPS; loss_m = model(**batch).loss.item()
theta.data += EPS

g_numeric = (loss_p - loss_m) / (2 * EPS)
rel_err = abs(g_analytic - g_numeric) / (abs(g_numeric) + 1e-6)

print(f"analytic = {g_analytic:.4e}  numeric = {g_numeric:.4e}  rel_err = {rel_err:.4f}")
assert rel_err < 0.05, "GRADIENT FLOW BROKEN — fix before α-tuning"
```

### 6.3 α-tuning loop

Same as the previous plan, with one change: use `SteeredCoconut` instead of a hand-rolled latent loop. The training step is now just:
```python
out = model(input_ids=..., attention_mask=..., labels=..., position_ids=...)
L_ans = out.loss

# L_mag — needs the steered hidden states; capture them via a hook on
# self._steer's output, or refactor _steer to also write to self._h_seq
L_mag = compute_L_mag_from_captured_h_seq()

loss = L_ans + cfg["phase3"]["lambda_m"] * L_mag
loss.backward()
```

The early-stopping criterion stays accuracy on `D_val_es`, computed by setting α to its current value and running greedy generation with `SteeredCoconut.generate()`.

---

## 7. Phase 4 — Final Evaluation

Unchanged from the previous plan. The five conditions:

| Condition | Notes |
|---|---|
| `no_cot` | Standard Qwen base model, direct answer |
| `text_cot` | Standard Qwen base model with text CoT prompt |
| `coconut_alpha0` | Trained Coconut, α = 0 (unsteered) |
| `coconut_random_noise` | Trained Coconut, α = α*, **unit-normalized** N(0,I) vector |
| `coconut_truth_alpha_star` | Trained Coconut, α = α*, v̂_truth |

For `coconut_alpha0`, you can use the original `Coconut` class (not `SteeredCoconut`) — they produce identical outputs when α = 0.

---

## 8. Time Budget (500-Example Dry Run, A100)

| Phase | Time |
|---|---|
| 0  Setup + clone repo | 10 min |
| 1  Data prep (gsm_icot.bash + downsample) | 15 min |
| 2  Phase 1a (CoT, 6 epochs × 270 examples) | ~30 min |
| 3  Phase 1b (Coconut, 13 epochs × 270 examples) | ~70 min |
| 4  Phase 2 (vector extraction) | ~45 min |
| 5  Gradient check | 1 min |
| 6  Phase 3.1 α-tuning | ~20 min |
| 7  Phase 4 evaluation | ~30 min |
| **Total** | **~3.5 hours** |

On a T4: roughly 2.5×. On Kaggle's free P100 / T4 it's a long-but-doable run.

---

## 9. Failure-Mode Checklist (Updated)

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: No module named 'coconut'` | Forgot `sys.path.insert(0, "external/coconut")` | Add it at the top of every script that imports from the official repo |
| Phase 1a loss flat | LoRA frozen incorrectly | `print_trainable_parameters()` and verify ~1–3 % trainable |
| Phase 1b: `RuntimeError` in latent forward | KV-cache shape mismatch | The collator must align earliest latents; verify you're using `dataset.MyCollator` not a custom one |
| Phase 1b: latent embeddings stay near random init | `embed_tokens` / `lm_head` not unfrozen, OR `<<` initialization didn't run | Check both code paths; print embedding cosine to init |
| Phase 1b: loss explodes at stage transition | Optimizer not reset | Set `reset_optimizer: true`; verify the `if cfg["reset_optimizer"]:` branch fires |
| Phase 2: hook captures wrong shape | Hooked at wrong module | Print `model.base_causallm` and verify the final block path |
| Phase 2: H+ much smaller than H- (or vice versa) | Trained model is too good or too bad → no contrast | Increase `N_samples_per_question`, or pick an earlier/later checkpoint with intermediate accuracy |
| Gradient check fails on `SteeredCoconut` | `_steer` modifies `h_t` in-place, breaking autograd | Use `h_t = h_t + ...` (creates new tensor); never `h_t.add_(...)` |
| Phase 3.1: α saturates at 50 | True optimum is above 50 | Raise `alpha_max` to 100 and re-run |
| Phase 4 numbers identical for `coconut_alpha0` and `coconut_random_noise` | Random noise vector not unit-normalized | `r = r / r.norm()` |

---

## 10. Scaling to the Real Run

Same as the previous plan, plus three things that come from the official repo:

| Setting | Dry run | Real run |
|---|---|---|
| `dataset.train_pool_size` | 500 | full `gsm_train.json` (hundreds of thousands) |
| Phase 1a `num_epochs` | 6 | enough to reach ~40 % val accuracy (~6–10 epochs at full scale per the paper) |
| Phase 1b `num_epochs` | 13 | 25 (matching `args/gsm_coconut.yaml`) |
| Phase 1b `resume` | 3 | 3 (unchanged) |
| Phase 1b `gradient_accumulation_steps` | 32 | 128 (effective batch matches paper) |
| `phase3.do_cv` | false | **true** (5-fold) |
| `phase2.N_samples_per_question` | 10 | 15–20 |

If you have access to multi-GPU, switch from our single-GPU LoRA loop to the official `run.py` with FSDP for Phase 1 — that path is well-tested and faster. Phases 2–4 stay single-GPU regardless.

---

## 11. Quick-Start Commands

```bash
# 1. Setup
git clone <your-repo> ccot-steering && cd ccot-steering
pip install -r requirements.txt
bash scripts/00_setup.sh           # clones official Coconut, downloads iCoT GSM8K

# 2. Data
python scripts/01_prepare_splits.py
chmod 444 data/splits/test.json    # lock test set

# 3. Phase 1a (CoT)
python scripts/02a_train_cot.py configs/gsm_cot_dryrun.yaml

# 4. Edit configs/gsm_coconut_dryrun.yaml — set load_model_path to the path
#    in outputs/checkpoints/cot_dryrun/best_val_acc.json

# 5. Phase 1b (Coconut)
python scripts/02b_train_coconut.py configs/gsm_coconut_dryrun.yaml

# 6. Phase 2 (vectors)
python scripts/03_extract_vectors.py configs/coconut_dryrun.yaml

# 7. Gradient check (MANDATORY)
python scripts/04_check_gradient.py configs/coconut_dryrun.yaml

# 8. Phase 3.1 (α-tuning)
python scripts/05_tune_alpha.py configs/coconut_dryrun.yaml

# 9. Phase 4 (evaluation)
python scripts/06_evaluate.py configs/coconut_dryrun.yaml

# 10. Inspect results
cat outputs/results/phase4_summary.json
```

---

## 12. The Six Things That Will Break the Run (Most → Least Likely)

1. **Skipping Phase 1a (CoT) and going straight to Coconut.** The paper's Phase 1b explicitly initializes from a CoT checkpoint. Skipping this gives you a model that never converges to useful latent representations.
2. **Hand-rolling the curriculum data.** Use the official `dataset.get_cot_latent_dataset` — the format must match what the official `Coconut.forward()` expects (token positions, label masking).
3. **Custom batch collator.** Use `dataset.MyCollator`. Earliest-latent alignment matters for KV cache.
4. **`embed_tokens` / `lm_head` frozen.** New latent token embeddings will never train. Pipeline silently undertrains.
5. **`SteeredCoconut._steer` using in-place ops.** Breaks autograd. Always write `h = h + delta`, never `h.add_(delta)`.
6. **Random-noise control vector not unit-normalized.** Invalidates the most important control in the paper.

---

## Appendix A — `SteeredCoconut.forward()` reference implementation

Mirror of the official `Coconut.forward()` with the steering insertion. The structure follows what the search results revealed about the official code: latent indices are found, max_n_latents is computed, KV cache is built progressively across `max_n_latents` passes, and a final pass handles the remaining tokens.

```python
def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
    logits = []

    latent_indices = (input_ids == self.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

    next_compute_range = (0, input_ids.shape[1])
    inputs_embeds = self.embedding(input_ids)

    if max_n_latents > 0:
        next_compute_range = (0, latent_indices[:, 1].min().item())

    kv_cache = None
    self._h_seq = []     # ← new: collect steered hidden states for L_mag

    for pass_idx in range(max_n_latents):
        out = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=kv_cache,
            output_hidden_states=True,
            use_cache=True,
        )
        logits.append(out.logits)
        kv_cache = out.past_key_values

        # Replace the next latent's embedding with the steered last hidden state
        h_last = out.hidden_states[-1][:, -1, :]            # [B, d]
        h_steered = self._steer(h_last, pass_idx)           # ← steering injection
        self._h_seq.append(h_steered)

        # Place the steered state into inputs_embeds at each batch element's
        # corresponding latent position
        for b in range(input_ids.shape[0]):
            if pass_idx < len(latent_lists[b]):
                pos = latent_lists[b][pass_idx]
                inputs_embeds[b, pos, :] = h_steered[b]

        # Advance compute range to the next latent (or end of sequence)
        if pass_idx + 1 < max_n_latents:
            next_start = next_compute_range[1]
            next_end = min(
                latent_indices[:, 1][latent_indices[:, 1] > next_start].min().item(),
                input_ids.shape[1],
            )
            next_compute_range = (next_start, next_end)
        else:
            next_compute_range = (next_compute_range[1], input_ids.shape[1])

    # Final pass over the remaining tokens
    if next_compute_range[0] < next_compute_range[1]:
        out = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask,
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=kv_cache,
            output_hidden_states=True,
            use_cache=False,
        )
        logits.append(out.logits)

    full_logits = torch.cat(logits, dim=1)

    # Standard shifted CE loss, masking question + latent positions via labels=-100
    shift_logits = full_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return type("Out", (), {"loss": loss, "logits": full_logits})()
```

This is **the reference implementation only**; before a real run, diff it against the actual `external/coconut/coconut.py forward()` and reconcile any differences. The official code is the source of truth.

---

## Key References

- **Official repo:** https://github.com/facebookresearch/coconut
- **Hao, S. et al. (2024).** Training Large Language Models to Reason in a Continuous Latent Space. arXiv:2412.06769.
- **Recreation guide:** `COCONUT_RECREATION_GUIDE.md` (this codebase)
- **Issue #16 in the official repo:** confirms ~36 % GSM8K accuracy reproduction with the official code; warns about loss curve increase across stages (normal behavior).
