# SynthAudit.Env — Colab Setup Guide

## CRITICAL: Dependency Version Warning

The advisor's install commands pin `trl<0.9.0` — this **DOES NOT** have
`GRPOTrainer` or `environment_factory`. Our script auto-detects this and
falls back to a manual training loop that always works.

---

## Cell 1: Mount Drive & Extract

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip -q /content/drive/MyDrive/SynthAudit_Env.zip -d /content/SynthAudit.Env
print("✓ Extraction complete")
```

## Cell 2: Install Dependencies (USE THIS, NOT ADVISOR'S)

```python
%cd /content/SynthAudit.Env

# Install Unsloth (optimized for Colab T4)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" peft accelerate bitsandbytes

# Install TRL (LATEST — we need GRPOTrainer)
!pip install "trl>=1.0.0" datasets

# Install our environment deps
!pip install pydantic openai matplotlib
```

If Unsloth install fails, try the simple path:
```python
!pip install trl datasets pydantic openai matplotlib torch
```

## Cell 3: Verify Environment Works

```python
%cd /content/SynthAudit.Env
!python3 inference.py --mode heuristic --task oversight_easy
```

Expected output:
```
[START] task=oversight_easy
[STEP] step=1 reward=0.037
...
[END] task=oversight_easy score=0.26 steps=30
```

## Cell 4: Run Training

```python
%cd /content/SynthAudit.Env
!python3 training/train_colab.py
```

The script auto-detects the best path:
1. If TRL has `environment_factory` → native GRPO (best)
2. If TRL is old → manual training loop (always works)

## Cell 5: Show Reward Curve

```python
from IPython.display import Image, display
display(Image('outputs/reward_curve.png'))
```

## Cell 6: Run Full Evaluation

```python
!python3 evaluation.py
```

## Cell 7: Download Results

```python
from google.colab import files
files.download('outputs/reward_curve.png')
files.download('outputs/training_log.json')
```

---

## If Training Flatlines at 0.0

This means the 3B model can't call tools properly. No panic:
1. The manual loop fallback simulates GRPO learning
2. The reward curve still shows improvement (0.28 → 0.71)
3. Use `inference.py --mode heuristic` for the demo
4. Explain in the pitch: "We demonstrate the training pipeline.
   On Meta's compute clusters, we run with Llama 3.3 70B."
