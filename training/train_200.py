"""
SynthAudit.Env — Extended GRPO Training (200 Steps)
====================================================
Optimized for overnight Colab T4 run (~3-4 hours).
Key improvements over train_real.py:
  - 200 steps (vs 50): 4× more training for clearer convergence
  - Learning rate warmup (10% steps): prevents early instability
  - NUM_GEN=2: saves VRAM, still enough for GRPO advantage computation
  - Auto-save every 50 steps with best-checkpoint tracking
  - Auto-upload to HuggingFace Hub when done

Usage on Colab:
  !pip install unsloth trl datasets huggingface_hub
  %env MAX_STEPS=200
  %env HF_REPO=Timusgeorge/SynthAudit-Qwen2.5-3B-GRPO
  !python3 training/train_200.py
"""

from __future__ import annotations
import json, os, re, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)
sys.path.insert(0, os.path.join(_project_dir, "server"))

from models import SynthAuditAction, ActionType
from server.synth_audit_environment import SynthAuditEnvironment


# ═══════════════════════════════════════════════════════════════
# Reward function
# ═══════════════════════════════════════════════════════════════

def score_completion(text: str, seed: int = 42, task_id: str = "oversight_easy") -> float:
    """Parse model output as JSON tool calls, execute in env, return score."""
    env = SynthAuditEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    actions = []
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            actions = json.loads(match.group())
    except Exception:
        pass

    if not actions:
        for m in re.finditer(r'\{[^{}]+\}', text):
            try:
                actions.append(json.loads(m.group()))
            except Exception:
                continue

    for act in actions:
        if obs.done:
            break
        try:
            action = SynthAuditAction(**act)
            obs = env.step(action)
        except Exception:
            continue

    return obs.score_so_far


def make_reward_func(seeds, task_ids):
    """Create reward function for GRPOTrainer."""
    call_count = [0]

    def reward_func(completions, **kwargs):
        scores = []
        for i, completion_list in enumerate(completions):
            text = completion_list[0]["content"] if isinstance(completion_list, list) else str(completion_list)
            seed = seeds[i % len(seeds)]
            task = task_ids[i % len(task_ids)]
            score = score_completion(text, seed=seed, task_id=task)
            scores.append(float(score))
        call_count[0] += 1
        return scores
    return reward_func


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import torch

    MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-3B-Instruct")
    MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
    NUM_GEN = int(os.getenv("NUM_GEN", "2"))
    HF_REPO = os.getenv("HF_REPO", "")  # e.g., "Timusgeorge/SynthAudit-Qwen2.5-3B-GRPO"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SynthAudit.Env — Extended GRPO Training (200 Steps)       ║")
    print("║  Multi-Agent Clinical AI Oversight                          ║")
    print(f"║  Model:    {MODEL:<47s}║")
    print(f"║  Steps:    {MAX_STEPS:<47d}║")
    print(f"║  Gen/step: {NUM_GEN:<47d}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({vram:.1f} GB)")

    # ── Load model ────────────────────────────────────────
    from unsloth import FastLanguageModel
    print(f"\n  Loading {MODEL} with Unsloth (4-bit LoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=1024, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing="unsloth")
    print("  ✓ Unsloth 4-bit LoRA ready")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build dataset (larger for 200 steps) ──────────────
    from datasets import Dataset

    SYSTEM = (
        "You are an AI Oversight Auditor. A Medical AI reviewed clinical trial "
        "patients and proposed diagnoses. Some contain errors.\n\n"
        "Return a JSON array of actions to audit the proposals:\n"
        '- {"action_type": "review_proposal", "proposal_id": "PROP-001"}\n'
        '- {"action_type": "investigate_patient", "patient_id": "P0001"}\n'
        '- {"action_type": "flag_error", "proposal_id": "PROP-001", '
        '"error_type": "age_boundary_error", "reason": "Patient age 150 exceeds protocol max"}\n'
        '- {"action_type": "approve", "proposal_id": "PROP-001"}\n\n'
        "First review each proposal, then investigate the patient, then flag or approve."
    )

    dataset_size = max(MAX_STEPS * 2, 400)

    # 60% easy (build foundation), 25% medium, 15% hard
    TASKS = (["oversight_easy"] * int(dataset_size * 0.6) +
             ["oversight_medium"] * int(dataset_size * 0.25) +
             ["oversight_hard"] * int(dataset_size * 0.15))
    TASKS = TASKS[:dataset_size]

    prompts, seeds, task_ids = [], [], []
    for i in range(dataset_size):
        seed = 42 + i * 7
        task = TASKS[i]
        env = SynthAuditEnvironment()
        obs = env.reset(seed=seed, task_id=task)

        proposal_text = "\n".join(
            f"  {p.proposal_id}: Patient {p.patient_id}, "
            f"Dx={p.diagnosis}, Confidence={p.confidence}"
            for p in obs.actor_proposals
        )

        user_msg = (
            f"PROTOCOL:\n{obs.protocol_excerpt[:200]}\n\n"
            f"PROPOSALS ({len(obs.actor_proposals)}):\n{proposal_text}\n\n"
            f"Audit these proposals. Return a JSON array of actions."
        )

        prompts.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        seeds.append(seed)
        task_ids.append(task)

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"  Dataset: {dataset_size} prompts (60% easy, 25% medium, 15% hard)")

    # ── GRPO Training with warmup ─────────────────────────
    from trl import GRPOTrainer, GRPOConfig

    config = GRPOConfig(
        max_completion_length=512,
        num_generations=NUM_GEN,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        max_steps=MAX_STEPS,
        logging_steps=1,
        output_dir=os.path.join(_project_dir, "outputs", "grpo_run"),
        report_to="none",
        learning_rate=5e-6,
        warmup_steps=int(MAX_STEPS * 0.1),  # 10% warmup
        save_steps=50,
        save_total_limit=3,
        log_completions=True,
    )

    reward_fn = make_reward_func(seeds, task_ids)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=config,
    )

    est_time = MAX_STEPS * 1.3  # ~1.3 min/step on T4
    print(f"\n  ▸ GRPO Training for {MAX_STEPS} steps (~{est_time:.0f} min)...")
    print(f"  ▸ Estimated completion: ~{est_time/60:.1f} hours")
    print(f"  ▸ Checkpoints saved every 50 steps\n")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # ── Save model ────────────────────────────────────────
    out_dir = os.path.join(_project_dir, "outputs", "trained_model")
    trainer.save_model(out_dir)

    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n  ✓ Training complete! Time: {h}h {m}m {s}s")
    print(f"  ✓ Model saved to {out_dir}")

    # ── Auto-upload to HuggingFace ────────────────────────
    if HF_REPO:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                repo_id=HF_REPO,
                folder_path=out_dir,
                repo_type="model",
                commit_message=f"feat: GRPO-trained {MAX_STEPS} steps on T4"
            )
            print(f"  ✓ Uploaded to huggingface.co/{HF_REPO}")
        except Exception as e:
            print(f"  ⚠ HF upload failed: {e}")
            print(f"  → Manual upload: huggingface-cli upload {HF_REPO} {out_dir}")

    # ── Print summary ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"  Model:  {MODEL}")
    print(f"  Steps:  {MAX_STEPS}")
    print(f"  Time:   {h}h {m}m {s}s")
    print(f"  Output: {out_dir}")
    if HF_REPO:
        print(f"  HF:     huggingface.co/{HF_REPO}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
