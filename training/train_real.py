"""
SynthAudit.Env — REAL GRPO Training (Unsloth + TRL)
=====================================================
ACTUALLY trains the model. Weights update. Rewards improve.

Run on Colab T4:
  !pip install unsloth
  !pip install trl datasets
  !python3 training/train_real.py
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
# Reward function: runs a FULL episode from model's completion
# ═══════════════════════════════════════════════════════════════

def score_completion(text: str, seed: int = 42, task_id: str = "oversight_easy") -> float:
    """Parse model output as JSON tool calls, execute in env, return score."""
    env = SynthAuditEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    # Try to parse JSON array of actions
    actions = []
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            actions = json.loads(match.group())
    except Exception:
        pass

    # Fallback: parse individual JSON objects
    if not actions:
        for m in re.finditer(r'\{[^{}]+\}', text):
            try:
                actions.append(json.loads(m.group()))
            except Exception:
                continue

    # Execute parsed actions
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
    def reward_func(completions, **kwargs):
        scores = []
        for i, completion_list in enumerate(completions):
            text = completion_list[0]["content"] if isinstance(completion_list, list) else str(completion_list)
            seed = seeds[i % len(seeds)]
            task = task_ids[i % len(task_ids)]
            score = score_completion(text, seed=seed, task_id=task)
            scores.append(float(score))
        return scores
    return reward_func


# ═══════════════════════════════════════════════════════════════
# Main Training
# ═══════════════════════════════════════════════════════════════

def main():
    import torch

    MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-3B-Instruct")
    MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
    NUM_GEN = int(os.getenv("NUM_GEN", "4"))

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SynthAudit.Env — REAL GRPO Training (Unsloth + TRL)       ║")
    print("║  Multi-Agent Clinical AI Oversight                          ║")
    print(f"║  Model:    {MODEL:<47s}║")
    print(f"║  Steps:    {MAX_STEPS:<47d}║")
    print(f"║  Gen/step: {NUM_GEN:<47d}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({vram:.1f} GB)")

    # ── Load model with Unsloth ───────────────────────────
    try:
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
        USE_UNSLOTH = True
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n  Loading {MODEL} with transformers...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, device_map="auto")
        USE_UNSLOTH = False
        print("  ⚠ No Unsloth — using raw transformers (higher VRAM)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build dataset ─────────────────────────────────────
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

    # Generate varied prompts by running env resets
    prompts = []
    seeds = []
    task_ids = []
    dataset_size = max(MAX_STEPS * 2, 64)

    TASKS = ["oversight_easy"] * (dataset_size // 2) + \
            ["oversight_medium"] * (dataset_size // 4) + \
            ["oversight_hard"] * (dataset_size - dataset_size // 2 - dataset_size // 4)

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
    print(f"  Dataset: {dataset_size} prompts (50% easy, 25% medium, 25% hard)")

    # ── Try GRPO Training ─────────────────────────────────
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
        save_steps=25,
        save_total_limit=2,
        log_completions=True,
    )

    reward_fn = make_reward_func(seeds, task_ids)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=config,
    )

    print(f"\n  ▸ GRPO Training for {MAX_STEPS} steps...")
    print(f"  ▸ This is REAL training — weights are being updated!\n")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # ── Save model ────────────────────────────────────────
    out_dir = os.path.join(_project_dir, "outputs", "trained_model")
    trainer.save_model(out_dir)

    # ── Extract metrics ───────────────────────────────────
    rewards = [h["train/reward"] for h in trainer.state.log_history
               if "train/reward" in h]
    losses = [h["train/loss"] for h in trainer.state.log_history
              if "train/loss" in h]

    results = {
        "model": MODEL,
        "method": "GRPO",
        "max_steps": MAX_STEPS,
        "num_generations": NUM_GEN,
        "elapsed_seconds": round(elapsed),
        "rewards": rewards,
        "losses": losses,
        "final_reward": rewards[-1] if rewards else None,
        "best_reward": max(rewards) if rewards else None,
    }

    os.makedirs(os.path.join(_project_dir, "outputs"), exist_ok=True)
    with open(os.path.join(_project_dir, "outputs", "training_log.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if rewards:
            steps = list(range(1, len(rewards) + 1))
            w = min(5, len(rewards))
            avg = []
            for i in range(len(rewards)):
                s = max(0, i - w + 1)
                avg.append(sum(rewards[s:i+1]) / (i - s + 1))

            axes[0].plot(steps, rewards, 'b-', alpha=0.3, linewidth=1)
            axes[0].plot(steps, avg, 'r-', linewidth=2.5, label=f'Running Avg (w={w})')
            axes[0].fill_between(steps, rewards, alpha=0.1, color='blue')
            axes[0].set_xlabel("Training Step")
            axes[0].set_ylabel("Reward (Episode Score)")
            axes[0].set_title("GRPO Reward Curve", fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        if losses:
            axes[1].plot(range(1, len(losses)+1), losses, 'g-', linewidth=1.5)
            axes[1].set_xlabel("Training Step")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Training Loss", fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"SynthAudit.Env — GRPO Training ({MODEL.split('/')[-1]})\n"
                     f"{MAX_STEPS} steps, {elapsed/60:.0f} min, REAL weight updates",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = os.path.join(_project_dir, "outputs", "reward_curve.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Reward curve: {path}")
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print(f"  REAL GRPO Training Complete")
    print(f"  Time:         {elapsed/60:.1f} min")
    print(f"  Steps:        {MAX_STEPS}")
    print(f"  Best reward:  {max(rewards) if rewards else 'N/A'}")
    print(f"  Final reward: {rewards[-1] if rewards else 'N/A'}")
    print(f"  Model saved:  {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
