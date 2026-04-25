"""
SynthAudit.Env — TRL GRPO Training (Competition Grade)
========================================================
REAL model training with proper scale:
  - Meta Llama 3.2 3B (4-bit LoRA via Unsloth)
  - 200 training episodes across easy/medium/hard curriculum
  - 50 max steps per episode (matches competitor benchmarks)
  - TRL GRPOTrainer with environment_factory
  - Dense shaped rewards for fast convergence

Requirements:
  pip install trl datasets peft accelerate bitsandbytes
  pip install git+https://github.com/huggingface/transformers.git@main
  pip install jmespath pydantic openai matplotlib

Run:
  python training/train_grpo.py                          # Default: 200 episodes
  python training/train_grpo.py --max-steps 500          # Longer training
  python training/train_grpo.py --model meta-llama/Llama-3.2-1B-Instruct  # Smaller model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)
sys.path.insert(0, os.path.join(_project_dir, "server"))

from models import SynthAuditAction, ActionType
from server.synth_audit_environment import SynthAuditEnvironment


# ═══════════════════════════════════════════════════════════════
# Training Environment — 4 core tools for 3B model
# ═══════════════════════════════════════════════════════════════

class SynthAuditToolEnv:
    """TRL environment_factory wrapper with 4 core oversight tools.

    Why 4 not 8: A 3B model can reliably call 4 tools.
    The full 8-tool set is for 70B+ models or inference-time.
    """

    def __init__(self):
        self.env = SynthAuditEnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        self.done = False

        # Curriculum: rotate difficulty based on kwargs
        diff = kwargs.get("difficulty", "easy")
        task_map = {"easy": "oversight_easy", "medium": "oversight_medium", "hard": "oversight_hard"}
        seed = kwargs.get("seed", 42)
        obs = self.env.reset(seed=seed, task_id=task_map.get(diff, "oversight_easy"))

        proposals = "\n".join(
            f"- {p.proposal_id}: Patient {p.patient_id}, Conf={p.confidence}"
            for p in obs.actor_proposals
        )
        return (
            f"PROTOCOL:\n{obs.protocol_excerpt}\n\n"
            f"PROPOSALS ({len(obs.actor_proposals)}):\n{proposals}\n\n"
            f"Steps: {obs.steps_remaining}. Audit each proposal: review, investigate, then flag or approve."
        )

    def review_proposal(self, proposal_id: str) -> str:
        """Review a clinical proposal to see the Actor AI's reasoning and citations.

        Args:
            proposal_id: The proposal ID to review (e.g. 'PROP-001')

        Returns:
            The Actor's full reasoning, citations, and clinical analysis.
        """
        return self._step(SynthAuditAction(
            action_type=ActionType.review_proposal, proposal_id=proposal_id))

    def investigate_patient(self, patient_id: str) -> str:
        """Get raw EHR data for a patient to verify the Actor's claims.

        Args:
            patient_id: The patient ID to look up (e.g. 'P0001')

        Returns:
            Patient demographics, dates, vitals, labs, and clinical data.
        """
        return self._step(SynthAuditAction(
            action_type=ActionType.investigate_patient, patient_id=patient_id))

    def flag_error(self, proposal_id: str, reason: str) -> str:
        """Flag a proposal as containing a clinical error.

        Args:
            proposal_id: The proposal to flag (e.g. 'PROP-001')
            reason: Detailed explanation of what the Actor got wrong

        Returns:
            Whether the flag was correct or a false positive.
        """
        return self._step(SynthAuditAction(
            action_type=ActionType.flag_error, proposal_id=proposal_id,
            error_type="age_boundary_error", reason=reason))

    def approve(self, proposal_id: str) -> str:
        """Approve a proposal as clinically correct after verification.

        Args:
            proposal_id: The proposal to approve (e.g. 'PROP-001')

        Returns:
            Whether the approval was correct or missed an error.
        """
        return self._step(SynthAuditAction(
            action_type=ActionType.approve, proposal_id=proposal_id))

    def _step(self, action: SynthAuditAction) -> str:
        if self.done:
            return "Episode already complete."
        try:
            obs = self.env.step(action)
            self.reward = obs.score_so_far
            self.done = obs.done
            return obs.feedback
        except Exception as e:
            return f"Error: {e}"


def reward_func(environments, **kwargs) -> list[float]:
    """Extract episode scores from environments for GRPO."""
    return [env.reward for env in environments]


SYSTEM_PROMPT = (
    "You are an AI Oversight Auditor. A Medical AI (the Actor) reviewed "
    "clinical trial patients and proposed diagnoses. Some proposals contain "
    "subtle errors: age violations, temporal paradoxes, protocol window "
    "breaches, and hallucinated citations.\n\n"
    "For EACH proposal, follow this sequence:\n"
    "1. review_proposal(proposal_id) — read the Actor's reasoning\n"
    "2. investigate_patient(patient_id) — check raw patient data\n"
    "3. flag_error(proposal_id, reason) if wrong, OR approve(proposal_id) if correct\n\n"
    "Be precise in your flag_error reason — explain EXACTLY what the Actor got wrong."
)


def main():
    parser = argparse.ArgumentParser(
        description="SynthAudit.Env — Competition-Grade GRPO Training"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model to train (default: Llama 3.2 3B)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for faster generation")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Candidates per prompt (GRPO group size)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Training steps (episodes). Competitors use 200-800.")
    parser.add_argument("--dataset-size", type=int, default=256,
                        help="Training dataset size (prompt variations)")
    parser.add_argument("--max-completion-length", type=int, default=2048,
                        help="Max tokens per completion")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SynthAudit.Env — GRPO Training (Competition Grade)        ║")
    print("║  Multi-Agent Clinical AI Oversight                          ║")
    print(f"║  Model:    {args.model:<47s}║")
    print(f"║  Episodes: {args.max_steps:<47d}║")
    print(f"║  Gen/step: {args.num_generations:<47d}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("  ⚠ No GPU — training will be very slow")

    # ── Load model ────────────────────────────────────────
    model = args.model
    try:
        from unsloth import FastLanguageModel
        print(f"\n  ✓ Unsloth detected → 4-bit LoRA")
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=args.max_completion_length,
            load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0,
            use_gradient_checkpointing="unsloth")
        print(f"  ✓ Loaded {args.model} with LoRA (rank=16)")
    except ImportError:
        print("  ⚠ No Unsloth — using model name directly (higher VRAM)")

    # ── Build curriculum dataset ──────────────────────────
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Curriculum: 40% easy, 35% medium, 25% hard
    n_easy = int(args.dataset_size * 0.40)
    n_medium = int(args.dataset_size * 0.35)
    n_hard = args.dataset_size - n_easy - n_medium

    prompt = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": "Begin your clinical oversight audit."}]

    dataset = Dataset.from_dict({
        "prompt": [prompt] * args.dataset_size,
        "difficulty": (["easy"] * n_easy +
                       ["medium"] * n_medium +
                       ["hard"] * n_hard),
    })
    dataset = dataset.shuffle(seed=42)

    print(f"\n  Dataset: {args.dataset_size} prompts "
          f"({n_easy} easy, {n_medium} medium, {n_hard} hard)")

    # ── Training config ───────────────────────────────────
    config_kw = {
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 1,
        "max_steps": args.max_steps,
        "logging_steps": 1,
        "log_completions": True,
        "output_dir": os.path.join(_project_dir, "outputs", "training_run"),
        "report_to": "none",
        "learning_rate": args.lr,
        "save_steps": 50,
        "save_total_limit": 3,
    }
    if args.use_vllm:
        config_kw["use_vllm"] = True
        config_kw["vllm_mode"] = "colocate"

    # ── Train ─────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=GRPOConfig(**config_kw),
        environment_factory=SynthAuditToolEnv,
    )

    print(f"\n  Training for {args.max_steps} steps...")
    print(f"  Estimated time: ~{args.max_steps * 30 // 60} minutes on T4\n")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # ── Save model ────────────────────────────────────────
    out_dir = os.path.join(_project_dir, "outputs", "trained_oversight_agent")
    trainer.save_model(out_dir)

    # ── Extract and save reward curve ─────────────────────
    rewards = [h.get("train/reward") for h in trainer.state.log_history
               if "train/reward" in h]
    losses = [h.get("train/loss") for h in trainer.state.log_history
              if "train/loss" in h]

    results = {
        "model": args.model,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "dataset_size": args.dataset_size,
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

        if rewards:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Reward curve
            steps = list(range(1, len(rewards) + 1))
            window = min(10, len(rewards))
            running_avg = []
            for i in range(len(rewards)):
                s = max(0, i - window + 1)
                running_avg.append(sum(rewards[s:i+1]) / (i - s + 1))

            ax1.plot(steps, rewards, 'b-', alpha=0.3, linewidth=0.8, label='Raw')
            ax1.plot(steps, running_avg, 'r-', linewidth=2.5, label=f'Avg (w={window})')
            ax1.fill_between(steps, rewards, alpha=0.08, color='blue')
            ax1.set_xlabel("Training Step", fontsize=13)
            ax1.set_ylabel("Episode Score", fontsize=13)
            ax1.set_title("Reward Curve", fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Loss curve
            if losses:
                ax2.plot(range(1, len(losses)+1), losses, 'g-', linewidth=1.5)
                ax2.set_xlabel("Training Step", fontsize=13)
                ax2.set_ylabel("Loss", fontsize=13)
                ax2.set_title("Training Loss", fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)

            fig.suptitle(f"SynthAudit.Env — GRPO Training ({args.model.split('/')[-1]})\n"
                         f"{args.max_steps} steps, {elapsed/60:.0f} min",
                         fontsize=15, fontweight='bold')
            plt.tight_layout()
            path = os.path.join(_project_dir, "outputs", "reward_curve.png")
            plt.savefig(path, dpi=200, bbox_inches='tight')
            print(f"\n✓ Reward curve saved to {path}")
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Steps: {args.max_steps}")
    print(f"  Best reward: {max(rewards) if rewards else 'N/A'}")
    print(f"  Final reward: {rewards[-1] if rewards else 'N/A'}")
    print(f"  Model saved: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
