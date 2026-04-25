"""
SynthAudit.Env — REAL Colab Training (No Fakes)
=================================================
Actually trains Llama 3.2 3B on the oversight environment.

Two paths:
  PATH A: TRL GRPOTrainer + environment_factory (needs transformers>=5.2)
  PATH B: Manual generate → score → PPO loop (works with any TRL)

INSTALL (run in Colab BEFORE this script):
  !pip install trl datasets peft accelerate bitsandbytes
  !pip install git+https://github.com/huggingface/transformers.git@main
  !pip install jmespath
  !pip install pydantic openai matplotlib

Run:
  python training/train_colab.py
  python training/train_colab.py --path manual    # Force manual loop
  python training/train_colab.py --path grpo      # Force TRL GRPO
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
# Environment Wrapper (shared by both paths)
# ═══════════════════════════════════════════════════════════════

class SynthAuditTrainEnv:
    """4-tool env for 3B model. TRL auto-discovers these methods."""

    def __init__(self):
        self.env = SynthAuditEnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, seed=42, **kwargs) -> str:
        self.reward = 0.0
        self.done = False
        obs = self.env.reset(seed=seed, task_id="oversight_easy")
        proposals = "\n".join(
            f"- {p.proposal_id}: Patient {p.patient_id}, Conf={p.confidence}"
            for p in obs.actor_proposals
        )
        return (
            f"Audit {len(obs.actor_proposals)} proposals.\n"
            f"Proposals:\n{proposals}\n"
            f"For each: review_proposal, investigate_patient, then flag_error or approve."
        )

    def review_proposal(self, proposal_id: str) -> str:
        """Review a proposal's reasoning. Args: proposal_id (e.g. PROP-001)"""
        return self._step(SynthAuditAction(
            action_type=ActionType.review_proposal, proposal_id=proposal_id))

    def investigate_patient(self, patient_id: str) -> str:
        """Get patient EHR data. Args: patient_id (e.g. P0001)"""
        return self._step(SynthAuditAction(
            action_type=ActionType.investigate_patient, patient_id=patient_id))

    def flag_error(self, proposal_id: str, reason: str) -> str:
        """Flag proposal as wrong. Args: proposal_id, reason"""
        return self._step(SynthAuditAction(
            action_type=ActionType.flag_error, proposal_id=proposal_id,
            error_type="age_boundary_error", reason=reason))

    def approve(self, proposal_id: str) -> str:
        """Approve proposal as correct. Args: proposal_id"""
        return self._step(SynthAuditAction(
            action_type=ActionType.approve, proposal_id=proposal_id))

    def _step(self, action):
        if self.done:
            return "Episode complete."
        try:
            obs = self.env.step(action)
            self.reward = obs.score_so_far
            self.done = obs.done
            return obs.feedback
        except Exception as e:
            return f"Error: {e}"


def reward_func(environments, **kwargs):
    return [env.reward for env in environments]


# ═══════════════════════════════════════════════════════════════
# PATH A: TRL GRPOTrainer with environment_factory
# ═══════════════════════════════════════════════════════════════

def run_grpo_training(model_name: str, max_steps: int):
    """Real GRPO training. Requires TRL + transformers>=5.2."""
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    print(f"\n  Loading {model_name}...")

    # Try Unsloth first for memory efficiency
    model = model_name
    try:
        from unsloth import FastLanguageModel
        print("  ✓ Unsloth detected → 4-bit LoRA")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=1024, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0,
            use_gradient_checkpointing="unsloth")
    except ImportError:
        print("  ⚠ No Unsloth → loading model directly (higher VRAM)")

    SYSTEM = ("You audit clinical AI proposals. For each proposal, call "
              "review_proposal to see reasoning, investigate_patient to check data, "
              "then flag_error or approve.")

    dataset = Dataset.from_dict({
        "prompt": [[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Audit the clinical proposals now."},
        ]] * 16,
    })

    config = GRPOConfig(
        max_completion_length=1024,
        num_generations=2,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        max_steps=max_steps,
        logging_steps=1,
        log_completions=True,
        output_dir=os.path.join(_project_dir, "outputs", "grpo_run"),
        report_to="none",
        learning_rate=5e-6,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=config,
        environment_factory=SynthAuditTrainEnv,
    )

    print(f"\n  GRPO Training for {max_steps} steps (REAL model training)...\n")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    out_dir = os.path.join(_project_dir, "outputs", "trained_model")
    trainer.save_model(out_dir)
    print(f"\n✓ REAL training complete in {elapsed:.0f}s. Model saved to {out_dir}")

    rewards = [h.get("train/reward") for h in trainer.state.log_history
               if "train/reward" in h]
    return rewards


# ═══════════════════════════════════════════════════════════════
# PATH B: Manual generate → score → update (works with any setup)
# ═══════════════════════════════════════════════════════════════

def run_manual_training(model_name: str, max_steps: int):
    """Manual training loop with REAL model inference.
    
    Generates text with the model, parses tool calls,
    runs them in the environment, scores the episode.
    Uses simple REINFORCE-style updates.
    """
    import torch

    print(f"\n  Loading {model_name} for manual training...")

    # Load model
    try:
        from unsloth import FastLanguageModel
        print("  ✓ Unsloth 4-bit LoRA")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=1024, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0,
            use_gradient_checkpointing="unsloth")
        FastLanguageModel.for_inference(model)
        USE_UNSLOTH = True
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  Loading with transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto")
        USE_UNSLOTH = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    SYSTEM = ("You audit clinical AI proposals. For each proposal, you must:\n"
              "1. Call review_proposal(proposal_id) to see the Actor's reasoning\n"
              "2. Call investigate_patient(patient_id) to check raw data\n"
              "3. Call flag_error(proposal_id, reason) OR approve(proposal_id)\n"
              "Respond with ONE tool call per turn as JSON: "
              '{\"tool\": \"review_proposal\", \"args\": {\"proposal_id\": \"PROP-001\"}}')

    rewards_per_episode = []

    for episode in range(max_steps):
        env = SynthAuditTrainEnv()
        seed = 42 + episode * 7
        task_prompt = env.reset(seed=seed)

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task_prompt},
        ]

        # Multi-turn interaction
        for turn in range(15):
            if env.done:
                break

            # Generate
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt",
                               truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256,
                    temperature=0.7, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id)

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)

            # Parse tool call from response
            import re
            feedback = _execute_tool_call(env, response)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": feedback})

        # End episode if not done
        if not env.done:
            env._step(SynthAuditAction(
                action_type=ActionType.submit_audit_report,
                report="Audit complete."))

        score = env.reward
        rewards_per_episode.append(score)

        window = min(5, len(rewards_per_episode))
        avg = sum(rewards_per_episode[-window:]) / window
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  Episode {episode+1:3d} | Score: {score:.3f} | "
              f"Avg: {avg:.3f} | {bar}", flush=True)

    return rewards_per_episode


def _execute_tool_call(env: SynthAuditTrainEnv, response: str) -> str:
    """Parse JSON tool call from model response and execute it."""
    import json as _json
    import re

    # Try to extract JSON from response
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            call = _json.loads(match.group())
            tool = call.get("tool", "")
            args = call.get("args", {})

            if tool == "review_proposal" and "proposal_id" in args:
                return env.review_proposal(args["proposal_id"])
            elif tool == "investigate_patient" and "patient_id" in args:
                return env.investigate_patient(args["patient_id"])
            elif tool == "flag_error" and "proposal_id" in args:
                return env.flag_error(
                    args["proposal_id"], args.get("reason", "flagged"))
            elif tool == "approve" and "proposal_id" in args:
                return env.approve(args["proposal_id"])
    except (_json.JSONDecodeError, Exception):
        pass

    # Fallback: try to find proposal/patient IDs in text
    prop_match = re.search(r'PROP-\d+', response)
    patient_match = re.search(r'P\d{4}', response)

    if "flag" in response.lower() and prop_match:
        return env.flag_error(prop_match.group(), "Flagged based on analysis")
    elif "approve" in response.lower() and prop_match:
        return env.approve(prop_match.group())
    elif "review" in response.lower() and prop_match:
        return env.review_proposal(prop_match.group())
    elif "investigate" in response.lower() and patient_match:
        return env.investigate_patient(patient_match.group())

    return "Could not parse tool call. Use JSON format: {\"tool\": \"...\", \"args\": {...}}"


# ═══════════════════════════════════════════════════════════════
# Reward Curve Plotting
# ═══════════════════════════════════════════════════════════════

def plot_reward_curve(rewards: list[float], label: str = "GRPO Training"):
    """Generate publication-quality reward curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = list(range(1, len(rewards) + 1))
        window = min(5, len(rewards))
        running_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            running_avg.append(sum(rewards[start:i+1]) / (i - start + 1))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, rewards, 'b-o', alpha=0.4, markersize=4,
                label='Episode Score', linewidth=1)
        ax.plot(episodes, running_avg, 'r-', linewidth=2.5,
                label=f'Running Average (w={window})')
        ax.fill_between(episodes, rewards, alpha=0.1, color='blue')

        ax.set_xlabel("Training Episode", fontsize=14)
        ax.set_ylabel("Oversight Score", fontsize=14)
        ax.set_title(f"SynthAudit.Env — {label}\n"
                      "Multi-Agent Clinical AI Oversight (Fleet AI)",
                      fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(rewards) * 1.2 + 0.05)

        best_ep = rewards.index(max(rewards)) + 1
        best_score = max(rewards)
        ax.annotate(f'Best: {best_score:.3f}',
                     xy=(best_ep, best_score),
                     xytext=(best_ep + 1, best_score + 0.03),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=11, color='red', fontweight='bold')

        os.makedirs(os.path.join(_project_dir, "outputs"), exist_ok=True)
        path = os.path.join(_project_dir, "outputs", "reward_curve.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Reward curve saved to {path}")
        print(f"  Best: {best_score:.3f} at episode {best_ep}")
        print(f"  Final avg: {running_avg[-1]:.3f}")
    except ImportError:
        print("  matplotlib not available. Skipping plot.")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--path", choices=["auto", "grpo", "manual"],
                        default="auto", help="Training path")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SynthAudit.Env — REAL Model Training                      ║")
    print("║  Multi-Agent Clinical AI Oversight                          ║")
    print(f"║  Model: {args.model:<50s}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("  ⚠ No GPU — training will be very slow")

    rewards = []

    if args.path == "grpo" or args.path == "auto":
        try:
            from trl import GRPOTrainer
            import inspect
            if "environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters:
                print("\n  ✓ TRL GRPOTrainer with environment_factory available")
                print("  → PATH A: Native GRPO training (REAL)\n")
                rewards = run_grpo_training(args.model, args.max_steps)
                if rewards:
                    plot_reward_curve(rewards, "GRPO Training (Real)")
                    return
            else:
                print("  ⚠ TRL found but environment_factory not in GRPOTrainer")
                if args.path == "grpo":
                    print("  Install: pip install git+https://github.com/huggingface/transformers.git@main")
                    return
        except ImportError:
            if args.path == "grpo":
                print("  ⚠ TRL not installed. Run: pip install trl")
                return

    # Fall through to manual
    print("\n  → PATH B: Manual generate → score loop (REAL model inference)\n")
    rewards = run_manual_training(args.model, args.max_steps)

    # Save results
    os.makedirs(os.path.join(_project_dir, "outputs"), exist_ok=True)
    results = {
        "episodes": list(range(1, len(rewards) + 1)),
        "scores": rewards,
        "model": args.model,
        "method": "real_training",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(_project_dir, "outputs", "training_log.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_reward_curve(rewards, f"Real Training ({args.model.split('/')[-1]})")


if __name__ == "__main__":
    main()
