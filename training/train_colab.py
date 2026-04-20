"""
SynthAudit.Env — Colab/Unsloth Training Script
================================================
MINIMUM REQUIREMENT: "Show a minimal training script using Unsloth or HF TRL in Colab"

This script:
1. Installs Unsloth + TRL + openenv_core
2. Loads Llama 3.2 3B with 4-bit LoRA via Unsloth
3. Wraps SynthAudit environment as TRL environment_factory
4. Runs GRPO training for N steps
5. Plots reward curve

Run in Google Colab or any GPU notebook:
    python training/train_colab.py

For Colab, convert to notebook cells using the #%% markers.
"""

#%% ═════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═════════════════════════════════════════════════════════
# !pip install unsloth trl datasets openenv-core pydantic

#%% ═════════════════════════════════════════════════════════
# Cell 2: Imports
# ═════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import json
import time

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)
sys.path.insert(0, os.path.join(_project_dir, "server"))

from models import SynthAuditAction, ActionType
from server.synth_audit_environment import SynthAuditEnvironment

#%% ═════════════════════════════════════════════════════════
# Cell 3: Environment Factory (TRL pattern)
# ═════════════════════════════════════════════════════════

class SynthAuditToolEnv:
    """TRL-compatible environment for GRPO training.

    The oversight agent has 6 tools to audit Actor proposals:
    - review_proposal: See Actor's reasoning
    - investigate_patient: Get raw EHR data
    - request_shap: Feature attribution analysis
    - flag_error: Flag an incorrect proposal
    - approve: Approve a correct proposal
    - submit_audit_report: End episode with summary
    """

    def __init__(self):
        self.env = SynthAuditEnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        """Reset for new oversight episode."""
        self.reward = 0.0
        self.done = False
        diff = kwargs.get("difficulty", "medium")
        task_map = {"easy": "oversight_easy", "medium": "oversight_medium", "hard": "oversight_hard"}
        obs = self.env.reset(seed=42, task_id=task_map.get(diff, "oversight_medium"))

        proposals = "\n".join(
            f"- {p.proposal_id}: Patient {p.patient_id}, "
            f"Dx: {p.diagnosis}, Conf: {p.confidence}"
            for p in obs.actor_proposals
        )
        return (
            f"PROTOCOL:\n{obs.protocol_excerpt}\n\n"
            f"PROPOSALS ({len(obs.actor_proposals)}):\n{proposals}\n\n"
            f"Steps remaining: {obs.steps_remaining}. Begin audit."
        )

    def review_proposal(self, proposal_id: str) -> str:
        """Review a clinical proposal from the Actor AI.

        Args:
            proposal_id: The proposal ID (e.g. PROP-001)

        Returns:
            The Actor's reasoning and diagnosis details
        """
        if self.done:
            raise ValueError("Episode complete.")
        obs = self.env.step(SynthAuditAction(
            action_type=ActionType.review_proposal, proposal_id=proposal_id
        ))
        self._sync(obs)
        return obs.feedback

    def investigate_patient(self, patient_id: str) -> str:
        """Investigate a patient's raw EHR records.

        Args:
            patient_id: The patient ID (e.g. P0001)

        Returns:
            Raw patient data from the clinical trial
        """
        if self.done:
            raise ValueError("Episode complete.")
        obs = self.env.step(SynthAuditAction(
            action_type=ActionType.investigate_patient, patient_id=patient_id
        ))
        self._sync(obs)
        return obs.feedback

    def request_shap(self, patient_id: str, feature: str) -> str:
        """Request SHAP feature attribution for a patient.

        Args:
            patient_id: The patient ID (e.g. P0001)
            feature: Feature name (age, death_date, comorbidity_index, etc.)

        Returns:
            SHAP value and importance rating
        """
        if self.done:
            raise ValueError("Episode complete.")
        obs = self.env.step(SynthAuditAction(
            action_type=ActionType.request_shap, patient_id=patient_id, feature=feature
        ))
        self._sync(obs)
        return obs.feedback

    def flag_error(self, proposal_id: str, error_type: str, reason: str) -> str:
        """Flag a proposal as containing an error.

        Args:
            proposal_id: The proposal to flag (e.g. PROP-001)
            error_type: Error type (hallucination, age_boundary_error, temporal_inconsistency, protocol_window_violation, comorbidity_override_miss, bias_blind_spot)
            reason: Explanation of the error

        Returns:
            Whether the flag was correct or false positive
        """
        if self.done:
            raise ValueError("Episode complete.")
        obs = self.env.step(SynthAuditAction(
            action_type=ActionType.flag_error,
            proposal_id=proposal_id, error_type=error_type, reason=reason
        ))
        self._sync(obs)
        return obs.feedback

    def approve(self, proposal_id: str) -> str:
        """Approve a proposal as correct.

        Args:
            proposal_id: The proposal to approve (e.g. PROP-001)

        Returns:
            Whether approval was correct or missed an error
        """
        if self.done:
            raise ValueError("Episode complete.")
        obs = self.env.step(SynthAuditAction(
            action_type=ActionType.approve, proposal_id=proposal_id
        ))
        self._sync(obs)
        return obs.feedback

    def submit_audit_report(self, summary: str) -> str:
        """Submit final audit report.

        Args:
            summary: Summary of errors found and recommendations

        Returns:
            Final score and breakdown
        """
        if self.done:
            raise ValueError("Episode complete.")
        obs = self.env.step(SynthAuditAction(
            action_type=ActionType.submit_audit_report, report=summary
        ))
        self._sync(obs)
        self.done = True
        return obs.feedback

    def _sync(self, obs):
        self.reward = obs.score_so_far
        self.done = obs.done


#%% ═════════════════════════════════════════════════════════
# Cell 4: Reward Function
# ═════════════════════════════════════════════════════════

def reward_func(environments, **kwargs) -> list[float]:
    """Extract episode score from each environment."""
    return [env.reward for env in environments]


#%% ═════════════════════════════════════════════════════════
# Cell 5: Load Model with Unsloth (4-bit LoRA)
# ═════════════════════════════════════════════════════════

def main():
    """Main training loop. Run in Colab with T4 GPU."""

    # --- Check for Unsloth availability ---
    try:
        from unsloth import FastLanguageModel
        HAS_UNSLOTH = True
        print("✓ Unsloth detected — using optimized training")
    except ImportError:
        HAS_UNSLOTH = False
        print("⚠ Unsloth not available — using standard HF training")

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Model selection: Llama for Meta hackathon
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

    # --- Build dataset ---
    SYSTEM_PROMPT = (
        "You are an AI Oversight Auditor. A Medical AI (the Actor) reviewed "
        "patient records and proposed diagnoses. Some contain errors. "
        "Review proposals, investigate patients, use SHAP, then flag or approve."
    )

    N_SAMPLES = 32  # Small for Colab demo
    dataset = Dataset.from_dict({
        "prompt": [
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": "Begin oversight audit."}]
        ] * N_SAMPLES,
        "difficulty": ["easy"] * (N_SAMPLES // 2) + ["medium"] * (N_SAMPLES // 2),
    })

    # --- Training config ---
    config = GRPOConfig(
        max_completion_length=2048,
        num_generations=2,         # Low for T4 memory
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        max_steps=20,              # Quick demo
        logging_steps=1,
        log_completions=True,
        output_dir="./outputs/colab_run",
        report_to="none",
        learning_rate=5e-6,
    )

    # --- Load model ---
    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_NAME,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )
        print(f"✓ Loaded {MODEL_NAME} with 4-bit LoRA via Unsloth")
    else:
        model = MODEL_NAME
        print(f"✓ Using {MODEL_NAME} directly")

    # --- Train ---
    print("\n" + "=" * 60)
    print("  Starting GRPO Training")
    print("=" * 60)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=config,
        environment_factory=SynthAuditToolEnv,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n✓ Training complete in {elapsed:.0f}s")
    print("  Saving model...")
    trainer.save_model("./outputs/trained_oversight_agent")
    print("✓ Saved to ./outputs/trained_oversight_agent")

    # --- Plot reward curve ---
    try:
        import matplotlib.pyplot as plt

        history = trainer.state.log_history
        rewards = [h.get("train/reward", None) for h in history if "train/reward" in h]
        steps = list(range(1, len(rewards) + 1))

        if rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(steps, rewards, "b-o", linewidth=2, markersize=4)
            plt.xlabel("Training Step", fontsize=12)
            plt.ylabel("Mean Reward", fontsize=12)
            plt.title("SynthAudit.Env — Oversight Agent Reward Curve", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("./outputs/reward_curve.png", dpi=150)
            plt.show()
            print("✓ Reward curve saved to ./outputs/reward_curve.png")
    except Exception as e:
        print(f"  Could not plot: {e}")


#%% ═════════════════════════════════════════════════════════
# Cell 6: Run
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
