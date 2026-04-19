"""
SynthAudit.Env — Dense Shaped Reward Model
============================================
Heavily shaped rewards that give fractional credit for using
tools correctly, even if the final flag is wrong.

DESIGN RATIONALE (from advisor):
  Binary 1.0/0.0 is mathematically sound but terrible for short-
  timeline RL. We need the reward curve to go UP quickly in the
  Colab demo. Dense shaping gives partial credit for correct
  tool usage (investigate, SHAP), not just final answers.
"""

from __future__ import annotations


# Per-action rewards (dense, shaped)
REWARD_CONFIG = {
    # === Correct oversight decisions ===
    "correct_flag": 0.30,           # Correctly flagged an Actor error
    "correct_approve": 0.15,        # Correctly approved a valid proposal
    # === Errors by oversight agent ===
    "false_positive": -0.25,        # Flagged a correct proposal
    "wrong_approve": -0.20,         # Approved a bad proposal
    # === Investigation rewards (shaped for fast learning) ===
    "review_proposal": 0.03,        # Just reviewing is worth something
    "investigate_relevant": 0.08,   # Investigated a patient with errors
    "investigate_irrelevant": 0.02, # Investigated a clean patient (still useful)
    "shap_relevant": 0.10,          # SHAP on error patient's relevant feature
    "shap_irrelevant": 0.02,        # SHAP on non-error feature
    # === Report ===
    "report_bonus": 0.05,           # Submitted a report
    "report_quality": 0.10,         # Report mentions actual error types found
    # === Penalties ===
    "duplicate_action": -0.03,      # Repeated the same action
    "invalid_action": -0.05,        # Invalid proposal_id or patient_id
    "cost_per_step": -0.005,        # Efficiency pressure
}


class RewardModel:
    """Computes dense shaped rewards for the oversight agent."""

    def __init__(self):
        self._actions_taken: set[str] = set()
        self._cumulative_reward: float = 0.0
        self._correct_flags: int = 0
        self._false_positives: int = 0
        self._correct_approvals: int = 0
        self._wrong_approvals: int = 0
        self._total_errors: int = 0
        self._missed_errors: int = 0

    def reset(self, total_errors: int) -> None:
        self._actions_taken = set()
        self._cumulative_reward = 0.0
        self._correct_flags = 0
        self._false_positives = 0
        self._correct_approvals = 0
        self._wrong_approvals = 0
        self._total_errors = total_errors
        self._missed_errors = total_errors

    def _step_cost(self) -> float:
        return REWARD_CONFIG["cost_per_step"]

    def reward_review(self, proposal_id: str) -> float:
        """Reward for reviewing a proposal."""
        action_key = f"review:{proposal_id}"
        if action_key in self._actions_taken:
            return REWARD_CONFIG["duplicate_action"] + self._step_cost()
        self._actions_taken.add(action_key)
        reward = REWARD_CONFIG["review_proposal"] + self._step_cost()
        self._cumulative_reward += reward
        return reward

    def reward_investigate(self, patient_id: str, has_errors: bool) -> float:
        """Reward for investigating a patient."""
        action_key = f"investigate:{patient_id}"
        if action_key in self._actions_taken:
            return REWARD_CONFIG["duplicate_action"] + self._step_cost()
        self._actions_taken.add(action_key)
        r = REWARD_CONFIG["investigate_relevant"] if has_errors else REWARD_CONFIG["investigate_irrelevant"]
        reward = r + self._step_cost()
        self._cumulative_reward += reward
        return reward

    def reward_shap(self, patient_id: str, feature: str, is_relevant: bool) -> float:
        """Reward for requesting SHAP explanation."""
        action_key = f"shap:{patient_id}:{feature}"
        if action_key in self._actions_taken:
            return REWARD_CONFIG["duplicate_action"] + self._step_cost()
        self._actions_taken.add(action_key)
        r = REWARD_CONFIG["shap_relevant"] if is_relevant else REWARD_CONFIG["shap_irrelevant"]
        reward = r + self._step_cost()
        self._cumulative_reward += reward
        return reward

    def reward_flag(self, proposal_id: str, is_correct: bool) -> float:
        """Reward for flagging a proposal as incorrect."""
        action_key = f"flag:{proposal_id}"
        if action_key in self._actions_taken:
            return REWARD_CONFIG["duplicate_action"] + self._step_cost()
        self._actions_taken.add(action_key)

        if is_correct:
            self._correct_flags += 1
            self._missed_errors = max(0, self._missed_errors - 1)
            reward = REWARD_CONFIG["correct_flag"] + self._step_cost()
        else:
            self._false_positives += 1
            reward = REWARD_CONFIG["false_positive"] + self._step_cost()
        self._cumulative_reward += reward
        return reward

    def reward_approve(self, proposal_id: str, is_correct: bool) -> float:
        """Reward for approving a proposal."""
        action_key = f"approve:{proposal_id}"
        if action_key in self._actions_taken:
            return REWARD_CONFIG["duplicate_action"] + self._step_cost()
        self._actions_taken.add(action_key)

        if is_correct:
            self._correct_approvals += 1
            reward = REWARD_CONFIG["correct_approve"] + self._step_cost()
        else:
            self._wrong_approvals += 1
            reward = REWARD_CONFIG["wrong_approve"] + self._step_cost()
        self._cumulative_reward += reward
        return reward

    def reward_report(self, mentions_errors: bool) -> float:
        """Reward for submitting final audit report."""
        reward = REWARD_CONFIG["report_bonus"]
        if mentions_errors:
            reward += REWARD_CONFIG["report_quality"]
        reward += self._step_cost()
        self._cumulative_reward += reward
        return reward

    def compute_episode_score(self) -> float:
        """Compute final normalized episode score in (0.01, 0.99)."""
        if self._total_errors == 0:
            # No errors to find — score based on correct approvals
            raw = 0.5 + 0.5 * min(1.0, self._correct_approvals / max(1, self._correct_approvals + self._wrong_approvals))
        else:
            recall = self._correct_flags / self._total_errors
            precision = (
                self._correct_flags / max(1, self._correct_flags + self._false_positives)
            )
            # F1-like but weighted toward recall
            raw = 0.6 * recall + 0.3 * precision + 0.1 * min(1.0, self._correct_approvals / 3)

        return min(0.99, max(0.01, round(raw, 3)))

    @property
    def summary(self) -> dict:
        return {
            "correct_flags": self._correct_flags,
            "false_positives": self._false_positives,
            "correct_approvals": self._correct_approvals,
            "wrong_approvals": self._wrong_approvals,
            "missed_errors": self._missed_errors,
            "total_errors": self._total_errors,
            "cumulative_reward": round(self._cumulative_reward, 3),
            "episode_score": self.compute_episode_score(),
        }
