"""
SynthAudit.Env — Pydantic Models
================================
Type-safe Action, Observation, and State models for the
Multi-Agent Clinical AI Oversight Environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Action Types
# ═══════════════════════════════════════════════════════════════

class ActionType(str, Enum):
    review_proposal = "review_proposal"
    investigate_patient = "investigate_patient"
    request_shap = "request_shap"
    flag_error = "flag_error"
    approve = "approve"
    submit_audit_report = "submit_audit_report"


class ErrorType(str, Enum):
    hallucination = "hallucination"
    age_boundary_error = "age_boundary_error"
    temporal_inconsistency = "temporal_inconsistency"
    protocol_window_violation = "protocol_window_violation"
    bias_blind_spot = "bias_blind_spot"
    comorbidity_override_miss = "comorbidity_override_miss"
    incorrect_diagnosis = "incorrect_diagnosis"


class SynthAuditAction(BaseModel):
    """Action the oversight agent can take."""
    action_type: ActionType
    proposal_id: Optional[str] = None
    patient_id: Optional[str] = None
    feature: Optional[str] = None       # For request_shap
    error_type: Optional[str] = None    # For flag_error
    reason: Optional[str] = None        # For flag_error
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    report: Optional[str] = None        # For submit_audit_report


# ═══════════════════════════════════════════════════════════════
# Observation
# ═══════════════════════════════════════════════════════════════

class ActorProposal(BaseModel):
    """A clinical proposal made by the Actor agent."""
    proposal_id: str
    patient_id: str
    diagnosis: str
    reasoning: str
    confidence: float
    recommended_action: str
    status: str = "pending"  # pending, flagged, approved


class SynthAuditObservation(BaseModel):
    """What the oversight agent sees after each step."""
    done: bool = False
    reward: float = 0.0
    task_id: str = ""
    difficulty: str = "medium"
    protocol_excerpt: str = ""
    actor_proposals: list[ActorProposal] = Field(default_factory=list)
    current_proposal_detail: Optional[dict] = None
    patient_data: Optional[dict] = None
    shap_result: Optional[dict] = None
    feedback: str = ""
    score_so_far: float = 0.01
    proposals_reviewed: int = 0
    errors_flagged: int = 0
    correct_flags: int = 0
    false_positives: int = 0
    approvals: int = 0
    correct_approvals: int = 0
    steps_taken: int = 0
    steps_remaining: int = 0
    phase: str = "review"  # review, investigation, reporting


# ═══════════════════════════════════════════════════════════════
# State (episode metadata)
# ═══════════════════════════════════════════════════════════════

class SynthAuditState(BaseModel):
    """Episode state tracking."""
    episode_id: str = ""
    step_count: int = 0
    current_score: float = 0.01
    proposals_total: int = 0
    proposals_reviewed: int = 0
    errors_flagged: int = 0
    correct_flags: int = 0
    false_positives: int = 0
    approvals: int = 0
    correct_approvals: int = 0
    missed_errors: int = 0
    shap_requests: int = 0
    investigations: int = 0
    phase: str = "review"
    score_breakdown: dict = Field(default_factory=dict)
