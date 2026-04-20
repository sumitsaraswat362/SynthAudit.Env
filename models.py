"""
SynthAudit.Env — Pydantic Models (Competition Grade)
=====================================================
Type-safe Action, Observation, and State models for the
Multi-Agent Clinical AI Oversight Environment.

8 tool actions for the Oversight Agent:
  review_proposal, investigate_patient, request_shap,
  cohort_analysis, temporal_audit, flag_error, approve,
  submit_audit_report
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Action Types — 8 Oversight Tools
# ═══════════════════════════════════════════════════════════════

class ActionType(str, Enum):
    review_proposal = "review_proposal"
    investigate_patient = "investigate_patient"
    request_shap = "request_shap"
    cohort_analysis = "cohort_analysis"
    temporal_audit = "temporal_audit"
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
    statistical_hallucination = "statistical_hallucination"
    citation_fabrication = "citation_fabrication"


class SynthAuditAction(BaseModel):
    """Action the oversight agent can take. Supports 8 tool types."""
    action_type: ActionType
    proposal_id: Optional[str] = None       # For review/flag/approve
    patient_id: Optional[str] = None        # For investigate/shap/temporal
    feature: Optional[str] = None           # For shap/cohort
    error_type: Optional[str] = None        # For flag_error
    reason: Optional[str] = None            # For flag_error (Theory-of-Mind)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    report: Optional[str] = None            # For submit_audit_report


# ═══════════════════════════════════════════════════════════════
# Actor Proposal (what the Actor agent produces)
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


# ═══════════════════════════════════════════════════════════════
# Observation — what the Oversight Agent sees
# ═══════════════════════════════════════════════════════════════

class SynthAuditObservation(BaseModel):
    """Rich observation returned after each step."""
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
    phase: str = "review"  # review, investigation, reporting, complete


# ═══════════════════════════════════════════════════════════════
# State — episode-level tracking
# ═══════════════════════════════════════════════════════════════

class SynthAuditState(BaseModel):
    """Episode state for monitoring and curriculum tracking."""
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
