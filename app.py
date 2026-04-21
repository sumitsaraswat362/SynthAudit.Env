"""
SynthAudit.Env — HuggingFace Spaces Interactive Demo
======================================================
Gradio app showing the oversight environment in action.
Judges can play with it live.

Deploy: huggingface-cli upload-folder . <space-name> --repo-type space
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

import gradio as gr

from models import SynthAuditAction, ActionType
from server.synth_audit_environment import SynthAuditEnvironment

# Global state
_env = None
_obs = None
_history = []
_step_count = 0


def reset_environment(difficulty, seed):
    """Start a new oversight episode."""
    global _env, _obs, _history, _step_count

    task_map = {"Easy": "oversight_easy", "Medium": "oversight_medium", "Hard": "oversight_hard"}
    task_id = task_map.get(difficulty, "oversight_medium")

    _env = SynthAuditEnvironment()
    _obs = _env.reset(seed=int(seed), task_id=task_id)
    _history = []
    _step_count = 0

    proposals_md = "## Actor Proposals\n\n"
    proposals_md += "| ID | Patient | Diagnosis | Confidence |\n"
    proposals_md += "|---|---|---|---|\n"
    for p in _obs.actor_proposals:
        proposals_md += f"| {p.proposal_id} | {p.patient_id} | {p.diagnosis} | {p.confidence:.2f} |\n"

    protocol_md = f"## Protocol\n\n```\n{_obs.protocol_excerpt}\n```"

    status = (
        f"**Difficulty**: {difficulty} | "
        f"**Proposals**: {len(_obs.actor_proposals)} | "
        f"**Steps**: {_obs.steps_remaining} | "
        f"**Score**: {_obs.score_so_far:.3f}"
    )

    return protocol_md, proposals_md, status, "", "Episode started. Use tools to audit."


def execute_action(action_type, proposal_id, patient_id, feature, reason):
    """Execute one oversight action."""
    global _env, _obs, _history, _step_count

    if _env is None:
        return "⚠️ Reset the environment first.", "", ""
    if _obs and _obs.done:
        return "Episode complete. Reset to start a new one.", "", ""

    try:
        at = ActionType(action_type)
        action = SynthAuditAction(
            action_type=at,
            proposal_id=proposal_id or None,
            patient_id=patient_id or None,
            feature=feature or None,
            reason=reason or None,
            error_type="age_boundary_error" if at == ActionType.flag_error else None,
            report=reason if at == ActionType.submit_audit_report else None,
        )
        _obs = _env.step(action)
        _step_count += 1
    except Exception as e:
        return f"❌ Error: {e}", "", ""

    _history.append({
        "step": _step_count,
        "action": action_type,
        "reward": _obs.reward,
        "score": _obs.score_so_far,
    })

    # Format feedback
    feedback = _obs.feedback

    # Status bar
    status = (
        f"**Step**: {_obs.steps_taken}/{_obs.steps_taken + _obs.steps_remaining} | "
        f"**Score**: {_obs.score_so_far:.3f} | "
        f"**Reward**: {_obs.reward:+.3f} | "
        f"**Flags**: {_obs.correct_flags}✓ {_obs.false_positives}✗ | "
        f"**Approvals**: {_obs.correct_approvals}✓ | "
        f"{'🔴 DONE' if _obs.done else '🟢 LIVE'}"
    )

    # History table
    hist_md = "## Action History\n\n"
    hist_md += "| Step | Action | Reward | Score |\n|---|---|---|---|\n"
    for h in _history[-15:]:
        emoji = "✅" if h["reward"] > 0 else "❌" if h["reward"] < 0 else "➖"
        hist_md += f"| {h['step']} | {h['action']} | {emoji} {h['reward']:+.3f} | {h['score']:.3f} |\n"

    return feedback, status, hist_md


def run_full_heuristic(difficulty, seed):
    """Run the smart heuristic agent automatically."""
    global _env, _obs, _history, _step_count

    task_map = {"Easy": "oversight_easy", "Medium": "oversight_medium", "Hard": "oversight_hard"}
    task_id = task_map.get(difficulty, "oversight_medium")

    _env = SynthAuditEnvironment()
    _obs = _env.reset(seed=int(seed), task_id=task_id)
    _history = []
    _step_count = 0

    proposals = _obs.actor_proposals

    # Phase 1: Review
    for p in proposals:
        if _obs.done:
            break
        _obs = _env.step(SynthAuditAction(action_type=ActionType.review_proposal, proposal_id=p.proposal_id))
        _step_count += 1
        _history.append({"step": _step_count, "action": "review", "reward": _obs.reward, "score": _obs.score_so_far})

    # Phase 2: Investigate
    for p in proposals:
        if _obs.done:
            break
        _obs = _env.step(SynthAuditAction(action_type=ActionType.investigate_patient, patient_id=p.patient_id))
        _step_count += 1
        _history.append({"step": _step_count, "action": "investigate", "reward": _obs.reward, "score": _obs.score_so_far})

    # Phase 3: Temporal audit
    for p in proposals:
        if _obs.done:
            break
        _obs = _env.step(SynthAuditAction(action_type=ActionType.temporal_audit, patient_id=p.patient_id))
        _step_count += 1
        _history.append({"step": _step_count, "action": "temporal", "reward": _obs.reward, "score": _obs.score_so_far})

    # Phase 4: SHAP
    for p in proposals:
        if _obs.done:
            break
        _obs = _env.step(SynthAuditAction(action_type=ActionType.request_shap, patient_id=p.patient_id, feature="age"))
        _step_count += 1
        _history.append({"step": _step_count, "action": "shap", "reward": _obs.reward, "score": _obs.score_so_far})

    # Phase 5: Flag/Approve
    for p in proposals:
        if _obs.done:
            break
        if p.confidence < 0.85:
            _obs = _env.step(SynthAuditAction(action_type=ActionType.flag_error, proposal_id=p.proposal_id, error_type="age_boundary_error", reason="Low confidence indicates Actor error"))
            _history.append({"step": _step_count + 1, "action": "flag", "reward": _obs.reward, "score": _obs.score_so_far})
        else:
            _obs = _env.step(SynthAuditAction(action_type=ActionType.approve, proposal_id=p.proposal_id))
            _history.append({"step": _step_count + 1, "action": "approve", "reward": _obs.reward, "score": _obs.score_so_far})
        _step_count += 1

    # Submit report
    if not _obs.done:
        _obs = _env.step(SynthAuditAction(action_type=ActionType.submit_audit_report, report="Heuristic audit. Flagged low-confidence for age/temporal/protocol errors."))
        _step_count += 1
        _history.append({"step": _step_count, "action": "report", "reward": _obs.reward, "score": _obs.score_so_far})

    status = (
        f"**Step**: {_obs.steps_taken} | **Final Score**: {_obs.score_so_far:.3f} | "
        f"**Flags**: {_obs.correct_flags}✓ {_obs.false_positives}✗ | "
        f"**Approvals**: {_obs.correct_approvals}✓ | 🔴 DONE"
    )

    hist_md = "## Heuristic Agent Run\n\n"
    hist_md += "| Step | Action | Reward | Score |\n|---|---|---|---|\n"
    for h in _history:
        emoji = "✅" if h["reward"] > 0 else "❌" if h["reward"] < 0 else "➖"
        hist_md += f"| {h['step']} | {h['action']} | {emoji} {h['reward']:+.3f} | {h['score']:.3f} |\n"

    return _obs.feedback, status, hist_md


# ═══════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════

DESCRIPTION = """
# 🩺 SynthAudit.Env — Multi-Agent Clinical AI Oversight

**Theme**: Fleet AI — Scalable Oversight | **Meta PyTorch OpenEnv Hackathon × Scaler SST**

An AI Actor reviews clinical trial patients and proposes diagnoses.
Some contain **subtle errors** — hallucinated protocol amendments, temporal paradoxes,
and 2-hop reasoning failures. Your job: audit the AI.

**8 Oversight Tools**: review, investigate, SHAP, cohort analysis, temporal audit,
flag, approve, submit report.
"""

with gr.Blocks(
    title="SynthAudit.Env — Clinical AI Oversight",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        difficulty = gr.Dropdown(["Easy", "Medium", "Hard"], value="Easy", label="Difficulty")
        seed = gr.Number(value=42, label="Seed", precision=0)
        reset_btn = gr.Button("🔄 Reset Episode", variant="primary")
        auto_btn = gr.Button("🤖 Run Heuristic Agent", variant="secondary")

    with gr.Row():
        status_bar = gr.Markdown("*Click Reset to start.*")

    with gr.Row():
        with gr.Column(scale=1):
            protocol_box = gr.Markdown("## Protocol\n*Reset to load.*")
            proposals_box = gr.Markdown("## Proposals\n*Reset to load.*")

        with gr.Column(scale=1):
            gr.Markdown("### Execute Action")
            action_type = gr.Dropdown(
                [a.value for a in ActionType],
                value="review_proposal", label="Action Type"
            )
            proposal_id = gr.Textbox(label="Proposal ID (e.g. PROP-001)", placeholder="PROP-001")
            patient_id = gr.Textbox(label="Patient ID (e.g. P0001)", placeholder="P0001")
            feature = gr.Textbox(label="Feature (for SHAP/cohort)", placeholder="age")
            reason = gr.Textbox(label="Reason / Report text", placeholder="Explain your reasoning...")
            exec_btn = gr.Button("▶ Execute", variant="primary")

    with gr.Row():
        feedback_box = gr.Textbox(label="Environment Feedback", lines=8, interactive=False)

    with gr.Row():
        history_box = gr.Markdown("## Action History\n*No actions yet.*")

    # Bind events
    reset_btn.click(
        reset_environment, [difficulty, seed],
        [protocol_box, proposals_box, status_bar, feedback_box, history_box]
    )
    exec_btn.click(
        execute_action, [action_type, proposal_id, patient_id, feature, reason],
        [feedback_box, status_bar, history_box]
    )
    auto_btn.click(
        run_full_heuristic, [difficulty, seed],
        [feedback_box, status_bar, history_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
