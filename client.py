"""
SynthAudit.Env — EnvClient
"""

from openenv.core.env_client import EnvClient
from .models import SynthAuditAction, SynthAuditObservation


class SynthAuditEnv(EnvClient[SynthAuditAction, SynthAuditObservation]):
    ACTION_TYPE = SynthAuditAction
    OBSERVATION_TYPE = SynthAuditObservation
