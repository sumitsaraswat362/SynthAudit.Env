"""
SynthAudit.Env — FastAPI Server
"""

import sys
import os

_server_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_server_dir)
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

try:
    from openenv.core.env_server import create_app
except (ImportError, TypeError):
    from openenv_compat import create_app

from synth_audit_environment import SynthAuditEnvironment
from models import SynthAuditAction, SynthAuditObservation


app = create_app(
    lambda: SynthAuditEnvironment(),
    SynthAuditAction,
    SynthAuditObservation,
    max_concurrent_envs=64,
)


def main():
    """Entry point for OpenEnv server deployment."""
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
