"""
OpenEnv Compatibility Shim
===========================
Minimal Environment ABC that mirrors the openenv-core interface.
Used for local dev on Python 3.9. In Docker/Colab (Python 3.10+),
the real openenv-core takes over automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """Minimal OpenEnv Environment base class."""

    @abstractmethod
    def reset(self, **kwargs) -> Any:
        ...

    @abstractmethod
    def step(self, action: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def state(self) -> Any:
        ...


def create_app(env_factory, action_type, observation_type, max_concurrent_envs=1):
    """Create a FastAPI app wrapping the environment."""
    from fastapi import FastAPI
    import json

    app = FastAPI(title="SynthAudit.Env")
    envs = {}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/reset")
    async def reset_env(body: dict = {}):
        env = env_factory()
        eid = id(env)
        envs[eid] = env
        obs = env.reset(**body)
        return {"env_id": eid, "observation": obs.dict() if hasattr(obs, 'dict') else obs.model_dump()}

    @app.post("/step/{env_id}")
    async def step_env(env_id: int, action: dict):
        env = envs.get(env_id)
        if not env:
            return {"error": "env not found"}
        act = action_type(**action)
        obs = env.step(act)
        return {"observation": obs.dict() if hasattr(obs, 'dict') else obs.model_dump()}

    @app.get("/state/{env_id}")
    async def get_state(env_id: int):
        env = envs.get(env_id)
        if not env:
            return {"error": "env not found"}
        s = env.state()
        return {"state": s.dict() if hasattr(s, 'dict') else s.model_dump()}

    return app
