from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_EXTRACTION_STAGE_ORDER = (
    'build_prompt',
    'llm_call',
    'parse_response',
    'validate_extraction_with_schema',
    'filtering',
    'postprocessing',
)


@dataclass
class TemporalWorkflowPayload:
    """Serializable payload for the registry-driven Temporal workflow."""

    pipeline_type: str
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    ctx_args: dict[str, Any] = field(default_factory=dict)
    stage_order: list[str] = field(default_factory=list)
    stage_timeout_seconds: int = 30

    def to_payload(self) -> dict[str, Any]:
        return {
            'pipeline_type': self.pipeline_type,
            'pipeline_config': self.pipeline_config,
            'ctx_args': self.ctx_args,
            'stage_order': self.stage_order,
            'stage_timeout_seconds': self.stage_timeout_seconds,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TemporalWorkflowPayload:
        return cls(
            pipeline_type=payload['pipeline_type'],
            pipeline_config=dict(payload.get('pipeline_config', {})),
            ctx_args=dict(payload.get('ctx_args', {})),
            stage_order=list(payload.get('stage_order', [])),
            stage_timeout_seconds=int(payload.get('stage_timeout_seconds', 30)),
        )


@dataclass
class StageActivityPayload:
    """Serializable payload exchanged between workflow and stage activity."""

    pipeline_type: str
    pipeline_config: dict[str, Any]
    stage_name: str
    state: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            'pipeline_type': self.pipeline_type,
            'pipeline_config': self.pipeline_config,
            'stage_name': self.stage_name,
            'state': self.state,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StageActivityPayload:
        return cls(
            pipeline_type=payload['pipeline_type'],
            pipeline_config=dict(payload.get('pipeline_config', {})),
            stage_name=payload['stage_name'],
            state=dict(payload.get('state', {})),
        )


@dataclass
class StageActivityResult:
    """Serializable result returned by the stage activity."""

    state: dict[str, Any]
    stage_result: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            'state': self.state,
            'stage_result': self.stage_result,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StageActivityResult:
        return cls(
            state=dict(payload.get('state', {})),
            stage_result=dict(payload.get('stage_result', {})),
        )
