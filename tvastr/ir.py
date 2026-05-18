from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list[str] = field(default_factory=list)
    dtype: str = "f32"
    type: str = "tensor"
    memory: str = "global"
    layout: str = "row_major"


@dataclass(frozen=True)
class ScheduleSpace:
    parameters: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class LoweringSpec:
    backend: str
    template: str


@dataclass(frozen=True)
class TestingSpec:
    reference: str = "torch"
    atol: float = 1e-2
    rtol: float = 1e-2
    shapes: list[dict[str, int]] = field(default_factory=list)


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    version: str
    description: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    semantics: dict[str, Any]
    constraints: dict[str, list[str]]
    performance_contract: dict[str, Any]
    schedule_space: ScheduleSpace
    lowering: LoweringSpec
    testing: TestingSpec