import yaml
from pathlib import Path

from tvastr.ir import (
    TensorSpec,
    ScheduleSpace,
    LoweringSpec,
    TestingSpec,
    RewriteSpec,
    OpSpec,
)


def _parse_tensor(x: dict) -> TensorSpec:
    return TensorSpec(
        name=x["name"],
        type=x.get("type", "tensor"),
        shape=x["shape"],
        dtype=x["dtype"],
        memory=x.get("memory", "global"),
        layout=x.get("layout", "row_major"),
    )


def parse_spec(path: str | Path) -> OpSpec:
    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    inputs = [_parse_tensor(x) for x in data.get("inputs", [])]
    outputs = [_parse_tensor(x) for x in data.get("outputs", [])]

    schedule_data = data.get("schedule_space", {})

    rewrites = [
        RewriteSpec(
            name=x["name"],
            from_ops=x.get("from", []),
            to=x["to"],
            legal_if=x.get("legal_if", []),
        )
        for x in data.get("rewrites", [])
    ]

    return OpSpec(
        name=data["name"],
        kind=data.get("kind", "tensor_op"),
        version=str(data.get("version", "0.1")),
        description=data.get("description", ""),
        inputs=inputs,
        outputs=outputs,
        semantics=data.get("semantics", {}),
        constraints=data.get("constraints", {}),
        performance_contract=data.get("performance_contract", {}),
        schedule_space=ScheduleSpace(
            parameters=schedule_data.get("parameters", {}),
            constraints=schedule_data.get("constraints", []),
        ),
        lowering=LoweringSpec(
            backend=data["lowering"]["backend"],
            template=data["lowering"]["template"],
        ),
        testing=TestingSpec(
            reference=data.get("testing", {}).get("reference", "torch"),
            atol=float(data.get("testing", {}).get("atol", 1e-2)),
            rtol=float(data.get("testing", {}).get("rtol", 1e-2)),
            shapes=data.get("testing", {}).get("shapes", []),
        ),
        rewrites=rewrites,
        layouts=data.get("layouts", {}),
        layout_contract=data.get("layout_contract", {}),
        layout_transforms=data.get("layout_transforms", []),
    )