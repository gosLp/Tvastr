import yaml
from pathlib import Path

from tvastr.ir import (
    TensorSpec,
    ScheduleSpace,
    LoweringSpec,
    TestingSpec,
    OpSpec
)

def parse_spec(path: str | Path) -> OpSpec:
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    
    inputs = [
        TensorSpec(
            name=x["name"],
            type=x.get("type", "tensor"),
            shape=x["shape"],
            dtype=x["dtype"],
            memory=x.get("memory", "global"),
            layout=x.get("layout", "row_major"),
        )
        for x in data.get("inputs", [])
    ]

    outputs = [
        TensorSpec(
            name=x["name"],
            type=x.get("type", "tensor"),
            shape=x["shape"],
            dtype=x["dtype"],
            memory=x.get("memory", "global"),
            layout=x.get("layout", "row_major"),
        )
        for x in data.get("outputs", [])
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
            parameters=data.get("schedule_space", {}).get("parameters", {})
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
    )
    