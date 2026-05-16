from tvastr.ir import OpSpec

def validate_spec(spec: OpSpec) -> None:
    if not spec.name:
        raise ValueError("Spec must have a name")

    if not spec.inputs:
        raise ValueError(f"{spec.name}: spec must have at least one input")

    if not spec.outputs:
        raise ValueError(f"{spec.name}: spec must have at least one output")

    if spec.lowering.backend not in {"triton", "mlir_stub"}:
        raise ValueError(
            f"{spec.name}: unsupported backend {spec.lowering.backend}"
        )

    if not spec.schedule_space.parameters:
        raise ValueError(f"{spec.name}: schedule_space.parameters is empty")

    required = {"BLOCK_M", "BLOCK_N", "BLOCK_K"}
    params = set(spec.schedule_space.parameters.keys())
    missing = required - params

    if spec.name.startswith("matmul") and missing:
        raise ValueError(
            f"{spec.name}: matmul-like ops require schedule params {missing}"
        )