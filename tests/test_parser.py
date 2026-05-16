from tvastr.parser import parse_spec

def test_parse_matmul_bias_relu():
    spec = parse_spec("specs/matmul_bias_relu.yaml")

    assert spec.name == "matmul_bias_relu"
    assert spec.kind == "tensor_op"
    assert len(spec.inputs) == 3
    assert len(spec.outputs) == 1

    names = [x.name for x in spec.inputs]
    assert names == ["A", "B", "bias"]

    assert spec.outputs[0].name == "C"
    assert spec.lowering.backend == "triton"
    assert "BLOCK_M" in spec.schedule_space.parameters