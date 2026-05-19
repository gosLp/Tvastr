from tvastr.parser import parse_spec


def test_parse_rewrites():
    spec = parse_spec("specs/matmul_bias_relu.yaml")

    assert len(spec.rewrites) == 1

    rw = spec.rewrites[0]
    assert rw.name == "fuse_matmul_bias_relu"
    assert rw.from_ops == ["matmul", "bias_add", "relu"]
    assert rw.to == "matmul_bias_relu"
    assert len(rw.legal_if) > 0