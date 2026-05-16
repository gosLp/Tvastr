from tvastr.parser import parse_spec
from tvastr.validate import validate_spec


def test_validate_matmul_bias_relu():
    spec = parse_spec("specs/matmul_bias_relu.yaml")
    validate_spec(spec)