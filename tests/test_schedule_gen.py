from pathlib import Path

from tvastr.parser import parse_spec
from tvastr.generators.schedule_gen import generate_schedules


def test_generate_schedules(tmp_path: Path):
    spec = parse_spec("specs/matmul_bias_relu.yaml")
    schedules = generate_schedules(spec, tmp_path)

    assert len(schedules) == 108
    assert (tmp_path / "schedules.json").exists()

    first = schedules[0]
    assert "BLOCK_M" in first
    assert "BLOCK_N" in first
    assert "BLOCK_K" in first
    assert "num_warps" in first
    assert "num_stages" in first