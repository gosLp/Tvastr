from pathlib import Path

from tvastr.parser import parse_spec
from tvastr.generators.schedule_gen import generate_schedules


def test_generate_constrained_ranked_schedules(tmp_path: Path):
    spec = parse_spec("specs/matmul_bias_relu.yaml")
    schedules = generate_schedules(spec, tmp_path)

    assert len(schedules) == 72

    assert (tmp_path / "schedules.json").exists()
    assert (tmp_path / "schedule_summary.json").exists()
    assert (tmp_path / "schedule_rank_report.json").exists()

    for cfg in schedules:
        area = cfg["BLOCK_M"] * cfg["BLOCK_N"]
        assert area >= 4096
        assert area <= 16384
        assert cfg["BLOCK_K"] in {32, 64}

    first = schedules[0]
    assert first["BLOCK_M"] * first["BLOCK_N"] >= 4096