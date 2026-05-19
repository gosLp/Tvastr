from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from tvastr.ir import OpSpec
from tvastr.generators.schedule_gen import enumerate_schedules


def _pick_default_config(spec: OpSpec) -> dict:
    _raw, ranked, _report = enumerate_schedules(spec)

    if not ranked:
        raise ValueError(f"{spec.name}: no valid schedules after constraints")

    best = ranked[0]

    return {
        "BLOCK_M": int(best["BLOCK_M"]),
        "BLOCK_N": int(best["BLOCK_N"]),
        "BLOCK_K": int(best["BLOCK_K"]),
        "num_warps": int(best["num_warps"]),
        "num_stages": int(best["num_stages"]),
    }


def generate_triton(spec: OpSpec, template_dir: Path, out_dir: Path) -> None:
    if spec.lowering.backend != "triton":
        return

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(spec.lowering.template)
    rendered = template.render(
        spec=spec,
        default_config=_pick_default_config(spec),
    )

    with (out_dir / "kernel_triton.py").open("w") as f:
        f.write(rendered)