from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from tvastr.ir import OpSpec
from tvastr.generators.triton_gen import _pick_default_config


def generate_run(spec: OpSpec, template_dir: Path, out_dir: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("run.py.j2")
    rendered = template.render(
        spec=spec,
        default_config=_pick_default_config(spec),
    )

    with (out_dir / "run.py").open("w") as f:
        f.write(rendered)