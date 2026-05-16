from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from tvastr.ir import OpSpec

def generate_reference(spec: OpSpec, template_dir: Path, out_dir: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("reference.py.j2")
    rendered = template.render(spec=spec)

    with (out_dir / "reference.py").open("w") as f:
        f.write(rendered)