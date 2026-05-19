import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from tvastr.ir import OpSpec


def generate_layout_artifacts(
    spec: OpSpec,
    template_dir: Path,
    out_dir: Path,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("layout_guards.py.j2")
    rendered = template.render(spec=spec)

    with (out_dir / "layout_guards.py").open("w") as f:
        f.write(rendered)

    report = {
        "op": spec.name,
        "layouts": spec.layouts,
        "layout_contract": spec.layout_contract,
        "layout_transforms": spec.layout_transforms,
    }

    with (out_dir / "layout_report.json").open("w") as f:
        json.dump(report, f, indent=2)