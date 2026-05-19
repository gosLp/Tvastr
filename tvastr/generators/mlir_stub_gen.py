from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from tvastr.ir import OpSpec


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def generate_mlir_stub(spec: OpSpec, template_dir: Path, out_dir: Path) -> None:
    mlir_dir = out_dir / "mlir_stub"
    mlir_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    class_name = _class_name(spec.name)

    example = env.get_template("mlir_example.mlir.j2").render(
        spec=spec,
        class_name=class_name,
    )

    ops_td = env.get_template("mlir_ops.td.j2").render(
        spec=spec,
        class_name=class_name,
    )

    if not ops_td.strip():
        raise RuntimeError("Generated MLIR ODS file is empty")

    with (mlir_dir / f"{spec.name}.mlir").open("w") as f:
        f.write(example)

    with (mlir_dir / f"{class_name}Ops.td").open("w") as f:
        f.write(ops_td)