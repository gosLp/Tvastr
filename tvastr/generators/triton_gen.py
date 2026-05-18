from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from tvastr.ir import OpSpec

def _pick_default_config(spec: OpSpec)  -> dict:
    params = spec.schedule_space.parameters

    def pick(name: str, preferred, fallback_index: int = 0):
        values = params.get(name, [])

        if preferred in values:
            return preferred
        if values:
            return values[fallback_index]
        raise ValueError(f"{spec.name}: missing schedule parameter: {name}")
    
    return {
        "BLOCK_M": pick("BLOCK_M", 128),
        "BLOCK_N": pick("BLOCK_N", 128),
        "BLOCK_K": pick("BLOCK_K", 32),
        "num_warps": pick("num_warps", 4),
        "num_stages": pick("num_stages", 2),
    }

def generate_triton(spec: OpSpec, template_dir: Path, out_dir: Path) -> None:
    if spec.lowering.backend != "triton":
        return

    env = Environment(
        loader = FileSystemLoader(str(template_dir)),
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