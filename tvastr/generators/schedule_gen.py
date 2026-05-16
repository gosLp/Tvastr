import itertools
import json
from pathlib import Path

from tvastr.ir import OpSpec

def generate_schedules(spec: OpSpec, out_dir: Path) -> list[dict]:
    params = spec.schedule_space.parameters

    if not params:
        raise ValueError(f"{spec.name}: schedule_space.parameters is empty")
    
    keys = list(params.keys())
    values = [params[k] for k in keys]


    schedules = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    out_path = out_dir / "schedules.json"

    with out_path.open("w") as f:
        json.dump(schedules, f, indent=2)
    
    return schedules
