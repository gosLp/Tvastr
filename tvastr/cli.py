from pathlib import Path
import argparse
from dataclasses import asdict
import json


from tvastr.parser import parse_spec
from tvastr.validate import validate_spec
from tvastr.generators.schedule_gen import generate_schedules
from tvastr.generators.reference_gen import generate_reference
from tvastr.generators.verifier_gen import generate_verifier
from tvastr.generators.triton_gen import generate_triton
from tvastr.generators.run_gen import generate_run
from tvastr.generators.autotune_gen import generate_autotune
from tvastr.generators.mlir_stub_gen import generate_mlir_stub
from tvastr.generators.rewrite_guards_gen import generate_rewrite_guards
from tvastr.generators.mlir_stub_gen import generate_mlir_stub
from tvastr.generators.layout_gen import generate_layout_artifacts
from tvastr.generators.lowering_plan_gen import generate_lowering_plan


def generate_cmd(args: argparse.Namespace) -> None: 

    spec = parse_spec(args.spec)
    validate_spec(spec)

    out_dir = Path(args.out) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "contract.json").open("w") as f:
        json.dump(asdict(spec), f, indent=2)
    
    template_dir = Path(__file__).parent / "templates"

    schedules = generate_schedules(spec, out_dir)

    generate_reference(spec, template_dir, out_dir)
    generate_verifier(spec, template_dir, out_dir)
    generate_triton(spec, template_dir, out_dir)
    generate_run(spec, template_dir, out_dir)
    generate_autotune(spec, template_dir, out_dir)
    generate_rewrite_guards(spec, template_dir, out_dir)
    generate_mlir_stub(spec, template_dir, out_dir)
    generate_layout_artifacts(spec, template_dir, out_dir)
    generate_lowering_plan(spec, out_dir)


    print(f"[ok] parsed: {spec.name}")
    print(f"[ok] backend: {spec.lowering.backend}")
    print(f"[ok] generated: {out_dir / 'contract.json'}")
    print(f"[ok] generated: {out_dir / 'schedules.json'}")
    print(f"[ok] generated: {out_dir / 'reference.py'}")
    print(f"[ok] generated: {out_dir / 'verifier.py'}")
    print(f"[ok] generated: {out_dir / 'kernel_triton.py'}")
    print(f"[ok] generated: {out_dir / 'run.py'}")
    print(f"[ok] schedules: {len(schedules)}")
    print(f"[ok] generated: {out_dir / 'autotune.py'}")
    print(f"[ok] generated: {out_dir / 'rewrite_guards.py'}")
    print(f"[ok] generated: {out_dir / 'mlir_stub.py'}")
    print(f"[ok] generated: {out_dir / 'schedule_summary.json'}")
    print(f"[ok] generated: {out_dir / 'schedule_rank_report.json'}")
    print(f"[ok] generated: {out_dir / 'layout_guards.py'}")
    print(f"[ok] generated: {out_dir / 'layout_report.json'}")
    print(f"[ok] generated: {out_dir / 'lowering_plan.json'}")

def main():
    parser = argparse.ArgumentParser(prog="tvastr")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("spec")
    gen.add_argument("--out", default="generated")
    gen.set_defaults(func=generate_cmd)

    args = parser.parse_args()
    args.func(args)