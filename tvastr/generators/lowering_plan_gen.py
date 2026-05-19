import json
from pathlib import Path


from tvastr.ir import OpSpec
from tvastr.generators.schedule_gen import enumerate_schedules


def generate_lowering_plan(spec: OpSpec, out_dir: Path) -> None:
    raw_schedules, ranked_schedules, ranked_report = enumerate_schedules(spec)

    required_checks = ["shape", "dtype"]

    if spec.layout_contract:
        required_checks.append("layout")
    
    if spec.rewrites:
        required_checks.append("rewrites")


    lowering_steps = [
        "parse_contract",
        "validate_contract",
        "verify_inputs",
    ]

    if spec.layout_contract:
        lowering_steps.append("verify_layouts")
    
    lowering_steps.extend(
        [
            "compute_reference",
            f"launch_{spec.lowering.backend}",
            "compare_outputs",
            "benchmark",
        ]
    )

    artifacts = [
        "contract.json",
        "schedules.json",
        "schedule_summary.json",
        "schedule_rank_report.json",
        "reference.py",
        "verifier.py",
        "kernel_triton.py",
        "run.py",
        "autotune.py",
    ]


    if spec.layout_contract:
        artifacts.extend(
            [
                "layout_guards.py",
                "layout_report.json",
            ]
        )
    
    if spec.rewrites:
        artifacts.append("rewrite_guards.py")
    
    artifacts.append("mlir_stub.py")

    plan = {
        "op": spec.name,
        "kind": spec.kind,
        "version": spec.version,
        "backend": spec.lowering.backend,
        "semantic_contract": spec.semantics,
        "performance_contract": spec.performance_contract,
        "required_checks": required_checks,
        "schedule_policy": {
            "raw_schedules": len(raw_schedules),
            "kept_schedules": len(ranked_schedules),
            "pruned_schedules": len(raw_schedules) - len(ranked_schedules),
            "ranked": True,
            "constraints": spec.schedule_space.constraints,
            "top_config": ranked_schedules[0] if ranked_schedules else None,
            "top_rank_explanation": ranked_report[0] if ranked_report else None,
        },
        "rewrite_policy": [
            {
                "name": rw.name,
                "from": rw.from_ops,
                "to": rw.to,
                "legal_if": rw.legal_if,
            }
            for rw in spec.rewrites
        ],
        "layout_policy": {
            "layouts": spec.layouts,
            "layout_contract": spec.layout_contract,
            "layout_transforms": spec.layout_transforms,
        },
        "lowering_steps": lowering_steps,
        "artifacts": artifacts,
        "note": (
            "This lowering plan is an intermediate compiler-style artifact. "
            "Backend templates are implementation details attached to this plan."
        ),
    }

    with open(out_dir / "lowering_plan.json", "w") as f:
        json.dump(plan, f, indent=2)