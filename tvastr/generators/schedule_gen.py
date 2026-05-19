import ast
import itertools
import json
import math
from pathlib import Path
from typing import Any

from tvastr.ir import OpSpec


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)


def _safe_eval(expr: str, env: dict[str, Any]) -> bool:
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"Unsupported expression in schedule constraint: {expr}"
            )

        if isinstance(node, ast.Name) and node.id not in env:
            raise ValueError(
                f"Unknown name {node.id!r} in schedule constraint: {expr}"
            )

    return bool(
        eval(
            compile(tree, "<schedule-constraint>", "eval"),
            {"__builtins__": {}},
            env,
        )
    )


def _passes_constraints(config: dict[str, Any], constraints: list[str]) -> bool:
    return all(_safe_eval(expr, config) for expr in constraints)


def _tile_area(config: dict[str, Any]) -> int:
    return int(config["BLOCK_M"]) * int(config["BLOCK_N"])


def _rank_score(config: dict[str, Any]) -> float:
    """
    Lower is better.

    Current simple heuristic:
    - prefer tile area in [8192, 16384]
    - prefer BLOCK_K in {32, 64}
    - prefer 4 warps for smaller tiles
    - prefer 8 warps for larger tiles
    - mildly prefer balanced M/N tiles
    - mildly prefer num_stages 3 or 4 over 5
    """

    bm = int(config["BLOCK_M"])
    bn = int(config["BLOCK_N"])
    bk = int(config["BLOCK_K"])
    warps = int(config["num_warps"])
    stages = int(config["num_stages"])

    area = bm * bn
    score = 0.0

    if 8192 <= area <= 16384:
        score += 0.0
    elif area < 8192:
        score += (8192 - area) / 1024.0
    else:
        score += (area - 16384) / 1024.0

    if bk not in {32, 64}:
        score += 10.0

    expected_warps = 4 if area <= 4096 else 8
    if warps != expected_warps:
        score += 1.0

    if stages == 5:
        score += 0.5
    elif stages not in {3, 4}:
        score += 1.0

    # Penalize very skinny tiles, but do not forbid them.
    aspect = max(bm, bn) / min(bm, bn)
    score += 0.25 * math.log2(aspect)

    # Stable tie-breakers:
    # prefer larger area, then larger BK.
    score -= area / 100000.0
    score -= bk / 1000000.0

    return score


def _rank_reason(config: dict[str, Any]) -> list[str]:
    area = _tile_area(config)
    warps = int(config["num_warps"])

    reasons = []

    if 8192 <= area <= 16384:
        reasons.append("preferred_tile_area")
    elif area == 4096:
        reasons.append("acceptable_tile_area")
    else:
        reasons.append("nonpreferred_tile_area")

    if area <= 4096 and warps == 4:
        reasons.append("warps_match_small_tile")
    elif area > 4096 and warps == 8:
        reasons.append("warps_match_large_tile")
    else:
        reasons.append("warps_not_preferred_for_tile_area")

    if int(config["BLOCK_K"]) in {32, 64}:
        reasons.append("preferred_BLOCK_K")

    return reasons


def enumerate_schedules(spec: OpSpec) -> tuple[list[dict], list[dict], list[dict]]:
    params = spec.schedule_space.parameters

    if not params:
        raise ValueError(f"{spec.name}: schedule_space.parameters is empty")

    keys = list(params.keys())
    values = [params[k] for k in keys]

    raw_schedules = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    constraints = spec.schedule_space.constraints

    kept = [
        cfg
        for cfg in raw_schedules
        if _passes_constraints(cfg, constraints)
    ]

    ranked = sorted(kept, key=_rank_score)

    ranked_report = [
        {
            "rank": i + 1,
            "score": _rank_score(cfg),
            "tile_area": _tile_area(cfg),
            "reasons": _rank_reason(cfg),
            "config": cfg,
        }
        for i, cfg in enumerate(ranked)
    ]

    return raw_schedules, ranked, ranked_report


def generate_schedules(spec: OpSpec, out_dir: Path) -> list[dict]:
    raw_schedules, ranked_schedules, ranked_report = enumerate_schedules(spec)

    schedules_path = out_dir / "schedules.json"
    with schedules_path.open("w") as f:
        json.dump(ranked_schedules, f, indent=2)

    rank_report_path = out_dir / "schedule_rank_report.json"
    with rank_report_path.open("w") as f:
        json.dump(ranked_report, f, indent=2)

    summary = {
        "op": spec.name,
        "raw_schedules": len(raw_schedules),
        "kept_schedules": len(ranked_schedules),
        "pruned_schedules": len(raw_schedules) - len(ranked_schedules),
        "constraints": spec.schedule_space.constraints,
        "ranking_heuristics": [
            "prefer tile area in [8192, 16384]",
            "prefer BLOCK_K in {32, 64}",
            "prefer 4 warps for smaller tiles",
            "prefer 8 warps for larger tiles",
            "mildly prefer balanced BLOCK_M/BLOCK_N",
            "mildly prefer num_stages 3 or 4",
        ],
        "top_10": ranked_report[:10],
    }

    summary_path = out_dir / "schedule_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return ranked_schedules