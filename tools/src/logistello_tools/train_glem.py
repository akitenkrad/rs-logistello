"""Phase 7: GLEM feature generation + training (design doc §4.3.6).

Reads a Rust-extracted ``GLX1`` corpus (Rust owns the base-literal
extraction), runs the §4.3.6 ``GenerateFeatures`` procedure with the B3
linear least-squares fit (reused Phase-4b GD-300 / muting / adjacent-5
smoothing), and writes the cross-language ``GLM1`` model file
(byte-identical to the Rust ``logistello_eval::GlemModel``; see
``crates/logistello-eval/GLEM_FORMAT.md``). Weights are in 1/128-disc units
(the leaf score is ``round(sum/128)`` with Edax bias rounding + ±63 clamp).

Usage::

    uv run logistello-tools train-glem \\
        --base-features cell64,mobility,corner --max-order 3 \\
        --support-threshold 0.001 [--weight-threshold W] \\
        [--method gd|ridge|sgd|logistic] --positions FILE.glx1 \\
        --output FILE.glm1
"""
from __future__ import annotations

import argparse

from logistello_tools.glem import generate_and_fit
from logistello_tools.glx1 import FAMILY_NAME_BY_WIRE, load_glx1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="logistello-tools train-glem",
        description=(
            "Phase 7: GLEM feature generation + B3 linear fit "
            "(design doc §4.3.6) -> GLM1"
        ),
    )
    parser.add_argument(
        "--base-features",
        default="cell64,mobility,corner",
        help=(
            "comma list of base-feature families "
            "(cell64,mobility,corner); must match the GLX1 corpus spec"
        ),
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=3,
        help="max conjunction order K (§6: 1,2,3,4); 1 = plain linear",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.001,
        help="τ_support: min training frequency to keep a generated "
        "conjunction (§6: 1e-4 .. 1e-2)",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.0,
        help="τ_weight: prune features whose max |weight| (disc units) is "
        "below this; 0 keeps all (default)",
    )
    parser.add_argument(
        "--method",
        choices=["gd", "ridge", "sgd", "logistic"],
        default="gd",
        help=(
            "gd = faithful B3 GD-300 + muting (default); ridge/sgd = "
            "scikit-learn least-squares (§7); logistic = optional "
            "Logistello-1 path (Objective-5 only)"
        ),
    )
    parser.add_argument(
        "--positions", required=True, help="input GLX1 extract file"
    )
    parser.add_argument(
        "--output", required=True, help="output GLM1 model file"
    )
    args = parser.parse_args(argv)

    ex = load_glx1(args.positions)
    want = [s.strip() for s in args.base_features.split(",") if s.strip()]
    have = ex.family_names
    if want != have:
        raise SystemExit(
            f"--base-features {want} disagrees with the GLX1 corpus spec "
            f"{have} (regenerate the corpus with `glem-extract "
            f"--base-features {','.join(have)}` or pass that spec)"
        )

    model, report = generate_and_fit(
        ex,
        max_order=args.max_order,
        support_threshold=args.support_threshold,
        weight_threshold=args.weight_threshold,
        method=args.method,
    )
    model.save(args.output)
    fam = ",".join(FAMILY_NAME_BY_WIRE[w] for w in ex.family_ids)
    print(
        f"train-glem base-features={fam} max-order={args.max_order} "
        f"support>={args.support_threshold} weight>={args.weight_threshold} "
        f"method={args.method} positions={ex.n_records}"
    )
    print(
        f"  #base={report.n_base} #generated={report.n_generated} "
        f"#after-support={report.n_after_support} "
        f"#after-weight-prune={report.n_after_weight_prune} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
