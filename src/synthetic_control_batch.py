from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd

from src.synthetic_control import SyntheticControl  # noqa: E402


def benchmark_sweep(
    specs: List[Dict[str, Any]],
    methods: List[Dict[str, Any]],
    direction: str = "column",
    sc_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run multiple methods across multiple simulation specifications.

    For every (spec, method) pair a ``SyntheticControl`` object is
    constructed and ``evaluate_all_columns`` (or ``evaluate_all_rows``)
    is called.  The Pearson correlation (simple mean) is extracted and
    assembled into summary DataFrames.

    Args:
        specs: Each dict must contain:
            ``"name"`` (str): display name for this simulation spec.
            ``"real"`` (np.ndarray): real matrix (n_rows, n_cols).
            ``"synthetic"`` (np.ndarray): synthetic matrix, same shape.
            Optional ``"additional_baseline"`` (np.ndarray): same shape.
        methods: Each dict must contain:
            ``"label"`` (str): display name for the method.
            All remaining keys are forwarded as kwargs to
            ``evaluate_all_columns`` / ``evaluate_all_rows``
            (must include ``"method"``).
        direction: ``"column"`` or ``"row"``.
        sc_kwargs: Shared ``SyntheticControl.__init__`` keyword arguments
            (e.g. ``imputation_rank``, ``min_col_std``).
        n_jobs: Parallel jobs passed to every ``evaluate_all_*`` call.
        verbose: If True, forward verbose flag to the evaluation calls
            (produces per-spec plots and diagnostics).

    Returns:
        Dict with keys:
            ``"correlation_mean"``: DataFrame (rows = specs, cols =
            ``["Baseline"] + method labels``), Pearson correlation
            (simple mean across columns/rows).
            ``"correlation_se"``: DataFrame of standard errors, same
            shape.
            ``"formatted"``: DataFrame of ``"mean ± SE"`` strings.
            ``"raw_results"``: nested dict
            ``{spec_name: {method_label: evaluate_all result}}``.
    """
    if sc_kwargs is None:
        sc_kwargs = {}

    spec_names = [s["name"] for s in specs]
    method_labels = [m["label"] for m in methods]
    all_col_labels = ["Baseline"] + method_labels

    mean_data: Dict[str, List[float]] = {lbl: [] for lbl in all_col_labels}
    se_data: Dict[str, List[float]] = {lbl: [] for lbl in all_col_labels}
    raw_results: Dict[str, Dict[str, Any]] = {}

    total = len(specs) * len(methods)
    counter = 0

    for spec_idx, spec in enumerate(specs, 1):
        name = spec["name"]
        raw_results[name] = {}

        print(f"\n{'=' * 70}")
        print(f"Specification {spec_idx}/{len(specs)}: {name}")
        print(f"{'=' * 70}")

        sc_init: Dict[str, Any] = {
            "real_matrix": spec["real"],
            "synthetic_matrix": spec["synthetic"],
            "dataset_name": name,
        }
        if "additional_baseline" in spec:
            sc_init["additional_baseline_matrix"] = spec["additional_baseline"]
        sc_init.update(sc_kwargs)
        sc = SyntheticControl(**sc_init)

        evaluate_fn = (
            sc.evaluate_all_columns if direction == "column" else sc.evaluate_all_rows
        )

        baseline_corr: Optional[tuple] = None
        spec_t0 = time.time()

        for method_config in methods:
            counter += 1
            label = method_config["label"]
            eval_kwargs = {k: v for k, v in method_config.items() if k != "label"}
            eval_kwargs["n_jobs"] = n_jobs
            eval_kwargs["verbose"] = verbose

            t0 = time.time()
            result = evaluate_fn(**eval_kwargs)
            elapsed = time.time() - t0
            raw_results[name][label] = result

            corr_arr = np.array(
                [m["correlation"] for m in result["metrics"]], dtype=float
            )
            corr_mean = float(np.nanmean(corr_arr))
            n = int(np.sum(~np.isnan(corr_arr)))
            corr_se = (
                float(np.nanstd(corr_arr, ddof=1) / np.sqrt(n))
                if n > 1
                else float("nan")
            )

            print(f"  [{counter}/{total}] {label:<30s}  {elapsed:6.1f}s")

            mean_data[label].append(corr_mean)
            se_data[label].append(corr_se)

            if baseline_corr is None:
                base_corr_arr = np.array(
                    [m["correlation"] for m in result["baseline_metrics"]],
                    dtype=float,
                )
                b_mean = float(np.nanmean(base_corr_arr))
                b_n = int(np.sum(~np.isnan(base_corr_arr)))
                b_se = (
                    float(np.nanstd(base_corr_arr, ddof=1) / np.sqrt(b_n))
                    if b_n > 1
                    else float("nan")
                )
                baseline_corr = (b_mean, b_se)

        spec_elapsed = time.time() - spec_t0
        print(f"  {'─' * 40}")
        print(f"  Total for this specification: {spec_elapsed:.1f}s")

        mean_data["Baseline"].append(baseline_corr[0])
        se_data["Baseline"].append(baseline_corr[1])

    corr_mean_df = pd.DataFrame(mean_data, index=spec_names)
    corr_se_df = pd.DataFrame(se_data, index=spec_names)

    formatted_data: Dict[str, List[str]] = {}
    for col in all_col_labels:
        formatted_data[col] = [
            f"{mean_data[col][i]:.4f} ± {se_data[col][i]:.4f}"
            for i in range(len(spec_names))
        ]
    formatted_df = pd.DataFrame(formatted_data, index=spec_names)

    print("\n" + "=" * 80)
    print("Benchmark Results — Pearson correlation (mean ± SE)")
    print("=" * 80)
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(formatted_df.to_string())

    return {
        "correlation_mean": corr_mean_df,
        "correlation_se": corr_se_df,
        "formatted": formatted_df,
        "raw_results": raw_results,
    }
