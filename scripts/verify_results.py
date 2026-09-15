from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Benchmark:
    directory: str
    dataset: str
    descriptor: str
    method_column: str
    method: str
    expected_rows: int
    rotation_threshold: float
    translation_threshold: float


BENCHMARKS = (
    Benchmark(
        "3dlomatch_fpfh", "3DLoMatch", "FPFH", "method", "sagereg_regen",
        1781, 15.0, 0.30,
    ),
    Benchmark(
        "3dmatch_fpfh", "3DMatch", "FPFH", "method", "sagereg_regen",
        1623, 15.0, 0.30,
    ),
    Benchmark(
        "3dmatch_fcgf", "3DMatch", "FCGF", "method", "sagereg_regen",
        1623, 15.0, 0.30,
    ),
    Benchmark(
        "kitti_fpfh", "KITTI", "FPFH", "method", "sagereg_regen",
        555, 5.0, 0.60,
    ),
    Benchmark(
        "kitti_fcgf", "KITTI", "FCGF", "method", "sagereg_regen",
        555, 5.0, 0.60,
    ),
    Benchmark(
        "eth_ablation", "ETH-TLS", "FPFH", "method", "a5_regeneration",
        713, 15.0, 0.30,
    ),
)


@dataclass(frozen=True)
class Metrics:
    count: int
    recall: float
    median_rotation: float
    median_translation: float
    mean_runtime: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"unexpected Boolean value: {value!r}")


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def summarize(rows: list[dict[str, str]]) -> Metrics:
    rotations = [float(row["rotation_error_deg"]) for row in rows]
    translations = [float(row["translation_error"]) for row in rows]
    successes = [as_bool(row["success"]) for row in rows]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    return Metrics(
        count=len(rows),
        recall=sum(successes) / len(successes),
        median_rotation=statistics.median(rotations),
        median_translation=statistics.median(translations),
        mean_runtime=statistics.fmean(runtimes),
    )


def verify_checksums() -> None:
    checksum_path = RESULTS / "SHA256SUMS"
    expected: dict[Path, str] = {}
    results_root = RESULTS.resolve()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise AssertionError(
                f"SHA256SUMS:{line_number}: malformed checksum entry"
            ) from error
        path = (RESULTS / relative.removeprefix("./")).resolve()
        if not path.is_relative_to(results_root):
            raise AssertionError(f"SHA256SUMS:{line_number}: path leaves results/")
        if path in expected:
            raise AssertionError(f"SHA256SUMS:{line_number}: duplicate path {relative}")
        expected[path] = digest.lower()

    supplied = set(RESULTS.rglob("*.csv"))
    if set(expected) != supplied:
        missing = sorted(
            str(path.relative_to(RESULTS)) for path in supplied - set(expected)
        )
        stale = sorted(
            str(path.relative_to(RESULTS)) for path in set(expected) - supplied
        )
        raise AssertionError(
            f"SHA256SUMS coverage mismatch; unlisted={missing}, missing_files={stale}"
        )
    for path, digest in expected.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"checksum mismatch: {path.relative_to(RESULTS)}")


def verify_summary_row(row: dict[str, str], metrics: Metrics) -> None:
    close(float(row["registration_recall"]), metrics.recall)
    close(float(row["median_rotation_error_deg"]), metrics.median_rotation)
    close(float(row["median_translation_error"]), metrics.median_translation)
    close(float(row["mean_runtime_seconds"]), metrics.mean_runtime)
    count_column = (
        "evaluated_pairs" if "evaluated_pairs" in row else "evaluated_samples"
    )
    if count_column in row and row[count_column]:
        if int(row[count_column]) != metrics.count:
            raise AssertionError(
                f"summary count {row[count_column]} != record count {metrics.count}"
            )


def verify_benchmark(spec: Benchmark) -> Metrics:
    directory = RESULTS / spec.directory
    all_rows = read_rows(directory / "pairs.csv")
    rows = [row for row in all_rows if row[spec.method_column] == spec.method]
    if spec.directory != "eth_ablation" and len(rows) != len(all_rows):
        raise AssertionError(f"{spec.directory}: non-SAGE-Reg rows are present")
    if len(rows) != spec.expected_rows:
        raise AssertionError(
            f"{spec.directory}: expected {spec.expected_rows} rows, found {len(rows)}"
        )
    rotations = [float(row["rotation_error_deg"]) for row in rows]
    translations = [float(row["translation_error"]) for row in rows]
    recorded = [as_bool(row["success"]) for row in rows]
    recomputed = [
        rotation <= spec.rotation_threshold and translation <= spec.translation_threshold
        for rotation, translation in zip(rotations, translations)
    ]
    if recorded != recomputed:
        raise AssertionError(f"{spec.directory}: success flags do not match thresholds")

    metrics = summarize(rows)
    summary = [
        row
        for row in read_rows(directory / "summary.csv")
        if row[spec.method_column] == spec.method
    ]
    if len(summary) != 1:
        raise AssertionError(f"{spec.directory}: summary must contain one data row")
    verify_summary_row(summary[0], metrics)
    return metrics


def verify_grouped_summary(
    record_file: str,
    summary_file: str,
    group_columns: tuple[str, ...],
) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(RESULTS / record_file):
        groups[tuple(row[column] for column in group_columns)].append(row)
    summary_rows = read_rows(RESULTS / summary_file)
    indexed = {
        tuple(row[column] for column in group_columns): row for row in summary_rows
    }
    if len(indexed) != len(summary_rows):
        raise AssertionError(f"{summary_file}: duplicate summary keys")
    if set(groups) != set(indexed):
        raise AssertionError(f"{summary_file}: summary keys do not match record groups")
    for key, rows in groups.items():
        verify_summary_row(indexed[key], summarize(rows))


def verify_method_only_diagnostics() -> None:
    eth_variants = {
        "a0_base_graph",
        "a1_dual_seeds",
        "a2_second_order",
        "a3_adaptive_expand",
        "a4_distribution_rank",
        "a5_regeneration",
    }
    sensitivity_variants = {
        "base",
        "hypotheses_5",
        "hypotheses_10",
        "hypotheses_30",
        "motif_0",
        "motif_075",
        "motif_225",
        "neighbors_24",
        "neighbors_36",
        "neighbors_72",
        "seeds_48",
        "seeds_72",
        "seeds_120",
    }
    checks = (
        ("eth_ablation/pairs.csv", "method", 4278, eth_variants),
        ("synthetic_structured/trials.csv", "method", 2400, {"one_hop", "sagereg"}),
        ("synthetic_factorial/trials.csv", "method", 7200, {"one_hop", "sagereg"}),
        ("sensitivity/pairs.csv", "variant", 1300, sensitivity_variants),
    )
    for relative, column, expected_rows, allowed in checks:
        rows = read_rows(RESULTS / relative)
        if len(rows) != expected_rows:
            raise AssertionError(
                f"{relative}: expected {expected_rows} rows, found {len(rows)}"
            )
        observed = {row[column] for row in rows}
        if observed != allowed:
            raise AssertionError(f"{relative}: unexpected variants {sorted(observed)}")

    verify_grouped_summary(
        "eth_ablation/pairs.csv", "eth_ablation/summary.csv", ("method",)
    )
    verify_grouped_summary(
        "sensitivity/pairs.csv", "sensitivity/summary.csv", ("variant",)
    )
    verify_grouped_summary(
        "synthetic_structured/trials.csv",
        "synthetic_structured/summary.csv",
        ("method", "inlier_ratio", "noise_sigma"),
    )
    verify_grouped_summary(
        "synthetic_factorial/trials.csv",
        "synthetic_factorial/summary.csv",
        (
            "method",
            "inlier_ratio",
            "noise_sigma",
            "coherent_outlier_probability",
            "coherent_outlier_spread",
            "unary_separation",
            "distractor_score_bonus",
        ),
    )


def verify_benchmark_index(metrics_by_key: dict[tuple[str, str], Metrics]) -> None:
    rows = read_rows(RESULTS / "benchmark_summary.csv")
    indexed = {(row["dataset"], row["descriptor"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(metrics_by_key):
        raise AssertionError("benchmark_summary.csv has unexpected or duplicate rows")
    specs = {(spec.dataset, spec.descriptor): spec for spec in BENCHMARKS}
    for key, metrics in metrics_by_key.items():
        row = indexed[key]
        spec = specs[key]
        if row["variant"] != spec.method:
            raise AssertionError(f"benchmark_summary.csv: unexpected variant for {key}")
        if int(row["evaluated_pairs"]) != metrics.count:
            raise AssertionError(f"benchmark_summary.csv: incorrect pair count for {key}")
        close(float(row["registration_recall_percent"]), 100.0 * metrics.recall)
        close(float(row["median_rotation_error_deg"]), metrics.median_rotation)
        close(float(row["median_translation_error"]), metrics.median_translation)
        close(float(row["mean_runtime_seconds"]), metrics.mean_runtime)


def verify_paired_comparisons(metrics_by_key: dict[tuple[str, str], Metrics]) -> None:
    rows = read_rows(RESULTS / "strongest_paired_comparisons.csv")
    indexed = {(row["dataset"], row["descriptor"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(metrics_by_key):
        raise AssertionError(
            "strongest_paired_comparisons.csv has unexpected or duplicate rows"
        )
    for key, metrics in metrics_by_key.items():
        row = indexed[key]
        if not row["baseline"].strip():
            raise AssertionError(f"paired comparison has an empty baseline for {key}")
        if int(row["evaluated_pairs"]) != metrics.count:
            raise AssertionError(f"paired comparison has an incorrect count for {key}")
        sagereg_rr = float(row["sagereg_rr_percent"])
        baseline_rr = float(row["baseline_rr_percent"])
        gain = float(row["sagereg_gain_points"])
        if not math.isclose(
            sagereg_rr, 100.0 * metrics.recall, abs_tol=0.015
        ):
            raise AssertionError(
                f"paired comparison has an incorrect SAGE-Reg RR for {key}"
            )
        if not math.isclose(gain, sagereg_rr - baseline_rr, abs_tol=0.015):
            raise AssertionError(f"paired comparison has an incorrect RR gain for {key}")
        discordant = int(row["sagereg_only_successes"]) + int(
            row["baseline_only_successes"]
        )
        if discordant > metrics.count:
            raise AssertionError(
                f"paired comparison has invalid discordant counts for {key}"
            )
        p_value = float(row["exact_two_sided_p"])
        if not 0.0 <= p_value <= 1.0:
            raise AssertionError(f"paired comparison has an invalid p-value for {key}")


def main() -> None:
    verify_checksums()
    metrics_by_key: dict[tuple[str, str], Metrics] = {}
    for benchmark in BENCHMARKS:
        metrics = verify_benchmark(benchmark)
        metrics_by_key[(benchmark.dataset, benchmark.descriptor)] = metrics
        print(
            f"{benchmark.directory:20s} "
            f"n={benchmark.expected_rows:4d} "
            f"RR={100.0 * metrics.recall:5.2f}% "
            f"RE={metrics.median_rotation:.4f} deg "
            f"TE={metrics.median_translation:.4f} m"
        )
    verify_benchmark_index(metrics_by_key)
    verify_paired_comparisons(metrics_by_key)
    verify_method_only_diagnostics()
    print("All CSV integrity, scope, and aggregate checks passed.")


if __name__ == "__main__":
    main()
