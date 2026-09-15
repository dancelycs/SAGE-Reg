from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from sagereg import (
    SAGEReg,
    SAGERegConfig,
    mutual_feature_matches,
    pose_guided_regeneration,
)


REQUIRED_KEYS = {
    "source_points",
    "target_points",
    "source_features",
    "target_features",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-regeneration", action="store_true")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"input archive is missing keys: {sorted(missing)}")
        source_points = np.asarray(archive["source_points"], dtype=np.float64)
        target_points = np.asarray(archive["target_points"], dtype=np.float64)
        source_features = np.asarray(archive["source_features"], dtype=np.float64)
        target_features = np.asarray(archive["target_features"], dtype=np.float64)

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    matched_source, matched_target, scores = mutual_feature_matches(
        source_points,
        target_points,
        source_features,
        target_features,
        max_correspondences=int(config.get("max_correspondences", 2000)),
    )
    if len(matched_source) < 3:
        raise RuntimeError("fewer than three mutual descriptor matches were found")

    estimator = SAGEReg(
        SAGERegConfig(**config["estimator"]),
        device=str(config.get("device", "cpu")),
    )
    result = estimator.register(matched_source, matched_target, scores)
    transform = result.transform
    regeneration_seconds = 0.0
    regeneration = {
        "regeneration_updates": 0,
        "regenerated_correspondences": 0,
    }
    if not args.no_regeneration:
        start = perf_counter()
        transform, regeneration = pose_guided_regeneration(
            source_points,
            target_points,
            source_features,
            target_features,
            transform,
            radii=tuple(float(value) for value in config["regeneration_radii"]),
            spatial_neighbors=int(config.get("regeneration_neighbors", 8)),
            iterations_per_radius=int(config.get("regeneration_iterations", 2)),
        )
        regeneration_seconds = perf_counter() - start

    report = {
        "num_initial_correspondences": int(len(matched_source)),
        "backend_seconds": float(result.runtime_seconds),
        "regeneration_seconds": float(regeneration_seconds),
        "best_hops": int(result.diagnostics["best_hops"]),
        "best_effective_rank": float(result.diagnostics["best_effective_rank"]),
        **regeneration,
        "transform": transform.tolist(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, transform=transform)


if __name__ == "__main__":
    main()
