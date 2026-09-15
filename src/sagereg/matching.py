from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    residuals,
    rotation_error_deg,
    transform_points,
    translation_error,
    weighted_procrustes,
)


def _normalize_features(features: np.ndarray) -> np.ndarray:
    features = np.nan_to_num(np.asarray(features, dtype=np.float64))
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def mutual_feature_matches(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    max_correspondences: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_features = _normalize_features(source_features)
    target_features = _normalize_features(target_features)
    if source_points.shape != (len(source_features), 3):
        raise ValueError("source points/features have inconsistent shapes")
    if target_points.shape != (len(target_features), 3):
        raise ValueError("target points/features have inconsistent shapes")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if len(source_features) < 1 or len(target_features) < 2:
        raise ValueError("matching requires at least one source and two target features")

    target_tree = cKDTree(target_features)
    distance, target_index = target_tree.query(source_features, k=2, workers=1)
    source_tree = cKDTree(source_features)
    _, reverse_index = source_tree.query(target_features, k=1, workers=1)
    source_index = np.arange(len(source_features))
    mutual = reverse_index[target_index[:, 0]] == source_index
    if not np.any(mutual):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    source_index = source_index[mutual]
    target_index_1 = target_index[mutual, 0]
    nearest = distance[mutual, 0]
    second = distance[mutual, 1]
    ratio_quality = np.clip(1.0 - nearest / np.maximum(second, 1e-8), 0.0, 1.0)
    distance_quality = np.exp(-nearest / max(np.median(nearest), 1e-8))
    scores = 0.55 * ratio_quality + 0.30 * distance_quality + 0.15
    if len(scores) > max_correspondences:
        keep = np.argsort(scores)[-max_correspondences:]
        source_index = source_index[keep]
        target_index_1 = target_index_1[keep]
        scores = scores[keep]
    return (
        source_points[source_index],
        target_points[target_index_1],
        scores.astype(np.float64),
    )


def pose_guided_regeneration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    initial_transform: np.ndarray,
    radii: tuple[float, ...] = (0.15, 0.10, 0.075),
    spatial_neighbors: int = 8,
    iterations_per_radius: int = 2,
) -> tuple[np.ndarray, dict]:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_features = _normalize_features(source_features)
    target_features = _normalize_features(target_features)
    if source_points.shape != (len(source_features), 3):
        raise ValueError("source points/features have inconsistent shapes")
    if target_points.shape != (len(target_features), 3):
        raise ValueError("target points/features have inconsistent shapes")
    current = np.asarray(initial_transform, dtype=np.float64).copy()
    if current.shape != (4, 4):
        raise ValueError("initial_transform must have shape (4, 4)")

    target_tree = cKDTree(target_points)
    accepted_updates = 0
    generated_counts: list[int] = []

    for radius in radii:
        for _ in range(iterations_per_radius):
            warped = transform_points(source_points, current)
            source_tree = cKDTree(warped)
            k_target = min(spatial_neighbors, len(target_points))
            k_source = min(spatial_neighbors, len(source_points))
            if k_target == 0 or k_source == 0:
                break

            f_dist, f_index = target_tree.query(
                warped, k=k_target, distance_upper_bound=radius, workers=1
            )
            r_dist, r_index = source_tree.query(
                target_points, k=k_source, distance_upper_bound=radius, workers=1
            )
            if f_index.ndim == 1:
                f_index, f_dist = f_index[:, None], f_dist[:, None]
            if r_index.ndim == 1:
                r_index, r_dist = r_index[:, None], r_dist[:, None]

            safe_f = np.minimum(f_index, len(target_points) - 1)
            forward_similarity = np.einsum(
                "nd,nkd->nk", source_features, target_features[safe_f]
            )
            forward_similarity[~np.isfinite(f_dist)] = -np.inf
            f_choice = np.argmax(forward_similarity, axis=1)
            f_valid = np.isfinite(
                forward_similarity[np.arange(len(source_points)), f_choice]
            )
            f_source = np.flatnonzero(f_valid)
            f_target = f_index[f_source, f_choice[f_source]]

            safe_r = np.minimum(r_index, len(source_points) - 1)
            reverse_similarity = np.einsum(
                "nd,nkd->nk", target_features, source_features[safe_r]
            )
            reverse_similarity[~np.isfinite(r_dist)] = -np.inf
            r_choice = np.argmax(reverse_similarity, axis=1)
            r_valid = np.isfinite(
                reverse_similarity[np.arange(len(target_points)), r_choice]
            )
            r_target = np.flatnonzero(r_valid)
            r_source = r_index[r_target, r_choice[r_target]]

            encoded = np.concatenate(
                [
                    f_source * len(target_points) + f_target,
                    r_source * len(target_points) + r_target,
                ]
            )
            encoded = np.unique(encoded)
            pair_source = encoded // len(target_points)
            pair_target = encoded % len(target_points)
            generated_counts.append(int(len(encoded)))
            if len(encoded) < 6:
                break

            spatial_error = residuals(
                source_points[pair_source], target_points[pair_target], current
            )
            similarity = np.einsum(
                "nd,nd->n",
                source_features[pair_source],
                target_features[pair_target],
            )
            weights = np.clip(similarity, 0.0, 1.0) ** 2
            weights *= np.exp(-0.5 * (spatial_error / max(radius, 1e-8)) ** 2)
            if np.count_nonzero(weights > 1e-6) < 6:
                break
            updated = weighted_procrustes(
                source_points[pair_source], target_points[pair_target], weights
            )
            new_error = residuals(
                source_points[pair_source], target_points[pair_target], updated
            )
            stable = (
                rotation_error_deg(updated, current) <= 10.0
                and translation_error(updated, current) <= 2.0 * radius
            )
            improves = np.median(new_error) + 1e-8 < np.median(spatial_error)
            if not (stable and improves):
                break
            current = updated
            accepted_updates += 1

    return current, {
        "regeneration_updates": accepted_updates,
        "regenerated_correspondences": max(generated_counts, default=0),
    }
