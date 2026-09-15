from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def weighted_procrustes(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    src_center = (weights[:, None] * source).sum(axis=0)
    tgt_center = (weights[:, None] * target).sum(axis=0)
    src_zero = source - src_center
    tgt_zero = target - tgt_center
    covariance = (weights[:, None] * src_zero).T @ tgt_zero
    u, _, vt = np.linalg.svd(covariance, full_matrices=True)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = tgt_center - rotation @ src_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def residuals(source: np.ndarray, target: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.linalg.norm(transform_points(source, transform) - target, axis=1)


def rotation_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    relative = estimated[:3, :3].T @ reference[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_error(estimated: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(estimated[:3, 3] - reference[:3, 3]))


def covariance_spectrum(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros(3, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    weights = weights / max(weights.sum(), 1e-12)
    center = (weights[:, None] * points).sum(axis=0)
    covariance = (weights[:, None] * (points - center)).T @ (points - center)
    return np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)


def effective_rank(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    probabilities = probabilities[probabilities > 1e-12]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(probabilities) else 0.0
