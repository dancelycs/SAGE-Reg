from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.special import expit, logit


@dataclass
class CompatibilityGraph:
    adjacency: sparse.csr_matrix
    edge_weight: sparse.csr_matrix
    unary_probability: np.ndarray
    structural_probability: np.ndarray
    fused_probability: np.ndarray


def _robust_probability(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median)) * 1.4826 + 1e-8
    z = (scores - median) / mad
    if not higher_is_better:
        z = -z
    return np.clip(expit(z), 1e-4, 1.0 - 1e-4)


def _entropy_reliability(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    return float(np.clip(1.0 - entropy.mean(), 0.05, 0.95))


def _pairwise_distances(points: np.ndarray, device: str) -> np.ndarray:
    try:
        import torch

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        tensor = torch.as_tensor(points, dtype=torch.float32, device="cuda" if use_cuda else "cpu")
        distances = torch.cdist(tensor, tensor)
        result = distances.cpu().numpy()
        del tensor, distances
        if use_cuda:
            torch.cuda.empty_cache()
        return result
    except (ImportError, RuntimeError):
        delta = points[:, None, :] - points[None, :, :]
        return np.linalg.norm(delta, axis=-1)


def build_compatibility_graph(
    source: np.ndarray,
    target: np.ndarray,
    descriptor_scores: np.ndarray | None,
    sigma: float,
    threshold: float,
    max_neighbors: int,
    device: str = "cpu",
    second_order_strength: float = 0.0,
) -> CompatibilityGraph:
    n = len(source)
    if descriptor_scores is None:
        descriptor_scores = np.ones(n, dtype=np.float64)
    unary = _robust_probability(descriptor_scores, higher_is_better=True)
    src_dist = _pairwise_distances(np.asarray(source, np.float32), device)
    tgt_dist = _pairwise_distances(np.asarray(target, np.float32), device)
    discrepancy = np.abs(src_dist - tgt_dist)
    compatibility = np.exp(-0.5 * (discrepancy / max(sigma, 1e-8)) ** 2)
    np.fill_diagonal(compatibility, 0.0)
    valid = discrepancy < threshold
    np.fill_diagonal(valid, False)

    if max_neighbors > 0 and max_neighbors < n:
        candidate = np.where(valid, compatibility, -np.inf)
        keep_indices = np.argpartition(candidate, -max_neighbors, axis=1)[:, -max_neighbors:]
        bounded = np.zeros_like(valid)
        bounded[np.arange(n)[:, None], keep_indices] = True
        valid &= bounded | bounded.T

    rows, cols = np.nonzero(valid)
    values = compatibility[rows, cols].astype(np.float32)
    weights = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    weights = weights.maximum(weights.T)
    adjacency = weights.copy()
    adjacency.data[:] = 1.0

    if second_order_strength > 0.0 and adjacency.nnz:
        common = (adjacency @ adjacency).multiply(adjacency).tocsr()
        row_max = np.asarray(common.max(axis=1).toarray()).ravel()
        common_coo = common.tocoo()
        denom = np.sqrt(
            np.maximum(row_max[common_coo.row], 1.0)
            * np.maximum(row_max[common_coo.col], 1.0)
        )
        motif = sparse.csr_matrix(
            (common_coo.data / denom, (common_coo.row, common_coo.col)),
            shape=common.shape,
        )
        factor = adjacency * 0.5 + motif * second_order_strength
        weights = weights.multiply(factor).tocsr()

    degree = np.asarray(weights.sum(axis=1)).ravel()
    propagated = np.asarray(weights @ unary).ravel() / np.maximum(degree, 1e-8)
    structural_raw = 0.55 * degree / max(np.percentile(degree, 75), 1e-8) + 0.45 * propagated
    structural = _robust_probability(structural_raw, higher_is_better=True)

    unary_weight = _entropy_reliability(unary)
    structural_weight = _entropy_reliability(structural)
    normalizer = unary_weight + structural_weight
    fused_logit = (unary_weight * logit(unary) + structural_weight * logit(structural)) / normalizer
    fused = np.clip(expit(fused_logit), 1e-4, 1.0 - 1e-4)
    return CompatibilityGraph(adjacency, weights, unary, structural, fused)


def binary_entropy(probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return float(np.mean(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))))
