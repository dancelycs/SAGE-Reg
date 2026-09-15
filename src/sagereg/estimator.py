from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse.csgraph import shortest_path

from .geometry import covariance_spectrum, effective_rank, residuals, weighted_procrustes
from .graph import CompatibilityGraph, binary_entropy, build_compatibility_graph


@dataclass
class SAGERegConfig:
    compatibility_sigma: float = 0.05
    compatibility_threshold: float = 0.10
    max_neighbors: int = 48
    second_order_strength: float = 1.5
    dual_channel_seeds: bool = True
    num_seeds: int = 96
    seed_graph_distance: int = 2
    seed_spatial_radius: float = 0.10
    max_hops: int = 2
    frontier_decay: float = 0.50
    min_hypothesis_size: int = 6
    posterior_threshold: float = 0.55
    entropy_gain_threshold: float = 0.0015
    inlier_threshold: float = 0.10
    top_hypotheses: int = 20
    refinement_iterations: int = 5
    max_correspondences: int = 2000
    coverage_volume_floor: float = 0.015
    concentration_penalty_weight: float = 0.55
    concentration_activation_fraction: float = 0.15
    degeneracy_rank_floor: float = 1.10
    degeneracy_penalty_weight: float = 0.50
    rank_tiebreak_weight: float = 0.15
    rank_tiebreak_cap: float = 2.50


@dataclass
class PoseHypothesis:
    transform: np.ndarray
    indices: np.ndarray
    score: float
    inlier_count: int
    effective_rank: float
    median_residual: float
    hops: int


@dataclass
class RegistrationResult:
    transform: np.ndarray
    inlier_mask: np.ndarray
    score: float
    runtime_seconds: float
    num_correspondences: int
    num_hypotheses: int
    diagnostics: dict


class SAGEReg:
    def __init__(self, config: SAGERegConfig | None = None, device: str = "cpu"):
        self.config = config or SAGERegConfig()
        self.device = device

    def _subsample(
        self, source: np.ndarray, target: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(source)
        if n <= self.config.max_correspondences:
            indices = np.arange(n)
        else:
            order = np.argsort(scores)[::-1]
            high = int(self.config.max_correspondences * 0.75)
            tail = order[high:]
            rng = np.random.default_rng(0)
            sampled = rng.choice(tail, self.config.max_correspondences - high, replace=False)
            indices = np.concatenate([order[:high], sampled])
        return source[indices], target[indices], scores[indices], indices

    def _diverse_seeds(
        self, source: np.ndarray, graph: CompatibilityGraph
    ) -> np.ndarray:
        n = len(source)
        target_count = min(self.config.num_seeds, n)
        candidate_count = min(n, max(target_count * 6, target_count))
        if target_count == 0:
            return np.empty(0, dtype=int)
        per_channel = max(1, candidate_count // 2)
        if self.config.dual_channel_seeds:
            candidates = np.unique(
                np.concatenate(
                    [
                        np.argsort(graph.unary_probability)[-per_channel:],
                        np.argsort(graph.structural_probability)[-per_channel:],
                    ]
                )
            )
        else:
            candidates = np.argsort(graph.fused_probability)[-candidate_count:]
        candidate_count = len(candidates)
        candidate_quality = np.maximum(
            graph.unary_probability[candidates], graph.structural_probability[candidates]
        )
        graph_dist = shortest_path(graph.adjacency, directed=False, unweighted=True, indices=candidates)
        selected_local = [int(np.argmax(candidate_quality))]
        while len(selected_local) < target_count:
            remaining = np.setdiff1d(np.arange(candidate_count), selected_local, assume_unique=True)
            if len(remaining) == 0:
                break
            topo = np.min(
                graph_dist[np.asarray(selected_local)[:, None], candidates[remaining][None, :]], axis=0
            )
            topo = np.where(np.isfinite(topo), topo, self.config.seed_graph_distance + 2.0)
            spatial = np.min(
                np.linalg.norm(
                    source[candidates[remaining]][:, None, :]
                    - source[candidates[np.asarray(selected_local)]][None, :, :],
                    axis=-1,
                ),
                axis=1,
            )
            diversity = np.minimum(topo / max(self.config.seed_graph_distance, 1), 2.0)
            diversity *= np.minimum(spatial / max(self.config.seed_spatial_radius, 1e-8), 2.0)
            quality = candidate_quality[remaining]
            selected_local.append(int(remaining[np.argmax(quality * (0.25 + diversity))]))
        return candidates[np.asarray(selected_local)]

    def _expand_seed(
        self,
        seed: int,
        graph: CompatibilityGraph,
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        selected = np.asarray([seed], dtype=int)
        frontier = selected.copy()
        previous_entropy = binary_entropy(graph.unary_probability[selected])
        hops_used = 0
        for hop in range(1, self.config.max_hops + 1):
            candidate_mask = np.zeros(graph.adjacency.shape[0], dtype=bool)
            for node in frontier:
                begin, end = graph.adjacency.indptr[node : node + 2]
                candidate_mask[graph.adjacency.indices[begin:end]] = True
            candidate_mask[selected] = False
            candidates = np.flatnonzero(candidate_mask)
            if len(candidates) == 0:
                break
            if hop == 1:
                induced = graph.edge_weight[candidates][:, candidates]
                spectral = np.ones(len(candidates), dtype=np.float64)
                spectral /= max(np.linalg.norm(spectral), 1e-12)
                for _ in range(10):
                    spectral = induced @ spectral
                    spectral /= max(np.linalg.norm(spectral), 1e-12)
                spectral /= max(spectral.max(), 1e-12)
                posterior = spectral * np.maximum(graph.unary_probability[candidates], 0.25)
                keep = min(len(candidates), max(6, int(np.ceil(np.log2(len(source))))))
                order = np.argsort(posterior)[-keep:]
                accepted = candidates[order]
                accepted_posterior = posterior[order]
            else:
                if len(selected) < 3:
                    sub = graph.edge_weight[candidates][:, selected]
                    support = np.asarray(sub.max(axis=1).toarray()).ravel()
                    posterior = graph.unary_probability[candidates] * np.maximum(support, 1e-6)
                    keep = min(len(candidates), 3 - len(selected))
                    order = np.argsort(posterior)[-keep:]
                    accepted = candidates[order]
                    accepted_posterior = posterior[order]
                    if len(accepted) == 0:
                        break
                    selected = np.unique(np.concatenate([selected, accepted]))
                    frontier = accepted
                    hops_used = hop
                    previous_entropy = binary_entropy(accepted_posterior)
                    continue
                provisional = weighted_procrustes(
                    source[selected], target[selected], graph.unary_probability[selected]
                )
                sub = graph.edge_weight[candidates][:, selected].toarray()
                support_order = min(5, len(selected))
                strongest = np.partition(sub, -support_order, axis=1)[:, -support_order:]
                support = strongest.mean(axis=1) * np.sqrt((sub > 0).mean(axis=1))
                base = graph.unary_probability[candidates]
                posterior = 1.0 - (1.0 - base) * (1.0 - support)
                dynamic_threshold = self.config.posterior_threshold + 0.03 * max(hop - 2, 0)
                pose_gate = residuals(source[candidates], target[candidates], provisional)
                accepted_mask = (
                    (posterior >= dynamic_threshold)
                    & (pose_gate < 1.5 * self.config.inlier_threshold)
                )
                accepted = candidates[accepted_mask]
                accepted_posterior = posterior[accepted_mask]
                if len(accepted):
                    order = np.argsort(accepted_posterior)[::-1]
                    base_budget = max(8, self.config.max_neighbors // 3)
                    hop_budget = max(4, int(np.ceil(
                        base_budget * self.config.frontier_decay ** max(hop - 2, 0)
                    )))
                    order = order[:hop_budget]
                    accepted = accepted[order]
                    accepted_posterior = accepted_posterior[order]
            if len(accepted) == 0:
                break
            updated = np.unique(np.concatenate([selected, accepted]))
            current_entropy = binary_entropy(accepted_posterior)
            confidence_gain = float(accepted_posterior.mean()) / max(len(updated), 1)
            entropy_gain = previous_entropy - current_entropy + confidence_gain
            selected, frontier = updated, accepted
            hops_used = hop
            previous_entropy = current_entropy
            if hop >= 2 and entropy_gain < self.config.entropy_gain_threshold:
                break
        return selected, hops_used

    def _pose_score(
        self, transform: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> tuple[float, np.ndarray, float, float]:
        errors = residuals(source, target, transform)
        mask = errors < self.config.inlier_threshold
        count = int(mask.sum())
        if count < max(3, self.config.min_hypothesis_size):
            return -np.inf, mask, 0.0, float("inf")
        spectrum_src = covariance_spectrum(source[mask])
        spectrum_tgt = covariance_spectrum(target[mask])
        rank = min(effective_rank(spectrum_src), effective_rank(spectrum_tgt))
        scale = max(np.trace(np.cov(source.T)), 1e-8)
        volume = np.prod(spectrum_src + 1e-8) ** (1.0 / 3.0) / scale
        median = float(np.median(errors[mask]))
        volume_shortfall = max(
            0.0,
            np.log(self.config.coverage_volume_floor / max(volume, 1e-12)),
        )
        consensus_fraction = count / max(len(source), 1)
        activation = np.clip(
            (consensus_fraction - self.config.concentration_activation_fraction)
            / max(self.config.concentration_activation_fraction, 1e-8),
            0.0,
            1.0,
        )
        concentration_penalty = (
            self.config.concentration_penalty_weight
            * activation
            * np.log1p(count)
            * volume_shortfall
        )
        rank_shortfall = max(0.0, self.config.degeneracy_rank_floor - rank)
        degeneracy_penalty = self.config.degeneracy_penalty_weight * rank_shortfall
        score = (
            np.log1p(count)
            - 1.5 * median / max(self.config.inlier_threshold, 1e-8)
            - concentration_penalty
            - degeneracy_penalty
            + self.config.rank_tiebreak_weight
            * min(rank, self.config.rank_tiebreak_cap)
        )
        return float(score), mask, rank, median

    def _refine(
        self, transform: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        current = transform.copy()
        for _ in range(self.config.refinement_iterations):
            errors = residuals(source, target, current)
            mask = errors < self.config.inlier_threshold
            if mask.sum() < 3:
                break
            weights = np.exp(-0.5 * (errors[mask] / self.config.inlier_threshold) ** 2)
            updated = weighted_procrustes(source[mask], target[mask], weights)
            if np.linalg.norm(updated - current) < 1e-8:
                current = updated
                break
            current = updated
        return current

    def register(
        self,
        source: np.ndarray,
        target: np.ndarray,
        descriptor_scores: np.ndarray | None = None,
    ) -> RegistrationResult:
        start = perf_counter()
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source and target correspondences must have shape (N, 3)")
        if len(source) < 3:
            raise ValueError("at least three correspondences are required")
        scores = np.ones(len(source)) if descriptor_scores is None else np.asarray(descriptor_scores, dtype=float)
        sub_src, sub_tgt, sub_scores, original_indices = self._subsample(source, target, scores)
        subsample_end = perf_counter()
        graph_start = perf_counter()
        graph = build_compatibility_graph(
            sub_src,
            sub_tgt,
            sub_scores,
            self.config.compatibility_sigma,
            self.config.compatibility_threshold,
            self.config.max_neighbors,
            self.device,
            self.config.second_order_strength,
        )
        graph_end = perf_counter()
        seed_start = perf_counter()
        seeds = self._diverse_seeds(sub_src, graph)
        seed_end = perf_counter()
        search_start = perf_counter()
        hypotheses: list[PoseHypothesis] = []
        hop_counts: list[int] = []
        for seed in seeds:
            indices, hops = self._expand_seed(int(seed), graph, sub_src, sub_tgt)
            hop_counts.append(hops)
            if len(indices) < self.config.min_hypothesis_size:
                neighbors = graph.edge_weight.getrow(seed).indices
                order = np.argsort(graph.fused_probability[neighbors])[::-1]
                indices = np.unique(np.concatenate([[seed], neighbors[order[: self.config.min_hypothesis_size - 1]]]))
            if len(indices) < 3:
                continue
            try:
                transform = weighted_procrustes(sub_src[indices], sub_tgt[indices], graph.fused_probability[indices])
            except np.linalg.LinAlgError:
                continue
            score, mask, rank, median = self._pose_score(transform, sub_src, sub_tgt)
            hypotheses.append(PoseHypothesis(transform, indices, score, int(mask.sum()), rank, median, hops))
        if not hypotheses:
            fallback = np.argsort(graph.fused_probability)[-min(max(3, self.config.min_hypothesis_size), len(sub_src)) :]
            transform = weighted_procrustes(sub_src[fallback], sub_tgt[fallback], graph.fused_probability[fallback])
            score, mask, rank, median = self._pose_score(transform, sub_src, sub_tgt)
            hypotheses.append(PoseHypothesis(transform, fallback, score, int(mask.sum()), rank, median, 0))
        search_end = perf_counter()

        hypotheses.sort(key=lambda item: item.score, reverse=True)
        refinement_start = perf_counter()
        refined: list[PoseHypothesis] = []
        for hypothesis in hypotheses[: self.config.top_hypotheses]:
            transform = self._refine(hypothesis.transform, sub_src, sub_tgt)
            score, mask, rank, median = self._pose_score(transform, sub_src, sub_tgt)
            refined.append(PoseHypothesis(transform, hypothesis.indices, score, int(mask.sum()), rank, median, hypothesis.hops))
        best = max(refined, key=lambda item: item.score)
        refinement_end = perf_counter()
        all_errors = residuals(source, target, best.transform)
        all_mask = all_errors < self.config.inlier_threshold
        diagnostics = {
            "mean_unary_probability": float(graph.unary_probability.mean()),
            "mean_structural_probability": float(graph.structural_probability.mean()),
            "mean_fused_probability": float(graph.fused_probability.mean()),
            "mean_hops": float(np.mean(hop_counts)) if hop_counts else 0.0,
            "best_hops": int(best.hops),
            "best_effective_rank": float(best.effective_rank),
            "best_median_residual": float(best.median_residual),
            "subsample_indices": original_indices.tolist(),
            "subsample_seconds": subsample_end - start,
            "graph_seconds": graph_end - graph_start,
            "seed_seconds": seed_end - seed_start,
            "search_seconds": search_end - search_start,
            "refinement_seconds": refinement_end - refinement_start,
        }
        return RegistrationResult(
            best.transform,
            all_mask,
            best.score,
            perf_counter() - start,
            len(source),
            len(hypotheses),
            diagnostics,
        )
