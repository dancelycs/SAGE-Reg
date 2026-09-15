import numpy as np

from sagereg import SAGEReg, SAGERegConfig, pose_guided_regeneration
from sagereg.geometry import rotation_error_deg, translation_error


def test_graph_backend_recovers_a_clean_transform():
    rng = np.random.default_rng(11)
    source = rng.uniform(-1.0, 1.0, size=(60, 3))
    transform = np.eye(4)
    transform[:3, 3] = [0.10, -0.03, 0.02]
    target = source + transform[:3, 3]
    scores = np.linspace(0.4, 1.0, len(source))
    config = SAGERegConfig(
        compatibility_sigma=0.025,
        compatibility_threshold=0.07,
        num_seeds=20,
        max_hops=2,
        inlier_threshold=0.035,
        max_correspondences=300,
    )
    result = SAGEReg(config, "cpu").register(source, target, scores)
    assert rotation_error_deg(result.transform, transform) < 1e-4
    assert translation_error(result.transform, transform) < 1e-4


def test_regeneration_accepts_a_stable_improvement():
    rng = np.random.default_rng(7)
    source = rng.uniform(-1.0, 1.0, size=(20, 3))
    target = source + np.asarray([0.10, -0.03, 0.02])
    features = np.eye(20)
    initial = np.eye(4)
    initial[:3, 3] = [0.07, -0.01, 0.00]
    transform, diagnostics = pose_guided_regeneration(
        source, target, features, features, initial, radii=(0.15, 0.08)
    )
    np.testing.assert_allclose(transform[:3, 3], [0.10, -0.03, 0.02], atol=1e-6)
    assert diagnostics["regeneration_updates"] >= 1
