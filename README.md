# SAGE-Reg

Implementation and result records for **“SAGE-Reg: Sampling-Aware Graph
Expansion and Regeneration for Low-Overlap Point Cloud Registration.”**

SAGE-Reg is a training-free correspondence-based registration method. It uses
dual-channel seed selection, common-neighbor consistency, bounded
pose-constrained graph expansion, spatial-support-aware hypothesis ranking,
and safeguarded coarse-to-fine correspondence regeneration.

## Installation

    git clone https://github.com/dancelycs/SAGE-Reg.git
    cd SAGE-Reg
    conda env create -f environment.yml
    conda activate PCR_PonitCloud
    pip install -e .

The environment contains the CPU dependencies. CUDA distance computation is
used automatically when a CUDA-enabled PyTorch installation is available;
otherwise NumPy is used.

## Reproduce the reported metrics

Run the result verifier from the repository root:

    python scripts/verify_results.py

Expected benchmark output:

```text
3dlomatch_fpfh       n=1781 RR=38.52% RE=40.3165 deg TE=0.9813 m
3dmatch_fpfh         n=1623 RR=81.70% RE=1.7453 deg TE=0.0558 m
3dmatch_fcgf         n=1623 RR=91.87% RE=1.6438 deg TE=0.0502 m
kitti_fpfh           n= 555 RR=92.07% RE=0.1145 deg TE=0.0277 m
kitti_fcgf           n= 555 RR=96.04% RE=0.1935 deg TE=0.1585 m
eth_ablation         n= 713 RR=48.39% RE=2.8114 deg TE=0.4695 m
All CSV integrity, scope, and aggregate checks passed.
```

The verifier checks every CSV against `results/SHA256SUMS`, reconstructs the
benchmark and diagnostic summaries from pair/trial records, checks success
labels against the evaluation thresholds, and validates the paired-comparison
table. Indoor and ETH registration recall uses RE $\leq 15$ deg and TE $\leq
0.30$ m; KITTI uses RE $\leq 5$ deg and TE $\leq 0.60$ m.

Run the implementation tests with:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

## Register one point-cloud pair

The input is a NumPy `.npz` archive containing:

- `source_points`: source coordinates, shape `(N, 3)`;
- `target_points`: target coordinates, shape `(M, 3)`;
- `source_features`: source descriptors, shape `(N, D)`;
- `target_features`: target descriptors, shape `(M, D)`.

Run the indoor configuration with:

    python scripts/run_pair.py \
      --input data/pair.npz \
      --config configs/indoor.yaml \
      --output outputs/transform.npz

Use `configs/outdoor.yaml` for KITTI-scale point clouds. The estimated
homogeneous transform maps source points into the target frame. It is printed
as JSON and stored under the `transform` key when `--output` is specified.
Passing `--no-regeneration` returns the graph-estimation result before local
correspondence regeneration.

## Repository structure

- `src/sagereg/`: SAGE-Reg implementation;
- `configs/`: indoor and outdoor evaluation parameters;
- `scripts/run_pair.py`: single-pair registration entry point;
- `scripts/verify_results.py`: numerical-result verification;
- `tests/`: implementation tests;
- `results/`: pair/trial records and derived summaries.

The result-file schemas and directory contents are described in
[results/README.md](results/README.md).

## Data

Benchmark point clouds and descriptors are not redistributed. Download
3DMatch, 3DLoMatch, ETH, and KITTI from their official providers and follow
their respective licenses. The repository accepts externally prepared points
and descriptors through the `.npz` interface described above.

## Citation and terms

Citation metadata are provided in [CITATION.cff](CITATION.cff). Usage terms are
provided in [NOTICE.md](NOTICE.md).
