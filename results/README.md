# Result records

The directory contains the pair- and trial-level SAGE-Reg records used to
derive the reported metrics. Comparator results are represented only by the
aggregate statistics in `strongest_paired_comparisons.csv`; comparator source
code and pair-level outputs are not part of this repository.

## Benchmark records

Each benchmark directory contains `pairs.csv` and `summary.csv`:

| Directory | Dataset and descriptor | Rows |
|---|---|---:|
| `3dlomatch_fpfh/` | 3DLoMatch, FPFH | 1,781 |
| `3dmatch_fpfh/` | 3DMatch, FPFH | 1,623 |
| `3dmatch_fcgf/` | 3DMatch, FCGF | 1,623 |
| `kitti_fpfh/` | KITTI, FPFH | 555 |
| `kitti_fcgf/` | KITTI, FCGF | 555 |

The benchmark `pairs.csv` files contain the final `sagereg_regen` result for
every evaluated pair. `benchmark_summary.csv` provides a common index of these
records and the full SAGE-Reg configuration in the ETH ablation.

## Ablation and diagnostic records

- `eth_ablation/`: six cumulative configurations on 713 ETH-TLS pairs;
- `synthetic_structured/`: one-stage and full SAGE-Reg results for 1,200 trials
  per variant;
- `synthetic_factorial/`: one-stage and full SAGE-Reg results for 3,600 trials
  per variant;
- `sensitivity/`: 13 parameter settings on a predefined 100-pair subset.

Each directory contains the trial/pair records and the corresponding grouped
summary. The synthetic and sensitivity files support the robustness,
mechanism, and parameter analyses.

## Fields

- `pair_index`, `scene`, `source_id`, `target_id`, and `pair_file` identify an
  evaluated pair or trial;
- `method` and `variant` identify the SAGE-Reg configuration;
- `num_correspondences` and `putative_inlier_ratio` describe the input;
- `rotation_error_deg` and `translation_error` contain pose errors;
- `success` applies the dataset-specific joint RE/TE criterion;
- `runtime_seconds` contains the measured backend time;
- graph depth, effective rank, regeneration activity, and stage timings are
  provided when available.

`strongest_paired_comparisons.csv` contains the aggregate paired outcomes used
for the statistical comparison. `SHA256SUMS` covers every CSV in this
directory.

## Verification

From the repository root, run:

    python scripts/verify_results.py

For checksum verification alone, run the following from `results/`:

    sha256sum -c SHA256SUMS
