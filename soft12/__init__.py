"""Soft 12-subtype guidance module (Plan C).

Plan C differs from Plan B (pseudo12) by replacing hard integer labels with
a continuous soft distribution over 12 subtypes, supervised with KL divergence.

12 subtypes mapping (fixed order):
  0: 0/-    1: 0/0    2: 0/1
  3: 1/0    4: 1/1    5: 1/2
  6: 2/1    7: 2/2    8: 2/3
  9: 3/2   10: 3/3   11: 3/+
"""

from soft12.soft_labels import (
    compute_severity_score,
    compute_normalization_stats,
    compute_intra_r,
    compute_intra_soft_dist,
    embed_soft12_target,
    generate_soft12_targets,
    print_soft12_diagnostics,
    save_norm_stats,
    load_norm_stats,
    SUBTYPE_NAMES,
    NUM_SUBTYPES,
)

__all__ = [
    'compute_severity_score',
    'compute_normalization_stats',
    'compute_intra_r',
    'compute_intra_soft_dist',
    'embed_soft12_target',
    'generate_soft12_targets',
    'print_soft12_diagnostics',
    'save_norm_stats',
    'load_norm_stats',
    'SUBTYPE_NAMES',
    'NUM_SUBTYPES',
]
