"""Pseudo 12-subtype label generation for ordinal boundary guidance (方案B)."""

from pseudo12.pseudo_labels import (
    compute_severity_score,
    calibrate_thresholds,
    assign_pseudo12_label,
    generate_pseudo12_labels,
    SUBTYPE_NAMES,
    NUM_SUBTYPES,
)
