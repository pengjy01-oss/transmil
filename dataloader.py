"""Backward-compatible wrapper: re-exports all public symbols from new locations.

External scripts (scripts/build_lung_region_cache.py, debug/diagnose_collapse.py)
import from this module. This wrapper ensures they continue to work without changes.
"""

# Slice & case path utilities
from datasets.ct_preprocess.slice_utils import (  # noqa: F401
    CACHE_VERSION,
    get_case_id_from_path as _get_case_id_from_path,
    iter_case_paths,
    select_case_slice_indices,
)

# Cache I/O
from datasets.ct_preprocess.cache_io import (  # noqa: F401
    build_case_preprocess_cache,
    load_case_cache,
    pack_region_context_cache,
    save_case_cache,
    save_region_context_debug,
    unpack_region_context_cache,
)

# Lung mask
from datasets.ct_preprocess.lung_mask import (  # noqa: F401
    build_pseudo_lung_mask,
    split_left_right_lung,
)

# Lung regions
from datasets.ct_preprocess.lung_regions import (  # noqa: F401
    get_lung_region_ranges,
    get_region_bbox,
    get_six_lung_regions,
    get_valid_region_centers,
    get_valid_region_slices,
)

# Instance builder
from datasets.ct_preprocess.instance_builder import (  # noqa: F401
    build_2p5d_region_instance,
    build_lung_region_context_from_mask,
    generate_lung_region_instances,
    sample_region_centers,
)

# Dataset classes
from datasets.ct_pne_dataset import CTPneNiiBags  # noqa: F401
from datasets.mnist_bags import MnistBags  # noqa: F401
