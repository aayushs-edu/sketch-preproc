# src/sketch_preproc/common/config.py
"""Configuration dataclasses for preprocessing pipelines."""

from dataclasses import dataclass


@dataclass
class PreprocCfg:
    """Base configuration for preprocessing pipelines."""
    
    # Image processing
    max_size: int = 1024
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    guided_r: int = 8
    guided_eps: float = 0.01
    
    # Binarization
    wolf_k: float = 0.5
    wolf_w: int = 25
    
    # Morphological operations
    morph_k: int = 3
    spur_len: int = 4
    keep_pct: float = 2.0
    min_len_factor: float = 0.02
    outer_band: int = 24
    
    # Edge detection
    edge_model: str = "auto"
    use_shading_removal: bool = True
    shading_method: str = "bilateral"
    use_sam_band: bool = True
    use_local_seeding: bool = True
    local_tile_size: int = 64
    min_area: int = 10
    max_eccentricity: float = 0.95
    roundness_threshold: float = 0.4
    morph_close_size: int = 3
    morph_open_size: int = 2
    use_saliency_fusion: bool = True
    saliency_weight: float = 0.3
    use_dual_threshold: bool = True
    weak_sigma: float = 0.5
    enclosed_thr: float = 0.65
    
    # Salience extraction
    alpha: float = 0.7
    tau_primary: float = 0.30
    tau_detail: float = 0.20
    tau_lo_primary: float = 0.18
    tau_lo_detail: float = 0.12


@dataclass
class PreprocCfgV4(PreprocCfg):
    """V4-specific configuration extending base config."""
    
    # Rolling guidance parameters
    rgf_sigma_s: float = 16.0
    rgf_sigma_r: float = 0.2
    rgf_iters: int = 2
    
    # Enhanced binarization
    binarize_mode: str = "SAUVOLA"
    
    # Jagged cleanup
    pre_morph_open: bool = True
    fur_max_len: int = 25
    fur_angle_thresh: float = 20.0
    rdp_epsilon: float = 0.5
    
    # Shade processing
    shade_normalise_strong: bool = True
    tau_shade: float = 0.10
    tau_lo_shade: float = 0.05