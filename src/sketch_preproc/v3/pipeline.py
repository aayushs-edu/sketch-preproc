# src/sketch_preproc/v3/pipeline.py
"""V3 preprocessing pipeline implementation."""

import cv2
import numpy as np
from typing import Dict, Any
import time

from ..common.utils import (
    check_ximgproc, remove_shading, coherent_hysteresis_fn,
    clean_skeleton, fuse_collinear, prune_fur_spikes, diagnostic_stats
)
from ..common.edge_detection import get_edge_detector, stroke_saliency
from .salience import extract_salience_masks


def process(
    ref_img: np.ndarray,
    user_img: np.ndarray,
    cfg,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Process images with V3 pipeline.
    
    Args:
        ref_img: Reference image (BGR)
        user_img: User sketch image (BGR)
        cfg: PreprocCfg configuration
        device: Processing device
        
    Returns:
        Dictionary with processing results
    """
    # Check dependencies
    check_ximgproc()
    
    # Initialize edge detector
    edge_detector = get_edge_detector()
    
    result = {}
    result['original'] = user_img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)
    result['gray'] = gray
    
    # Shading removal
    if cfg.use_shading_removal:
        gray_processed = remove_shading(gray, method=cfg.shading_method)
        result['shading_removed'] = gray_processed
    else:
        gray_processed = gray
    
    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
    )
    gray_clahe = clahe.apply(gray_processed)
    result['clahe'] = gray_clahe
    
    # Guided filter
    guided = cv2.ximgproc.guidedFilter(
        gray_clahe, gray_clahe,
        cfg.guided_r, cfg.guided_eps
    )
    result['guided'] = guided
    
    # Edge detection
    guided_bgr = cv2.cvtColor(guided, cv2.COLOR_GRAY2BGR)
    edge_prob = edge_detector(guided_bgr)
    result['edge_prob'] = edge_prob
    
    # Saliency fusion (if enabled)
    if cfg.use_saliency_fusion:
        saliency = stroke_saliency(guided)
        result['saliency'] = saliency
        
        edge_prob_fused = edge_prob * (1 - cfg.saliency_weight) + saliency * cfg.saliency_weight
        edge_prob_fused = np.clip(edge_prob_fused, 0, 1)
        result['edge_fused'] = edge_prob_fused
        result['final_edge_prob'] = edge_prob_fused
    else:
        result['final_edge_prob'] = edge_prob
    
    # Handle size mismatch
    if result['final_edge_prob'].shape[:2] != gray.shape[:2]:
        h_edge, w_edge = result['final_edge_prob'].shape[:2]
        gray_resized = cv2.resize(gray, (w_edge, h_edge), interpolation=cv2.INTER_LINEAR)
    else:
        gray_resized = gray
    
    # Extract dual masks
    edges_primary, edges_detail = extract_salience_masks(
        result['final_edge_prob'],
        gray_resized,
        alpha=cfg.alpha,
        tau_primary=cfg.tau_primary,
        tau_detail=cfg.tau_detail,
        tau_lo_primary=cfg.tau_lo_primary,
        tau_lo_detail=cfg.tau_lo_detail,
        spur_len=cfg.spur_len
    )
    
    result['edges_primary'] = edges_primary
    result['edges_detail'] = edges_detail
    result['edges_combined'] = cv2.bitwise_or(edges_primary, edges_detail)
    
    # Compute stats
    result['stats'] = diagnostic_stats(
        edges_primary, edges_detail, gray_resized, result['final_edge_prob']
    )
    
    return result