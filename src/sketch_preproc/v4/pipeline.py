# src/sketch_preproc/v4/pipeline.py
"""V4 preprocessing pipeline implementation."""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import time

from ..common.utils import check_ximgproc, diagnostic_stats
from ..common.edge_detection import get_edge_detector, stroke_saliency
from .preprocess import rolling_guidance_normalise, boost_line_ridges
from .salience import extract_salience_masks_v4


def process(
    ref_img: np.ndarray,
    user_img: np.ndarray,
    cfg,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Process images with V4 pipeline.
    
    Args:
        ref_img: Reference image (BGR)
        user_img: User sketch image (BGR)
        cfg: PreprocCfgV4 configuration
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
    
    # V4: Rolling guidance normalisation
    gray_normalised = rolling_guidance_normalise(gray, cfg)
    result['gray_normalised'] = gray_normalised
    
    # V4: Line ridge enhancement
    gray_enhanced = boost_line_ridges(gray_normalised)
    result['gray_enhanced'] = gray_enhanced
    
    # V4: Pre-edge morphological cleanup
    if cfg.pre_morph_open:
        cleaned = gray_enhanced.copy()
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_h)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_v)
        result['pre_morphed'] = cleaned
        gray_for_edge = cleaned
    else:
        gray_for_edge = gray_enhanced
    
    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
    )
    gray_clahe = clahe.apply(gray_for_edge)
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
    
    # V4: Prepare shade preprocessing
    if cfg.shade_normalise_strong:
        from dataclasses import replace
        shade_cfg = replace(cfg, rgf_iters=max(3, cfg.rgf_iters), rgf_sigma_r=0.1)
        shade_preproc = rolling_guidance_normalise(gray, shade_cfg)
        result['shade_preproc'] = shade_preproc
    else:
        shade_preproc = gray_normalised
    
    # Saliency fusion
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
        shade_resized = cv2.resize(shade_preproc, (w_edge, h_edge), interpolation=cv2.INTER_LINEAR)
    else:
        gray_resized = gray
        shade_resized = shade_preproc
    
    # V4: Extract triple masks
    edges_primary, edges_detail, edges_shade = extract_salience_masks_v4(
        result['final_edge_prob'],
        gray_resized,
        shade_resized,
        cfg,
        alpha=cfg.alpha,
        tau_primary=cfg.tau_primary,
        tau_detail=cfg.tau_detail,
        tau_shade=cfg.tau_shade,
        tau_lo_primary=cfg.tau_lo_primary,
        tau_lo_detail=cfg.tau_lo_detail,
        tau_lo_shade=cfg.tau_lo_shade,
        spur_len=cfg.spur_len
    )
    
    result['edges_primary'] = edges_primary
    result['edges_detail'] = edges_detail
    result['edges_shade'] = edges_shade
    result['edges_combined'] = cv2.bitwise_or(edges_primary, edges_detail)
    
    # Compute stats
    result['stats'] = diagnostic_stats(
        edges_primary, edges_detail, gray_resized, result['final_edge_prob']
    )
    
    return result