# src/sketch_preproc/v4/salience.py
"""V4 salience mask extraction with shade support."""

import cv2
import numpy as np
from typing import Optional, Tuple
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from ..common.utils import (
    coherent_hysteresis_fn, clean_skeleton, fuse_collinear
)
from .preprocess import prune_fur_spikes_v4


def extract_salience_masks_v4(
    prob_map: np.ndarray,
    gray: np.ndarray,
    shade_preproc: Optional[np.ndarray],
    cfg,
    alpha: float = 0.7,
    tau_primary: float = 0.30,
    tau_detail: float = 0.20,
    tau_shade: float = 0.10,
    tau_lo_primary: float = 0.18,
    tau_lo_detail: float = 0.12,
    tau_lo_shade: float = 0.05,
    spur_len: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V4: Extract three salience masks with enhanced shade processing."""
    
    # Use V4 thresholds if available
    if hasattr(cfg, 'tau_shade'):
        tau_shade = cfg.tau_shade
        tau_lo_shade = cfg.tau_lo_shade
    
    # PidiNet cleanup
    prob_blur = cv2.medianBlur((prob_map * 255).astype(np.uint8), 3).astype(np.float32) / 255
    
    # Coherence-based filtering
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # 11×11 coherence
    Jxx = cv2.boxFilter(gx*gx, ddepth=-1, ksize=(11,11))
    Jyy = cv2.boxFilter(gy*gy, ddepth=-1, ksize=(11,11))
    Jxy = cv2.boxFilter(gx*gy, ddepth=-1, ksize=(11,11))
    
    lam1 = 0.5*(Jxx+Jyy + np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2))
    lam2 = 0.5*(Jxx+Jyy - np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2))
    coherence_11 = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
    
    P_clean = np.where(coherence_11 > 0.4, prob_blur, 0.0)
    
    # 3×3 coherence
    Jxx_3 = cv2.GaussianBlur(gx*gx, (3,3), 0)
    Jyy_3 = cv2.GaussianBlur(gy*gy, (3,3), 0)
    Jxy_3 = cv2.GaussianBlur(gx*gy, (3,3), 0)
    lam1_3 = 0.5*(Jxx_3+Jyy_3 + np.sqrt((Jxx_3-Jyy_3)**2 + 4*Jxy_3**2))
    lam2_3 = 0.5*(Jxx_3+Jyy_3 - np.sqrt((Jxx_3-Jyy_3)**2 + 4*Jxy_3**2))
    coherence_3 = (lam1_3 - lam2_3) / (lam1_3 + lam2_3 + 1e-6)
    
    P_clean = P_clean * (coherence_3 ** 0.5)
    
    # Contrast term
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag /= (grad_mag.max() + 1e-6)
    
    # Primary salience
    sal = (P_clean ** alpha) * (grad_mag ** (1 - alpha))
    
    # Shade salience (V4 enhancement)
    if shade_preproc is not None and cfg.shade_normalise_strong:
        gx_shade = cv2.Sobel(shade_preproc, cv2.CV_32F, 1, 0, ksize=3)
        gy_shade = cv2.Sobel(shade_preproc, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_shade = np.sqrt(gx_shade**2 + gy_shade**2)
        grad_mag_shade /= (grad_mag_shade.max() + 1e-6)
        sal_shade = (P_clean ** 0.5) * (grad_mag_shade ** 0.5)
    else:
        sal_shade = sal
    
    # Extract masks
    masks = []
    
    for (hi, lo, sal_map) in [
        (tau_primary, tau_lo_primary, sal),
        (tau_detail, tau_lo_detail, sal),
        (tau_shade, tau_lo_shade, sal_shade)
    ]:
        m = coherent_hysteresis_fn(sal_map, hi, lo, max_gap=2)
        
        # Size filtering
        ncc, lbl = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        
        if hi == tau_primary:
            min_dim = 5
            keep = np.zeros_like(m, dtype=bool)
            for cc_id in range(1, ncc):
                ys, xs = np.where(lbl == cc_id)
                if ys.size == 0: continue
                if max(xs.max()-xs.min(), ys.max()-ys.min()) >= min_dim:
                    keep |= (lbl == cc_id)
            m = keep.astype(np.uint8)
            
        elif hi == tau_detail:
            small = np.zeros_like(m, dtype=bool)
            for cc_id in range(1, ncc):
                cc = (lbl == cc_id)
                sk = skeletonize(cc).astype(bool)
                if sk.sum() < 8:
                    small |= cc
            m = m.astype(np.uint8)
            m[small] = 0
        
        # Clean skeleton
        m = clean_skeleton(m, min_branch=7)
        m = fuse_collinear(m, max_angle=15)
        
        # V4: Enhanced fur pruning
        if cfg is not None:
            m = prune_fur_spikes_v4(m, cfg)
        else:
            from ..common.utils import prune_fur_spikes
            m = prune_fur_spikes(m, max_branch_len=20, angle_thresh=25)
        
        m = (m > 0).astype(np.uint8) * 255
        masks.append(m)
    
    primary, detail, shade = masks
    
    # Final cleanup
    lbl = label(detail, connectivity=2)
    for r in regionprops(lbl):
        if r.perimeter < 6:
            detail[lbl == r.label] = 0
    
    return primary, detail, shade