# src/sketch_preproc/v3/salience.py
"""Salience mask extraction for V3 pipeline."""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from ..common.utils import (
    coherent_hysteresis_fn, clean_skeleton, fuse_collinear, prune_fur_spikes
)


def extract_salience_masks(
    prob_map: np.ndarray,
    gray: np.ndarray,
    alpha: float = 0.7,
    tau_primary: float = 0.30,
    tau_detail: float = 0.20,
    tau_shade: float = 0.08,
    tau_lo_primary: float = 0.18,
    tau_lo_detail: float = 0.12,
    spur_len: int = 3
):
    """Extract primary and detail masks from edge probability map."""
    
    # PidiNet cleanup
    prob_blur = cv2.medianBlur((prob_map*255).astype(np.uint8), 3).astype(np.float32)/255
    
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
    
    # Salience
    sal = (P_clean ** alpha) * (grad_mag ** (1 - alpha))
    
    # Extract masks
    masks = []
    for (hi, lo) in ((tau_primary, tau_lo_primary),
                     (tau_detail, tau_lo_detail),
                     (tau_shade, 0.08)):
        m = coherent_hysteresis_fn(sal, hi, lo, max_gap=2)
        
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
        
        m = clean_skeleton(m, min_branch=7)
        m = fuse_collinear(m, max_angle=15)
        m = prune_fur_spikes(m, max_branch_len=20, angle_thresh=25)
        m = (m > 0).astype(np.uint8) * 255
        masks.append(m)
    
    primary, detail, shade = masks
    
    # Final cleanup
    lbl = label(detail, connectivity=2)
    for r in regionprops(lbl):
        if r.perimeter < 6:
            detail[lbl == r.label] = 0
    
    return primary, detail