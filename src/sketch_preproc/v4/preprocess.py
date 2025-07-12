# src/sketch_preproc/v4/preprocess.py
"""V4-specific preprocessing functions."""

import cv2
import numpy as np
import warnings

from ..common.utils import check_ximgproc


def rolling_guidance_normalise(gray: np.ndarray, cfg) -> np.ndarray:
    """Apply rolling guidance filter for shading normalisation."""
    if cfg.rgf_iters == 0:
        # Fallback to CLAHE only
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
        )
        return clahe.apply(gray)
    
    check_ximgproc()
    
    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
    )
    gray_clahe = clahe.apply(gray)
    
    # Rolling Guidance Filter
    base = cv2.ximgproc.rollingGuidanceFilter(
        gray_clahe.astype(np.float32),
        d=-1,
        sigmaColor=cfg.rgf_sigma_r * 255,
        sigmaSpace=cfg.rgf_sigma_s,
        numOfIter=cfg.rgf_iters
    )
    
    # Division normalisation
    base_safe = np.maximum(base, 1.0)
    normalised = (gray_clahe.astype(np.float32) / base_safe) * 128.0
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    
    return normalised


def boost_line_ridges(gray: np.ndarray) -> np.ndarray:
    """Enhance line structures using multi-orientation top-hat morphology."""
    angles = np.arange(0, 180, 22.5)
    ridge_response = np.zeros_like(gray, dtype=np.float32)
    
    for angle in angles:
        kernel_size = 5
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        
        angle_rad = np.deg2rad(angle)
        dx = np.cos(angle_rad) * (kernel_size // 2)
        dy = np.sin(angle_rad) * (kernel_size // 2)
        
        center = kernel_size // 2
        x0, y0 = int(center - dx), int(center - dy)
        x1, y1 = int(center + dx), int(center + dy)
        cv2.line(kernel, (x0, y0), (x1, y1), 1, 1)
        
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        ridge_response += tophat.astype(np.float32)
    
    ridge_response = ridge_response / len(angles)
    enhanced = gray.astype(np.float32) * 0.7 + ridge_response * 0.3
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def prune_fur_spikes_v4(skel: np.ndarray, cfg) -> np.ndarray:
    """Enhanced fur/spike pruning with RDP simplification."""
    from ..common.utils import prune_fur_spikes
    
    # Apply V3 pruning
    cleaned = prune_fur_spikes(
        skel,
        max_branch_len=cfg.fur_max_len,
        angle_thresh=cfg.fur_angle_thresh
    )
    
    if cfg.rdp_epsilon <= 0:
        return cleaned
    
    # Apply RDP simplification
    try:
        import sknw
        
        G = sknw.build_sknw(cleaned, multi=False)
        simplified = np.zeros_like(cleaned)
        
        for (u, v) in G.edges():
            if G[u][v].get('pts') is not None:
                pts = G[u][v]['pts']
                
                if len(pts) > 3:
                    pts_float = pts.astype(np.float32)
                    simplified_pts = rdp_simplify(pts_float, cfg.rdp_epsilon)
                    
                    for i in range(len(simplified_pts) - 1):
                        cv2.line(
                            simplified,
                            (int(simplified_pts[i][1]), int(simplified_pts[i][0])),
                            (int(simplified_pts[i+1][1]), int(simplified_pts[i+1][0])),
                            255, 1
                        )
                else:
                    for i in range(len(pts) - 1):
                        cv2.line(
                            simplified,
                            (int(pts[i][1]), int(pts[i][0])),
                            (int(pts[i+1][1]), int(pts[i+1][0])),
                            255, 1
                        )
        
        return simplified
        
    except Exception as e:
        warnings.warn(f"RDP simplification failed: {e}")
        return cleaned


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker algorithm implementation."""
    if len(points) < 3:
        return points
    
    start, end = points[0], points[-1]
    dists = []
    
    for i in range(1, len(points) - 1):
        dist = point_line_distance(points[i], start, end)
        dists.append(dist)
    
    if not dists or max(dists) < epsilon:
        return np.array([start, end])
    
    index = np.argmax(dists) + 1
    
    left = rdp_simplify(points[:index + 1], epsilon)
    right = rdp_simplify(points[index:], epsilon)
    
    return np.vstack([left[:-1], right])


def point_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Calculate perpendicular distance from point to line."""
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    
    if proj_length < 0.0:
        return np.linalg.norm(point - line_start)
    elif proj_length > line_len:
        return np.linalg.norm(point - line_end)
    else:
        proj_point = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - proj_point)