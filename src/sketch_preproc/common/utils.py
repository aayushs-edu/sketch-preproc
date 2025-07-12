# src/sketch_preproc/common/utils.py
"""Shared utility functions for both pipelines."""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
import warnings


def check_ximgproc():
    """Check if cv2.ximgproc is available."""
    try:
        import cv2.ximgproc
        return True
    except (ImportError, AttributeError):
        raise ImportError(
            "OpenCV contrib (ximgproc) is required but not installed.\n"
            "Install with: pip install opencv-contrib-python>=4.10"
        )


def remove_shading(gray, method='bilateral', wls_lambda=2.0, wls_sigma=0.5):
    """Remove shading/illumination from grayscale image."""
    if method == 'wls':
        try:
            check_ximgproc()
            base = cv2.ximgproc.weightedMedianFilter(
                gray.astype(np.uint8), gray.astype(np.uint8), 15
            )
            guide = gray.astype(np.uint8)
            base_wls = cv2.ximgproc.fastGlobalSmootherFilter(
                guide, gray, wls_lambda, wls_sigma
            )
            detail = gray.astype(np.float32) - base_wls.astype(np.float32)
            detail = np.clip(detail + 128, 0, 255).astype(np.uint8)
            return detail
        except:
            method = 'bilateral'
    
    if method == 'bilateral':
        base = cv2.bilateralFilter(gray, 25, 80, 80)
        detail = gray.astype(np.float32) - base.astype(np.float32)
        detail = detail + 128
        blur = cv2.GaussianBlur(detail, (0, 0), 2.0)
        enhanced = cv2.addWeighted(detail, 1.5, blur, -0.5, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return gray


def coherent_hysteresis_fn(sal, tau_hi, tau_lo, max_gap=2):
    """Link weak pixels to strong ones based on gradient orientation alignment."""
    strong = (sal >= tau_hi).astype(np.uint8)
    weak = ((sal >= tau_lo) & (sal < tau_hi)).astype(np.uint8)
    
    gx = cv2.Sobel(sal, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sal, cv2.CV_32F, 0, 1, ksize=3)
    ori = np.arctan2(gy, gx)
    
    mask = strong.copy()
    for _ in range(max_gap):
        dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        bridge = dil & weak
        agree = (np.abs(np.sin(ori - cv2.blur(ori, (3, 3)))) <= np.sin(np.deg2rad(15))).astype(np.uint8)
        bridge = bridge & agree
        if not bridge.any():
            break
        mask = mask | bridge
        weak = weak & (~bridge)
    return mask.astype(bool)


def clean_skeleton(bin_mask, min_branch=7):
    """Remove spurs and smooth skeleton."""
    try:
        import sknw
        from scipy.signal import savgol_filter
        
        skel = skeletonize(bin_mask).astype(np.uint8)
        G = sknw.build_sknw(skel, multi=False)
        skel_clean = np.zeros_like(skel)
        
        for (u, v) in G.edges():
            if G[u][v].get('pts') is not None:
                pts = G[u][v]['pts']
                
                if len(pts) < min_branch and (G.degree(u) == 1 or G.degree(v) == 1):
                    continue
                
                if len(pts) >= 9:
                    xs, ys = pts[:, 1], pts[:, 0]
                    try:
                        xs_smooth = savgol_filter(xs.astype(float), min(7, len(xs)-2), 2)
                        ys_smooth = savgol_filter(ys.astype(float), min(7, len(ys)-2), 2)
                        pts_smooth = np.column_stack([ys_smooth, xs_smooth]).astype(np.int32)
                    except:
                        pts_smooth = pts
                else:
                    pts_smooth = pts
                
                for i in range(len(pts_smooth)-1):
                    cv2.line(skel_clean,
                            (pts_smooth[i][1], pts_smooth[i][0]),
                            (pts_smooth[i+1][1], pts_smooth[i+1][0]),
                            255, 1)
        
        return skel_clean
        
    except (ImportError, Exception):
        skel = skeletonize(bin_mask).astype(np.uint8)
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        filtered = cv2.filter2D(skel, -1, kernel)
        cleaned = ((filtered >= 12) & (skel > 0)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        final = skeletonize(closed > 0).astype(np.uint8)
        return final


def fuse_collinear(skel, max_angle=15):
    """Fuse nearby collinear segments."""
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    filtered = cv2.filter2D(skel, -1, kernel)
    endpoints = (skel > 0) & (filtered == 11)
    ep_coords = np.column_stack(np.where(endpoints))
    
    if len(ep_coords) < 2:
        return skel
    
    result = skel.copy()
    
    for i in range(len(ep_coords)):
        for j in range(i+1, len(ep_coords)):
            p1 = ep_coords[i]
            p2 = ep_coords[j]
            
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist > 5 or dist < 2:
                continue
            
            y1, x1 = p1
            y2, x2 = p2
            
            neighbors1 = []
            neighbors2 = []
            
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue
                    ny1, nx1 = y1 + dy, x1 + dx
                    ny2, nx2 = y2 + dy, x2 + dx
                    
                    if 0 <= ny1 < skel.shape[0] and 0 <= nx1 < skel.shape[1] and skel[ny1, nx1] > 0:
                        neighbors1.append((ny1, nx1))
                    if 0 <= ny2 < skel.shape[0] and 0 <= nx2 < skel.shape[1] and skel[ny2, nx2] > 0:
                        neighbors2.append((ny2, nx2))
            
            if len(neighbors1) > 0 and len(neighbors2) > 0:
                n1 = neighbors1[0]
                n2 = neighbors2[0]
                
                v1 = np.array([x1 - n1[1], y1 - n1[0]])
                v2 = np.array([n2[1] - x2, n2[0] - y2])
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
                    
                    if angle < max_angle:
                        cv2.line(result, (x1, y1), (x2, y2), 255, 1)
    
    return result


def prune_fur_spikes(skel, max_branch_len=20, angle_thresh=25):
    """Remove short protruding branches that stick out from backbone."""
    try:
        import sknw
        
        G = sknw.build_sknw(skel, multi=False)
        drop_edges = []
        
        def edge_len(pts):
            return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        
        def unit_vec(p0, p1):
            v = p1 - p0
            n = np.linalg.norm(v)
            return v / n if n else v
        
        for (u, v) in G.edges():
            if G[u][v].get('pts') is None:
                continue
            
            pts = G[u][v]['pts']
            L = edge_len(pts)
            
            if L >= max_branch_len:
                continue
            
            if (G.degree(u) == 1 and G.degree(v) >= 3) or (G.degree(v) == 1 and G.degree(u) >= 3):
                j = v if G.degree(v) >= 3 else u
                ep = u if j == v else v
                
                dirs = []
                for nbr in G.neighbors(j):
                    if nbr == ep:
                        continue
                    edge_data = G[j][nbr]
                    if edge_data.get('pts') is not None:
                        pts_j = edge_data['pts']
                        if len(pts_j) >= 2:
                            if np.array_equal(pts_j[0], G.nodes[j]['o']):
                                dirs.append(unit_vec(pts_j[0], pts_j[-1]))
                            else:
                                dirs.append(unit_vec(pts_j[-1], pts_j[0]))
                
                if not dirs:
                    continue
                
                main_dir = np.mean(dirs, axis=0)
                main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-6)
                
                branch_pts = G[j][ep]['pts']
                if len(branch_pts) >= 2:
                    if np.array_equal(branch_pts[0], G.nodes[j]['o']):
                        branch_dir = unit_vec(branch_pts[0], branch_pts[-1])
                    else:
                        branch_dir = unit_vec(branch_pts[-1], branch_pts[0])
                    
                    dot_product = np.clip(np.dot(main_dir, branch_dir), -1, 1)
                    ang = np.degrees(np.arccos(abs(dot_product)))
                    
                    if ang >= angle_thresh:
                        drop_edges.append((u, v))
        
        for e in drop_edges:
            if G.has_edge(*e):
                G.remove_edge(*e)
        
        out = np.zeros_like(skel)
        for (u, v) in G.edges():
            if G[u][v].get('pts') is not None:
                pts = G[u][v]['pts']
                for i in range(len(pts)-1):
                    cv2.line(out,
                            (int(pts[i][1]), int(pts[i][0])),
                            (int(pts[i+1][1]), int(pts[i+1][0])),
                            255, 1)
        
        return out
        
    except (ImportError, Exception):
        return skel


def diagnostic_stats(edges_primary, edges_detail, gray_img, prob_map):
    """Compute diagnostic statistics for edge detection quality."""
    stats = {}
    
    combo = cv2.bitwise_or(edges_primary, edges_detail)
    num_labels, labels = cv2.connectedComponents(combo)
    
    stroke_lengths = []
    tiny_count = 0
    total_pixels = 0
    
    for i in range(1, num_labels):
        component = (labels == i)
        size = np.sum(component)
        total_pixels += size
        
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            stroke_lengths.append(perimeter)
            
            if perimeter < 5:
                tiny_count += size
    
    stats['median_stroke_length'] = np.median(stroke_lengths) if stroke_lengths else 0
    stats['mean_stroke_length'] = np.mean(stroke_lengths) if stroke_lengths else 0
    stats['num_strokes'] = len(stroke_lengths)
    stats['tiny_stroke_fraction'] = tiny_count / total_pixels if total_pixels > 0 else 0
    
    stats['primary_density'] = np.mean(edges_primary > 0)
    stats['detail_density'] = np.mean(edges_detail > 0)
    stats['total_density'] = np.mean(combo > 0)
    
    h, w = edges_primary.shape
    center_mask = np.zeros_like(edges_primary, dtype=bool)
    center_mask[h//4:3*h//4, w//4:3*w//4] = True
    
    stats['center_coverage'] = np.mean(combo[center_mask] > 0)
    stats['periphery_coverage'] = np.mean(combo[~center_mask] > 0)
    
    return stats