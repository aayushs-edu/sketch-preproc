# src/sketch_preproc/common/edge_detection.py
"""Edge detection utilities."""

import torch
import numpy as np
import cv2
import warnings


def get_edge_detector():
    """Get best available edge detector: PiDiNet > DexiNed > Canny."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try PiDiNet first
    try:
        from controlnet_aux import PidiNetDetector
        pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        if torch.cuda.is_available():
            pidinet = pidinet.to(device)
        
        def pidinet_detector(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            edges = pidinet(img_rgb, detect_resolution=512, image_resolution=512)
            edges_np = np.array(edges)
            
            if len(edges_np.shape) == 3:
                edges_np = cv2.cvtColor(edges_np, cv2.COLOR_RGB2GRAY)
            
            return edges_np.astype(np.float32) / 255.0
        
        return pidinet_detector
        
    except Exception as e:
        warnings.warn(f"PiDiNet not available: {e}")
    
    # Fallback: Enhanced Canny
    def canny_detector(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 50, 50)
        
        otsu_thresh, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower = 0.5 * otsu_thresh
        upper = otsu_thresh
        
        edges1 = cv2.Canny(filtered, lower, upper)
        edges2 = cv2.Canny(cv2.GaussianBlur(filtered, (5,5), 1.0), lower*0.5, upper*0.5)
        
        edges = cv2.bitwise_or(edges1, edges2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges.astype(np.float32) / 255.0
    
    return canny_detector


def stroke_saliency(gray):
    """Extract stroke saliency/structure from grayscale image."""
    try:
        from controlnet_aux import LineartDetector
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        if device == 'cuda':
            detector = detector.to(device)
        
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result = detector(gray_rgb)
        result_np = np.array(result)
        
        if len(result_np.shape) == 3:
            result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)
        
        return result_np.astype(np.float32) / 255.0
        
    except Exception:
        # Fallback: Laplacian of Gaussian
        scales = [1.0, 2.0, 4.0]
        responses = []
        
        for sigma in scales:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            laplacian = cv2.Laplacian(blurred, cv2.CV_32F)
            laplacian = laplacian * (sigma ** 2)
            responses.append(np.abs(laplacian))
        
        saliency = np.maximum.reduce(responses)
        saliency = cv2.GaussianBlur(saliency, (0, 0), 2.0)
        
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency