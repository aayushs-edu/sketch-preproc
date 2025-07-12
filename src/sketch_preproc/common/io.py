# src/sketch_preproc/common/io.py
"""I/O utilities for loading and saving images."""

from typing import Union
from pathlib import Path
import numpy as np
import cv2


def load_image(image: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
    """
    Load image from various sources.
    
    Args:
        image: Image source (path, array, or bytes)
        
    Returns:
        BGR image as numpy array
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Failed to load image from: {image}")
        return img
    
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            # Grayscale to BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                return image
            elif image.shape[2] == 4:
                # RGBA to BGR
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        raise ValueError(f"Invalid image array shape: {image.shape}")
    
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        return img
    
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def save_masks(output_dir: Union[str, Path], prefix: str, **masks):
    """Save mask images to directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, mask in masks.items():
        if mask is not None:
            path = output_dir / f"{prefix}_{name}.png"
            cv2.imwrite(str(path), mask)