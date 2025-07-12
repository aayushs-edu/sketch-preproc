"""Sketch preprocessing package with V3 and V4 pipelines."""

from typing import Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import yaml
import warnings

from .common.io import load_image
from .common.config import PreprocCfg, PreprocCfgV4


def preprocess(
    ref_image: Union[str, Path, np.ndarray, bytes],
    user_image: Union[str, Path, np.ndarray, bytes],
    pipeline: str = "v4",
    preset: Optional[str] = None,
    config: Optional[PreprocCfg] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Preprocess sketch images using V3 or V4 pipeline.
    
    Args:
        ref_image: Reference image (path, array, or bytes)
        user_image: User sketch image (path, array, or bytes)
        pipeline: Pipeline version ("v3" or "v4")
        preset: Preset name to load from presets/ directory
        config: Custom configuration (overrides preset)
        device: Device for processing ("cpu" or "cuda")
    
    Returns:
        Dictionary with keys:
            - primary_mask: Primary edge mask (uint8)
            - detail_mask: Detail edge mask (uint8)
            - shade_mask: Shade mask (uint8, None for V3)
            - debug: Debug information dict
    """
    # Validate pipeline
    if pipeline not in ["v3", "v4"]:
        raise ValueError(f"Unknown pipeline: {pipeline}. Use 'v3' or 'v4'")
    
    # Load images
    ref_img = load_image(ref_image)
    user_img = load_image(user_image)
    
    # Load configuration
    if config is None:
        if preset is None:
            preset = f"default_{pipeline}"
        
        # Load preset from YAML
        preset_path = Path(__file__).parent / "presets" / f"{preset}.yml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_path}")
        
        with open(preset_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Create appropriate config class
        if pipeline == "v4":
            config = PreprocCfgV4(**config_dict)
        else:
            config = PreprocCfg(**config_dict)
    
    # Import and run appropriate pipeline
    if pipeline == "v3":
        from .v3.pipeline import process
    else:
        from .v4.pipeline import process
    
    # Process images
    result = process(ref_img, user_img, config, device)
    
    # Ensure consistent output format
    output = {
        "primary_mask": result["edges_primary"],
        "detail_mask": result["edges_detail"],
        "shade_mask": result.get("edges_shade", None),
        "debug": {
            k: v for k, v in result.items() 
            if k not in ["edges_primary", "edges_detail", "edges_shade"]
        }
    }
    
    return output


__all__ = ["preprocess"]
