# tests/test_pipelines.py
"""Basic tests for preprocessing pipelines."""

import pytest
import numpy as np
import cv2
from pathlib import Path

from sketch_preproc import preprocess


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a 256x256 image with some basic shapes
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Draw a circle
    cv2.circle(img, (128, 128), 50, (0, 0, 0), 2)
    
    # Draw a rectangle
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 0), 2)
    
    # Draw a line
    cv2.line(img, (0, 0), (255, 255), (0, 0, 0), 2)
    
    return img


def test_v3_pipeline(sample_image):
    """Test V3 pipeline processing."""
    result = preprocess(
        sample_image,
        sample_image,
        pipeline="v3",
        preset="default_v3"
    )
    
    assert "primary_mask" in result
    assert "detail_mask" in result
    assert "shade_mask" in result
    assert "debug" in result
    
    assert result["primary_mask"].shape == (256, 256)
    assert result["detail_mask"].shape == (256, 256)
    assert result["shade_mask"] is None  # V3 doesn't have shade
    
    assert result["primary_mask"].dtype == np.uint8
    assert result["detail_mask"].dtype == np.uint8


def test_v4_pipeline(sample_image):
    """Test V4 pipeline processing."""
    result = preprocess(
        sample_image,
        sample_image,
        pipeline="v4",
        preset="default_v4"
    )
    
    assert "primary_mask" in result
    assert "detail_mask" in result
    assert "shade_mask" in result
    assert "debug" in result
    
    assert result["primary_mask"].shape == (256, 256)
    assert result["detail_mask"].shape == (256, 256)
    assert result["shade_mask"].shape == (256, 256)
    
    assert result["primary_mask"].dtype == np.uint8
    assert result["detail_mask"].dtype == np.uint8
    assert result["shade_mask"].dtype == np.uint8


def test_input_types(sample_image):
    """Test different input types."""
    # Test with numpy array
    result1 = preprocess(sample_image, sample_image)
    assert result1 is not None
    
    # Test with bytes
    _, buffer = cv2.imencode('.png', sample_image)
    img_bytes = buffer.tobytes()
    result2 = preprocess(img_bytes, img_bytes)
    assert result2 is not None
    
    # Test edge density should be similar
    density1 = np.mean(result1["primary_mask"] > 0)
    density2 = np.mean(result2["primary_mask"] > 0)
    assert abs(density1 - density2) < 0.1


def test_custom_config(sample_image):
    """Test with custom configuration."""
    from sketch_preproc.common.config import PreprocCfgV4
    
    config = PreprocCfgV4(
        rgf_iters=3,
        fur_max_len=30,
        rdp_epsilon=1.0,
        tau_primary=0.25
    )
    
    result = preprocess(
        sample_image,
        sample_image,
        config=config
    )
    
    assert result is not None
    assert "primary_mask" in result


def test_invalid_pipeline(sample_image):
    """Test invalid pipeline name."""
    with pytest.raises(ValueError):
        preprocess(sample_image, sample_image, pipeline="v5")


def test_performance(sample_image):
    """Test performance requirement."""
    import time
    
    # Create a larger test image (1024x1024)
    large_image = cv2.resize(sample_image, (1024, 1024))
    
    start_time = time.perf_counter()
    result = preprocess(large_image, large_image, pipeline="v4")
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Should complete within 150ms on reasonable hardware
    # Allow some margin for CI environments
    assert elapsed_ms < 500, f"Processing took {elapsed_ms:.1f}ms, expected < 500ms"