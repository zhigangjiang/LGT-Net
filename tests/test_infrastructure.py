"""
Validation tests to verify the testing infrastructure is working correctly.
"""
import pytest
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def test_pytest_working():
    """Basic test to verify pytest is working."""
    assert True


def test_pytest_markers():
    """Test that custom pytest markers are working."""
    pass


@pytest.mark.unit
def test_unit_marker():
    """Test unit marker is working."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_integration_marker():
    """Test integration marker is working."""
    assert isinstance([], list)


@pytest.mark.slow
def test_slow_marker():
    """Test slow marker is working."""
    import time
    time.sleep(0.01)
    assert True


def test_temp_directory_fixture(temp_dir):
    """Test temporary directory fixture."""
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello, testing!")
    
    assert test_file.exists()
    assert test_file.read_text() == "Hello, testing!"


def test_mock_config_fixture(mock_config):
    """Test mock configuration fixture."""
    assert hasattr(mock_config, 'MODEL')
    assert mock_config.MODEL.NAME == "lgt_net"
    assert mock_config.TRAIN.BATCH_SIZE == 4


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
def test_numpy_fixtures(sample_image, sample_layout, sample_pano_data):
    """Test numpy-dependent fixtures."""
    assert isinstance(sample_image, np.ndarray)
    assert sample_image.shape == (480, 960, 3)
    
    assert "corners" in sample_layout
    assert "boundaries" in sample_layout
    assert "room_layout" in sample_layout
    
    assert "image" in sample_pano_data
    assert "depth" in sample_pano_data
    assert "layout" in sample_pano_data


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_torch_fixtures(sample_tensor, sample_dataset_item, sample_checkpoint):
    """Test torch-dependent fixtures."""
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (1, 3, 480, 960)
    
    assert "image" in sample_dataset_item
    assert "room_id" in sample_dataset_item
    
    assert "model_state_dict" in sample_checkpoint
    assert "epoch" in sample_checkpoint


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_torch_available():
    """Test that torch is available and working."""
    x = torch.tensor([1, 2, 3])
    assert x.sum().item() == 6


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
def test_numpy_available():
    """Test that numpy is available and working."""
    x = np.array([1, 2, 3])
    assert x.sum() == 6


def test_mock_model_fixture(mock_model):
    """Test mock model fixture."""
    assert mock_model.eval() is None
    assert mock_model.train() is None
    assert mock_model.forward() is not None


class TestInfrastructure:
    """Test class to verify class-based testing works."""
    
    def test_class_method(self):
        """Test method in test class."""
        assert True
    
    def test_with_fixture(self, mock_config):
        """Test class method with fixture."""
        assert mock_config.TRAIN.BATCH_SIZE == 4