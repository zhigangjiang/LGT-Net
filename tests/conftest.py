import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    if not HAS_NUMPY:
        pytest.skip("numpy not available")
    return np.random.randint(0, 255, (480, 960, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    """Create a sample torch tensor for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    return torch.randn(1, 3, 480, 960)


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.MODEL = Mock()
    config.DATASET = Mock()
    config.TRAIN = Mock()
    config.TEST = Mock()
    
    config.MODEL.NAME = "lgt_net"
    config.MODEL.PRETRAINED = False
    config.DATASET.NAME = "mp3d"
    config.DATASET.ROOT_DIR = "/tmp/dataset"
    config.TRAIN.BATCH_SIZE = 4
    config.TRAIN.LEARNING_RATE = 0.001
    config.TEST.BATCH_SIZE = 1
    
    return config


@pytest.fixture
def sample_layout():
    """Create sample layout data for testing."""
    if not HAS_NUMPY:
        pytest.skip("numpy not available")
    return {
        "corners": np.array([[100, 200], [200, 200], [300, 250], [250, 300]]),
        "boundaries": np.array([0, 1, 1, 0]),
        "room_layout": np.random.rand(256, 1024)
    }


@pytest.fixture
def sample_pano_data():
    """Create sample panoramic data for testing."""
    if not HAS_NUMPY:
        pytest.skip("numpy not available")
    return {
        "image": np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8),
        "depth": np.random.rand(512, 1024),
        "layout": np.random.rand(256, 1024)
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.eval.return_value = None
    model.train.return_value = None
    if HAS_TORCH:
        model.forward.return_value = torch.randn(1, 1, 256, 1024)
    else:
        model.forward.return_value = Mock()
    return model


@pytest.fixture
def sample_dataset_item():
    """Create a sample dataset item."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    return {
        "image": torch.randn(3, 512, 1024),
        "depth": torch.randn(1, 512, 1024),
        "boundaries": torch.randn(1, 256, 1024),
        "room_id": "room_001",
        "image_path": "/fake/path/image.jpg"
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "1"
    yield
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture
def sample_checkpoint():
    """Create a sample model checkpoint."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    return {
        "model_state_dict": {"layer1.weight": torch.randn(10, 5)},
        "optimizer_state_dict": {"param_groups": [{"lr": 0.001}]},
        "epoch": 10,
        "best_loss": 0.5
    }