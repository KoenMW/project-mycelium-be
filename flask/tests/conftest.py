import pytest
import os
import sys
from unittest.mock import Mock, patch

# Set environment variables before any imports
os.environ['SKIP_MODEL_LOADING'] = 'true'
os.environ['TESTING'] = 'true'

# Add the flask directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def mock_all_models():
    """Mock all model loading to prevent file not found errors during testing"""
    
    # Import modules after setting environment variables
    import prediction.predictor
    import clustering.clusterer
    import segmentation.segment_mycelium
    
    patches = []
    
    # Only patch if the attributes exist (they might not exist due to conditional imports)
    if hasattr(prediction.predictor, 'load_model'):
        patches.append(patch('prediction.predictor.load_model'))
    
    if hasattr(clustering.clusterer, 'load_model'):
        patches.append(patch('clustering.clusterer.load_model'))
    
    if hasattr(clustering.clusterer, 'joblib'):
        patches.append(patch('clustering.clusterer.joblib.load'))
    
    if hasattr(segmentation.segment_mycelium, 'YOLO'):
        patches.append(patch('segmentation.segment_mycelium.YOLO'))
    
    # Start all patches
    mocks = []
    for p in patches:
        mock = p.start()
        mocks.append(mock)
    
    # Configure mocks if they exist
    if len(mocks) >= 1 and hasattr(prediction.predictor, 'load_model'):
        # Mock prediction model
        mock_pred_model = Mock()
        mock_pred_model.predict.return_value = [[0.1, 0.9] + [0.0] * 12]
        mocks[0].return_value = mock_pred_model
    
    if len(mocks) >= 2 and hasattr(clustering.clusterer, 'load_model'):
        # Mock clustering models
        mock_encoder = Mock()
        mock_encoder.predict.return_value = [[0.1] * 512]
        mocks[1].return_value = mock_encoder
    
    if len(mocks) >= 3 and hasattr(clustering.clusterer, 'joblib'):
        mock_clusterer = Mock()
        mock_pca = Mock()
        mock_pca.transform.return_value = [[0.1] * 100]
        mocks[2].side_effect = [mock_clusterer, mock_pca]
    
    if len(mocks) >= 4 and hasattr(segmentation.segment_mycelium, 'YOLO'):
        # Mock YOLO
        mock_yolo_instance = Mock()
        mock_result = Mock()
        mock_mask = Mock()
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value = [[0.1] * 100]
        mock_mask.data = [mock_tensor]
        mock_result.masks = mock_mask
        mock_yolo_instance.predict.return_value = [mock_result]
        mocks[3].return_value = mock_yolo_instance
    
    yield
    
    # Stop all patches
    for p in patches:
        p.stop()

@pytest.fixture
def client():
    """Create test client"""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client