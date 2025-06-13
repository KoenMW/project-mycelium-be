import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the flask directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def mock_all_models():
    """Mock all model loading to prevent file not found errors during testing"""
    with patch('prediction.predictor.load_model') as mock_pred_load, \
         patch('clustering.clusterer.load_model') as mock_cluster_load, \
         patch('clustering.clusterer.joblib.load') as mock_joblib, \
         patch('segmentation.segment_mycelium.YOLO') as mock_yolo:
        
        # Mock prediction model
        mock_pred_model = Mock()
        mock_pred_model.predict.return_value = [[0.1, 0.9] + [0.0] * 12]
        mock_pred_load.return_value = mock_pred_model
        
        # Mock clustering models
        mock_encoder = Mock()
        mock_encoder.predict.return_value = [[0.1] * 512]
        mock_cluster_load.return_value = mock_encoder
        
        mock_clusterer = Mock()
        mock_pca = Mock()
        mock_pca.transform.return_value = [[0.1] * 100]
        mock_joblib.side_effect = [mock_clusterer, mock_pca]
        
        # Mock YOLO
        mock_yolo_instance = Mock()
        mock_yolo_instance.predict.return_value = [Mock(masks=Mock(data=[Mock()]))]
        mock_yolo.return_value = mock_yolo_instance
        
        yield

@pytest.fixture
def client():
    """Create test client"""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client