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
    
    patches = []
    
    # Only patch if the attributes exist (they might not exist due to conditional imports)
    if hasattr(prediction.predictor, 'load_model'):
        patches.append(patch('prediction.predictor.load_model'))
    
    if hasattr(clustering.clusterer, 'load_model'):
        patches.append(patch('clustering.clusterer.load_model'))
    
    if hasattr(clustering.clusterer, 'joblib'):
        patches.append(patch('clustering.clusterer.joblib.load'))
    
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