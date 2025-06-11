import pytest
import numpy as np
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

# Import the clustering functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from clustering.clusterer import load_clustering_models, cluster_image


class TestClustering:
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes for testing"""
        image = Image.new('RGB', (224, 224), color='green')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder model"""
        mock_encoder = Mock()
        # Mock encoder output - latent vector of size 512
        mock_latent = np.random.random((1, 512))
        mock_encoder.predict.return_value = mock_latent
        return mock_encoder
    
    @pytest.fixture
    def mock_clusterer(self):
        """Create a mock HDBSCAN clusterer"""
        mock_clusterer = Mock()
        return mock_clusterer
    
    @pytest.fixture
    def mock_pca(self):
        """Create a mock PCA model"""
        mock_pca = Mock()
        # Mock PCA output - reduced to 100 dimensions
        mock_pca_output = np.random.random((1, 100))
        mock_pca.transform.return_value = mock_pca_output
        return mock_pca
    
    @patch('clustering.clusterer.joblib.load')
    @patch('clustering.clusterer.load_model')
    @patch('os.path.exists')
    def test_load_clustering_models_default_success(self, mock_exists, mock_load_model, mock_joblib_load):
        """Test successful loading of default clustering models"""
        mock_exists.return_value = True
        mock_encoder = Mock()
        mock_clusterer = Mock()
        mock_pca = Mock()
        
        mock_load_model.return_value = mock_encoder
        mock_joblib_load.side_effect = [mock_clusterer, mock_pca]
        
        # Clear the model cache first
        from clustering.clusterer import _encoders, _clusterers, _pcas
        _encoders.clear()
        _clusterers.clear()
        _pcas.clear()
        
        encoder, clusterer, pca = load_clustering_models("default")
        
        assert encoder == mock_encoder
        assert clusterer == mock_clusterer
        assert pca == mock_pca
        assert "default" in _encoders
        assert "default" in _clusterers
        assert "default" in _pcas
    
    @patch('clustering.clusterer.joblib.load')
    @patch('clustering.clusterer.load_model')
    @patch('os.path.exists')
    def test_load_clustering_models_custom_version(self, mock_exists, mock_load_model, mock_joblib_load):
        """Test loading of custom clustering model version"""
        mock_exists.return_value = True
        mock_encoder = Mock()
        mock_clusterer = Mock()
        mock_pca = Mock()
        
        mock_load_model.return_value = mock_encoder
        mock_joblib_load.side_effect = [mock_clusterer, mock_pca]
        
        # Clear the model cache first
        from clustering.clusterer import _encoders, _clusterers, _pcas
        _encoders.clear()
        _clusterers.clear()
        _pcas.clear()
        
        encoder, clusterer, pca = load_clustering_models("v2.0")
        
        assert encoder == mock_encoder
        assert clusterer == mock_clusterer
        assert pca == mock_pca
        
        # Verify correct paths were checked
        expected_calls = [
            os.path.join("model_versions", "v2.0", "encoder_model.keras"),
            os.path.join("model_versions", "v2.0", "hdbscan_clusterer.pkl"),
            os.path.join("model_versions", "v2.0", "pca_model.pkl")
        ]
        mock_exists.assert_any_call(expected_calls[0])
        mock_exists.assert_any_call(expected_calls[1])
        mock_exists.assert_any_call(expected_calls[2])
    
    @patch('os.path.exists')
    def test_load_clustering_models_missing_files(self, mock_exists):
        """Test FileNotFoundError when some model files are missing"""
        def exists_side_effect(path):
            if "encoder" in path:
                return True
            return False  # clusterer and pca missing
        
        mock_exists.side_effect = exists_side_effect
        
        # Clear the model cache first
        from clustering.clusterer import _encoders, _clusterers, _pcas
        _encoders.clear()
        _clusterers.clear()
        _pcas.clear()
        
        with pytest.raises(FileNotFoundError) as excinfo:
            load_clustering_models("missing")
        
        error_msg = str(excinfo.value)
        assert "Missing model files for version 'missing'" in error_msg
        assert "clusterer:" in error_msg
        assert "pca:" in error_msg
    
    def test_load_clustering_models_cache(self):
        """Test that model caching works correctly"""
        from clustering.clusterer import _encoders, _clusterers, _pcas
        
        # Add mock models to cache
        mock_encoder = Mock()
        mock_clusterer = Mock()
        mock_pca = Mock()
        
        _encoders["cached_version"] = mock_encoder
        _clusterers["cached_version"] = mock_clusterer
        _pcas["cached_version"] = mock_pca
        
        encoder, clusterer, pca = load_clustering_models("cached_version")
        
        assert encoder == mock_encoder
        assert clusterer == mock_clusterer
        assert pca == mock_pca
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_success(self, mock_load_models, mock_approximate_predict, 
                                  sample_image_bytes, mock_encoder, mock_clusterer, mock_pca):
        """Test successful image clustering"""
        # Setup mocks
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        mock_approximate_predict.return_value = (2, np.array([0.1, 0.2, 0.8, 0.1]))  # cluster 2, probabilities
        
        result, error = cluster_image(sample_image_bytes, hour=100, version="default")
        
        assert error is None
        assert result is not None
        assert result["cluster"] == 2
        assert result["confidence"] == 0.8
        assert result["hour"] == 100
        assert result["normalized_hour"] == round(100/360, 4)
        assert result["version"] == "default"
        
        mock_encoder.predict.assert_called_once()
        mock_pca.transform.assert_called_once()
        mock_approximate_predict.assert_called_once()
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_no_hour(self, mock_load_models, mock_approximate_predict,
                                  sample_image_bytes, mock_encoder, mock_clusterer, mock_pca):
        """Test clustering without hour parameter"""
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        mock_approximate_predict.return_value = (1, np.array([0.3, 0.7, 0.1]))
        
        result, error = cluster_image(sample_image_bytes, version="default")
        
        assert error is None
        assert result["hour"] is None
        assert result["normalized_hour"] == 0.0
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_different_version(self, mock_load_models, mock_approximate_predict,
                                           sample_image_bytes, mock_encoder, mock_clusterer, mock_pca):
        """Test clustering with different model version"""
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        mock_approximate_predict.return_value = (0, np.array([0.9, 0.1]))
        
        result, error = cluster_image(sample_image_bytes, version="v2.0")
        
        assert error is None
        assert result["version"] == "v2.0"
        mock_load_models.assert_called_with("v2.0")
    
    def test_cluster_image_invalid_image(self):
        """Test clustering with invalid image bytes"""
        invalid_bytes = io.BytesIO(b"not an image")
        
        result, error = cluster_image(invalid_bytes, version="default")
        
        assert result is None
        assert error is not None
        assert "cannot identify image file" in error.lower() or "invalid" in error.lower()
    
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_model_loading_error(self, mock_load_models, sample_image_bytes):
        """Test clustering when model loading fails"""
        mock_load_models.side_effect = FileNotFoundError("Models not found")
        
        result, error = cluster_image(sample_image_bytes, version="nonexistent")
        
        assert result is None
        assert error == "Models not found"
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_encoder_error(self, mock_load_models, mock_approximate_predict,
                                        sample_image_bytes, mock_clusterer, mock_pca):
        """Test clustering when encoder prediction fails"""
        mock_encoder = Mock()
        mock_encoder.predict.side_effect = Exception("Encoder failed")
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        
        result, error = cluster_image(sample_image_bytes, version="default")
        
        assert result is None
        assert error == "Encoder failed"
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_pca_error(self, mock_load_models, mock_approximate_predict,
                                    sample_image_bytes, mock_encoder, mock_clusterer):
        """Test clustering when PCA transformation fails"""
        mock_pca = Mock()
        mock_pca.transform.side_effect = Exception("PCA failed")
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        
        result, error = cluster_image(sample_image_bytes, version="default")
        
        assert result is None
        assert error == "PCA failed"
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_clustering_error(self, mock_load_models, mock_approximate_predict,
                                          sample_image_bytes, mock_encoder, mock_clusterer, mock_pca):
        """Test clustering when HDBSCAN prediction fails"""
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        mock_approximate_predict.side_effect = Exception("Clustering failed")
        
        result, error = cluster_image(sample_image_bytes, version="default")
        
        assert result is None
        assert error == "Clustering failed"
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_hour_normalization(self, mock_load_models, mock_approximate_predict,
                                             sample_image_bytes, mock_encoder, mock_clusterer, mock_pca):
        """Test hour normalization edge cases"""
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        mock_approximate_predict.return_value = (0, np.array([1.0]))
        
        # Test maximum hour
        result, error = cluster_image(sample_image_bytes, hour=360, version="default")
        assert error is None
        assert result["normalized_hour"] == 1.0
        
        # Test zero hour
        result, error = cluster_image(sample_image_bytes, hour=0, version="default")
        assert error is None
        assert result["normalized_hour"] == 0.0
        
        # Test mid-range hour
        result, error = cluster_image(sample_image_bytes, hour=180, version="default")
        assert error is None
        assert result["normalized_hour"] == 0.5
    
    @patch('clustering.clusterer.approximate_predict')
    @patch('clustering.clusterer.load_clustering_models')
    def test_cluster_image_confidence_rounding(self, mock_load_models, mock_approximate_predict,
                                              sample_image_bytes, mock_encoder, mock_clusterer, mock_pca):
        """Test that confidence values are properly rounded"""
        mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
        
        # Test precise confidence value
        mock_approximate_predict.return_value = (3, np.array([0.1, 0.2, 0.3, 0.456789]))
        
        result, error = cluster_image(sample_image_bytes, version="default")
        
        assert error is None
        assert result["confidence"] == 0.4568  # Rounded to 4 decimal places
        assert result["cluster"] == 3
    
    def test_cluster_image_return_types(self, sample_image_bytes):
        """Test that clustering returns correct types"""
        with patch('clustering.clusterer.load_clustering_models') as mock_load_models:
            with patch('clustering.clusterer.approximate_predict') as mock_approximate_predict:
                mock_encoder = Mock()
                mock_encoder.predict.return_value = np.random.random((1, 512))
                mock_clusterer = Mock()
                mock_pca = Mock()
                mock_pca.transform.return_value = np.random.random((1, 100))
                
                mock_load_models.return_value = (mock_encoder, mock_clusterer, mock_pca)
                mock_approximate_predict.return_value = (1, np.array([0.2, 0.8]))
                
                result, error = cluster_image(sample_image_bytes, hour=50, version="default")
                
                assert isinstance(result, dict) or result is None
                assert isinstance(error, str) or error is None
                assert not (result is None and error is None)  # At least one should be set
                
                if result:
                    assert isinstance(result["cluster"], int)
                    assert isinstance(result["confidence"], (int, float))
                    assert isinstance(result["hour"], (int, type(None)))
                    assert isinstance(result["normalized_hour"], (int, float))
                    assert isinstance(result["version"], str)