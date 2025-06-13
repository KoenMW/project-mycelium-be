import pytest
import numpy as np
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Import the clustering functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from clustering.clusterer import cluster_image


class TestClustering:
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes for testing"""
        image = Image.new('RGB', (224, 224), color='green')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    def test_cluster_image_testing_mode(self, sample_image_bytes):
        """Test clustering in testing mode (should return mock data)"""
        result, error = cluster_image(sample_image_bytes, hour=100, version="default")
        
        assert error is None
        assert result is not None
        assert isinstance(result, dict)
        assert "cluster" in result
        assert "confidence" in result
        assert "hour" in result
        assert "version" in result
        assert result["hour"] == 100
        assert result["version"] == "default"
    
    def test_cluster_image_no_hour(self, sample_image_bytes):
        """Test clustering without hour parameter"""
        result, error = cluster_image(sample_image_bytes, version="default")
        
        assert error is None
        assert result is not None
        assert result["hour"] is None
        assert result["normalized_hour"] == 0.0
    
    def test_cluster_image_different_version(self, sample_image_bytes):
        """Test clustering with different model version in testing mode"""
        # In testing mode, this should return mock data regardless of version
        result, error = cluster_image(sample_image_bytes, version="v2.0")
        
        # In testing mode, should return mock data and ignore version mismatch
        assert error is None
        assert result is not None
        assert result["version"] == "v2.0"
    
    def test_cluster_image_hour_normalization(self, sample_image_bytes):
        """Test hour normalization"""
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
    
    def test_cluster_image_return_types(self, sample_image_bytes):
        """Test that clustering returns correct types"""
        result, error = cluster_image(sample_image_bytes, hour=50, version="default")
        
        assert isinstance(result, dict) or result is None
        assert isinstance(error, str) or error is None
        
        if result:
            assert isinstance(result["cluster"], int)
            assert isinstance(result["confidence"], (int, float))
            assert isinstance(result["hour"], (int, type(None)))
            assert isinstance(result["normalized_hour"], (int, float))
            assert isinstance(result["version"], str)
    
    def test_cluster_image_confidence_range(self, sample_image_bytes):
        """Test that confidence is within valid range"""
        result, error = cluster_image(sample_image_bytes, hour=100, version="default")
        
        assert error is None
        assert result is not None
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_cluster_image_various_hours(self, sample_image_bytes):
        """Test clustering with various hour values"""
        test_hours = [0, 50, 100, 180, 270, 360]
        
        for hour in test_hours:
            result, error = cluster_image(sample_image_bytes, hour=hour, version="default")
            assert error is None
            assert result is not None
            assert result["hour"] == hour
            expected_normalized = hour / 360
            assert abs(result["normalized_hour"] - expected_normalized) < 0.001
    
    def test_cluster_image_invalid_bytes(self):
        """Test clustering with invalid image bytes"""
        invalid_bytes = io.BytesIO(b"not an image")
        
        # In testing mode, this should still return mock data
        # because the environment check happens before image processing
        result, error = cluster_image(invalid_bytes, version="default")
        
        # Should return mock data in testing mode
        if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
            assert result is not None
            assert error is None
        else:
            # In production mode, should handle the error
            assert result is None
            assert error is not None
    
    def test_cluster_image_multiple_calls(self, sample_image_bytes):
        """Test that multiple calls to cluster_image work consistently"""
        results = []
        for i in range(3):
            result, error = cluster_image(sample_image_bytes, hour=100, version="default")
            assert error is None
            assert result is not None
            results.append(result)
        
        # In testing mode, all results should be consistent
        for result in results:
            assert result["hour"] == 100
            assert result["version"] == "default"
    
    def test_cluster_image_edge_case_hours(self, sample_image_bytes):
        """Test clustering with edge case hour values"""
        # Test negative hour (should still work)
        result, error = cluster_image(sample_image_bytes, hour=-10, version="default")
        assert error is None
        assert result is not None
        assert result["hour"] == -10
        
        # Test very large hour
        result, error = cluster_image(sample_image_bytes, hour=1000, version="default")
        assert error is None
        assert result is not None
        assert result["hour"] == 1000
    
    def test_cluster_image_version_handling(self, sample_image_bytes):
        """Test that version parameter is handled correctly in testing mode"""
        versions = ["default", "v1.0", "v2.0", "custom_version"]
        
        for version in versions:
            result, error = cluster_image(sample_image_bytes, hour=100, version=version)
            # In testing mode, all versions should work and return mock data
            assert error is None
            assert result is not None
            assert result["version"] == version
    
    def test_load_clustering_models_skip_loading(self):
        """Test that load_clustering_models returns None values when skipping"""
        from clustering.clusterer import load_clustering_models
        
        # In testing mode with SKIP_MODEL_LOADING=true, should return None values
        result = load_clustering_models("default")
        assert result == (None, None, None)
    
    def test_clustering_constants(self):
        """Test that clustering constants are properly defined"""
        from clustering.clusterer import IMG_SIZE, MAX_HOUR, PCA_COMPONENTS, SEED
        
        assert IMG_SIZE == (224, 224)
        assert MAX_HOUR == 360
        assert PCA_COMPONENTS == 100
        assert SEED == 42
    
    def test_clustering_model_cache_exists(self):
        """Test that model cache dictionaries exist"""
        from clustering.clusterer import _encoders, _clusterers, _pcas
        
        assert isinstance(_encoders, dict)
        assert isinstance(_clusterers, dict)
        assert isinstance(_pcas, dict)