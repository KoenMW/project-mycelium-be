import pytest
import numpy as np
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import os

# Import the prediction functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prediction.predictor import predict_growth_stage


class TestPrediction:
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes for testing"""
        # Create a simple test image
        image = Image.new('RGB', (224, 224), color='red')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer.getvalue()
    
    def test_predict_growth_stage_testing_mode(self, sample_image_bytes):
        """Test prediction in testing mode (should return mock data)"""
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert error is None
        assert result is not None
        assert isinstance(result, dict)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "version" in result
        assert result["predicted_class"] == "5"
        assert result["confidence"] == 0.92
        assert result["version"] == "default"
    
    def test_predict_growth_stage_different_version(self, sample_image_bytes):
        """Test prediction with different model version in testing mode"""
        result, error = predict_growth_stage(sample_image_bytes, "v2.0")
        
        # In testing mode, should return mock data regardless of version
        assert error is None
        assert result is not None
        assert result["version"] == "v2.0"
    
    def test_predict_growth_stage_return_types(self, sample_image_bytes):
        """Test that prediction returns correct types"""
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert isinstance(result, dict) or result is None
        assert isinstance(error, str) or error is None
        
        if result:
            assert isinstance(result["predicted_class"], str)
            assert isinstance(result["confidence"], (int, float))
            assert isinstance(result["version"], str)
    
    def test_predict_growth_stage_confidence_range(self, sample_image_bytes):
        """Test that confidence is within valid range"""
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert error is None
        assert result is not None
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_predict_growth_stage_various_versions(self, sample_image_bytes):
        """Test prediction with various model versions in testing mode"""
        versions = ["default", "v1.0", "v2.0", "custom_version"]
        
        for version in versions:
            result, error = predict_growth_stage(sample_image_bytes, version)
            # In testing mode, all versions should work and return mock data
            assert error is None
            assert result is not None
            assert result["version"] == version
    
    def test_predict_growth_stage_invalid_bytes(self):
        """Test prediction with invalid image bytes"""
        invalid_bytes = b"not an image"
        
        # In testing mode, this should still return mock data
        # because the environment check happens before image processing
        result, error = predict_growth_stage(invalid_bytes, "default")
        
        # Should return mock data in testing mode
        if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
            assert result is not None
            assert error is None
        else:
            # In production mode, should handle the error
            assert result is None
            assert error is not None
    
    def test_predict_growth_stage_multiple_calls(self, sample_image_bytes):
        """Test that multiple calls to predict_growth_stage work consistently"""
        results = []
        for i in range(3):
            result, error = predict_growth_stage(sample_image_bytes, "default")
            assert error is None
            assert result is not None
            results.append(result)
        
        # In testing mode, all results should be consistent
        for result in results:
            assert result["predicted_class"] == "5"
            assert result["confidence"] == 0.92
            assert result["version"] == "default"
    
    def test_load_prediction_model_skip_loading(self):
        """Test that load_prediction_model returns None when skipping"""
        from prediction.predictor import load_prediction_model
        
        # In testing mode with SKIP_MODEL_LOADING=true, should return None
        result = load_prediction_model("default")
        assert result is None
    
    def test_prediction_constants(self):
        """Test that prediction constants are properly defined"""
        from prediction.predictor import IMG_SIZE, CLASSES, DEFAULT_MODEL_PATH
        
        assert IMG_SIZE == (224, 224)
        assert len(CLASSES) == 14
        assert CLASSES == [str(i) for i in range(14)]
        assert DEFAULT_MODEL_PATH == "../models/best_hybrid_model.keras"
    
    def test_prediction_model_cache_exists(self):
        """Test that model cache dictionary exists"""
        from prediction.predictor import _models
        
        assert isinstance(_models, dict)
    
    def test_predict_growth_stage_edge_cases(self, sample_image_bytes):
        """Test prediction with edge case scenarios"""
        # Test with empty version string
        result, error = predict_growth_stage(sample_image_bytes, "")
        assert error is None
        assert result is not None
        assert result["version"] == ""
        
        # Test with very long version string
        long_version = "v" + "x" * 100
        result, error = predict_growth_stage(sample_image_bytes, long_version)
        assert error is None
        assert result is not None
        assert result["version"] == long_version
    
    def test_predict_growth_stage_class_range(self, sample_image_bytes):
        """Test that predicted class is within expected range"""
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert error is None
        assert result is not None
        
        predicted_class = int(result["predicted_class"])
        assert 0 <= predicted_class <= 13  # Should be in range 0-13
    
    def test_predict_growth_stage_consistency(self, sample_image_bytes):
        """Test that predictions are consistent in testing mode"""
        # Make multiple predictions with the same input
        results = []
        for _ in range(5):
            result, error = predict_growth_stage(sample_image_bytes, "default")
            assert error is None
            results.append(result)
        
        # All results should be identical in testing mode
        first_result = results[0]
        for result in results[1:]:
            assert result["predicted_class"] == first_result["predicted_class"]
            assert result["confidence"] == first_result["confidence"]
            assert result["version"] == first_result["version"]
    
    def test_predict_growth_stage_different_image_sizes(self):
        """Test prediction with different image sizes"""
        # Test with different image dimensions
        sizes = [(100, 100), (300, 300), (500, 200)]
        
        for width, height in sizes:
            image = Image.new('RGB', (width, height), color='blue')
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            image_bytes = img_buffer.getvalue()
            
            result, error = predict_growth_stage(image_bytes, "default")
            
            # In testing mode, should return mock data regardless of input size
            assert error is None
            assert result is not None
            assert result["predicted_class"] == "5"
            assert result["confidence"] == 0.92