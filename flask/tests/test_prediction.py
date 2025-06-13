import pytest
import numpy as np
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

# Import the prediction functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prediction.predictor import load_prediction_model, predict_growth_stage


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
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock Keras model"""
        mock_model = Mock()
        # Mock prediction output - shape (1, 14) for 14 classes
        mock_predictions = np.array([[0.1, 0.05, 0.8, 0.02, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]])
        mock_model.predict.return_value = mock_predictions
        return mock_model
    
    @patch('prediction.predictor.load_model')
    @patch('os.path.exists')
    def test_load_prediction_model_default_success(self, mock_exists, mock_load_model):
        """Test successful loading of default model"""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_load_model.return_value = mock_model
    
        # Clear the model cache first
        from prediction.predictor import _models
        _models.clear()
    
        result = load_prediction_model("default")
    
        assert result == mock_model
        mock_load_model.assert_called_once()
        assert "default" in _models
    
    @patch('prediction.predictor.load_model')
    @patch('os.path.exists')
    def test_load_prediction_model_custom_version(self, mock_exists, mock_load_model):
        """Test loading of custom model version"""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Clear the model cache first
        from prediction.predictor import _models
        _models.clear()
        
        result = load_prediction_model("v2.0")
        
        assert result == mock_model
        expected_path = os.path.join("model_versions", "v2.0", "hybrid_model.keras")
        mock_exists.assert_called_with(expected_path)
    
    @patch('os.path.exists')
    def test_load_prediction_model_file_not_found(self, mock_exists):
        """Test FileNotFoundError when model doesn't exist"""
        mock_exists.return_value = False
        
        # Clear the model cache first
        from prediction.predictor import _models
        _models.clear()
        
        with pytest.raises(FileNotFoundError) as excinfo:
            load_prediction_model("nonexistent")
        
        assert "Model not found for version 'nonexistent'" in str(excinfo.value)
    
    def test_load_prediction_model_cache(self):
        """Test that model caching works correctly"""
        from prediction.predictor import _models
        
        # Add a mock model to cache
        mock_model = Mock()
        _models["cached_version"] = mock_model
        
        result = load_prediction_model("cached_version")
        
        assert result == mock_model
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_success(self, mock_load_model, sample_image_bytes, mock_model):
        """Test successful prediction"""
        mock_load_model.return_value = mock_model
        
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert error is None
        assert result is not None
        assert "predicted_class" in result
        assert "confidence" in result
        assert "version" in result
        assert result["predicted_class"] == "2"  # Index 2 has highest probability (0.8)
        assert result["confidence"] == 0.8
        assert result["version"] == "default"
        mock_model.predict.assert_called_once()
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_different_version(self, mock_load_model, sample_image_bytes, mock_model):
        """Test prediction with different model version"""
        mock_load_model.return_value = mock_model
        
        result, error = predict_growth_stage(sample_image_bytes, "v2.0")
        
        assert error is None
        assert result["version"] == "v2.0"
        mock_load_model.assert_called_with("v2.0")
    
    def test_predict_growth_stage_invalid_image(self):
        """Test prediction with invalid image bytes"""
        invalid_bytes = b"not an image"
        
        result, error = predict_growth_stage(invalid_bytes, "default")
        
        assert result is None
        assert error is not None
        assert "cannot identify image file" in error.lower() or "invalid" in error.lower()
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_model_error(self, mock_load_model, sample_image_bytes):
        """Test prediction when model raises an exception"""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model prediction failed")
        mock_load_model.return_value = mock_model
        
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert result is None
        assert error == "Model prediction failed"
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_load_model_error(self, mock_load_model, sample_image_bytes):
        """Test prediction when model loading fails"""
        mock_load_model.side_effect = FileNotFoundError("Model not found")
        
        result, error = predict_growth_stage(sample_image_bytes, "nonexistent")
        
        assert result is None
        assert error == "Model not found"
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_edge_case_predictions(self, mock_load_model, sample_image_bytes):
        """Test prediction with edge case model outputs"""
        mock_model = Mock()
        
        # Test with all equal probabilities
        equal_probs = np.array([[1/14] * 14])
        mock_model.predict.return_value = equal_probs
        mock_load_model.return_value = mock_model
        
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert error is None
        assert result["predicted_class"] == "0"  # First class should be selected
        assert 0.07 <= result["confidence"] <= 0.08  # Should be around 1/14
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_confidence_rounding(self, mock_load_model, sample_image_bytes):
        """Test that confidence values are properly rounded"""
        mock_model = Mock()
        
        # Create predictions with precise floating point values
        predictions = np.zeros((1, 14))
        predictions[0, 5] = 0.123456789  # This should be rounded to 4 decimal places
        mock_model.predict.return_value = predictions
        mock_load_model.return_value = mock_model
        
        result, error = predict_growth_stage(sample_image_bytes, "default")
        
        assert error is None
        assert result["confidence"] == 0.1235  # Rounded to 4 decimal places
        assert result["predicted_class"] == "5"
    
    @patch('prediction.predictor.load_prediction_model')
    def test_predict_growth_stage_image_preprocessing(self, mock_load_model, mock_model):
        """Test that image preprocessing works correctly"""
        mock_load_model.return_value = mock_model
        
        # Create an image with different dimensions
        large_image = Image.new('RGB', (512, 512), color='blue')
        img_buffer = io.BytesIO()
        large_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        large_image_bytes = img_buffer.getvalue()
        
        result, error = predict_growth_stage(large_image_bytes, "default")
        
        assert error is None
        assert result is not None
        # Verify that model.predict was called (image was processed successfully)
        mock_model.predict.assert_called_once()
        
        # Verify the input shape is correct (1, 224, 224, 3)
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 2  # Two inputs for hybrid model
        assert call_args[0].shape == (1, 224, 224, 3)
        assert call_args[1].shape == (1, 224, 224, 3)
    
    def test_predict_growth_stage_return_types(self, sample_image_bytes):
        """Test that prediction returns correct types"""
        with patch('prediction.predictor.load_prediction_model') as mock_load_model:
            mock_model = Mock()
            mock_predictions = np.array([[0.1, 0.9] + [0.0] * 12])
            mock_model.predict.return_value = mock_predictions
            mock_load_model.return_value = mock_model
            
            result, error = predict_growth_stage(sample_image_bytes, "default")
            
            assert isinstance(result, dict) or result is None
            assert isinstance(error, str) or error is None
            assert not (result is None and error is None)  # At least one should be set
            
            if result:
                assert isinstance(result["predicted_class"], str)
                assert isinstance(result["confidence"], (int, float))
                assert isinstance(result["version"], str)