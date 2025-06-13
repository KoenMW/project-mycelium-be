import pytest
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

# Import the segmentation functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from segmentation.segment_mycelium import segment_image, segment_and_save


class TestSegmentation:
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes for testing"""
        # Create a simple test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [255, 255, 255]  # White square in center
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary image file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[25:75, 25:75] = [255, 255, 255]
            cv2.imwrite(tmp.name, image)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_segment_image_testing_mode(self, sample_image_bytes):
        """Test segmentation in testing mode (should return mock data)"""
        result, error = segment_image(sample_image_bytes)
        
        # In testing mode, should return mock data
        assert error is None
        assert result is not None
        assert isinstance(result, BytesIO)
    
    def test_segment_and_save_testing_mode(self, sample_image_path):
        """Test segment_and_save in testing mode"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_out:
            output_path = tmp_out.name
        
        try:
            result = segment_and_save(sample_image_path, output_path)
            # In testing mode, should return True without actually processing
            assert result is True
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_production_mode_success(self, mock_yolo, sample_image_bytes):
        """Test successful image segmentation in production mode"""
        # Create proper mock tensor structure
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value = np.ones((50, 50))
    
        # Mock YOLO results
        mock_mask = Mock()
        mock_mask.data = [mock_tensor]  # List of mock tensors
    
        mock_result = Mock()
        mock_result.masks = mock_mask
        mock_yolo.predict.return_value = [mock_result]

        # Test the function
        result, error = segment_image(sample_image_bytes)

        # Assertions
        assert error is None
        assert result is not None
        assert isinstance(result, BytesIO)
        mock_yolo.predict.assert_called_once()
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_no_mask_detected(self, mock_yolo, sample_image_bytes):
        """Test when no mask is detected in production mode"""
        # Mock YOLO results with no masks
        mock_result = Mock()
        mock_result.masks = None
        mock_yolo.predict.return_value = [mock_result]
        
        # Test the function
        result, error = segment_image(sample_image_bytes)
        
        # Assertions
        assert result is None
        assert error == "No mask detected"
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_empty_masks(self, mock_yolo, sample_image_bytes):
        """Test when masks are empty in production mode"""
        # Mock YOLO results with empty masks
        mock_mask = Mock()
        mock_mask.data = []
        mock_result = Mock()
        mock_result.masks = mock_mask
        mock_yolo.predict.return_value = [mock_result]
        
        # Test the function
        result, error = segment_image(sample_image_bytes)
        
        # Assertions
        assert result is None
        assert error == "No mask detected"
    
    def test_segment_image_invalid_bytes(self):
        """Test with invalid image bytes in testing mode"""
        invalid_bytes = b"not an image"
        
        result, error = segment_image(invalid_bytes)
        
        # In testing mode, should return mock data regardless of input
        if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
            assert result is not None
            assert error is None
        else:
            assert result is None
            assert error == "Invalid image bytes"
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_exception_handling(self, mock_yolo, sample_image_bytes):
        """Test exception handling in segment_image"""
        # Make YOLO predict raise an exception
        mock_yolo.predict.side_effect = Exception("YOLO error")
        
        result, error = segment_image(sample_image_bytes)
        
        assert result is None
        assert error == "YOLO error"
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    @patch('cv2.imread')
    def test_segment_and_save_production_mode_success(self, mock_imread, mock_yolo, sample_image_path):
        """Test successful segment_and_save operation in production mode"""
        # Mock cv2.imread
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Create proper mock tensor structure
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value = np.ones((50, 50))
        
        # Mock YOLO results
        mock_mask = Mock()
        mock_mask.data = [mock_tensor]
        
        mock_result = Mock()
        mock_result.masks = mock_mask
        mock_yolo.predict.return_value = [mock_result]
        
        # Test with temporary output path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_out:
            output_path = tmp_out.name
        
        try:
            result = segment_and_save(sample_image_path, output_path)
            
            assert result is True
            assert os.path.exists(output_path)
            mock_yolo.predict.assert_called_once()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('cv2.imread')
    def test_segment_and_save_invalid_image(self, mock_imread):
        """Test segment_and_save with invalid image path in production mode"""
        mock_imread.return_value = None
        
        result = segment_and_save("invalid_path.jpg", "output.png")
        
        assert result is False
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    @patch('cv2.imread')
    def test_segment_and_save_no_mask(self, mock_imread, mock_yolo):
        """Test segment_and_save when no mask is detected in production mode"""
        # Mock cv2.imread
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Mock YOLO results with no masks
        mock_result = Mock()
        mock_result.masks = None
        mock_yolo.predict.return_value = [mock_result]
        
        result = segment_and_save("test.jpg", "output.png")
        
        assert result is False
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('segmentation.segment_mycelium.yolo')
    @patch('cv2.imread')
    def test_segment_and_save_exception(self, mock_imread, mock_yolo):
        """Test exception handling in segment_and_save in production mode"""
        # Mock cv2.imread
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Make YOLO predict raise an exception
        mock_yolo.predict.side_effect = Exception("Processing error")
        
        result = segment_and_save("test.jpg", "output.png")
        
        assert result is False
    
    def test_segment_image_return_types(self, sample_image_bytes):
        """Test that segment_image returns correct types"""
        result, error = segment_image(sample_image_bytes)
        
        assert isinstance(result, BytesIO) or result is None
        assert isinstance(error, str) or error is None
        # In testing mode, should return mock data
        if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
            assert result is not None
            assert error is None
    
    def test_segment_image_consistency_testing_mode(self, sample_image_bytes):
        """Test that segmentation is consistent in testing mode"""
        results = []
        for _ in range(3):
            result, error = segment_image(sample_image_bytes)
            assert error is None
            assert result is not None
            results.append(result)
        
        # All results should be BytesIO objects in testing mode
        for result in results:
            assert isinstance(result, BytesIO)
    
    def test_segment_and_save_consistency_testing_mode(self, sample_image_path):
        """Test that segment_and_save is consistent in testing mode"""
        results = []
        for _ in range(3):
            result = segment_and_save(sample_image_path, "dummy_output.png")
            results.append(result)
        
        # All results should be True in testing mode
        for result in results:
            assert result is True
    
    def test_segmentation_with_different_formats(self):
        """Test segmentation with different image formats"""
        # Test with PNG format
        png_image = Image.new('RGB', (100, 100), color='blue')
        png_buffer = BytesIO()
        png_image.save(png_buffer, format='PNG')
        png_bytes = png_buffer.getvalue()
        
        result, error = segment_image(png_bytes)
        
        # In testing mode, should return mock data regardless of format
        if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
            assert result is not None
            assert error is None
        
        # Test with different size image
        large_image = Image.new('RGB', (512, 512), color='green')
        large_buffer = BytesIO()
        large_image.save(large_buffer, format='JPEG')
        large_bytes = large_buffer.getvalue()
        
        result, error = segment_image(large_bytes)
        
        # In testing mode, should return mock data regardless of size
        if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
            assert result is not None
            assert error is None