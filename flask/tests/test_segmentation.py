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
    
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_success(self, mock_yolo, sample_image_bytes):
        """Test successful image segmentation"""
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
    
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_no_mask_detected(self, mock_yolo, sample_image_bytes):
        """Test when no mask is detected"""
        # Mock YOLO results with no masks
        mock_result = Mock()
        mock_result.masks = None
        mock_yolo.predict.return_value = [mock_result]
        
        # Test the function
        result, error = segment_image(sample_image_bytes)
        
        # Assertions
        assert result is None
        assert error == "No mask detected"
    
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_empty_masks(self, mock_yolo, sample_image_bytes):
        """Test when masks are empty"""
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
        """Test with invalid image bytes"""
        invalid_bytes = b"not an image"
        
        result, error = segment_image(invalid_bytes)
        
        assert result is None
        assert error == "Invalid image bytes"
    
    @patch('segmentation.segment_mycelium.yolo')
    def test_segment_image_exception_handling(self, mock_yolo, sample_image_bytes):
        """Test exception handling in segment_image"""
        # Make YOLO predict raise an exception
        mock_yolo.predict.side_effect = Exception("YOLO error")
        
        result, error = segment_image(sample_image_bytes)
        
        assert result is None
        assert error == "YOLO error"
    
    @patch('segmentation.segment_mycelium.yolo')
    @patch('cv2.imread')
    def test_segment_and_save_success(self, mock_imread, mock_yolo, sample_image_path):
        """Test successful segment_and_save operation"""
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
    
    @patch('cv2.imread')
    def test_segment_and_save_invalid_image(self, mock_imread):
        """Test segment_and_save with invalid image path"""
        mock_imread.return_value = None
        
        result = segment_and_save("invalid_path.jpg", "output.png")
        
        assert result is False
    
    @patch('segmentation.segment_mycelium.yolo')
    @patch('cv2.imread')
    def test_segment_and_save_no_mask(self, mock_imread, mock_yolo):
        """Test segment_and_save when no mask is detected"""
        # Mock cv2.imread
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Mock YOLO results with no masks
        mock_result = Mock()
        mock_result.masks = None
        mock_yolo.predict.return_value = [mock_result]
        
        result = segment_and_save("test.jpg", "output.png")
        
        assert result is False
    
    @patch('segmentation.segment_mycelium.yolo')
    @patch('cv2.imread')
    def test_segment_and_save_exception(self, mock_imread, mock_yolo):
        """Test exception handling in segment_and_save"""
        # Mock cv2.imread
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Make YOLO predict raise an exception
        mock_yolo.predict.side_effect = Exception("Processing error")
        
        result = segment_and_save("test.jpg", "output.png")
        
        assert result is False
    
    def test_segment_image_return_types(self, sample_image_bytes):
        """Test that segment_image returns correct types"""
        with patch('segmentation.segment_mycelium.yolo') as mock_yolo:
            # Mock successful case
            mock_tensor = Mock()
            mock_tensor.cpu.return_value.numpy.return_value = np.ones((50, 50))
            
            mock_mask = Mock()
            mock_mask.data = [mock_tensor]
            
            mock_result = Mock()
            mock_result.masks = mock_mask
            mock_yolo.predict.return_value = [mock_result]
            
            result, error = segment_image(sample_image_bytes)
            
            assert isinstance(result, BytesIO) or result is None
            assert isinstance(error, str) or error is None
            assert not (result is None and error is None)  # At least one should be set