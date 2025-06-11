import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, mock_open
import requests
from database.data import fetch_pocketbase_data, upload_to_pocketbase, POCKETBASE_URL, COLLECTION_NAME

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs - module level fixture"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestFetchPocketbaseData:
    """Test suite for fetch_pocketbase_data function"""
    
    @pytest.fixture
    def mock_response_data(self):
        """Mock PocketBase API response data"""
        return {
            "items": [
                {
                    "id": "test_id_1",
                    "original": "test1_h24_0.jpg",
                    "run": 1,
                    "hour": 24,
                    "angle": 0,
                    "estimatedDay": 1,
                    "trainingData": True
                },
                {
                    "id": "test_id_2", 
                    "segmented": "test2_h48_90.jpg",  # Test fallback to segmented
                    "run": 2,
                    "hour": 48,
                    "angle": 90,
                    "estimatedDay": 2,
                    "trainingData": False
                }
            ],
            "totalPages": 1,
            "page": 1
        }
    
    @patch('database.data.requests.get')
    def test_fetch_pocketbase_data_success(self, mock_get, mock_response_data, temp_output_dir):
        """Test successful data fetching from PocketBase"""
        # Mock API response
        mock_api_response = Mock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = mock_response_data
        
        # Mock image download response
        mock_img_response = Mock()
        mock_img_response.status_code = 200
        mock_img_response.content = b"fake_image_data"
        
        mock_get.side_effect = [mock_api_response, mock_img_response, mock_img_response]
        
        # Execute function
        fetch_pocketbase_data(training_data_only=True, output_dir=temp_output_dir)
        
        # Verify API call
        expected_url = f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records?filter=(trainingData=true)&page=1&perPage=500"
        mock_get.assert_any_call(expected_url)
        
        # Verify files were created
        expected_files = ["test1_h24_0.jpg", "test2_h48_90.jpg"]
        for file_name in expected_files:
            file_path = os.path.join(temp_output_dir, file_name)
            assert os.path.exists(file_path), f"File {file_name} should exist"
            
            with open(file_path, 'rb') as f:
                assert f.read() == b"fake_image_data"
    
    @patch('database.data.requests.get')
    def test_fetch_pocketbase_data_api_error(self, mock_get, temp_output_dir):
        """Test handling of API errors"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response
        
        # Should not raise exception, just print error
        fetch_pocketbase_data(output_dir=temp_output_dir)
        
        # Verify directory was created but no files downloaded
        assert os.path.exists(temp_output_dir)
        assert len(os.listdir(temp_output_dir)) == 0
    
    @patch('database.data.requests.get')
    def test_fetch_pocketbase_data_incomplete_records(self, mock_get, temp_output_dir):
        """Test handling of incomplete records"""
        incomplete_data = {
            "items": [
                {"id": "incomplete_1", "original": "test.jpg"},  # Missing required fields
                {"id": "incomplete_2", "run": 1, "hour": 24},   # Missing angle and file
            ],
            "totalPages": 1,
            "page": 1
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = incomplete_data
        mock_get.return_value = mock_response
        
        fetch_pocketbase_data(output_dir=temp_output_dir)
        
        # No files should be downloaded due to incomplete records
        assert len(os.listdir(temp_output_dir)) == 0
    
    @patch('database.data.requests.get')
    def test_fetch_pocketbase_data_pagination(self, mock_get, temp_output_dir):
        """Test pagination handling"""
        # First page
        page1_data = {
            "items": [{"id": "test1", "original": "test1.jpg", "run": 1, "hour": 24, "angle": 0, "trainingData": True}],
            "totalPages": 2,
            "page": 1
        }
        
        # Second page
        page2_data = {
            "items": [{"id": "test2", "original": "test2.jpg", "run": 2, "hour": 48, "angle": 90, "trainingData": True}],
            "totalPages": 2,
            "page": 2
        }
        
        mock_responses = []
        for data in [page1_data, page2_data]:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = data
            mock_responses.append(mock_resp)
        
        # Add image download mocks
        for _ in range(2):
            mock_img = Mock()
            mock_img.status_code = 200
            mock_img.content = b"fake_image"
            mock_responses.append(mock_img)
        
        mock_get.side_effect = mock_responses
        
        fetch_pocketbase_data(output_dir=temp_output_dir)
        
        # Verify both pages were fetched
        assert mock_get.call_count == 4  # 2 API calls + 2 image downloads

class TestUploadToPocketbase:
    """Test suite for upload_to_pocketbase function"""
    
    @pytest.fixture
    def temp_image_file(self):
        """Create temporary image file for testing"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_file.write(b"fake_image_data")
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @patch('database.data.requests.post')
    def test_upload_to_pocketbase_success(self, mock_post, temp_image_file):
        """Test successful upload to PocketBase"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = upload_to_pocketbase(
            image_path=temp_image_file,
            test_num=1,
            hour=24,
            angle=0,
            estimated_day=1,
            is_training_data=True
        )
        
        assert result is True
        
        # Verify POST was called with correct URL
        expected_url = f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == expected_url
        
        # Verify form data
        assert 'files' in kwargs
        assert 'data' in kwargs
        assert kwargs['data']['run'] == 1
        assert kwargs['data']['hour'] == 24
        assert kwargs['data']['angle'] == 0
        assert kwargs['data']['estimatedDay'] == 1
        assert kwargs['data']['trainingData'] is True
    
    @patch('database.data.requests.post')
    def test_upload_to_pocketbase_failure(self, mock_post, temp_image_file):
        """Test upload failure handling"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        result = upload_to_pocketbase(
            image_path=temp_image_file,
            test_num=1,
            hour=24,
            angle=0
        )
        
        assert result is False
    
    @patch('prediction.predictor.predict_growth_stage')
    @patch('database.data.requests.post')
    def test_upload_with_prediction(self, mock_post, mock_predict, temp_image_file):
        """Test upload with day prediction"""
        # Mock successful prediction
        mock_predict.return_value = ({"predicted_class": "2"}, None)
        
        # Mock successful upload
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = upload_to_pocketbase(
            image_path=temp_image_file,
            test_num=1,
            hour=24,
            angle=0,
            predict_day=True
        )
        
        assert result is True
        
        # Verify prediction was called
        mock_predict.assert_called_once()
        
        # Verify estimated_day was set from prediction
        args, kwargs = mock_post.call_args
        assert kwargs['data']['estimatedDay'] == 2
    
    @patch('prediction.predictor.predict_growth_stage')
    @patch('database.data.requests.post')
    def test_upload_prediction_fallback(self, mock_post, mock_predict, temp_image_file):
        """Test fallback when prediction fails"""
        # Mock failed prediction
        mock_predict.return_value = (None, "Prediction error")
        
        # Mock successful upload
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = upload_to_pocketbase(
            image_path=temp_image_file,
            test_num=1,
            hour=48,  # Should fallback to day 2
            angle=0,
            predict_day=True
        )
        
        assert result is True
        
        # Verify fallback calculation (48 // 24 = 2)
        args, kwargs = mock_post.call_args
        assert kwargs['data']['estimatedDay'] == 2
    
    def test_upload_nonexistent_file(self):
        """Test upload with non-existent file"""
        result = upload_to_pocketbase(
            image_path="nonexistent.jpg",
            test_num=1,
            hour=24,
            angle=0
        )
        
        assert result is False

class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    @patch('database.data.requests.get')
    @patch('database.data.requests.post') 
    def test_fetch_and_upload_cycle(self, mock_post, mock_get, temp_output_dir):
        """Test complete fetch and upload cycle"""
        # Mock fetch response
        fetch_data = {
            "items": [
                {
                    "id": "test_id",
                    "original": "test_image.jpg",
                    "run": 1,
                    "hour": 24,
                    "angle": 0,
                    "estimatedDay": 1,
                    "trainingData": True
                }
            ],
            "totalPages": 1,
            "page": 1
        }
        
        mock_fetch_response = Mock()
        mock_fetch_response.status_code = 200
        mock_fetch_response.json.return_value = fetch_data
        
        mock_img_response = Mock()
        mock_img_response.status_code = 200
        mock_img_response.content = b"test_image_data"
        
        mock_get.side_effect = [mock_fetch_response, mock_img_response]
        
        # Mock upload response
        mock_upload_response = Mock()
        mock_upload_response.status_code = 200
        mock_post.return_value = mock_upload_response
        
        # Execute fetch
        fetch_pocketbase_data(output_dir=temp_output_dir)
        
        # Verify file was downloaded
        downloaded_file = os.path.join(temp_output_dir, "test1_h24_0.jpg")
        assert os.path.exists(downloaded_file)
        
        # Execute upload
        result = upload_to_pocketbase(
            image_path=downloaded_file,
            test_num=1,
            hour=24,
            angle=0,
            estimated_day=1
        )
        
        assert result is True

def test_constants():
    """Test that constants are properly defined"""
    assert POCKETBASE_URL == "https://mycelium-pb-g4c8fsf0bvetfacm.westeurope-01.azurewebsites.net"
    assert COLLECTION_NAME == "mycelium"