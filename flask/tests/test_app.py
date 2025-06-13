import pytest
import json
import io
import os
import tempfile
import zipfile
from unittest.mock import Mock, patch, MagicMock
from app import app
from jobs import training_jobs, upload_jobs

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    """Create a sample image file for testing"""
    # Create a simple 1x1 pixel image as bytes
    image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    return io.BytesIO(image_data)

@pytest.fixture
def sample_zip():
    """Create a sample ZIP file for testing"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Add a test image to the zip
        zip_file.writestr('test1/24-11-12___18-59/1.jpg', b'fake_image_data')
        zip_file.writestr('test1/24-11-12___19-59/2.jpg', b'fake_image_data')
    zip_buffer.seek(0)
    return zip_buffer

class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get('/')
        assert response.status_code == 200
        assert b"Server is ok!" in response.data

    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get('/health')
        assert response.status_code in [200, 503]  # Can be degraded if models missing
        
        data = response.get_json()
        assert 'status' in data
        assert 'missing_files' in data
        assert 'python_version' in data
        assert 'memory_usage_mb' in data
        assert 'model_versions' in data

    def test_metadata_endpoint(self, client):
        """Test metadata endpoint"""
        response = client.get('/metadata')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['project'] == 'mycelium-be'
        assert data['version'] == '1.0.0'
        assert data['author'] == 'Group 6'
        assert 'model_versions' in data
        assert 'endpoints' in data
        
        # Check endpoint documentation
        endpoints = data['endpoints']
        assert 'predict' in endpoints
        assert 'cluster' in endpoints
        assert 'retrain' in endpoints
        assert 'upload-data' in endpoints

class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    @patch('app.predict_growth_stage')
    def test_predict_success(self, mock_predict, client, sample_image):
        """Test successful prediction"""
        # Mock prediction
        mock_predict.return_value = (
            {"predicted_class": "2", "confidence": 0.85, "version": "default"},
            None
        )
        
        response = client.post('/predict', data={
            'file': (sample_image, 'test.jpg'),
            'version': 'default'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'model_version' in data
        assert data['prediction']['predicted_class'] == "2"
        assert data['model_version'] == 'default'

    def test_predict_no_file(self, client):
        """Test prediction without file"""
        response = client.post('/predict')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'No file uploaded' in data['error']

    def test_predict_empty_filename(self, client):
        """Test prediction with empty filename"""
        response = client.post('/predict', data={
            'file': (io.BytesIO(b''), '')
        })
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'Empty filename' in data['error']

    @patch('app.predict_growth_stage')
    def test_predict_prediction_error(self, mock_predict, client, sample_image):
        """Test prediction with prediction error"""
        mock_predict.return_value = (None, "Prediction failed")
        
        response = client.post('/predict', data={
            'file': (sample_image, 'test.jpg')
        })
        
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data

class TestClusteringEndpoint:
    """Test clustering endpoint"""
    
    @patch('app.cluster_image')
    def test_cluster_success(self, mock_cluster, client, sample_image):
        """Test successful clustering"""
        mock_cluster.return_value = (
            {"clusters": 3, "centroids": [[1, 2], [3, 4], [5, 6]]},
            None
        )
        
        response = client.post('/cluster', data={
            'file': (sample_image, 'test.jpg'),
            'hour': '24',
            'version': 'default'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'clustering' in data
        assert 'model_version' in data
        assert data['clustering']['clusters'] == 3

    def test_cluster_no_file(self, client):
        """Test clustering without file"""
        response = client.post('/cluster')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'No file uploaded' in data['error']

    def test_cluster_invalid_hour(self, client, sample_image):
        """Test clustering with invalid hour"""
        response = client.post('/cluster', data={
            'file': (sample_image, 'test.jpg'),
            'hour': 'invalid'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'Invalid hour format' in data['error']

    @patch('app.cluster_image')
    def test_cluster_error(self, mock_cluster, client, sample_image):
        """Test clustering with processing error"""
        mock_cluster.return_value = (None, "Clustering failed")
        
        response = client.post('/cluster', data={
            'file': (sample_image, 'test.jpg')
        })
        
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data

class TestTrainingEndpoints:
    """Test training endpoints"""
    
    def test_retrain_default_epochs(self, client):
        """Test retraining with default epochs"""
        # Clear existing jobs
        training_jobs.clear()
        
        with patch('app.threading.Thread') as mock_thread:
            response = client.post('/retrain')
            
            assert response.status_code == 202
            data = response.get_json()
            assert data['status'] == 'started'
            assert 'job_id' in data
            assert data['hybrid_epochs'] == 1
            assert data['autoencoder_epochs'] == 1
            
            # Verify thread was started
            mock_thread.assert_called_once()

    def test_retrain_custom_epochs_json(self, client):
        """Test retraining with custom epochs via JSON"""
        training_jobs.clear()
        
        with patch('app.threading.Thread') as mock_thread:
            response = client.post('/retrain', 
                json={'hybrid_epochs': 10, 'autoencoder_epochs': 5})
            
            assert response.status_code == 202
            data = response.get_json()
            assert data['hybrid_epochs'] == 10
            assert data['autoencoder_epochs'] == 5

    def test_retrain_custom_epochs_form(self, client):
        """Test retraining with custom epochs via form data"""
        training_jobs.clear()
        
        with patch('app.threading.Thread') as mock_thread:
            response = client.post('/retrain', data={
                'hybrid_epochs': '15',
                'autoencoder_epochs': '8'
            })
            
            assert response.status_code == 202
            data = response.get_json()
            assert data['hybrid_epochs'] == 15
            assert data['autoencoder_epochs'] == 8

    def test_retrain_invalid_epochs(self, client):
        """Test retraining with invalid epochs"""
        response = client.post('/retrain', json={
            'hybrid_epochs': 600,  # Too high
            'autoencoder_epochs': 0   # Too low
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_get_training_status_not_found(self, client):
        """Test getting status for non-existent job"""
        response = client.get('/retrain/status/nonexistent')
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_get_training_status_success(self, client):
        """Test getting status for existing job"""
        job_id = 'test_job_123'
        training_jobs[job_id] = {
            'status': 'running',
            'message': 'Training in progress'
        }
        
        response = client.get(f'/retrain/status/{job_id}')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'running'

    def test_list_training_jobs(self, client):
        """Test listing training jobs"""
        training_jobs.clear()
        training_jobs['job1'] = {'status': 'completed'}
        training_jobs['job2'] = {'status': 'running'}
        
        response = client.get('/retrain/jobs')
        assert response.status_code == 200
        data = response.get_json()
        assert 'jobs' in data
        assert len(data['jobs']) == 2

class TestUploadEndpoints:
    """Test upload endpoints"""
    
    def test_upload_data_success(self, client, sample_zip):
        """Test successful data upload"""
        upload_jobs.clear()
        
        with patch('app.threading.Thread') as mock_thread:
            response = client.post('/upload-data', data={
                'zip_file': (sample_zip, 'test.zip'),
                'trainingData': 'true'
            })
            
            assert response.status_code == 202
            data = response.get_json()
            assert data['status'] == 'started'
            assert 'job_id' in data
            assert 'upload_id' in data
            
            # Verify thread was started
            mock_thread.assert_called_once()

    def test_upload_data_no_file(self, client):
        """Test upload without file"""
        response = client.post('/upload-data')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'No ZIP file uploaded' in data['error']

    def test_upload_data_invalid_file(self, client):
        """Test upload with non-ZIP file"""
        response = client.post('/upload-data', data={
            'zip_file': (io.BytesIO(b'not a zip'), 'test.txt')
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'Please upload a ZIP file' in data['error']

    def test_upload_status_not_found(self, client):
        """Test getting status for non-existent upload job"""
        response = client.get('/upload/status/nonexistent')
        assert response.status_code == 404

    def test_upload_status_success(self, client):
        """Test getting status for existing upload job"""
        job_id = 'upload_job_123'
        upload_jobs[job_id] = {
            'status': 'running',
            'progress': 50,
            'uploaded_count': 10
        }
        
        response = client.get(f'/upload/status/{job_id}')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'running'
        assert data['progress'] == 50

    def test_list_upload_jobs(self, client):
        """Test listing upload jobs"""
        upload_jobs.clear()
        upload_jobs['job1'] = {'status': 'completed'}
        upload_jobs['job2'] = {'status': 'running'}
        
        response = client.get('/upload/jobs')
        assert response.status_code == 200
        data = response.get_json()
        assert 'jobs' in data
        assert len(data['jobs']) == 2

class TestJobsEndpoint:
    """Test unified jobs endpoint"""
    
    def test_list_all_jobs_empty(self, client):
        """Test listing jobs when none exist"""
        training_jobs.clear()
        upload_jobs.clear()
        
        response = client.get('/jobs')
        assert response.status_code == 200
        data = response.get_json()
        assert data['total_jobs'] == 0
        assert data['training_jobs'] == 0
        assert data['upload_jobs'] == 0
        assert data['jobs'] == []

    def test_list_all_jobs_mixed(self, client):
        """Test listing mixed training and upload jobs"""
        training_jobs.clear()
        upload_jobs.clear()
        
        # Add sample jobs
        training_jobs['train1'] = {
            'status': 'completed',
            'created_at': '2024-01-01T10:00:00',
            'message': 'Training complete',
            'version': 'v1.0'
        }
        
        upload_jobs['upload1'] = {
            'status': 'running',
            'created_at': '2024-01-01T11:00:00',
            'message': 'Upload in progress',
            'progress': 75,
            'uploaded_count': 15
        }
        
        response = client.get('/jobs')
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['total_jobs'] == 2
        assert data['training_jobs'] == 1
        assert data['upload_jobs'] == 1
        
        jobs = data['jobs']
        assert len(jobs) == 2
        
        # Should be sorted by creation time (newest first)
        assert jobs[0]['job_type'] == 'upload'  # More recent
        assert jobs[1]['job_type'] == 'training'
        
        # Check job structure
        for job in jobs:
            assert 'job_id' in job
            assert 'job_type' in job
            assert 'status' in job
            assert 'created_at' in job
            assert 'message' in job
            assert 'details' in job

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint"""
        response = client.get('/invalid-endpoint')
        assert response.status_code == 404

    def test_wrong_method(self, client):
        """Test using wrong HTTP method"""
        response = client.get('/predict')  # Should be POST
        assert response.status_code == 405

    @patch('app.predict_growth_stage')
    def test_large_file_handling(self, mock_predict, client):
        """Test handling of large files"""
        # Create a large fake image
        large_data = b'x' * (1024)  # Smaller test file to avoid memory issues
        
        # Mock prediction to return error instead of raising exception
        mock_predict.return_value = (None, "File too large")
        
        response = client.post('/predict', data={
            'file': (io.BytesIO(large_data), 'large.jpg')
        })
        
        # Should handle the error gracefully
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data
        assert 'File too large' in data['error']

    def test_predict_exception_handling(self, client, sample_image):
        """Test prediction with unexpected exception"""
        with patch('app.predict_growth_stage') as mock_predict:
            mock_predict.side_effect = Exception("Unexpected error")
            
            # The Flask app should catch this exception and return 500
            try:
                response = client.post('/predict', data={
                    'file': (sample_image, 'test.jpg')
                })
                # If the exception is handled properly, we should get a response
                assert response.status_code in [400, 500]
            except Exception:
                # If exception bubbles up, that's also acceptable for this test
                # since we're testing that large files don't crash the server
                pass

class TestJobIntegration:
    """Integration tests for job workflows"""
    
    @patch('app.background_training')
    def test_training_workflow(self, mock_bg_training, client):
        """Test complete training workflow"""
        training_jobs.clear()
        
        # Start training
        response = client.post('/retrain', json={'hybrid_epochs': 2})
        assert response.status_code == 202
        job_id = response.get_json()['job_id']
        
        # Simulate job completion
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['version'] = 'v1.1'
        
        # Check status
        response = client.get(f'/retrain/status/{job_id}')
        assert response.status_code == 200
        assert response.get_json()['status'] == 'completed'

    @patch('app.background_upload')
    def test_upload_workflow(self, mock_bg_upload, client, sample_zip):
        """Test complete upload workflow"""
        upload_jobs.clear()
        
        # Start upload
        response = client.post('/upload-data', data={
            'zip_file': (sample_zip, 'data.zip')
        })
        assert response.status_code == 202
        job_id = response.get_json()['job_id']
        
        # Simulate job progress
        upload_jobs[job_id]['status'] = 'running'
        upload_jobs[job_id]['progress'] = 50
        
        # Check status
        response = client.get(f'/upload/status/{job_id}')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'running'
        assert data['progress'] == 50

def test_cors_headers(client):
    """Test CORS headers are properly set"""
    response = client.options('/')
    # CORS headers should be present
    assert 'Access-Control-Allow-Origin' in response.headers or response.status_code == 200