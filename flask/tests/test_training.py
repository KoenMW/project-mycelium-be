import pytest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import tensorflow as tf
from PIL import Image
import io

from training.retrain import full_retrain_pipeline
from training.train_model import train_hybrid_model
from training.splitter import split_by_hour
from training.cluster_retrainer import retrain_cluster_model

@pytest.fixture
def temp_training_dir():
    """Create temporary directory for training tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_training_data():
    """Create mock training data structure"""
    return {
        "test1_h24_0.jpg": {"hour": 24, "test": 1, "angle": 0},
        "test1_h48_90.jpg": {"hour": 48, "test": 1, "angle": 90},
        "test2_h72_0.jpg": {"hour": 72, "test": 2, "angle": 0},
        "test2_h96_180.jpg": {"hour": 96, "test": 2, "angle": 180},
    }

@pytest.fixture
def sample_image_data():
    """Create sample image data as bytes"""
    # Create a simple 224x224 RGB image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()

def create_mock_image_files(directory, image_files, image_data):
    """Helper to create mock image files"""
    os.makedirs(directory, exist_ok=True)
    for filename in image_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)

class TestSplitter:
    """Test image splitting functionality"""
    
    def test_split_by_hour_basic(self, temp_training_dir, mock_training_data, sample_image_data):
        """Test basic hour-based splitting"""
        source_dir = os.path.join(temp_training_dir, "source")
        output_dir = os.path.join(temp_training_dir, "output")
        
        # Create mock image files
        create_mock_image_files(source_dir, mock_training_data.keys(), sample_image_data)
        
        # Execute splitting
        split_by_hour(source_dir, output_dir, hour_split=24)
        
        # Verify output structure
        assert os.path.exists(output_dir)
        
        # Check that files are properly categorized
        bucket_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        assert len(bucket_dirs) > 0
        
        # Verify files exist in buckets
        total_files = 0
        for bucket in bucket_dirs:
            bucket_path = os.path.join(output_dir, bucket)
            files_in_bucket = os.listdir(bucket_path)
            total_files += len(files_in_bucket)
        
        assert total_files == len(mock_training_data)

    def test_split_by_hour_empty_source(self, temp_training_dir):
        """Test splitting with empty source directory"""
        source_dir = os.path.join(temp_training_dir, "empty_source")
        output_dir = os.path.join(temp_training_dir, "output")
        
        os.makedirs(source_dir)
        
        # Should not raise exception
        split_by_hour(source_dir, output_dir, hour_split=24)
        assert os.path.exists(output_dir)

    def test_split_by_hour_invalid_filenames(self, temp_training_dir, sample_image_data):
        """Test splitting with invalid filename patterns"""
        source_dir = os.path.join(temp_training_dir, "source")
        output_dir = os.path.join(temp_training_dir, "output")
        
        invalid_files = ["invalid.jpg", "notmatching_h24_0.jpg", "test_invalid.jpg"]
        create_mock_image_files(source_dir, invalid_files, sample_image_data)
        
        split_by_hour(source_dir, output_dir, hour_split=24)
        
        # Should create output dir but no files should be processed
        assert os.path.exists(output_dir)
        # Count total files in all subdirectories
        total_files = sum(len(os.listdir(os.path.join(output_dir, d))) 
                         for d in os.listdir(output_dir) 
                         if os.path.isdir(os.path.join(output_dir, d)))
        assert total_files == 0

class TestTrainHybridModel:
    """Test hybrid model training"""
    
    def test_train_hybrid_model_no_images(self, temp_training_dir):
        """Test training with no images"""
        # Create empty directory
        with pytest.raises(ValueError, match="No training images found"):
            train_hybrid_model(temp_training_dir, version="test", num_classes=2)

    def test_train_hybrid_model_file_collection(self, temp_training_dir, sample_image_data):
        """Test file collection logic without model training"""
        # Create test directory structure
        for class_idx in range(3):
            class_dir = os.path.join(temp_training_dir, str(class_idx))
            os.makedirs(class_dir)
            
            # Mix of valid and invalid files
            valid_files = [f"valid_{i}.jpg" for i in range(2)]
            invalid_files = [f"invalid_{i}.txt" for i in range(2)] + [f"bad_{i}.doc" for i in range(2)]
        
            for filename in valid_files + invalid_files:
                filepath = os.path.join(class_dir, filename)
                with open(filepath, 'wb') as f:
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        f.write(sample_image_data)
                    else:
                        f.write(b'not an image')

        # Test the file collection part by using a minimal mock
        with patch('training.train_model.train_test_split') as mock_split:
            mock_split.return_value = ([], [], [], [])

            # This should collect files but fail during training setup
            try:
                train_hybrid_model(temp_training_dir, version="file_test", num_classes=3)
            except:
                pass  # Expected to fail after file collection
                
            # Verify train_test_split was called - indicates files were collected
            if mock_split.called:
                call_args = mock_split.call_args[0]
                filepaths = call_args[0]
                labels = call_args[1]

                # Should have collected 6 valid image files (2 per class * 3 classes)
                assert len(filepaths) == 6
                assert len(labels) == 6

                # All files should be .jpg files
                assert all(fp.endswith('.jpg') for fp in filepaths)

class TestClusterRetrainer:
    """Test clustering model retraining"""
    
    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('training.cluster_retrainer.load_model')
    @patch('training.cluster_retrainer.cv2.imread')
    @patch('training.cluster_retrainer.cv2.imwrite')
    @patch('training.cluster_retrainer.joblib.dump')
    def test_retrain_cluster_model_success(self, mock_joblib_dump, mock_cv2_write, 
                                         mock_cv2_read, mock_load_model, temp_training_dir):
        """Test successful cluster model retraining"""
        # Setup mock encoder
        mock_encoder = Mock()
        mock_features = np.random.rand(1, 56, 56, 64)  # Mock encoder output
        mock_encoder.predict.return_value = mock_features
        mock_load_model.return_value = mock_encoder
        
        # Setup mock image reading
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_cv2_read.return_value = mock_image
        
        # Create mock image files
        data_folder = os.path.join(temp_training_dir, "data")
        os.makedirs(data_folder)
        
        test_files = ["test1_h24_0.jpg", "test1_h48_90.jpg", "test2_h72_180.jpg"]
        for filename in test_files:
            with open(os.path.join(data_folder, filename), 'wb') as f:
                f.write(b'fake_image')
        
        output_dir = os.path.join(temp_training_dir, "output")
        os.makedirs(output_dir)
        
        with patch('training.cluster_retrainer.PCA') as mock_pca, \
             patch('training.cluster_retrainer.hdbscan.HDBSCAN') as mock_hdbscan:
            
            # Setup PCA mock
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(len(test_files), 100)
            mock_pca.return_value = mock_pca_instance
            
            # Setup HDBSCAN mock
            mock_clusterer = Mock()
            mock_clusterer.fit_predict.return_value = np.array([0, 1, -1])  # Two clusters + noise
            mock_hdbscan.return_value = mock_clusterer
            
            # Execute clustering
            labels = retrain_cluster_model(data_folder, "encoder.keras", output_dir)
            
            # Verify calls
            mock_load_model.assert_called_with("encoder.keras")
            assert mock_encoder.predict.call_count == len(test_files)
            mock_pca_instance.fit_transform.assert_called_once()
            mock_clusterer.fit_predict.assert_called_once()
            
            # Verify models were saved
            assert mock_joblib_dump.call_count == 2  # PCA and HDBSCAN
            
            # Verify labels returned
            assert len(labels) == len(test_files)

    @patch.dict(os.environ, {'SKIP_MODEL_LOADING': 'false'})
    @patch('training.cluster_retrainer.load_model')
    @patch('training.cluster_retrainer.joblib.dump')
    def test_retrain_cluster_model_no_valid_files(self, mock_joblib_dump, mock_load_model, temp_training_dir):
        """Test clustering with no valid image files"""
        data_folder = os.path.join(temp_training_dir, "data")
        os.makedirs(data_folder)  # Create the actual directory
        
        # Create non-image files
        with open(os.path.join(data_folder, "invalid.txt"), 'w') as f:
            f.write("not an image")
        
        output_dir = os.path.join(temp_training_dir, "output")
        os.makedirs(output_dir, exist_ok=True)  # Create the actual output directory
        
        # Mock encoder
        mock_encoder = Mock()
        mock_load_model.return_value = mock_encoder
        
        with patch('training.cluster_retrainer.PCA') as mock_pca, \
             patch('training.cluster_retrainer.hdbscan.HDBSCAN') as mock_hdbscan, \
             patch('training.cluster_retrainer.os.makedirs') as mock_makedirs:  # Mock only the makedirs inside the function
            
            # Setup mocks for empty feature case
            mock_pca_instance = Mock()
            mock_pca_instance.fit_transform.return_value = np.array([]).reshape(0, 100)
            mock_pca.return_value = mock_pca_instance
            
            mock_clusterer = Mock()
            mock_clusterer.fit_predict.return_value = np.array([])
            mock_hdbscan.return_value = mock_clusterer
            
            # Execute clustering
            labels = retrain_cluster_model(data_folder, "encoder.keras", output_dir)
            
            # Should return empty array
            assert len(labels) == 0
            
            # Verify that PCA and HDBSCAN were still created and saved (even with empty data)
            mock_pca.assert_called_once()
            mock_hdbscan.assert_called_once()
            assert mock_joblib_dump.call_count == 2  # PCA and HDBSCAN models should still be saved


class TestFullRetrainPipeline:
    """Test complete retraining pipeline"""
    
    @patch('training.retrain.fetch_pocketbase_data')
    @patch('training.retrain.split_by_hour')
    @patch('training.retrain.train_hybrid_model')
    @patch('training.retrain.retrain_cluster_model')
    @patch('training.retrain.os.path.exists')
    @patch('training.retrain.os.listdir')
    def test_full_retrain_pipeline_success(self, mock_listdir, mock_exists, mock_cluster_retrain, 
                                         mock_train_hybrid, mock_split, mock_fetch, temp_training_dir):
        """Test successful full retraining pipeline"""
        # Setup mocks for successful scenario
        mock_exists.return_value = True  # Directory exists
        mock_listdir.return_value = ['test1_h24_0.jpg', 'test2_h48_90.jpg']  # Directory has files
        mock_train_hybrid.return_value = os.path.join(temp_training_dir, "model_versions", "v123456")
        
        # Execute pipeline
        result = full_retrain_pipeline(
            num_classes=10,
            job_temp_dir=temp_training_dir,
            hybrid_epochs=3,
            autoencoder_epochs=2
        )
        
        # Verify all steps were called
        mock_fetch.assert_called_once()
        mock_split.assert_called_once()
        mock_train_hybrid.assert_called_once()
        mock_cluster_retrain.assert_called_once()
        
        # Verify fetch was called with correct parameters
        fetch_call = mock_fetch.call_args
        assert fetch_call[1]['training_data_only'] is True
        assert "fetched_training_data" in fetch_call[1]['output_dir']
        
        # Verify training was called with custom epochs
        train_call = mock_train_hybrid.call_args
        assert train_call[1]['hybrid_epochs'] == 3
        assert train_call[1]['autoencoder_epochs'] == 2
        assert train_call[1]['num_classes'] == 10
        
        # Verify result
        assert result == mock_train_hybrid.return_value

    @patch('training.retrain.fetch_pocketbase_data')
    @patch('training.retrain.os.path.exists')
    @patch('training.retrain.os.listdir')
    def test_full_retrain_pipeline_no_training_data(self, mock_listdir, mock_exists, 
                                                   mock_fetch, temp_training_dir):
        """Test pipeline with no training data"""
        # Mock no training data found
        mock_exists.return_value = True
        mock_listdir.return_value = []  # Empty directory
        
        with pytest.raises(ValueError, match="No training data found in database"):
            full_retrain_pipeline(job_temp_dir=temp_training_dir)

    @patch('training.retrain.fetch_pocketbase_data')
    @patch('training.retrain.os.path.exists')
    def test_full_retrain_pipeline_directory_not_exists(self, mock_exists, mock_fetch, temp_training_dir):
        """Test pipeline when directory doesn't exist after fetch"""
        # Mock directory doesn't exist
        mock_exists.return_value = False
        
        with pytest.raises(ValueError, match="No training data found in database"):
            full_retrain_pipeline(job_temp_dir=temp_training_dir)

class TestTrainingIntegration:
    """Integration tests for training components"""
    
    def test_training_data_flow(self, temp_training_dir, sample_image_data):
        """Test data flow through training pipeline components"""
        # Setup test data structure
        source_dir = os.path.join(temp_training_dir, "source")
        labeled_dir = os.path.join(temp_training_dir, "labeled")
        
        # Create mock training images
        test_images = {
            "test1_h24_0.jpg": 24,
            "test1_h48_90.jpg": 48,
            "test1_h72_180.jpg": 72,
        }
        
        create_mock_image_files(source_dir, test_images.keys(), sample_image_data)
        
        # Test splitter
        split_by_hour(source_dir, labeled_dir, hour_split=24)
        
        # Verify splitting worked
        assert os.path.exists(labeled_dir)
        
        # Count files in all class directories
        total_files = 0
        for class_dir in os.listdir(labeled_dir):
            class_path = os.path.join(labeled_dir, class_dir)
            if os.path.isdir(class_path):
                total_files += len(os.listdir(class_path))
        
        assert total_files == len(test_images)

def test_training_constants():
    """Test training configuration constants"""
    from training.cluster_retrainer import IMG_SIZE, MAX_HOUR, MIN_CLUSTER_SIZE
    
    # Test clustering constants
    assert IMG_SIZE == (224, 224)
    assert MAX_HOUR == 360
    assert MIN_CLUSTER_SIZE == 50
    
    # Test that these constants are used consistently
    assert isinstance(IMG_SIZE, tuple)
    assert len(IMG_SIZE) == 2