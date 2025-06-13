import os
import zipfile
import tempfile
import shutil
import re
from datetime import datetime
from prediction.predictor import predict_growth_stage
from database.data import upload_to_pocketbase

# === Global Job Tracking ===
# Dictionary to store status and metadata for all active/completed upload jobs
# Key: job_id (string), Value: dict with status, progress, messages, etc.
upload_jobs = {}

def collect_images_recursively(root_path):
    """
    Recursively collect all image files from a directory structure.
    
    This function traverses a directory tree and finds all image files,
    collecting metadata about their location and structure for processing.
    
    Args:
        root_path (str): Root directory to start the recursive search
        
    Returns:
        list: List of dictionaries containing image file information including
              file path, filename, relative path, and parent folder details
    """
    def recurse_directory(current_path, relative_path=""):
        """
        Recursive function to traverse directories and collect image files.
        
        This internal function handles the actual directory traversal,
        building relative paths and collecting file metadata.
        """
        files = []
        
        try:
            # Iterate through all items in the current directory
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                item_relative_path = os.path.join(relative_path, item) if relative_path else item
                
                if os.path.isfile(item_path):
                    # Check if it's an image file by extension
                    if item.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                        # Collect comprehensive file metadata
                        files.append({
                            "file_path": item_path,                                    # Full system path
                            "filename": item,                                          # Just the filename
                            "relative_path": item_relative_path,                       # Path relative to root
                            "parent_folder": os.path.basename(os.path.dirname(item_path)),  # Immediate parent folder
                            "full_relative_path": relative_path                        # Full relative directory path
                        })
                
                elif os.path.isdir(item_path):
                    # Recursively process subdirectories
                    files.extend(recurse_directory(item_path, item_relative_path))
                    
        except PermissionError as e:
            print(f"⚠️ Permission denied accessing {current_path}: {e}")
        except Exception as e:
            print(f"⚠️ Error processing {current_path}: {e}")
        
        return files
    
    return recurse_directory(root_path)

def extract_timestamp_info(folder_path):
    """
    Extract timestamp information from folder path using pattern matching.
    
    This function recognizes various timestamp formats commonly used in
    experimental folder naming conventions and converts them to datetime objects.
    
    Args:
        folder_path (str): Path containing timestamp information
        
    Returns:
        datetime or None: Parsed datetime object if pattern found, None otherwise
    """
    # Define various timestamp patterns found in experimental data
    timestamp_patterns = [
        r'(\d{2})-(\d{2})-(\d{2})___(\d{2})-(\d{2})',  # Format: 24-11-12___18-59
        r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})',    # Format: 2024-11-12_18-59
        r'(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})',       # Format: 241112_1859
    ]
    
    # Try each pattern until one matches
    for pattern in timestamp_patterns:
        match = re.search(pattern, folder_path)
        if match:
            groups = match.groups()
            if len(groups) == 5:
                # Handle different year formats (YY vs YYYY)
                if len(groups[0]) == 2:  # Two-digit year (YY)
                    year = int(f"20{groups[0]}")  # Assume 20XX
                else:  # Four-digit year (YYYY)
                    year = int(groups[0])
                
                # Extract other time components
                month = int(groups[1])
                day = int(groups[2])
                hour = int(groups[3])
                minute = int(groups[4])
                
                # Return constructed datetime object
                return datetime(year, month, day, hour, minute)
    
    # Return None if no timestamp pattern is found
    return None

def background_upload(job_id, zip_file_path, is_training_data, temp_dir):
    """
    Process uploaded ZIP file in the background, uploading images to database.
    
    This function handles the complete pipeline for processing experimental image data:
    1. Extract and analyze ZIP file structure
    2. Recursively collect all image files
    3. Extract experimental metadata (test number, timestamps)
    4. Process each image through prediction
    5. Upload processed data to the database
    
    Args:
        job_id (str): Unique identifier for tracking this upload job
        zip_file_path (str): Path to the uploaded ZIP file containing images
        is_training_data (bool): Whether these images should be marked as training data
        temp_dir (str): Temporary directory for processing files
        
    Returns:
        None: Updates global upload_jobs dictionary with progress and results
    """
    try:
        # === Job Initialization ===
        print(f"🚀 Starting background upload job {job_id}")
        upload_jobs[job_id]["status"] = "running"
        upload_jobs[job_id]["message"] = "Processing ZIP file..."
        
        # === ZIP File Extraction ===
        # Extract ZIP contents to a temporary directory for processing
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        upload_jobs[job_id]["message"] = "Collecting images recursively..."
        
        # === Image Collection ===
        # Recursively find all image files in the extracted directory structure
        all_images = collect_images_recursively(extract_dir)
        
        # Validate that images were found
        if not all_images:
            raise ValueError("No image files found in ZIP")
        
        print(f"📸 Found {len(all_images)} images recursively")
        
        # === Extract Experimental Metadata ===
        # Determine test number from directory structure
        test_number = 1  # Default fallback
        for image_info in all_images:
            # Look for "test" followed by numbers in the path
            test_match = re.search(r'test(\d+)', image_info["relative_path"], re.IGNORECASE)
            if test_match:
                test_number = int(test_match.group(1))
                break
        
        upload_jobs[job_id]["message"] = f"Found test {test_number}, analyzing timestamps..."
        
        # === Timestamp Analysis ===
        # Find reference timestamp for calculating relative hours
        all_timestamps = []
        for image_info in all_images:
            timestamp = extract_timestamp_info(image_info["full_relative_path"])
            if timestamp:
                all_timestamps.append(timestamp)
        
        # Use earliest timestamp as reference point (experiment start)
        reference_timestamp = min(all_timestamps) if all_timestamps else datetime.now()
        print(f"📅 Reference timestamp: {reference_timestamp}")
        
        # === Processing Setup ===
        upload_jobs[job_id]["total_images"] = len(all_images)
        upload_jobs[job_id]["message"] = f"Processing {len(all_images)} images..."
        
        # Initialize processing counters
        uploaded_count = 0
        failed_count = 0
        
        # === Image Processing Loop ===
        for i, image_info in enumerate(all_images, 1):
            # Update progress tracking
            progress = (i / len(all_images)) * 100
            upload_jobs[job_id]["progress"] = round(progress, 1)
            upload_jobs[job_id]["message"] = f"Processing image {i}/{len(all_images)} ({progress:.1f}%)"
            
            # === Extract Camera Angle ===
            # Try to extract angle number from filename (expected: 1, 2, 3, or 4)
            try:
                angle = int(os.path.splitext(image_info["filename"])[0])
                if angle not in [1, 2, 3, 4]:  # Validate angle range
                    angle = 1  # Default fallback
            except ValueError:
                angle = 1  # Default if filename isn't a number
            
            # === Calculate Time Offset ===
            # Calculate hours since experiment start based on timestamp
            timestamp = extract_timestamp_info(image_info["full_relative_path"])
            if timestamp and reference_timestamp:
                time_diff = timestamp - reference_timestamp
                calculated_hour = int(time_diff.total_seconds() / 3600)  # Convert to hours
            else:
                calculated_hour = 0  # Fallback if no timestamp found
            
            # === Growth Stage Prediction ===
            # Use ML model to predict growth stage from original image
            try:
                # Read original image as bytes for prediction
                with open(image_info["file_path"], 'rb') as img_file:
                    img_bytes = img_file.read()
                
                # Run growth stage prediction
                prediction_result, prediction_error = predict_growth_stage(img_bytes, version="default")
                
                if prediction_result and not prediction_error:
                    # Use predicted class as estimated day
                    predicted_class = int(prediction_result.get("predicted_class", "0"))
                    estimated_day = predicted_class
                    print(f"🔮 Predicted day {estimated_day} for {image_info['filename']}")
                else:
                    # Fallback: calculate day from hours if prediction fails
                    estimated_day = calculated_hour // 24
                    print(f"⚠️ Prediction failed for {image_info['filename']}, using timestamp fallback")
                    
            except Exception as e:
                # Fallback: calculate day from hours if prediction errors
                estimated_day = calculated_hour // 24
                print(f"⚠️ Prediction error for {image_info['filename']}: {e}")
            
            # === Database Upload ===
            # Upload original image and metadata to PocketBase database
            if upload_to_pocketbase(image_info["file_path"], test_number, calculated_hour, angle, estimated_day, is_training_data):
                uploaded_count += 1
                print(f"✅ [{i}/{len(all_images)}] Uploaded: {image_info['filename']} (test{test_number}_h{calculated_hour}_{angle})")
            else:
                failed_count += 1
                print(f"❌ [{i}/{len(all_images)}] Failed: {image_info['filename']}")
        
        # === Job Completion ===
        # Update job status with final results
        upload_jobs[job_id]["status"] = "completed"
        upload_jobs[job_id]["uploaded_count"] = uploaded_count
        upload_jobs[job_id]["failed_count"] = failed_count
        upload_jobs[job_id]["test_number"] = test_number
        upload_jobs[job_id]["progress"] = 100
        upload_jobs[job_id]["message"] = f"Upload completed! {uploaded_count} uploaded, {failed_count} failed"
        
        print(f"✅ Background upload job {job_id} completed")
        print(f"📊 Results: {uploaded_count} uploaded, {failed_count} failed from {len(all_images)} total images")
        
    except Exception as e:
        # === Error Handling ===
        # Capture and store any processing failures
        upload_jobs[job_id]["status"] = "failed"
        upload_jobs[job_id]["error"] = str(e)
        upload_jobs[job_id]["message"] = f"Upload failed: {str(e)}"
        print(f"❌ Background upload job {job_id} failed: {str(e)}")
    
    finally:
        # === Cleanup Process ===
        # Always clean up temporary directory regardless of success/failure
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temp directory for upload {job_id}")
            except Exception as e:
                print(f"⚠️ Could not clean up temp directory for upload {job_id}: {e}")