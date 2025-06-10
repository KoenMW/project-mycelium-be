import os
import zipfile
import tempfile
import shutil
import re
from datetime import datetime
from segmentation.segment_mycelium import segment_and_save
from prediction.predictor import predict_growth_stage
from database.data import upload_to_pocketbase

# Global dictionary to track upload jobs
upload_jobs = {}

def collect_images_recursively(root_path):
    """
    Recursively collect all image files from a directory structure
    Returns list of image file information
    """
    def recurse_directory(current_path, relative_path=""):
        """Recursive function to traverse directories - following the provided pattern"""
        files = []
        
        try:
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                item_relative_path = os.path.join(relative_path, item) if relative_path else item
                
                if os.path.isfile(item_path):
                    # Check if it's an image file
                    if item.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                        files.append({
                            "file_path": item_path,
                            "filename": item,
                            "relative_path": item_relative_path,
                            "parent_folder": os.path.basename(os.path.dirname(item_path)),
                            "full_relative_path": relative_path
                        })
                
                elif os.path.isdir(item_path):
                    # Recursively process subdirectory
                    files.extend(recurse_directory(item_path, item_relative_path))
                    
        except PermissionError as e:
            print(f"⚠️ Permission denied accessing {current_path}: {e}")
        except Exception as e:
            print(f"⚠️ Error processing {current_path}: {e}")
        
        return files
    
    return recurse_directory(root_path)

def extract_timestamp_info(folder_path):
    """Extract timestamp information from folder path"""
    # Look for timestamp patterns in the folder path
    timestamp_patterns = [
        r'(\d{2})-(\d{2})-(\d{2})___(\d{2})-(\d{2})',  # 24-11-12___18-59
        r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})',    # 2024-11-12_18-59
        r'(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})',       # 241112_1859
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, folder_path)
        if match:
            groups = match.groups()
            if len(groups) == 5:
                # Handle different date formats
                if len(groups[0]) == 2:  # YY format
                    year = int(f"20{groups[0]}")
                else:  # YYYY format
                    year = int(groups[0])
                
                month = int(groups[1])
                day = int(groups[2])
                hour = int(groups[3])
                minute = int(groups[4])
                
                return datetime(year, month, day, hour, minute)
    
    return None

def background_upload(job_id, zip_file_path, is_training_data, temp_dir):
    """Background upload function with recursive file collection"""
    try:
        print(f"🚀 Starting background upload job {job_id}")
        upload_jobs[job_id]["status"] = "running"
        upload_jobs[job_id]["message"] = "Processing ZIP file..."
        
        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        upload_jobs[job_id]["message"] = "Collecting images recursively..."
        
        # Recursively collect all image files
        all_images = collect_images_recursively(extract_dir)
        
        if not all_images:
            raise ValueError("No image files found in ZIP")
        
        print(f"📸 Found {len(all_images)} images recursively")
        
        # Extract test number from the first folder structure found
        test_number = 1
        for image_info in all_images:
            test_match = re.search(r'test(\d+)', image_info["relative_path"], re.IGNORECASE)
            if test_match:
                test_number = int(test_match.group(1))
                break
        
        upload_jobs[job_id]["message"] = f"Found test {test_number}, analyzing timestamps..."
        
        # Find reference timestamp from all collected images
        all_timestamps = []
        for image_info in all_images:
            timestamp = extract_timestamp_info(image_info["full_relative_path"])
            if timestamp:
                all_timestamps.append(timestamp)
        
        reference_timestamp = min(all_timestamps) if all_timestamps else datetime.now()
        print(f"📅 Reference timestamp: {reference_timestamp}")
        
        upload_jobs[job_id]["total_images"] = len(all_images)
        upload_jobs[job_id]["message"] = f"Processing {len(all_images)} images..."
        
        # Process each image
        segmented_dir = os.path.join(temp_dir, "segmented")
        os.makedirs(segmented_dir, exist_ok=True)
        
        uploaded_count = 0
        failed_count = 0
        
        for i, image_info in enumerate(all_images, 1):
            progress = (i / len(all_images)) * 100
            upload_jobs[job_id]["progress"] = round(progress, 1)
            upload_jobs[job_id]["message"] = f"Processing image {i}/{len(all_images)} ({progress:.1f}%)"
            
            # Extract angle from filename
            try:
                angle = int(os.path.splitext(image_info["filename"])[0])
                if angle not in [1, 2, 3, 4]:
                    angle = 1
            except ValueError:
                angle = 1
            
            # Calculate hour from timestamp
            timestamp = extract_timestamp_info(image_info["full_relative_path"])
            if timestamp and reference_timestamp:
                time_diff = timestamp - reference_timestamp
                calculated_hour = int(time_diff.total_seconds() / 3600)
            else:
                calculated_hour = 0
            
            # Segment and save temporarily
            temp_segmented_path = os.path.join(segmented_dir, f"temp_{image_info['filename']}")
            
            if segment_and_save(image_info["file_path"], temp_segmented_path):
                # Predict growth stage to get estimated day
                try:
                    with open(temp_segmented_path, 'rb') as img_file:
                        img_bytes = img_file.read()
                    
                    prediction_result, prediction_error = predict_growth_stage(img_bytes, version="default")
                    
                    if prediction_result and not prediction_error:
                        predicted_class = int(prediction_result.get("predicted_class", "0"))
                        estimated_day = predicted_class
                        print(f"🔮 Predicted day {estimated_day} for {image_info['filename']}")
                    else:
                        estimated_day = calculated_hour // 24
                        print(f"⚠️ Prediction failed for {image_info['filename']}, using timestamp fallback")
                        
                except Exception as e:
                    estimated_day = calculated_hour // 24
                    print(f"⚠️ Prediction error for {image_info['filename']}: {e}")
                
                # Upload to database
                if upload_to_pocketbase(temp_segmented_path, test_number, calculated_hour, angle, estimated_day, is_training_data):
                    uploaded_count += 1
                    print(f"✅ [{i}/{len(all_images)}] Uploaded: {image_info['filename']} (test{test_number}_h{calculated_hour}_{angle})")
                else:
                    failed_count += 1
                    print(f"❌ [{i}/{len(all_images)}] Failed: {image_info['filename']}")
                
                # Clean up temp file
                if os.path.exists(temp_segmented_path):
                    os.remove(temp_segmented_path)
            else:
                failed_count += 1
                print(f"❌ [{i}/{len(all_images)}] Segmentation failed: {image_info['filename']}")
        
        # Job completed successfully
        upload_jobs[job_id]["status"] = "completed"
        upload_jobs[job_id]["uploaded_count"] = uploaded_count
        upload_jobs[job_id]["failed_count"] = failed_count
        upload_jobs[job_id]["test_number"] = test_number
        upload_jobs[job_id]["progress"] = 100
        upload_jobs[job_id]["message"] = f"Upload completed! {uploaded_count} uploaded, {failed_count} failed"
        
        print(f"✅ Background upload job {job_id} completed")
        print(f"📊 Results: {uploaded_count} uploaded, {failed_count} failed from {len(all_images)} total images")
        
    except Exception as e:
        upload_jobs[job_id]["status"] = "failed"
        upload_jobs[job_id]["error"] = str(e)
        upload_jobs[job_id]["message"] = f"Upload failed: {str(e)}"
        print(f"❌ Background upload job {job_id} failed: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temp directory for upload {job_id}")
            except Exception as e:
                print(f"⚠️ Could not clean up temp directory for upload {job_id}: {e}")