import uuid
from database.data import fetch_pocketbase_data
from training.splitter import split_by_hour
from training.train_model import train_hybrid_model
from training.cluster_retrainer import retrain_cluster_model
from segmentation.segment_mycelium import segment_and_save
import os, shutil, re
from datetime import datetime

def full_retrain_pipeline(upload_folder: str, num_classes: int = 14, quick_mode: bool = False):
    assert os.path.exists(upload_folder), f"❌ Provided upload folder does not exist: {upload_folder}"

    # 1. Fetch PocketBase data (already segmented)
    print("📡 Fetching PocketBase data...")
    fetch_pocketbase_data()

    # 2. Segment uploaded images and rename them properly
    print("🔍 Segmenting uploaded images...")
    segmented_upload_folder = "segmented_uploaded_data"
    os.makedirs(segmented_upload_folder, exist_ok=True)
    
    # Extract test number from main folder name (e.g., "myceliumtest4" -> test number 4)
    main_folder_name = os.path.basename(upload_folder)
    test_match = re.search(r'test(\d+)', main_folder_name, re.IGNORECASE)
    if test_match:
        test_number = int(test_match.group(1))
        print(f"📝 Extracted test number: {test_number}")
    else:
        print(f"⚠️ Could not extract test number from folder name '{main_folder_name}', using default: 1")
        test_number = 1
    
    # Process uploaded images - segment them and give proper names
    for root, dirs, files in os.walk(upload_folder):
        # Skip the main folder, only process subfolders
        if root == upload_folder:
            continue
            
        # Extract hour from subfolder name (e.g., "24-11-12___18-59" -> hour based on timestamp)
        subfolder_name = os.path.basename(root)
        hour_match = re.search(r'(\d{2})-(\d{2})-(\d{2})___(\d{2})-(\d{2})', subfolder_name)
        
        if hour_match:
            # Convert timestamp to hours since start
            year = int(f"20{hour_match.group(1)}")  # Assuming 20xx format
            month = int(hour_match.group(2))
            day = int(hour_match.group(3))
            hour_of_day = int(hour_match.group(4))
            minute = int(hour_match.group(5))
            
            # Calculate hours since a reference point (you may want to adjust this)
            # For now, let's use a simple calculation based on day and hour
            calculated_hour = (day - 1) * 24 + hour_of_day
            print(f"📅 Subfolder '{subfolder_name}' -> calculated hour: {calculated_hour}")
        else:
            print(f"⚠️ Could not extract time from subfolder '{subfolder_name}', using hour: 0")
            calculated_hour = 0
        
        # Process images in this subfolder (should be named 1, 2, 3, 4 for angles)
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src_path = os.path.join(root, f)
                
                # Extract angle from filename (1, 2, 3, 4)
                filename_base = os.path.splitext(f)[0]
                try:
                    angle = int(filename_base)
                    if angle not in [1, 2, 3, 4]:
                        print(f"⚠️ Unexpected angle value '{angle}' in file '{f}', using as-is")
                except ValueError:
                    print(f"⚠️ Could not extract angle from filename '{f}', using default: 1")
                    angle = 1
                
                # Create proper filename: test{testNumber}_h{Hour}_{angle}
                proper_filename = f"test{test_number}_h{calculated_hour}_{angle}.jpg"
                output_path = os.path.join(segmented_upload_folder, proper_filename)
                
                # Handle filename conflicts by adding suffix
                counter = 1
                original_output_path = output_path
                while os.path.exists(output_path):
                    name, ext = os.path.splitext(proper_filename)
                    output_path = os.path.join(segmented_upload_folder, f"{name}_dup{counter}{ext}")
                    counter += 1
                
                # Segment the image
                print(f"🔍 Segmenting: {f} -> {os.path.basename(output_path)}")
                print(f"    📂 From: {subfolder_name} (hour: {calculated_hour}, angle: {angle})")
                
                success = segment_and_save(src_path, output_path)
                if not success:
                    print(f"⚠️ Failed to segment {f}, skipping...")

    # 3. Merge fetched data (already segmented) and newly segmented uploaded data
    print("📁 Merging segmented data...")
    combined_data = "segmented_training_data"
    if os.path.exists(combined_data):
        shutil.rmtree(combined_data)
    os.makedirs(combined_data, exist_ok=True)

    # Copy fetched data (already segmented)
    if os.path.exists("fetched_training_data"):
        for f in os.listdir("fetched_training_data"):
            src = os.path.join("fetched_training_data", f)
            dst = os.path.join(combined_data, f)
            shutil.copy2(src, dst)
            print(f"📋 Copied fetched: {f}")

    # Copy segmented uploaded data
    if os.path.exists(segmented_upload_folder):
        for f in os.listdir(segmented_upload_folder):
            src = os.path.join(segmented_upload_folder, f)
            dst = os.path.join(combined_data, f)
            
            # Handle filename conflicts
            counter = 1
            original_dst = dst
            while os.path.exists(dst):
                name, ext = os.path.splitext(f)
                dst = os.path.join(combined_data, f"{name}_copy{counter}{ext}")
                counter += 1
            
            shutil.copy2(src, dst)
            print(f"📋 Copied segmented: {f}")

    # 4. Split images by hour for classification (prediction model)
    print("📊 Splitting images by hour for classification...")
    split_by_hour(combined_data, "labeled_training_data", hour_split=24)

    # 5. Train prediction model (classifier)
    print("🧠 Training classification model...")
    version = f"v{uuid.uuid4().hex[:6]}"
    version_dir = train_hybrid_model("labeled_training_data", version=version, num_classes=num_classes, quick_mode=quick_mode)

    # 6. Train clustering model using all segmented images
    print("🔗 Training clustering model...")
    encoder_path = os.path.join(version_dir, "encoder_model.keras")
    retrain_cluster_model(combined_data, encoder_path, version_dir)

    # Clean up temporary directories
    cleanup_dirs = [segmented_upload_folder, "fetched_training_data", "labeled_training_data", "split"]
    for cleanup_dir in cleanup_dirs:
        if os.path.exists(cleanup_dir):
            try:
                shutil.rmtree(cleanup_dir)
                print(f"🧹 Cleaned up: {cleanup_dir}")
            except Exception as e:
                print(f"⚠️ Could not clean up {cleanup_dir}: {e}")

    print(f"✅ Full retraining pipeline completed! Version: {version}")
    return version_dir