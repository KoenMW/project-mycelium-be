import os
import requests

# === PocketBase API Configuration ===
# URL of the hosted PocketBase database instance on Azure
POCKETBASE_URL = "https://mycelium-pb-g4c8fsf0bvetfacm.westeurope-01.azurewebsites.net"
COLLECTION_NAME = "mycelium"  # Name of the database collection storing image records
OUTPUT_DIR = "fetched_training_data"  # Default directory to save downloaded images

def fetch_pocketbase_data(training_data_only=True, output_dir=None):
    """
    Download image data from PocketBase database with pagination support.
    
    This function retrieves metadata records from the database and downloads
    the corresponding image files to local storage for training or analysis.
    
    Args:
        training_data_only (bool): If True, only download images marked as training data
        output_dir (str): Custom directory path to save images, uses OUTPUT_DIR if None
        
    Returns:
        None: Function prints progress and saves files to disk
    """
    # Use default output directory if none specified
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("📡 Fetching records from PocketBase...")
    
    # Initialize pagination variables
    all_records = []  # Store all fetched database records
    page = 1         # Start from first page
    per_page = 500   # Number of records to fetch per API call
    
    # === Pagination Loop ===
    # Continue fetching until all pages are retrieved
    while True:
        # Build API URL with optional training data filter
        filter_param = f"?filter=(trainingData=true)" if training_data_only else "?"
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records{filter_param}&page={page}&perPage={per_page}"
        
        # Make API request to fetch current page
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"❌ Failed to fetch page {page}: {response.status_code} {response.text}")
            break

        # Parse JSON response and extract records
        data = response.json()
        records = data.get("items", [])
        
        # Stop if no more records found
        if not records:
            break
            
        # Add current page records to collection
        all_records.extend(records)
        print(f"📄 Fetched page {page}: {len(records)} records (Total: {len(all_records)})")
        
        # Check if we've reached the last page
        if page >= data.get("totalPages", 1):
            break
            
        page += 1

    print(f"✅ Retrieved {len(all_records)} total records (training_data_only={training_data_only}).")

    # === Image Download Process ===
    downloaded_count = 0
    
    for record in all_records:
        # === Handle Database Field Name Variations ===
        # Account for typos in field names, fallback to segmented image if original missing
        file_name = (record.get("original") or 
                    record.get("ooriginal") or 
                    record.get("orriginal") or 
                    record.get("segmented"))
        
        # Extract metadata fields for file naming and validation
        test = record.get("test")
        run = record.get("run")
        hour = record.get("hour")
        angle = record.get("angle")
        estimated_day = record.get("estimatedDay")
        is_training_data = record.get("trainingData", False)

        # Skip records with missing essential metadata
        if not file_name or run is None or hour is None or angle is None:
            print(f"⚠️ Skipping incomplete record: {record}")
            continue

        # === File Download ===
        # Construct download URL and standardized local filename
        file_url = f"{POCKETBASE_URL}/api/files/{COLLECTION_NAME}/{record['id']}/{file_name}"
        file_name_local = f"test{run}_h{hour}_{angle}.jpg"  # Standardized naming convention
        file_path = os.path.join(output_dir, file_name_local)

        try:
            # Download image file from PocketBase
            img_resp = requests.get(file_url)
            if img_resp.status_code == 200:
                # Save image to local file system
                with open(file_path, "wb") as f:
                    f.write(img_resp.content)
                print(f"⬇️ Downloaded: {file_name_local} (Training: {is_training_data})")
                downloaded_count += 1
            else:
                print(f"❌ Failed to download {file_name}: HTTP {img_resp.status_code}")
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")
    
    print(f"✅ Downloaded {downloaded_count} training images")

def upload_to_pocketbase(image_path, test_num, hour, angle, estimated_day=None, is_training_data=True, predict_day=False):
    """
    Upload a processed image file to the PocketBase database with metadata.
    
    This function handles uploading images and their associated metadata to the database.
    It can optionally use ML models to predict the growth stage if not provided.
    
    Args:
        image_path (str): Local file path to the image to upload
        test_num (int): Test/run number for organizing experiments
        hour (int): Hour timestamp when image was captured
        angle (str): Camera angle or position identifier
        estimated_day (int, optional): Growth day estimate, can be predicted if None
        is_training_data (bool): Whether this image should be used for model training
        predict_day (bool): If True, use ML model to predict estimated_day
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        # === Optional Growth Stage Prediction ===
        # Use ML model to predict growth stage if requested and not provided
        if predict_day and estimated_day is None:
            from prediction.predictor import predict_growth_stage
            
            # Read image file as bytes for prediction
            with open(image_path, 'rb') as img_file:
                img_bytes = img_file.read()
            
            # Run prediction model on the image
            prediction_result, prediction_error = predict_growth_stage(img_bytes, version="default")
            
            if prediction_result and not prediction_error:
                # Use predicted class as estimated day
                predicted_class = int(prediction_result.get("predicted_class", "0"))
                estimated_day = predicted_class  # Or use mapping function
                print(f"🔮 Predicted estimated day: {estimated_day}")
            else:
                # Fallback to simple hour-based calculation if prediction fails
                estimated_day = hour // 24  # Convert hours to approximate days
                print(f"⚠️ Prediction failed, using hour fallback: day {estimated_day}")

        # === Prepare Upload Data ===
        # Open image file for upload
        files = {'segmented': open(image_path, 'rb')}
        
        # Prepare metadata to accompany the image
        data = {
            'run': test_num,           # Experiment/test identifier
            'hour': hour,              # Time when image was captured
            'angle': angle,            # Camera angle/position
            'trainingData': is_training_data  # Whether to use for ML training
        }
        
        # Add estimated day if available
        if estimated_day is not None:
            data['estimatedDay'] = estimated_day

        # === Upload to Database ===
        # Send POST request with image file and metadata
        response = requests.post(
            f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records",
            files=files,    # Image file
            data=data       # Metadata fields
        )
        
        # Clean up file handle
        files['segmented'].close()
        
        # Check upload success and provide feedback
        if response.status_code == 200:
            print(f"✅ Uploaded to database: test{test_num}_h{hour}_{angle}.jpg (Training: {is_training_data}, Day: {estimated_day})")
            return True
        else:
            print(f"❌ Failed to upload to database: {response.status_code} {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading to database: {e}")
        return False