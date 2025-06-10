import os
import requests

POCKETBASE_URL = "https://mycelium-pb-g4c8fsf0bvetfacm.westeurope-01.azurewebsites.net"
COLLECTION_NAME = "mycelium"
OUTPUT_DIR = "fetched_training_data"

def fetch_pocketbase_data(training_data_only=True, output_dir=None):
    """
    Fetch data from PocketBase with pagination
    Args:
        training_data_only: If True, only fetch records where trainingData=true
        output_dir: Custom output directory, defaults to OUTPUT_DIR
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)

    print("📡 Fetching records from PocketBase...")
    
    all_records = []
    page = 1
    per_page = 500
    
    while True:
        # Add filter for training data if requested
        filter_param = f"?filter=(trainingData=true)" if training_data_only else "?"
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records{filter_param}&page={page}&perPage={per_page}"
        
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch page {page}: {response.status_code} {response.text}")
            break

        data = response.json()
        records = data.get("items", [])
        
        if not records:
            break
            
        all_records.extend(records)
        print(f"📄 Fetched page {page}: {len(records)} records (Total: {len(all_records)})")
        
        # Check if we've reached the end
        if page >= data.get("totalPages", 1):
            break
            
        page += 1

    print(f"✅ Retrieved {len(all_records)} total records (training_data_only={training_data_only}).")

    downloaded_count = 0
    for record in all_records:
        # Handle typos in field names and use segmented if original is empty
        file_name = (record.get("original") or 
                    record.get("ooriginal") or 
                    record.get("orriginal") or 
                    record.get("segmented"))
        
        test = record.get("test")
        run = record.get("run")
        hour = record.get("hour")
        angle = record.get("angle")
        estimated_day = record.get("estimatedDay")
        is_training_data = record.get("trainingData", False)

        if not file_name or run is None or hour is None or angle is None:
            print(f"⚠️ Skipping incomplete record: {record}")
            continue

        file_url = f"{POCKETBASE_URL}/api/files/{COLLECTION_NAME}/{record['id']}/{file_name}"
        file_name_local = f"test{run}_h{hour}_{angle}.jpg"
        file_path = os.path.join(output_dir, file_name_local)

        try:
            img_resp = requests.get(file_url)
            if img_resp.status_code == 200:
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
    Upload processed image to PocketBase
    Args:
        predict_day: If True, use model to predict estimated_day
    """
    try:
        # Predict estimated day if requested and not provided
        if predict_day and estimated_day is None:
            from prediction.predictor import predict_growth_stage
            
            with open(image_path, 'rb') as img_file:
                img_bytes = img_file.read()
            
            prediction_result, prediction_error = predict_growth_stage(img_bytes, version="default")
            
            if prediction_result and not prediction_error:
                predicted_class = int(prediction_result.get("predicted_class", "0"))
                estimated_day = predicted_class  # Or use mapping function
                print(f"🔮 Predicted estimated day: {estimated_day}")
            else:
                estimated_day = hour // 24  # Fallback to hour-based calculation
                print(f"⚠️ Prediction failed, using hour fallback: day {estimated_day}")

        # Prepare form data
        files = {'segmented': open(image_path, 'rb')}
        data = {
            'run': test_num,
            'hour': hour,
            'angle': angle,
            'trainingData': is_training_data
        }
        if estimated_day is not None:
            data['estimatedDay'] = estimated_day

        response = requests.post(
            f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records",
            files=files,
            data=data
        )
        
        files['segmented'].close()
        
        if response.status_code == 200:
            print(f"✅ Uploaded to database: test{test_num}_h{hour}_{angle}.jpg (Training: {is_training_data}, Day: {estimated_day})")
            return True
        else:
            print(f"❌ Failed to upload to database: {response.status_code} {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading to database: {e}")
        return False