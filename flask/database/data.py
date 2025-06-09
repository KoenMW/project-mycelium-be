import os
import requests

POCKETBASE_URL = "https://mycelium-pb-g4c8fsf0bvetfacm.westeurope-01.azurewebsites.net"
COLLECTION_NAME = "mycelium"
OUTPUT_DIR = "fetched_training_data"

def fetch_pocketbase_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📡 Fetching records from PocketBase...")
    response = requests.get(f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records?perPage=1000")
    if response.status_code != 200:
        print(f"❌ Failed to fetch records: {response.status_code} {response.text}")
        return

    records = response.json().get("items", [])
    print(f"✅ Retrieved {len(records)} records.")

    for record in records:
        # Handle typos in field names and use segmented if original is empty
        file_name = (record.get("original") or 
                    record.get("ooriginal") or 
                    record.get("orriginal") or 
                    record.get("segmented"))
        
        test = record.get("test")
        run = record.get("run")  # Use 'run' instead of 'test' based on your data
        hour = record.get("hour")
        angle = record.get("angle")

        if not file_name or run is None or hour is None or angle is None:
            print(f"⚠️ Skipping incomplete record: {record}")
            continue

        file_url = f"{POCKETBASE_URL}/api/files/{COLLECTION_NAME}/{record['id']}/{file_name}"
        file_name_local = f"test{run}_h{hour}_{angle}.jpg"  # Use 'run' instead of 'test'
        file_path = os.path.join(OUTPUT_DIR, file_name_local)

        try:
            img_resp = requests.get(file_url)
            if img_resp.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(img_resp.content)
                print(f"⬇️ Downloaded: {file_name_local}")
            else:
                print(f"❌ Failed to download {file_name}: HTTP {img_resp.status_code}")
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")