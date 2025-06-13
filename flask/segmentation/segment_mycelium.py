import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# === YOLO model init ===
YOLO_MODEL_PATH = "../models/yolo_segmenting_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = None

# Only load YOLO model if not in testing mode
if not os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
    from ultralytics import YOLO
    
    try:
        if os.path.exists(YOLO_MODEL_PATH):
            yolo = YOLO(YOLO_MODEL_PATH)
            yolo.to(device)
            print(f"✅ YOLO model loaded from: {YOLO_MODEL_PATH}")
        else:
            print(f"⚠️ YOLO model not found at: {YOLO_MODEL_PATH}")
            print("🔄 Using default YOLOv8 model (will download automatically)")
            yolo = YOLO('yolov8n-seg.pt')  # Use default YOLOv8 nano segmentation model
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("🔄 Falling back to default YOLOv8 model")
        yolo = YOLO('yolov8n-seg.pt')
else:
    print("🧪 Skipping YOLO model loading for testing")
    
# === 1. Single image segmentatie (API) ===
def segment_image(image_bytes):
    # Return mock data if in testing mode
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        output_buffer = BytesIO()
        # Create a simple mock image
        mock_image = Image.new('RGB', (100, 100), color='red')
        mock_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        return output_buffer, None
    
    try:
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Invalid image bytes"

        results = yolo.predict(image, device=device)
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            return None, "No mask detected"

        mask = masks.data[0].cpu().numpy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_image = image_rgb.copy()
        masked_image[resized_mask == 0] = 0

        output_buffer = BytesIO()
        Image.fromarray(masked_image).save(output_buffer, format="PNG")
        output_buffer.seek(0)
        return output_buffer, None
    except Exception as e:
        return None, str(e)

# === 2. Batch segmentatie voor retraining ===
def segment_and_save(image_path: str, output_path: str) -> bool:
    # Return mock success if in testing mode
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        return True
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to read: {image_path}")
            return False

        results = yolo.predict(image, device=device)
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            print(f"⚠️ No mask detected in {image_path}")
            return False

        mask = masks.data[0].cpu().numpy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_image = image_rgb.copy()
        masked_image[resized_mask == 0] = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(masked_image).save(output_path)
        print(f"✅ Saved segmented: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"❌ Error segmenting {image_path}: {e}")
        return False