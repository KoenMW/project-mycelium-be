# segmentor.py
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import os
from PIL import Image
from io import BytesIO

# === INIT YOLO MODEL ON GPU IF AVAILABLE ===
device = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_PATH = "model/yolo_segmenting_model"
yolo = YOLO(YOLO_MODEL_PATH)
yolo.to(device)

def segment_image(image_bytes):
    # Load image from bytes
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Predict
    results = yolo.predict(image, device=0)
    masks = results[0].masks

    if masks is None or len(masks.data) == 0:
        return None, "No mask detected"

    mask = masks.data[0].cpu().numpy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Apply mask
    masked_image = image_rgb.copy()
    masked_image[resized_mask == 0] = 0

    # Convert to PNG for response
    pil_image = Image.fromarray(masked_image)
    output_buffer = BytesIO()
    pil_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    
    return output_buffer, None
