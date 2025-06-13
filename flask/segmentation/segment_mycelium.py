import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# === YOLO Model Configuration ===
YOLO_MODEL_PATH = "../models/yolo_segmenting_model.pt"  # Path to custom-trained YOLO segmentation model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available for faster processing
yolo = None  # Global variable to store the loaded YOLO model

# === Model Loading with Fallback Strategy ===
# Only load YOLO model if not in testing mode to speed up tests
if not os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
    from ultralytics import YOLO  # YOLOv8 library for object detection and segmentation
    
    try:
        # Try to load custom-trained model first
        if os.path.exists(YOLO_MODEL_PATH):
            yolo = YOLO(YOLO_MODEL_PATH)  # Load custom mycelium segmentation model
            yolo.to(device)               # Move model to GPU/CPU
            print(f"✅ YOLO model loaded from: {YOLO_MODEL_PATH}")
        else:
            # Fallback to default YOLOv8 model if custom model not found
            print(f"⚠️ YOLO model not found at: {YOLO_MODEL_PATH}")
            print("🔄 Using default YOLOv8 model (will download automatically)")
            yolo = YOLO('yolov8n-seg.pt')  # Use default YOLOv8 nano segmentation model
    except Exception as e:
        # Last resort fallback if any loading errors occur
        print(f"❌ Error loading YOLO model: {e}")
        print("🔄 Falling back to default YOLOv8 model")
        yolo = YOLO('yolov8n-seg.pt')
else:
    print("🧪 Skipping YOLO model loading for testing")
    
# === API Endpoint Function: Single Image Segmentation ===
def segment_image(image_bytes):
    """
    Segment mycelium from a single image provided as bytes (for API endpoints).
    
    This function processes an uploaded image through the YOLO segmentation model
    to isolate mycelium regions by creating a mask that removes the background.
    
    Args:
        image_bytes: Raw image data as bytes (e.g., from HTTP request)
        
    Returns:
        tuple: (segmented_image_buffer, error_message)
            - segmented_image_buffer: BytesIO buffer containing PNG of segmented image
            - error_message: String describing any error, or None if successful
    """
    # Return mock data if in testing mode to avoid model dependencies
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        output_buffer = BytesIO()
        # Create a simple mock image for testing
        mock_image = Image.new('RGB', (100, 100), color='red')
        mock_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        return output_buffer, None
    
    try:
        # === Image Decoding ===
        # Convert bytes to OpenCV image format
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Invalid image bytes"

        # === YOLO Segmentation ===
        # Run YOLO model to detect and segment mycelium
        results = yolo.predict(image, device=device)
        masks = results[0].masks  # Extract segmentation masks from results
        
        # Check if any masks were detected
        if masks is None or len(masks.data) == 0:
            return None, "No mask detected"

        # === Mask Processing ===
        # Get the first (best) mask and convert to numpy array
        mask = masks.data[0].cpu().numpy()
        
        # Convert image from BGR (OpenCV) to RGB (PIL/web standard)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize mask to match original image dimensions
        resized_mask = cv2.resize(
            mask.astype(np.uint8), 
            (image_rgb.shape[1], image_rgb.shape[0]), 
            interpolation=cv2.INTER_NEAREST  # Use nearest neighbor to preserve binary mask
        )
        
        # === Apply Mask ===
        # Create segmented image by setting background pixels to black
        masked_image = image_rgb.copy()
        masked_image[resized_mask == 0] = 0  # Set non-mycelium areas to black

        # === Output Preparation ===
        # Convert numpy array back to PIL Image and save to BytesIO buffer
        output_buffer = BytesIO()
        Image.fromarray(masked_image).save(output_buffer, format="PNG")
        output_buffer.seek(0)  # Reset buffer position for reading
        return output_buffer, None
        
    except Exception as e:
        # Return error information if any step fails
        return None, str(e)

# === Batch Processing Function: File-to-File Segmentation ===
def segment_and_save(image_path: str, output_path: str) -> bool:
    """
    Segment mycelium from an image file and save the result to disk.
    
    This function is used for batch processing during training data preparation
    or background upload jobs. It reads from a file path and saves to another.
    
    Args:
        image_path (str): Path to the input image file
        output_path (str): Path where segmented image should be saved
        
    Returns:
        bool: True if segmentation and saving successful, False otherwise
    """
    # Return mock success if in testing mode
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        return True
    
    try:
        # === Load Image from File ===
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to read: {image_path}")
            return False

        # === YOLO Segmentation ===
        # Run YOLO model to detect and segment mycelium
        results = yolo.predict(image, device=device)
        masks = results[0].masks  # Extract segmentation masks
        
        # Check if any masks were detected
        if masks is None or len(masks.data) == 0:
            print(f"⚠️ No mask detected in {image_path}")
            return False

        # === Mask Processing ===
        # Get the first (best) mask and process it
        mask = masks.data[0].cpu().numpy()
        
        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize mask to match image dimensions
        resized_mask = cv2.resize(
            mask.astype(np.uint8), 
            (image_rgb.shape[1], image_rgb.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # === Apply Mask ===
        # Create segmented image by masking out background
        masked_image = image_rgb.copy()
        masked_image[resized_mask == 0] = 0  # Set background to black

        # === Save Result ===
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save segmented image as PIL Image
        Image.fromarray(masked_image).save(output_path)
        print(f"✅ Saved segmented: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        # Log error and return failure status
        print(f"❌ Error segmenting {image_path}: {e}")
        return False