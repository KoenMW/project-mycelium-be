import numpy as np
import io
import os

# === Configuration Constants ===
DEFAULT_MODEL_PATH = "../models/best_hybrid_model.keras"  # Path to the main production prediction model
IMG_SIZE = (224, 224)  # Standard image size required by the neural network
CLASSES = [str(i) for i in range(14)]  # Growth stage classes: 0-13 days

# === Global Model Cache ===
# Dictionary to store loaded models in memory to avoid reloading them repeatedly
# Key: version string, Value: loaded Keras model
_models = {}

# === Conditional Import System ===
# Only import heavy ML libraries when not in testing mode to speed up tests
# This allows the application to run tests without loading large model files
if not os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
    from keras.models import load_model  # For loading neural network models
    from keras.preprocessing.image import img_to_array  # Convert PIL images to numpy arrays
    from PIL import Image  # Python Imaging Library for image processing
else:
    print("🧪 Skipping ML library imports for testing")

def load_prediction_model(version="default"):
    """
    Load and cache a growth stage prediction model for a specific version.
    
    This function handles loading hybrid neural network models that predict
    mycelium growth stages from processed images. The models use both VGG
    features and custom encoder features for improved accuracy.
    
    Args:
        version (str): Model version to load ("default" or specific version name)
        
    Returns:
        keras.Model or None: Loaded prediction model, or None if testing
        
    Raises:
        FileNotFoundError: If the specified model version doesn't exist
    """
    # Skip model loading entirely during testing to improve test speed
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        print(f"🧪 Skipping prediction model loading for testing - version: {version}")
        return None
    
    # Check if model for this version is already cached in memory
    if version in _models:
        return _models[version]
    
    # Determine model file path based on version
    if version == "default":
        # Use default model path for the main production model
        model_path = DEFAULT_MODEL_PATH
    else:
        # Use version-specific directory structure for experimental models
        model_path = os.path.join("model_versions", version, "hybrid_model.keras")
    
    # Validate that the model file exists before attempting to load
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for version '{version}' at {model_path}")
    
    print(f"🔍 Loading prediction model for version: {version}")
    
    # Load the Keras model from file
    model = load_model(model_path)
    
    # Cache the loaded model in memory for future use
    _models[version] = model
    
    print(f"✅ Model loaded for version: {version}")
    return model

# === Startup Model Loading ===
# Automatically load default model when the module is imported (unless testing)
# This ensures the model is ready for immediate use without loading delays
if not os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
    try:
        load_prediction_model("default")
    except FileNotFoundError:
        print("⚠️ Default prediction model not found")

# === Main Prediction Function ===
def predict_growth_stage(image_bytes, version="default"):
    """
    Predict the growth stage of mycelium from an image.
    
    This function processes an image through a hybrid neural network that
    combines VGG16 features with custom encoder features to classify the
    mycelium growth stage into one of 14 classes (days 0-13).
    
    Args:
        image_bytes: Raw image data as bytes (e.g., from file upload)
        version (str): Model version to use for prediction
        
    Returns:
        tuple: (result_dict, error_message)
            - result_dict: Contains predicted_class, confidence, and version
            - error_message: String describing any error that occurred, or None
    """
    # Return mock data during testing to avoid model dependencies
    if os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true':
        return {
            "predicted_class": "5",      # Mock prediction: day 5
            "confidence": 0.92,          # Mock confidence score
            "version": version
        }, None
    
    try:
        # Load the prediction model for the specified version
        model = load_prediction_model(version)
        
        # === Image Preprocessing ===
        # Convert bytes to PIL Image and ensure RGB format (remove alpha channel if present)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to standard dimensions expected by the model
        image = image.resize(IMG_SIZE)
        # Convert to numpy array and normalize pixel values to [0,1] range
        array = img_to_array(image) / 255.0
        # Add batch dimension (model expects 4D input: batch_size, height, width, channels)
        input_data = np.expand_dims(array, axis=0)

        # === Model Prediction ===
        # Feed the same image data into both VGG and encoder inputs of the hybrid model
        # The model architecture expects two identical inputs for feature fusion
        predictions = model.predict([input_data, input_data])
        
        # === Extract Results ===
        # Find the class with the highest prediction probability
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        predicted_class = CLASSES[predicted_index]  # Convert index to class string
        confidence = float(np.max(predictions))     # Highest probability as confidence
        
        # Return structured result with prediction details
        return {
            "predicted_class": predicted_class,      # Growth stage (0-13 days)
            "confidence": round(confidence, 4),      # Prediction confidence (0-1)
            "version": version                       # Model version used
        }, None
        
    except Exception as e:
        # Return error information if any step in the prediction pipeline fails
        return None, str(e)