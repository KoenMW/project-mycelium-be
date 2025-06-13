import os, shutil, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam

def train_hybrid_model(input_dir, version="v1.0", num_classes=14, hybrid_epochs=1, autoencoder_epochs=1):
    """
    Train a hybrid neural network model for mycelium growth stage classification.
    
    This function implements a sophisticated training pipeline that combines:
    1. Autoencoder for unsupervised feature learning
    2. VGG16 for pre-trained visual features
    3. Hybrid model that fuses both feature types for classification
    
    Args:
        input_dir (str): Directory containing class-organized training images
        version (str): Unique version identifier for this model
        num_classes (int): Number of growth stages to classify (default: 14 days)
        hybrid_epochs (int): Training epochs for the final classification model
        autoencoder_epochs (int): Training epochs for the feature encoder
        
    Returns:
        str: Path to directory containing all saved models for this version
    """
    print(f"🚀 Starting training for version {version}")
    print(f"📊 Training configuration: Autoencoder={autoencoder_epochs} epochs, Hybrid={hybrid_epochs} epochs")
    
    # === Training Configuration ===
    IMG_SIZE = (224, 224)                                    # Standard input size for VGG16 and encoder
    BATCH_SIZE = 32                                          # Number of images per training batch
    EPOCHS = hybrid_epochs                                   # Main model training duration
    AUTOENCODER_EPOCHS = autoencoder_epochs                  # Feature encoder training duration
    CLASSES = [str(i) for i in range(num_classes)]          # Growth stage labels (0, 1, 2, ..., 13)
    SEED = 42                                                # Random seed for reproducible results
    
    # Set random seeds for reproducible training
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # === Data Collection and Validation ===
    print("📁 Collecting training files...")
    # Traverse class directories and collect image file paths with labels
    filepaths, labels = [], []
    for label in CLASSES:
        label_path = os.path.join(input_dir, label)
        if os.path.exists(label_path):
            # Find all image files in this class directory
            files_in_label = [f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"   Class {label}: {len(files_in_label)} images")
            
            # Add file paths and corresponding labels to training set
            for fname in files_in_label:
                filepaths.append(os.path.join(label_path, fname))
                labels.append(label)

    # Validate that training data was found
    if not filepaths:
        raise ValueError("❌ No training images found!")
    
    print(f"✅ Total images collected: {len(filepaths)}")

    # === Train/Validation Split ===
    # Split data while maintaining class distribution (stratified split)
    train_files, val_files, y_train, y_val = train_test_split(
        filepaths, labels, 
        test_size=0.2,           # 20% for validation
        stratify=labels,         # Maintain class balance in both sets
        random_state=SEED
    )
    print(f"📊 Train: {len(train_files)}, Validation: {len(val_files)}")

    # === Directory Structure Setup ===
    # Create organized directory structure expected by Keras ImageDataGenerator
    train_dir, val_dir = "split/train", "split/val"
    for path, files, lbls in [(train_dir, train_files, y_train), (val_dir, val_files, y_val)]:
        # Clean and recreate directory structure
        if os.path.exists(path): 
            shutil.rmtree(path)
        
        # Create class subdirectories
        for label in CLASSES: 
            os.makedirs(os.path.join(path, label), exist_ok=True)
        
        # Copy files to appropriate class folders
        for f, label in zip(files, lbls):
            shutil.copy(f, os.path.join(path, label, os.path.basename(f)))

    # === Data Augmentation Setup ===
    # Training data generator with augmentation to improve model generalization
    train_gen = ImageDataGenerator(
        rescale=1./255,           # Normalize pixel values to [0,1]
        rotation_range=15,        # Random rotation up to 15 degrees
        width_shift_range=0.1,    # Horizontal shift up to 10% of width
        height_shift_range=0.1,   # Vertical shift up to 10% of height
        shear_range=0.1,          # Shearing transformation
        zoom_range=0.1,           # Random zoom in/out
        horizontal_flip=True      # Random horizontal flipping
    )
    
    # Validation data generator with only normalization (no augmentation)
    val_gen = ImageDataGenerator(rescale=1./255)

    # Create data flow generators for training
    train_data = train_gen.flow_from_directory(
        train_dir, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # One-hot encoded labels
        classes=CLASSES           # Explicit class order
    )
    
    val_data = val_gen.flow_from_directory(
        val_dir, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,            # Keep validation order consistent
        classes=CLASSES
    )

    # === Class Weight Calculation ===
    # Calculate weights to handle class imbalance during training
    y_train_labels = train_data.classes
    weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.unique(y_train_labels), 
        y=y_train_labels
    )
    class_weights = dict(enumerate(weights))
    print(f"📊 Class weights: {class_weights}")

    # === Autoencoder Architecture ===
    # Build convolutional autoencoder for unsupervised feature learning
    print("🔧 Building autoencoder...")
    input_img = Input(shape=(224, 224, 3))
    
    # Encoder: Compress image to lower-dimensional representation
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)           # Bottleneck layer
    
    # Decoder: Reconstruct image from compressed representation
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create autoencoder (full) and encoder (feature extractor) models
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # === Autoencoder Training ===
    print("⚡ Training autoencoder...")
    
    def autoencoder_generator(data_generator):
        """
        Convert classification data generator to autoencoder format.
        Autoencoder trains to reconstruct input images, so input = target.
        """
        for batch_x, batch_y in data_generator:
            yield batch_x, batch_x  # Input and target are the same for reconstruction
    
    # Train autoencoder to learn meaningful image representations
    autoencoder.fit(
        autoencoder_generator(train_data), 
        steps_per_epoch=len(train_data),
        validation_data=autoencoder_generator(val_data), 
        validation_steps=len(val_data), 
        epochs=AUTOENCODER_EPOCHS, 
        verbose=1
    )

    # === Hybrid Model Architecture ===
    print("🧠 Building hybrid model...")
    
    # VGG16 branch: Pre-trained features from ImageNet
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # Fine-tune only the last few layers, freeze earlier layers
    for layer in vgg.layers[:-4]:
        layer.trainable = False
    
    vgg_flat = Flatten()(vgg.output)
    
    # Custom encoder branch: Domain-specific features
    encoder_input = Input(shape=(224, 224, 3))
    encoder_out = encoder(encoder_input)  # Use trained encoder
    encoder_flat = Flatten()(encoder_out)
    
    # === Feature Fusion and Classification ===
    # Combine VGG16 and encoder features for improved classification
    x = Concatenate()([vgg_flat, encoder_flat])  # Fuse both feature streams
    x = Dense(256)(x)                            # Dense classification layer
    x = BatchNormalization()(x)                  # Normalize activations
    x = Activation("relu")(x)                    # ReLU activation
    x = Dropout(0.3)(x)                          # Prevent overfitting
    output = Dense(len(CLASSES), activation="softmax")(x)  # Final classification layer
    
    # Create hybrid model with dual inputs (same image fed to both branches)
    hybrid = Model(inputs=[vgg.input, encoder_input], outputs=output)
    hybrid.compile(
        optimizer=Adam(1e-4),              # Lower learning rate for fine-tuning
        loss="categorical_crossentropy",   # Multi-class classification loss
        metrics=["accuracy"]               # Track accuracy during training
    )

    print(f"🎯 Training hybrid model ({EPOCHS} epochs)...")
    
    # === TensorFlow Dataset Conversion ===
    # Convert Keras generators to tf.data.Dataset for better class weight support
    def make_dual_input_dataset(generator):
        """
        Convert ImageDataGenerator to tf.data.Dataset with dual inputs.
        The hybrid model needs the same image fed to both VGG and encoder branches.
        """
        output_signature = (
            (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # VGG input
             tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)), # Encoder input
            tf.TensorSpec(shape=(None, len(CLASSES)), dtype=tf.float32)   # Labels
        )
        
        def gen():
            while True:
                x, y = next(generator)
                yield (x, x), y  # Same image for both inputs
        
        return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # Create optimized datasets for training
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = make_dual_input_dataset(train_data).repeat().prefetch(buffer_size=AUTOTUNE)
    val_dataset = make_dual_input_dataset(val_data).prefetch(buffer_size=AUTOTUNE)

    # === Model Versioning Setup ===
    version_dir = f"model_versions/{version}"
    os.makedirs(version_dir, exist_ok=True)
    
    # === Training Callbacks ===
    # Configure advanced training behaviors
    callbacks = [
        EarlyStopping(
            monitor='val_loss',           # Stop if validation loss stops improving
            patience=10,                  # Wait 10 epochs before stopping
            restore_best_weights=True,    # Revert to best weights
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',           # Reduce learning rate if validation loss plateaus
            factor=0.5,                   # Halve the learning rate
            patience=5,                   # Wait 5 epochs before reducing
            min_lr=1e-6,                  # Minimum learning rate
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(version_dir, "hybrid_model.keras"),  # Save best model
            monitor='val_loss',           # Based on validation loss
            save_best_only=True,          # Only save improvements
            verbose=1
        )
    ]

    # === Final Model Training ===
    # Train the hybrid model with class weights and callbacks
    history = hybrid.fit(
        train_dataset,
        steps_per_epoch=len(train_data),     # Full epoch = all training batches
        validation_data=val_dataset,
        validation_steps=len(val_data),      # Full validation = all validation batches
        epochs=EPOCHS,
        class_weight=class_weights,          # Handle class imbalance
        callbacks=callbacks,                 # Advanced training behaviors
        verbose=1
    )

    # === Model Persistence ===
    print("💾 Saving models...")
    # Save all three models for different use cases
    encoder.save(os.path.join(version_dir, "encoder_model.keras"))        # For clustering
    autoencoder.save(os.path.join(version_dir, "autoencoder_model.keras")) # For feature learning
    hybrid.save(os.path.join(version_dir, "hybrid_model.keras"))          # For classification

    print(f"✅ Training complete! Models saved in: {version_dir}")
    return version_dir