import os, shutil, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam

def train_hybrid_model(input_dir, version="v1.0", num_classes=14, quick_mode=False):
    print(f"🚀 Starting training for version {version}")
    if quick_mode:
        print("⚡ QUICK MODE ENABLED - Using minimal settings for fast training")
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64  # Increased for faster training
    
    # Quick mode settings
    if quick_mode:
        EPOCHS = 10      # Very short for testing
        AUTOENCODER_EPOCHS = 3
        AUTOENCODER_STEPS = 10
        VALIDATION_STEPS = 5
    else:
        EPOCHS = 80     
        AUTOENCODER_EPOCHS = 20
        AUTOENCODER_STEPS = 50
        VALIDATION_STEPS = 20
    
    CLASSES = [str(i) for i in range(num_classes)]
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("📁 Collecting training files...")
    # Collect files/labels
    filepaths, labels = [], []
    for label in CLASSES:
        label_path = os.path.join(input_dir, label)
        if os.path.exists(label_path):
            files_in_label = [f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"   Class {label}: {len(files_in_label)} images")
            for fname in files_in_label:
                filepaths.append(os.path.join(label_path, fname))
                labels.append(label)

    if not filepaths:
        raise ValueError("❌ No training images found!")
    
    print(f"✅ Total images collected: {len(filepaths)}")

    # Split train/val
    train_files, val_files, y_train, y_val = train_test_split(filepaths, labels, test_size=0.2, stratify=labels, random_state=SEED)
    print(f"📊 Train: {len(train_files)}, Validation: {len(val_files)}")

    # Setup dirs
    train_dir, val_dir = "split/train", "split/val"
    for path, files, lbls in [(train_dir, train_files, y_train), (val_dir, val_files, y_val)]:
        if os.path.exists(path): shutil.rmtree(path)
        for label in CLASSES: 
            os.makedirs(os.path.join(path, label), exist_ok=True)
        for f, label in zip(files, lbls):
            shutil.copy(f, os.path.join(path, label, os.path.basename(f)))

    # Image generators
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.05,
                                   height_shift_range=0.05, shear_range=0.05, zoom_range=0.05, horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                               class_mode="categorical", classes=CLASSES)
    val_data = val_gen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           class_mode="categorical", shuffle=False, classes=CLASSES)

    # Calculate class weights (for information, but won't use with generator)
    y_train_labels = train_data.classes
    class_weights = dict(enumerate(class_weight.compute_class_weight("balanced", classes=np.unique(y_train_labels), y=y_train_labels)))
    print(f"📊 Class weights calculated: {class_weights}")

    print("🔧 Building autoencoder...")
    # Simplified Autoencoder (faster training)
    input_img = Input(shape=(224, 224, 3))
    x = Conv2D(16, 3, activation="relu", padding="same")(input_img)  # Reduced filters
    x = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", padding="same")(x)  # Reduced filters
    encoded = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", padding="same")(encoded)
    x = UpSampling2D(2)(x)
    x = Conv2D(16, 3, activation="relu", padding="same")(x)
    x = UpSampling2D(2)(x)
    decoded = Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer=Adam(1e-3), loss="mse")
    
    print(f"⚡ Training autoencoder ({'QUICK' if quick_mode else 'NORMAL'})...")
    
    # Fixed autoencoder training generator - return tuple (input, target) where target = input
    def autoencoder_generator(data_generator):
        for batch_x, batch_y in data_generator:
            yield batch_x, batch_x  # For autoencoder: input = target
    
    # Fixed validation generator for autoencoder
    def autoencoder_val_generator(data_generator):
        for batch_x, batch_y in data_generator:
            yield batch_x, batch_x
    
    autoencoder.fit(
        autoencoder_generator(train_data), 
        steps_per_epoch=min(len(train_data), AUTOENCODER_STEPS),
        validation_data=autoencoder_val_generator(val_data), 
        validation_steps=min(len(val_data), VALIDATION_STEPS), 
        epochs=AUTOENCODER_EPOCHS, 
        verbose=1
    )

    print("🧠 Building hybrid model...")
    # Hybrid model
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in vgg.layers[:-2]: layer.trainable = False  # Fine-tune more layers
    vgg_flat = Flatten()(vgg.output)
    encoder_input = Input(shape=(224, 224, 3))
    encoder_out = encoder(encoder_input)
    encoder_flat = Flatten()(encoder_out)
    x = Concatenate()([vgg_flat, encoder_flat])
    x = Dense(128)(x)  # Reduced size
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASSES), activation="softmax")(x)
    hybrid = Model(inputs=[vgg.input, encoder_input], outputs=output)
    hybrid.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"🎯 Training hybrid model ({EPOCHS} epochs)...")
    # Training with faster dataset creation
    def make_dual_input_dataset(generator):
        for batch_x, batch_y in generator:
            yield (batch_x, batch_x), batch_y

    version_dir = f"model_versions/{version}"
    os.makedirs(version_dir, exist_ok=True)
    
    # Create a custom callback for progress
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")

    # Quick mode callbacks - more aggressive early stopping
    if quick_mode:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1),
            ProgressCallback()
        ]
    else:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
            ModelCheckpoint(os.path.join(version_dir, "hybrid_model.keras"), monitor='val_loss', save_best_only=True, verbose=1),
            ProgressCallback()
        ]

    # REMOVED class_weight parameter since it's not supported with generators
    history = hybrid.fit(
        make_dual_input_dataset(train_data), 
        steps_per_epoch=len(train_data),
        validation_data=make_dual_input_dataset(val_data), 
        validation_steps=len(val_data),
        epochs=EPOCHS, 
        # class_weight=class_weights,  # REMOVED - not supported with generators
        callbacks=callbacks,
        verbose=1
    )

    print("💾 Saving models...")
    encoder.save(os.path.join(version_dir, "encoder_model.keras"))
    autoencoder.save(os.path.join(version_dir, "autoencoder_model.keras"))
    hybrid.save(os.path.join(version_dir, "hybrid_model.keras"))

    print(f"✅ Training complete! Models saved in: {version_dir}")
    return version_dir