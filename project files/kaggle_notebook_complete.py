"""
GrainPalette Rice Classification - Complete Kaggle Notebook Implementation
Based on the provided Kaggle notebook structure using TensorFlow Hub MobileNetV2
"""

# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL.Image as Image
import cv2
import os
import numpy as np
import pathlib
import pandas as pd
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🌾 GrainPalette Rice Classification - Kaggle Implementation")
print("=" * 60)

# 1. Dataset Setup and Exploration
print("\n📂 Step 1: Dataset Setup and Exploration")
print("-" * 40)

# Kaggle dataset path
data_dir = "../input/rice-image-dataset/Rice_Image_Dataset"
data_dir = pathlib.Path(data_dir)
print(f"Dataset path: {data_dir}")

# Load image paths for each rice type (limiting to 600 images per class for balanced dataset)
arborio = list(data_dir.glob('Arborio/*'))[:600]
basmati = list(data_dir.glob('Basmati/*'))[:600]
ipsala = list(data_dir.glob('Ipsala/*'))[:600]
jasmine = list(data_dir.glob('Jasmine/*'))[:600]
karacadag = list(data_dir.glob('Karacadag/*'))[:600]

print(f"Arborio images: {len(arborio)}")
print(f"Basmati images: {len(basmati)}")
print(f"Ipsala images: {len(ipsala)}")
print(f"Jasmine images: {len(jasmine)}")
print(f"Karacadag images: {len(karacadag)}")

# 2. Data Organization
print("\n📊 Step 2: Data Organization")
print("-" * 30)

# Contains the images path
df_images = {
    'arborio': arborio,
    'basmati': basmati,
    'ipsala': ipsala,
    'jasmine': jasmine,
    'karacadag': karacadag
}

# Contains numerical labels for the categories
df_labels = {
    'arborio': 0,
    'basmati': 1,
    'ipsala': 2,
    'jasmine': 3,
    'karacadag': 4
}

print("Class labels mapping:")
for class_name, label in df_labels.items():
    print(f"  {class_name}: {label}")

# 3. Image Preview
print("\n📸 Step 3: Rice Category Preview")
print("-" * 32)

def show_rice_samples():
    """Display sample images from each rice category"""
    fig, ax = plt.subplots(ncols=5, figsize=(20, 5))
    fig.suptitle('Rice Category Samples', fontsize=16, fontweight='bold')
    
    # Load and display one image from each category
    arborio_image = img.imread(arborio[0])
    basmati_image = img.imread(basmati[0])
    ipsala_image = img.imread(ipsala[0])
    jasmine_image = img.imread(jasmine[0])
    karacadag_image = img.imread(karacadag[0])
    
    ax[0].set_title('Arborio', fontweight='bold')
    ax[1].set_title('Basmati', fontweight='bold')
    ax[2].set_title('Ipsala', fontweight='bold')
    ax[3].set_title('Jasmine', fontweight='bold')
    ax[4].set_title('Karacadag', fontweight='bold')
    
    ax[0].imshow(arborio_image)
    ax[1].imshow(basmati_image)
    ax[2].imshow(ipsala_image)
    ax[3].imshow(jasmine_image)
    ax[4].imshow(karacadag_image)
    
    # Remove axes for cleaner look
    for axis in ax:
        axis.axis('off')
    
    plt.tight_layout()
    plt.show()

show_rice_samples()

# 4. Image Preprocessing and Data Preparation
print("\n🔄 Step 4: Image Preprocessing and Data Preparation")
print("-" * 50)

def preprocess_images():
    """Preprocess images and prepare training data"""
    X, y = [], []  # X = images, y = labels
    
    print("Processing images...")
    total_images = sum(len(images) for images in df_images.values())
    processed = 0
    
    for label, images in df_images.items():
        for image_path in images:
            try:
                # Read and resize image
                img = cv2.imread(str(image_path))
                if img is not None:
                    # Resize to MobileNet input size (224, 224)
                    resized_img = cv2.resize(img, (224, 224))
                    X.append(resized_img)
                    y.append(df_labels[label])
                    
                processed += 1
                if processed % 500 == 0:
                    print(f"  Processed {processed}/{total_images} images...")
                    
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
    
    # Convert to numpy arrays and normalize
    X = np.array(X)
    X = X / 255.0  # Normalize pixel values to [0, 1]
    y = np.array(y)
    
    print(f"✅ Preprocessing complete!")
    print(f"  Final dataset shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X, y

# Check current image size before preprocessing
sample_img = cv2.imread(str(df_images['arborio'][0]))
print(f"Original image shape: {sample_img.shape}")

# Preprocess all images
X, y = preprocess_images()

# 5. Data Splitting
print("\n📊 Step 5: Data Splitting")
print("-" * 25)

# Split data into training, validation, and test sets
X_train, X_test_val, y_train, y_test_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Display class distribution
def show_class_distribution():
    """Show distribution of classes in each split"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = [
        (y_train, "Training Set"),
        (y_val, "Validation Set"),
        (y_test, "Test Set")
    ]
    
    class_names = list(df_labels.keys())
    
    for i, (y_split, title) in enumerate(splits):
        unique, counts = np.unique(y_split, return_counts=True)
        axes[i].bar([class_names[j] for j in unique], counts, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_xlabel('Rice Types')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

show_class_distribution()

# 6. Model Building with TensorFlow Hub MobileNetV2
print("\n🧠 Step 6: Model Building with TensorFlow Hub MobileNetV2")
print("-" * 55)

# Pre-trained CNN model as a Feature Extractor
mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobile_net = hub.KerasLayer(
    mobile_net_url, 
    input_shape=(224, 224, 3), 
    trainable=False  # Freeze the pre-trained weights
)

print("✅ MobileNetV2 feature extractor loaded from TensorFlow Hub")

# 7. Adding Dense Layer
print("\n🔗 Step 7: Adding Dense Classification Layer")
print("-" * 40)

num_classes = 5  # Number of rice types

model = keras.Sequential([
    mobile_net,
    keras.layers.Dense(num_classes, activation='softmax')
])

print("Model architecture:")
model.summary()

print(f"\nModel parameters:")
print(f"  Total params: {model.count_params():,}")
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"  Trainable params: {trainable_params:,}")
print(f"  Non-trainable params: {model.count_params() - trainable_params:,}")

# 8. Configure the Learning Process
print("\n⚙️ Step 8: Configure the Learning Process")
print("-" * 40)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

print("✅ Model compiled with:")
print("  Optimizer: Adam")
print("  Loss: SparseCategoricalCrossentropy")
print("  Metrics: Accuracy")

# 9. Train the Model
print("\n🚀 Step 9: Train the Model")
print("-" * 25)

# Add callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("✅ Training completed!")

# 10. Visualizing Training History
print("\n📈 Step 10: Visualizing Training Results")
print("-" * 40)

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n📊 Final Training Metrics:")
    print(f"  Training Accuracy: {final_train_acc:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Training Loss: {final_train_loss:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")

plot_training_history(history)

# 11. Model Evaluation
print("\n🔍 Step 11: Model Evaluation")
print("-" * 30)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions for detailed analysis
y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\n📋 Detailed Classification Report:")
class_names = list(df_labels.keys())
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\n📊 Per-class Accuracy:")
    for i, (class_name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f"  {class_name}: {acc:.4f}")

plot_confusion_matrix(y_test, y_pred_classes, class_names)

# 12. Test Individual Prediction
print("\n🧪 Step 12: Test Individual Prediction")
print("-" * 35)

def test_single_prediction(image_path=None, image_index=None):
    """Test prediction on a single image"""
    if image_path:
        # Load image from path
        test_img = cv2.imread(image_path)
    else:
        # Use image from test set
        if image_index is None:
            image_index = np.random.randint(0, len(X_test))
        test_img = (X_test[image_index] * 255).astype(np.uint8)
        actual_class = y_test[image_index]
        print(f"Actual class: {class_names[actual_class]}")
    
    # Preprocess image
    if test_img is not None:
        test_img_processed = cv2.resize(test_img, (224, 224))
        test_img_processed = np.array(test_img_processed)
        test_img_processed = test_img_processed / 255.0
        test_img_processed = np.expand_dims(test_img_processed, 0)
        
        # Make prediction
        pred = model.predict(test_img_processed, verbose=0)
        pred_class = pred.argmax()
        confidence = pred.max()
        
        print(f"Predicted class: {class_names[pred_class]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show image and prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Predicted: {class_names[pred_class]} (Confidence: {confidence:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
        
        return pred_class, confidence
    else:
        print("❌ Could not load image")
        return None, None

# Test with a random image from test set
print("Testing with random image from test set:")
test_single_prediction()

# 13. Save the Model
print("\n💾 Step 13: Save the Model")
print("-" * 25)

# Save the model
model.save("rice_classification_model.h5")
print("✅ Model saved as 'rice_classification_model.h5'")

# Also save in SavedModel format
model.save("rice_model_savedmodel")
print("✅ Model also saved in SavedModel format")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
print("✅ Training history saved as 'training_history.csv'")

print("\n🎉 Rice Classification Model Training Complete!")
print("=" * 50)
print("📊 Final Results Summary:")
print(f"  • Test Accuracy: {test_accuracy:.4f}")
print(f"  • Total Training Epochs: {len(history.history['accuracy'])}")
print(f"  • Model Size: {model.count_params():,} parameters")
print(f"  • Dataset Size: {len(X)} images")
print("\n🚀 Ready for deployment in Flask application!")
