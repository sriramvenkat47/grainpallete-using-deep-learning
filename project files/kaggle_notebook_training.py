"""
GrainPalette Rice Classification - Kaggle Notebook Training Script
This script is designed to run in a Kaggle notebook environment
with direct access to the Rice Image Dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 5
LEARNING_RATE = 0.0001

# Rice classes
RICE_CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Kaggle dataset path (update this based on your Kaggle environment)
KAGGLE_DATA_PATH = "/kaggle/input/rice-image-dataset"

def explore_dataset(data_dir):
    """Explore and visualize the dataset"""
    print("🔍 Exploring Dataset")
    print("=" * 30)
    
    # Count images in each class
    class_counts = {}
    total_images = 0
    
    for class_name in RICE_CLASSES:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(image_files)
            total_images += len(image_files)
            print(f"{class_name}: {len(image_files)} images")
        else:
            print(f"❌ {class_name} directory not found!")
            class_counts[class_name] = 0
    
    print(f"\nTotal images: {total_images}")
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_counts.keys(), class_counts.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    plt.title('Rice Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Rice Types')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    plt.title('Rice Class Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return class_counts, total_images

def visualize_sample_images(data_dir, samples_per_class=3):
    """Visualize sample images from each class"""
    print("\n📸 Sample Images from Each Class")
    print("=" * 35)
    
    fig, axes = plt.subplots(len(RICE_CLASSES), samples_per_class, 
                            figsize=(15, 3*len(RICE_CLASSES)))
    
    for i, class_name in enumerate(RICE_CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Select random samples
            selected_files = np.random.choice(image_files, 
                                            min(samples_per_class, len(image_files)), 
                                            replace=False)
            
            for j, img_file in enumerate(selected_files):
                img_path = os.path.join(class_path, img_file)
                img = load_img(img_path, target_size=IMG_SIZE)
                
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'{class_name}', fontweight='bold')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_data_generators(data_dir):
    """Create data generators with comprehensive augmentation"""
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator

def visualize_augmented_images(train_generator):
    """Visualize data augmentation effects"""
    print("\n🔄 Data Augmentation Examples")
    print("=" * 30)
    
    # Get a batch of images
    batch_images, batch_labels = next(train_generator)
    
    # Show original and augmented versions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Original-style image (less augmented)
        axes[0, i].imshow(batch_images[i])
        axes[0, i].set_title('Augmented Image', fontweight='bold')
        axes[0, i].axis('off')
        
        # Another augmented version
        axes[1, i].imshow(batch_images[i+5])
        axes[1, i].set_title('Augmented Image', fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_model():
    """Create CNN model with MobileNetV2 transfer learning"""
    
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def compile_model(model):
    """Compile the model with optimizer and metrics"""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model

def create_callbacks():
    """Create comprehensive training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            'best_rice_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    return callbacks

def plot_training_history(history, fine_tune_history=None):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    if fine_tune_history:
        axes[0, 0].plot(range(len(history.history['accuracy']), 
                             len(history.history['accuracy']) + len(fine_tune_history.history['accuracy'])),
                       fine_tune_history.history['accuracy'], 
                       label='Fine-tune Training Accuracy', linewidth=2, linestyle='--')
        axes[0, 0].plot(range(len(history.history['val_accuracy']), 
                             len(history.history['val_accuracy']) + len(fine_tune_history.history['val_accuracy'])),
                       fine_tune_history.history['val_accuracy'], 
                       label='Fine-tune Validation Accuracy', linewidth=2, linestyle='--')
    
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    if fine_tune_history:
        axes[0, 1].plot(range(len(history.history['loss']), 
                             len(history.history['loss']) + len(fine_tune_history.history['loss'])),
                       fine_tune_history.history['loss'], 
                       label='Fine-tune Training Loss', linewidth=2, linestyle='--')
        axes[0, 1].plot(range(len(history.history['val_loss']), 
                             len(history.history['val_loss']) + len(fine_tune_history.history['val_loss'])),
                       fine_tune_history.history['val_loss'], 
                       label='Fine-tune Validation Loss', linewidth=2, linestyle='--')
    
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot top-k accuracy if available
    if 'top_k_categorical_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_k_categorical_accuracy'], 
                       label='Training Top-K Accuracy', linewidth=2)
        axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], 
                       label='Validation Top-K Accuracy', linewidth=2)
        axes[1, 0].set_title('Top-K Categorical Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-K Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    axes[1, 1].text(0.5, 0.5, 'Training Completed!\n\nModel Performance:\n' + 
                   f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n" +
                   f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}",
                   transform=axes[1, 1].transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Training Summary', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model_comprehensive(model, validation_generator):
    """Comprehensive model evaluation"""
    print("\n📊 Comprehensive Model Evaluation")
    print("=" * 40)
    
    # Basic evaluation
    loss, accuracy, top_k_acc = model.evaluate(validation_generator, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Top-K Accuracy: {top_k_acc:.4f}")
    
    # Predictions for confusion matrix
    validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=RICE_CLASSES))
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=RICE_CLASSES, yticklabels=RICE_CLASSES)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return loss, accuracy, top_k_acc

def main():
    """Main training function for Kaggle environment"""
    print("🌾 GrainPalette Rice Classification - Kaggle Training")
    print("=" * 55)
    
    # Check dataset path
    if not os.path.exists(KAGGLE_DATA_PATH):
        print(f"❌ Dataset path {KAGGLE_DATA_PATH} not found!")
        print("Please ensure you have added the Rice Image Dataset to your Kaggle notebook.")
        return
    
    # Explore dataset
    class_counts, total_images = explore_dataset(KAGGLE_DATA_PATH)
    
    if total_images == 0:
        print("❌ No images found in dataset!")
        return
    
    # Visualize sample images
    visualize_sample_images(KAGGLE_DATA_PATH)
    
    # Create data generators
    print("\n📊 Creating data generators...")
    train_generator, validation_generator = create_data_generators(KAGGLE_DATA_PATH)
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    # Visualize augmented images
    visualize_augmented_images(train_generator)
    
    # Create and compile model
    print("\n🧠 Creating model...")
    model, base_model = create_model()
    model = compile_model(model)
    
    print(f"Model created with {model.count_params():,} parameters")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Display model summary
    model.summary()
    
    # Train the model
    print("\n🚀 Starting initial training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Fine-tuning
    print("\n🔧 Fine-tuning model...")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Continue training
    fine_tune_epochs = 10
    total_epochs = EPOCHS + fine_tune_epochs
    
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=EPOCHS,
        validation_data=validation_generator,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(history, history_fine)
    
    # Comprehensive evaluation
    evaluate_model_comprehensive(model, validation_generator)
    
    # Save the final model
    model.save('rice_classification_model.h5')
    print("\n✅ Model saved as 'rice_classification_model.h5'")
    
    # Save in SavedModel format
    model.save('rice_model_savedmodel')
    print("✅ Model also saved in SavedModel format")
    
    print("\n🎉 Training completed successfully!")
    print("Download the model files to use in your Flask application.")

if __name__ == "__main__":
    # Configure GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 Using GPU: {len(gpus)} device(s) available")
        except RuntimeError as e:
            print(e)
    else:
        print("💻 Using CPU for training")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()
