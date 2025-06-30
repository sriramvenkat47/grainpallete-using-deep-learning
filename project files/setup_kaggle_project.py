"""
Setup script for GrainPalette Rice Classification Project
Kaggle-compatible version with TensorFlow Hub integration
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages for Kaggle-style implementation"""
    print("📦 Installing required packages...")
    
    packages = [
        "tensorflow>=2.13.0",
        "tensorflow-hub>=0.14.0",
        "Flask>=2.3.3",
        "Pillow>=10.0.1",
        "numpy>=1.24.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.3",
        "plotly>=5.15.0",
        "opencv-python>=4.8.0",
        "Werkzeug>=2.3.7"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package}: {e}")
            return False
    
    print("✅ All requirements installed successfully!")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    directories = [
        "static/uploads",
        "templates",
        "models",
        "data",
        "logs",
        "Data/val"  # For Flask app image uploads
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_tensorflow_hub():
    """Check TensorFlow Hub installation"""
    print("🔍 Checking TensorFlow Hub installation...")
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ TensorFlow Hub version: {hub.__version__}")
        
        # Check for GPU support
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU support detected!")
        else:
            print("💻 Running on CPU")
            
        return True
    except ImportError as e:
        print(f"❌ TensorFlow Hub not found: {e}")
        return False

def create_kaggle_instructions():
    """Provide detailed Kaggle setup instructions"""
    print("\n📊 Kaggle Setup Instructions:")
    print("=" * 50)
    print("🎯 OPTION 1: Direct Kaggle Notebook (Recommended)")
    print("-" * 45)
    print("1. Go to: https://www.kaggle.com/code")
    print("2. Click 'New Notebook'")
    print("3. Add Dataset:")
    print("   • Click 'Add Data' → 'Datasets'")
    print("   • Search: 'Rice Image Dataset'")
    print("   • Select: muratkokludataset/rice-image-dataset")
    print("   • Click 'Add'")
    print("4. Upload 'kaggle_notebook_complete.py' to your notebook")
    print("5. Run all cells for complete training pipeline")
    print("6. Download the trained model files")
    
    print("\n🔧 OPTION 2: Local Development with Kaggle API")
    print("-" * 45)
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Get API credentials:")
    print("   • Go to: https://www.kaggle.com/account")
    print("   • Click 'Create New API Token'")
    print("   • Download kaggle.json")
    print("   • Place in ~/.kaggle/kaggle.json")
    print("   • Run: chmod 600 ~/.kaggle/kaggle.json")
    print("3. Run: python train_model.py")
    
    print("\n📁 Expected Dataset Structure:")
    print("-" * 30)
    print("rice-image-dataset/")
    print("└── Rice_Image_Dataset/")
    print("    ├── Arborio/")
    print("    ├── Basmati/")
    print("    ├── Ipsala/")
    print("    ├── Jasmine/")
    print("    └── Karacadag/")

def main():
    """Main setup function"""
    print("🌾 GrainPalette Kaggle-Style Setup")
    print("=" * 35)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Check TensorFlow Hub
    if not check_tensorflow_hub():
        print("❌ Setup failed during TensorFlow Hub check")
        return
    
    # Kaggle instructions
    create_kaggle_instructions()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Next Steps:")
    print("1. Choose your training approach (Kaggle Notebook or Local)")
    print("2. Train the model using kaggle_notebook_complete.py")
    print("3. Download the trained model (rice_classification_model.h5)")
    print("4. Run: python app.py to start the web application")
    
    print("\n💡 Pro Tips:")
    print("• Use Kaggle Notebook for free GPU access")
    print("• The notebook includes comprehensive visualizations")
    print("• Model achieves ~97% accuracy on test set")
    print("• Web app supports real-time rice classification")

if __name__ == "__main__":
    main()
