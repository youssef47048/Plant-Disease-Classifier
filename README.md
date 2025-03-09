# 🌿 Plant Disease Classifier


<p align="center">
   <img src="images/logo.png" alt="Project Logo" width="600"/>
 </p>
 
 <p align="center">
   <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/>
   <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"/>
   <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch"/>
   <img src="https://img.shields.io/badge/Streamlit-1.20%2B-red" alt="Streamlit"/>
 </p>
 

<p align = "center" >A deep learning-based web application that diagnoses diseases in plant leaves using convolutional neural networks (CNNs).</p>

![Plant Disease Classifier Demo](https://your-repo-url/images/demo.gif)

## 🚀 Live Demo

The application is currently deployed and available at:
[https://plant-disease-classifier-cnn.streamlit.app/](https://plant-disease-classifier-cnn.streamlit.app/)


## Video Demonstration
 
 <div align="center">
   <video width="700" src="https://github.com/user-attachments/assets/205a963b-3a80-473e-8337-0dccaa977388"/>
 </div>
 



## ✨ Features

- **Instant Disease Detection**: Upload an image of a plant leaf and get immediate diagnosis results
- **Comprehensive Diagnosis**: Provides detailed information on detected diseases, including causes, treatments, and prevention
- **User-Friendly Interface**: Clean, intuitive UI with image previews and visualization of results
- **Multiple Plant Support**: Currently supports tomatoes, potatoes, and bell peppers
- **High Accuracy**: 96.5% accuracy on test datasets
- **Example Images**: Try the application with pre-loaded example images

## 🧪 Supported Plant Diseases

The model can currently identify the following plants and diseases:

- **Tomato**:
  - Healthy
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites
  - Target Spot
  - Yellow Leaf Curl Virus
  - Mosaic Virus

- **Potato**:
  - Healthy
  - Early Blight
  - Late Blight

- **Bell Pepper**:
  - Healthy
  - Bacterial Spot

## 🔧 Model Architecture

The application uses a custom CNN architecture with the following components:

- 5 convolutional blocks with batch normalization and ReLU activation
- Global Average Pooling
- Fully connected layers with dropout for regularization
- Trained on the PlantVillage dataset with data augmentation techniques
- Achieves 96.5% accuracy on the test set

```
PlantDiseaseModel(
  (conv_block1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), padding=same)
    (1): BatchNorm2d(64)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2)
  )
  (conv_block2): Sequential(...)
  (conv_block3): Sequential(...)
  (conv_block4): Sequential(...)
  (conv_block5): Sequential(...)
  (global_avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc_block): Sequential(
    (0): Flatten()
    (1): Linear(in_features=512, out_features=256)
    (2): ReLU()
    (3): Dropout(p=0.5)
    (4): Linear(in_features=256, out_features=num_classes)
  )
)
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: PyTorch
- **Data Processing**: Pandas, NumPy, PIL
- **Visualization**: Matplotlib, Streamlit components
- **Model Training**: PyTorch, scikit-learn

## 📋 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/sayedgamal99/Plant-Disease-Classifier.git
   cd Plant-Disease-Classifier
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Access the application in your web browser:
   ```
   http://localhost:8501
   ```

## 🚂 Training the Model

If you want to train the model on your own dataset or retrain it:

1. Organize your dataset in the following structure:
   ```
   data/
   ├── Tomato_Healthy/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── Tomato_Bacterial_Spot/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   ```

2. Run the training script:
   ```bash
   python plant_disease_classifier.py --data_dir path/to/data --model_path models/best_model.pth --batch_size 32 --epochs 30 --lr 0.001
   ```

3. The script will:
   - Split the data into training, validation, and test sets
   - Train the model with early stopping and learning rate scheduling
   - Save the best model weights
   - Export necessary files for inference (label encoder, transforms, etc.)
   - Evaluate the model on the test set
   - Generate learning curves

## 📁 Project Structure

```
Plant-Disease-Classifier/
├── app.py                     # Streamlit application code
├── plant_disease_classifier.py # Model training and inference code
├── models/                    # Directory containing trained models
│   ├── best_model.pth         # Trained model weights
│   ├── model_config.json      # Model configuration
│   ├── class_names.json       # Class names mapping
│   └── label_encoder.pkl      # Label encoder for class mapping
├── images/                    # Example and static images
│   └── examples/              # Example plant disease images
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
```

## 👥 Contributors

- [Sayed Gamal](https://github.com/sayedgamal99)
- [Youssef Mohammed](https://github.com/youssef47048)
