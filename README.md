
# Distributed Attack Detection in IoT using Machine Learning and Deep Learning

## Overview
This project focuses on detecting distributed attacks in Internet of Things (IoT) environments using Machine Learning (ML) and Deep Learning (DL) techniques. The solution utilizes classification models to identify malicious activities and ensure robust network security.

## Features
- Data preprocessing and feature encoding using the KDD dataset.
- Implementation of Random Forest Classifier for baseline detection.
- Deep Learning model built with Keras and TensorFlow for advanced detection.
- Model performance evaluation using accuracy, loss metrics, and confusion matrix.
- Visualization of model performance over training epochs.

## Dataset
The project uses the **KDD Dataset** for intrusion detection. This dataset is processed with:
- Label Encoding for categorical variables.
- One-Hot Encoding for specific features (`duration`, `protocol_type`, `service`).
- Train-test split for model evaluation.

## Models Implemented
### 1. Random Forest Classifier
- Number of Estimators: 500
- Criterion: Entropy
- Out-of-bag (OOB) Score Enabled

### 2. Deep Learning Model
- Input Layer: 150 neurons, Tanh activation
- Hidden Layers: Relu and Sigmoid activations with Dropout for regularization
- Output Layer: Softmax activation for multi-class classification
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Training Epochs: 10

## Results
- **Random Forest Accuracy:** ~99.69%
- **Deep Learning Model Accuracy:** ~94.89%
- Visualization of training and validation accuracy and loss over epochs.

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/distributed-attack-detection.git
cd distributed-attack-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
# Run the model training
python train_model.py
```

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- pandas
- matplotlib
- numpy

## Future Work
- Integration with real-time IoT networks.
- Deployment in edge devices for on-device threat detection.
- Incorporation of additional datasets for broader evaluation.

## License
This project is licensed under the MIT License.

## Contact
For any queries, please contact [Your Name] at [your.email@example.com].
