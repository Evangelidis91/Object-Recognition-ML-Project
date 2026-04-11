# Multi-Label Object Recognition with CNN

A multi-label image classification system built with TensorFlow/Keras, trained on the [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) dataset. The model detects multiple object classes in a single image using a custom Convolutional Neural Network (CNN).

## Classes

The model recognizes the following 8 object classes:

| Class | Class |
|-------|-------|
| Man | Woman |
| Car | Wheel |
| Tree | Clothing |
| Mammal | Furniture |

## Project Structure

```
.
├── object_recognition.py              # Main script: data loading, model building, training & evaluation
├── OpenImagesDatasetPreparation.py    # Dataset download & analysis using FiftyOne
├── model_test.py                      # Inference script for testing the trained model on new images
├── multilabel_cnn_tfdata_7_6.keras    # Pre-trained model weights
└── README.md
```

## Model Architecture

A sequential CNN with 4 convolutional blocks followed by dense layers:

- **4 Conv2D blocks**: 32 → 64 → 128 → 256 filters, each with BatchNormalization, ReLU activation, and MaxPooling
- **Global Average Pooling** to reduce spatial dimensions
- **2 Dense layers** (256, 128 units) with BatchNormalization and Dropout (0.5, 0.3)
- **Output layer**: 8 units with sigmoid activation (multi-label classification)
- **Regularization**: L2 regularization, Dropout, EarlyStopping, ReduceLROnPlateau

**Input size**: 300x300 RGB images

## Requirements

- Python 3.10+
- TensorFlow 2.x
- NumPy
- scikit-learn
- FiftyOne (for dataset download)

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn fiftyone
```

## Usage

### Training the Model

1. Update the `DATASET_DIR` path in `object_recognition.py` to your local directory.
2. Run the training script:

```bash
python object_recognition.py
```

This will:
- Download the Open Images V7 dataset via FiftyOne (if not already cached)
- Extract image paths and multi-hot labels
- Split data into train/validation/test sets (64%/16%/20%)
- Train the CNN with early stopping and learning rate scheduling
- Evaluate on the test set and print a classification report
- Save the trained model as `multilabel_cnn_tfdata_7_6.keras`

### Testing the Model

To run inference on a single image using the pre-trained model:

1. Update `MODEL_PATH` and `IMAGE_PATH` in `model_test.py`.
2. Run:

```bash
python model_test.py
```

The script outputs confidence scores for each class and lists detections above the threshold (default: 0.5).

## Training Details

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy, AUC (multi-label)
- **Data augmentation**: Random horizontal flips, random brightness adjustments
- **Callbacks**: EarlyStopping (patience=10), ReduceLROnPlateau (patience=5)
- **Max epochs**: 200 (with early stopping)
