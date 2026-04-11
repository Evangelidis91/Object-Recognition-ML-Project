# Multi-Label Object Recognition with CNN

> Master's Degree project (DAMA61) — Multi-label image classification using a custom CNN architecture.

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
├── requirements.txt                   # Python dependencies
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
pip install -r requirements.txt
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

## Dataset

- **Source**: Open Images V7, downloaded via [FiftyOne](https://docs.voxel51.com/)
- **Samples**: 10,000 per split (train / validation / test = 30,000 total)
- **Usable samples after filtering**: 30,000 (all contained at least one target class)
- **Final split** (after re-splitting): Train 19,200 / Validation 4,800 / Test 6,000

## Training Details

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy, AUC (multi-label)
- **Data augmentation**: Random horizontal flips, random brightness adjustments
- **Callbacks**: EarlyStopping (patience=10), ReduceLROnPlateau (patience=5)
- **Max epochs**: 200 (with early stopping)
- **Actual epochs trained**: 136 (early stopping triggered, best weights restored from epoch 126)

## Performance

Evaluated on the held-out test set (6,000 images):

| Metric | Score |
|--------|-------|
| Test Loss | 0.3875 |
| Test Accuracy | 0.4578 |
| Test AUC | 0.8286 |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Man | 0.59 | 0.46 | 0.52 | 2,132 |
| Car | 0.91 | 0.86 | 0.89 | 1,784 |
| Wheel | 0.73 | 0.67 | 0.70 | 1,308 |
| Woman | 0.59 | 0.31 | 0.41 | 1,587 |
| Tree | 0.68 | 0.33 | 0.45 | 567 |
| Clothing | 0.61 | 0.60 | 0.61 | 2,565 |
| Mammal | 0.56 | 0.17 | 0.26 | 1,515 |
| Furniture | 0.00 | 0.00 | 0.00 | 113 |
| **Weighted Avg** | **0.65** | **0.51** | **0.56** | **11,571** |

**Key observations:**
- **Car** is the best-performing class (F1: 0.89), likely due to its visually distinctive features
- **Furniture** has zero performance due to very low support (113 samples) — insufficient training data
- The model achieves a strong **AUC of 0.83**, indicating good ranking ability across classes despite moderate accuracy
- Recall is generally lower than precision, suggesting the model is conservative in its predictions
