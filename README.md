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

## Tech Stack

| Technology | Role |
|------------|------|
| **TensorFlow / Keras** | Model building, training, and inference |
| **tf.data API** | Efficient input pipeline with parallel loading, batching, and prefetching |
| **FiftyOne** | Open Images V7 dataset download and management |
| **scikit-learn** | Data splitting (`train_test_split`) and evaluation (`classification_report`) |
| **NumPy** | Multi-hot label encoding and array operations |

## Project Structure

```
.
├── object_recognition.py              # Main script: data loading, model building, training & evaluation
├── OpenImagesDatasetPreparation.py    # Dataset download & analysis using FiftyOne
├── model_test.py                      # Inference script for testing the trained model on new images
├── multilabel_cnn_tfdata_7_7.keras    # Pre-trained model weights
├── requirements.txt                   # Python dependencies
└── README.md
```

## Model Architecture

A sequential CNN with 4 convolutional blocks followed by dense layers:

```
Input (300x300x3)
    |
    v
[Conv2D 32] -> BatchNorm -> ReLU -> MaxPool
    |
    v
[Conv2D 64] -> BatchNorm -> ReLU -> MaxPool
    |
    v
[Conv2D 128] -> BatchNorm -> ReLU -> MaxPool
    |
    v
[Conv2D 256] -> BatchNorm -> ReLU -> MaxPool
    |
    v
Global Average Pooling
    |
    v
Dense 256 -> BatchNorm -> Dropout (0.5)
    |
    v
Dense 128 -> BatchNorm -> Dropout (0.3)
    |
    v
Dense 8 (sigmoid) -> Multi-label output
```

- **Regularization**: L2 weight decay (1e-4), Dropout, EarlyStopping, ReduceLROnPlateau
- All convolutional layers use `same` padding and L2 regularization

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
- Save the trained model as `multilabel_cnn_tfdata_7_7.keras`

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
- **Final split**: Train 19,968 / Validation 4,992 / Test 19,968 (using FiftyOne's original splits)

## Training Details

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Binary cross-entropy
- **Metrics**: BinaryAccuracy, AUC, Precision, Recall
- **Data augmentation**: Random horizontal flips, random brightness, contrast, and saturation adjustments
- **Callbacks**: EarlyStopping (patience=20), ReduceLROnPlateau (patience=10)
- **Max epochs**: 200 (with early stopping)
- **Actual epochs trained**: 89 (early stopping triggered, best weights restored from epoch 69)

## Performance

Evaluated on the held-out test set (18,494 labeled instances):

| Metric | Score |
|--------|-------|
| Test Loss | 0.4383 |
| Binary Accuracy | 0.8251 |
| Test AUC | 0.8530 |
| Precision | 0.6750 |
| Recall | 0.4722 |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Man | 0.49 | 0.54 | 0.51 | 2,146 |
| Car | 0.83 | 0.87 | 0.85 | 2,895 |
| Wheel | 0.71 | 0.73 | 0.72 | 2,503 |
| Woman | 0.51 | 0.36 | 0.42 | 1,501 |
| Tree | 0.68 | 0.78 | 0.73 | 2,038 |
| Clothing | 0.68 | 0.29 | 0.41 | 3,869 |
| Mammal | 0.00 | 0.00 | 0.00 | 3,161 |
| Furniture | 0.00 | 0.00 | 0.00 | 381 |
| **Weighted Avg** | **0.54** | **0.47** | **0.49** | **18,494** |

**Key observations:**
- **Car** is the best-performing class (F1: 0.85), likely due to its visually distinctive features
- **Tree** improved significantly (F1: 0.73) with well-calibrated confidence scores
- **Mammal** and **Furniture** have zero performance — Furniture due to very low support, Mammal likely due to high visual diversity within the class
- The model achieves a strong **AUC of 0.85**, indicating good ranking ability across classes
- Recall is generally lower than precision, suggesting the model is conservative in its predictions

### Example Prediction

Running inference on a test image of a car:

```
--- 0a5f6925b7af0423.jpg ---
Scores:
  Man             : 0.0201
  Car             : 0.9584
  Wheel           : 0.7178
  Woman           : 0.0059
  Tree            : 0.0037
  Clothing        : 0.0176
  Mammal          : 0.0010
  Furniture       : 0.0004

Predicted    : ['Car', 'Wheel']
Ground Truth : ['Car', 'Wheel']
Correct      : ['Car', 'Wheel']
```

The model correctly identifies both Car and Wheel with high confidence while keeping unrelated classes near zero.

## Limitations & Future Work

- **Class imbalance**: Furniture had only 381 test samples vs. 3,869 for Clothing. Techniques like oversampling, class weighting, or focal loss could help underrepresented classes.
- **Custom CNN vs. transfer learning**: The model is built from scratch as a learning exercise. Using a pre-trained backbone (e.g., ResNet, EfficientNet) would likely improve performance significantly.
- **Low recall on some classes**: Woman (0.36) and Clothing (0.29) suffer from low recall. Threshold tuning per class or additional data augmentation could improve detection rates.
- **Mammal class failure**: Despite 3,161 test samples, Mammal achieves 0.00 F1 — likely due to the extreme visual diversity within the class (dogs, cats, horses, etc.).
- **Dataset scale**: Training on 30,000 images is relatively small for 8-class multi-label classification. Scaling to 100K+ samples would likely improve generalization.
- **No visual explainability**: Adding Grad-CAM or similar attention visualization would help interpret what regions the model focuses on for each class.
