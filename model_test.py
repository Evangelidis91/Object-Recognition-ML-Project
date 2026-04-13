import tensorflow as tf
import numpy as np
import os
import fiftyone as fo


def get_ground_truth_labels(image_path, class_names):
    """
    Looks up the ground truth labels for an image from FiftyOne datasets.

    Args:
        image_path: Absolute path to the image file.
        class_names: List of target class names to filter by.

    Returns:
        set: Set of ground truth class names found in the image, or None if not found.
    """
    # Search across all loaded FiftyOne datasets for this image
    for dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        # Match by filepath
        match = dataset.match({"filepath": image_path})
        if len(match) > 0:
            sample = match.first()
            if sample.ground_truth and sample.ground_truth.detections:
                labels = set()
                for detection in sample.ground_truth.detections:
                    if detection.label in class_names:
                        labels.add(detection.label)
                return labels
    return None


def load_and_preprocess_image(image_path, image_size):
    """
    Loads and preprocesses a single image for prediction.

    Args:
        image_path: Path to the image file.
        image_size: Tuple of (height, width) for resizing.

    Returns:
        tf.Tensor: Preprocessed image tensor with batch dimension.
    """
    img = tf.io.read_file(image_path)
    # Decode into uint8 then cast to float32 (keeps 0-255 range for Rescaling layer in model)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, image_size)
    img.set_shape([*image_size, 3])
    img = tf.expand_dims(img, axis=0)
    return img


def predict_single_image(model, image_path, class_names, image_size=(300, 300), threshold=0.5):
    """
    Runs prediction on a single image using an already-loaded model.

    Args:
        model: Loaded Keras model.
        image_path: Path to the image file.
        class_names: List of target class names.
        image_size: Tuple of (height, width) for resizing.
        threshold: Confidence threshold for detection.

    Returns:
        list: Detected classes with confidence, or None on error.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        img = load_and_preprocess_image(image_path, image_size)
        preds = model.predict(img, verbose=0)[0]

        detected = []
        print(f"\n--- {os.path.basename(image_path)} ---")
        print("Scores:")
        for i, conf in enumerate(preds):
            if i < len(class_names):
                print(f"  {class_names[i]:<16}: {conf:.4f}")

        for i, conf in enumerate(preds):
            if i < len(class_names) and conf > threshold:
                detected.append({'class': class_names[i], 'confidence': float(conf)})

        return detected

    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None


def compare_prediction(image_path, detected, class_names):
    """
    Compares model predictions against ground truth from FiftyOne.

    Args:
        image_path: Path to the image file.
        detected: List of detected class dicts from predict_single_image.
        class_names: List of target class names.
    """
    predicted_classes = {d['class'] for d in detected} if detected else set()
    ground_truth = get_ground_truth_labels(image_path, class_names)

    print(f"\nPredicted    : {sorted(predicted_classes) if predicted_classes else 'None'}")
    if ground_truth is not None:
        print(f"Ground Truth : {sorted(ground_truth)}")

        correct = predicted_classes & ground_truth
        missed = ground_truth - predicted_classes
        false_pos = predicted_classes - ground_truth

        if correct:
            print(f"Correct      : {sorted(correct)}")
        if missed:
            print(f"Missed       : {sorted(missed)}")
        if false_pos:
            print(f"False Pos    : {sorted(false_pos)}")

        return correct, missed, false_pos
    else:
        print("Ground truth not found in FiftyOne datasets.")
        return None, None, None


if __name__ == "__main__":
    MODEL_PATH = "multilabel_cnn_tfdata_7_7.keras"
    # Single image or list of images
    IMAGE_PATHS = [
        "/Users/konstantinosevangelidis/fiftyone/open-images-v7/test/data/0a4f0099934b081c.jpg",
        "/Users/konstantinosevangelidis/fiftyone/open-images-v7/test/data/0a4b20e6e5a52e49.jpg",
        "/Users/konstantinosevangelidis/fiftyone/open-images-v7/test/data/0a5f6925b7af0423.jpg",
        "/Users/konstantinosevangelidis/fiftyone/open-images-v7/test/data/0a8c37ddfc3ee652.jpg",
        "/Users/konstantinosevangelidis/fiftyone/open-images-v7/test/data/0a70f739f5470c48.jpg"
    ]
    CLASSES = ['Man', 'Car', 'Wheel', 'Woman', 'Tree', 'Clothing', 'Mammal', 'Furniture']
    IMG_SIZE = (300, 300)
    CONFIDENCE_THRESHOLD = 0.5

    # Load model once
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    # Track totals across all images
    total_correct = 0
    total_missed = 0
    total_false_pos = 0
    images_processed = 0

    for image_path in IMAGE_PATHS:
        detected = predict_single_image(model, image_path, CLASSES, IMG_SIZE, CONFIDENCE_THRESHOLD)
        if detected is not None:
            correct, missed, false_pos = compare_prediction(image_path, detected, CLASSES)
            if correct is not None:
                total_correct += len(correct)
                total_missed += len(missed)
                total_false_pos += len(false_pos)
                images_processed += 1

    # Print summary if multiple images were processed
    if images_processed > 1:
        print(f"\n{'=' * 50}")
        print(f"Summary ({images_processed} images)")
        print(f"  Correct predictions : {total_correct}")
        print(f"  Missed labels       : {total_missed}")
        print(f"  False positives     : {total_false_pos}")
        total_gt = total_correct + total_missed
        if total_gt > 0:
            print(f"  Recall              : {total_correct / total_gt:.2%}")
        total_pred = total_correct + total_false_pos
        if total_pred > 0:
            print(f"  Precision           : {total_correct / total_pred:.2%}")
