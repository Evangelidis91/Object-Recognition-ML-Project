import os
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers, optimizers, Input
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# --- Placeholder for Dataset Preparation ---
# This section attempts to import a custom module for dataset handling
try:
    from OpenImagesDatasetPreparation import OpenImagesDatasetPreparation

    print("Successfully imported OpenImagesDatasetPreparation.")
except ImportError:
    print("Warning: OpenImagesDatasetPreparation not found. Using placeholder.")


    class OpenImagesDatasetPreparation:
        def __init__(self, dataset_dir, classes):
            print("Placeholder: OpenImagesDatasetPreparation initialized.")
            self.dataset_dir = dataset_dir
            self.classes = classes

        def download_dataset(self, max_samples):
            print(f"Placeholder: Simulating download/load for {max_samples} samples per split.")
            # Return empty dict to allow script flow in placeholder mode
            return {'train': None, 'validation': None, 'test': None}
# --- End Placeholder ---


# --- Constants Definition ---
IMAGE_SIZE = (300, 300)  # Adjusted image size for the model input
BATCH_SIZE = 32  # Number of samples processed in each training iteration.
EPOCHS = 200  # Maximum number of times to iterate over the entire training dataset.
CLASSES = ['Man', 'Car', 'Wheel', 'Woman', 'Tree', 'Clothing', 'Mammal', 'Furniture']  # List of specific object classes to detect.
NUM_CLASSES = len(CLASSES)  # The total number of classes based on the CLASSES list.


# --- Data Extraction: Multi-hot Labels ---
def extract_paths_and_multihot_labels(datasets, classes):
    """
    Finds picture files and creates labels for them, preserving original splits.

    This function looks through the picture information per split.
    It only keeps pictures that show at least one of the items in CLASSES
    variable list and makes sure the picture file exists.
    For each picture, it creates a multi-hot label that includes info if the
    items from CLASSES variables exists into the picture.

    Args:
        datasets: Dictionary of fiftyone dataset splits.
        classes: List of target class names.
    Returns:
        split_data: dict mapping split name to (image_paths, labels_array).
    """
    print("\nExtracting file paths and multi-hot labels...")
    num_classes_local = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Check if datasets is empty or None
    if not datasets or all(v is None for v in datasets.values()):
        print("Warning: Dataset dictionary is empty or invalid. Cannot extract data.")
        return {}

    split_data = {}

    # Look at each of the set('train', 'validation', 'test')
    for split, dataset in datasets.items():
        # Skip if set is empty
        if dataset is None:
            print(f"Skipping empty split: {split}")
            continue

        print(f"Extracting from {split}...")
        image_paths = []
        labels_list = []

        # Go through each picture's information in the set.
        # Use iter_samples for potential memory efficiency with large fiftyone datasets
        for sample in dataset.iter_samples(autosave=False, progress=True):
            try:
                # --- Check if the picture info is usable ---
                # Make sure it tells us the file path and what's in the picture ('ground_truth').
                if not hasattr(sample, 'filepath') or not hasattr(sample, 'ground_truth'):
                    continue  # Skip this picture if info is missing.

                # Make sure there are 'detections' (found objects) listed.
                if not sample.ground_truth or not hasattr(sample.ground_truth, 'detections'):
                    continue  # Skip if no objects were listed for this picture.

                # --- Create the label ---
                label_vector = np.zeros(num_classes_local, dtype=int)
                has_target_class = False  # Flag to check if there is at least one thing

                # Look each of the objects found in the picture.
                for detection in sample.ground_truth.detections:
                    if hasattr(detection, 'label') and detection.label in class_to_idx:
                        # O(1) lookup instead of classes.index()
                        idx = class_to_idx[detection.label]
                        # 0 to 1 for this index position
                        label_vector[idx] = 1
                        # Change the flag
                        has_target_class = True

                # Check if we keep this picture and add info to the list variables
                if has_target_class and os.path.exists(sample.filepath):
                    image_paths.append(sample.filepath)
                    labels_list.append(label_vector)
            # Handle error
            except Exception as e:
                print(f"Error processing sample {sample.id}: {e}")

        print(f"Extracted {len(image_paths)} valid samples from {split}.")

        if image_paths:
            labels_array = np.array(labels_list, dtype=np.float32)
            split_data[split] = (image_paths, labels_array)

    total = sum(len(paths) for paths, _ in split_data.values())
    print(f"Total extracted samples: {total}")
    return split_data


# --- Prepare Data for TensorFlow ---
def create_tf_dataset(image_paths, labels, batch_size, augment=False):
    """
    Gets data ready for the model using Tensorflow

    This takes the lists of pictures paths and labels and makes a
    Tensorflow dataset, efficient for training. It handles:
    - Loading pictures from files.
    - Changing their size
    - Grouping them into batches.
    - Get the data ready

    Args:
        image_paths : List of file locations for the pictures.
        labels : The labels for each picture.
        batch_size : How many pictures to put in each group (batch).
        augment : If True, slightly change the training pictures (flip, change brightness).
                        Helps the model the learn better

    Returns:
        tf.data.Dataset: A TensorFlow Dataset, or None if no paths were given.
    """

    # Check if we have any picture path.
    if not image_paths:
        print("Error: No image paths provided to create dataset.")
        return None

    # Ensure labels are numpy array of correct type
    labels = np.array(labels, dtype=np.float32)

    # Create TensorFlow Datasets from the lists of paths and labels.
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    # --- Function to Load and Prepare One Picture ---
    # This function will be applied to every (path, label) pair.
    # Note: try/except does NOT work inside tf.data.map() (graph mode).
    # If an image fails to decode, TensorFlow will raise an op-level error.
    def load_and_preprocess(path, label):
        # Read the picture file.
        img = tf.io.read_file(path)
        # Decode into uint8 then cast to float32 (keeps 0-255 range for Rescaling layer in model)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        img.set_shape([None, None, 3])  # Set shape after decoding
        img = tf.image.resize(img, IMAGE_SIZE)  # Resize the image
        img.set_shape([*IMAGE_SIZE, 3])  # Set shape after resizing

        # Apply data augmentation if specified (training only)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        return img, label

    # --- Apply Steps to the Dataset ---
    # Run the 'load_and_preprocess' function on every item in the dataset.
    # num_parallel_calls=tf.data.AUTOTUNE helps TensorFlow do this faster using computer resources smartly.
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        # Decide how much data to load for shuffling.
        buffer_size = min(len(image_paths), 2000)
        ds = ds.shuffle(buffer_size=buffer_size)

    # Group the data into batches.
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# --- Build the Model ---
def build_cnn_model(input_shape, num_classes):
    """
        Builds the neural network model (the 'brain') for recognizing pictures.

        This model uses layers common in image recognition (Convolutional Neural Network - CNN).
        It includes techniques like Batch Normalization, Pooling, Dropout, and Regularization
        to help it learn better and avoid mistakes. The final layer is set up for finding
        multiple possible labels in one picture.

        Args:
            input_shape : The size of the pictures going into the model (height, width, channels).
            num_classes : How many different classes the model should look for.

        Returns:
            keras.Model: The ready-to-train Keras model.
        """

    l2_reg = 1e-4  # Reduced L2 factor

    # Create the model.
    model = models.Sequential([
        # Tell the model what size the input pictures will be.
        Input(shape=input_shape),

        # Normalize pixel values from [0, 255] to [0, 1] so the model is self-contained.
        layers.Rescaling(1./255),

        # --- Block 1: Find basic patterns ---
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # --- Block 2: Find more complex patterns ---
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # --- Block 3 ---
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # --- Block 4 ---
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        # Dense blocks follow same pattern as conv blocks: Linear→BN→ReLU
        layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label
    ])

    # --- Set up the Learning Process ---
    # Choose the Adam optimizer for learning. Use a slightly lower learning rate.
    optimizer = optimizers.Adam(learning_rate=1e-4)

    # --- Compile the Model ---
    # Get the model ready for training by telling it how to learn.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[
                      metrics.BinaryAccuracy(name='binary_accuracy'),
                      metrics.AUC(name='auc', multi_label=True),
                      metrics.Precision(name='precision'),
                      metrics.Recall(name='recall'),
                  ])
    return model


# --- Predict Classes for a Single Picture ---
def predict_multiple_classes(model, image_path, classes, threshold=0.5):
    """
        Uses the trained model to find classes in one specific picture file.

        Args:
            model : The trained model.
            image_path : The location of the picture file to check.
            classes : The list of class names the model knows.
            threshold : How sure the model needs to be (0.0 to 1.0) to say it found a class.
                               Default is 0.5.

        Returns:
            list: A list of found classes. Each item in the list tells the 'class' name
                  and the 'confidence'. Returns an empty list
                  if the picture file is not found or there's an error.
        """

    try:

        # --- Load and Prepare the Picture ---
        # Read image file
        img = tf.io.read_file(image_path)
        # turn it into pixel data (keep 0-255 range, Rescaling layer in model normalizes)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        img.set_shape([None, None, 3])
        # Resize it to the required size
        img = tf.image.resize(img, IMAGE_SIZE)
        img.set_shape([*IMAGE_SIZE, 3])
        # Add an extra dimension at the beginning (batch dimension). The model expects a batch, even if it's just one picture.
        img = tf.expand_dims(img, axis=0)
        """
        Notes
        Final shape is gonna be like this
        (1, height, width, channels)
        """

        # Predict and get first (only) batch item
        preds = model.predict(img, verbose=0)[0]

        # Empty list to store classes we detect.
        detected = []
        print("\n--- Predictions for Single Image ---")
        print(f"(Using threshold: {threshold})")
        print("Scores:")
        # Look at the probability score for each class.
        for i, conf in enumerate(preds):
            # Print the score for this class.
            print(f"- {classes[i]:<16}: {conf:.4f}")
            if conf > threshold:
                detected.append({'class': classes[i], 'confidence': float(conf)})

        print("\nDetected Classes above threshold:")
        if detected:
            for det in detected: print(f"- {det['class']} ({det['confidence']:.2f})")
        else:
            print("None")
        return detected

    # --- Handle Errors ---
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return []
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return []


# ==================
# Main Execution
# ==================
if __name__ == "__main__":
    # --- Reproducibility ---
    tf.random.set_seed(42)
    np.random.seed(42)

    # --- Configuration ---
    # Set these paths to match your local environment before running
    DATASET_DIR = "./data/open-images-v7"  # Path to Open Images V7 dataset
    TEST_IMAGE_PATH = "./data/open-images-v7/test/data/sample.jpg"  # Path to a test image
    MAX_SAMPLES = 10000  # Adjust number of samples to load per split
    MODEL_SAVE_PATH = "multilabel_cnn_tfdata_7_6.keras"
    # --- End Configuration ---

    # --- Basic Checks ---
    if not os.path.isdir(DATASET_DIR):
        print(
            "=" * 60 + f"\n!!! WARNING: DATASET_DIR '{DATASET_DIR}' does not exist! Cannot run. !!!\n" + "=" * 60)
        exit()

    # --- Load Dataset Information ---
    # Download/load the dataset meta-info
    try:
        # Create the dataset tool object
        dataset_prep = OpenImagesDatasetPreparation(DATASET_DIR, CLASSES)
        datasets = dataset_prep.download_dataset(max_samples=MAX_SAMPLES)
        if not datasets:
            raise ValueError("Dataset loading/download failed.")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # --- >>> NEW: Analyze Datasets <<< ---
    # Check if the dataset_prep object actually has the 'analyze_dataset' method
    # This prevents errors if you are running with the placeholder class definition
    if hasattr(dataset_prep, 'analyze_dataset'):
        print("\n--- Starting Dataset Analysis (if available) ---")
        # Iterate through the splits returned (e.g., 'train', 'validation', 'test')
        for split_name, fiftyone_dataset_split in datasets.items():
            if fiftyone_dataset_split is not None:
                print(f"\n--- Analyzing the '{split_name}' split ---")
                try:
                    # Call the analyze_dataset method on the dataset_prep object,
                    # passing the specific FiftyOne dataset split object
                    dataset_prep.analyze_dataset(fiftyone_dataset_split)
                    print(f"--- Finished analyzing '{split_name}' split ---")
                except Exception as e:
                    print(f"Error during analysis of '{split_name}' split: {e}")
                    # import traceback
                    # print(traceback.format_exc())
            else:
                print(f"\nSkipping analysis for '{split_name}' split because it is None.")
        print("\n--- Dataset Analysis Finished ---")
    else:
        print("\nSkipping dataset analysis: 'analyze_dataset' method not found (possibly using placeholder).")
    # --- >>> End of New Analysis Section <<< ---

    # --- Get Picture Paths and Labels ---
    # Extract filepaths and multi-hot labels, preserving original FiftyOne splits.
    split_data = extract_paths_and_multihot_labels(datasets, CLASSES)
    if not split_data:
        print("Error: No data extracted. Check dataset and classes.")
        exit()

    # --- Use Original Splits ---
    # FiftyOne already provides train/validation/test splits — no need to re-split.
    for required_split in ['train', 'validation', 'test']:
        if required_split not in split_data:
            print(f"Error: Missing '{required_split}' split in extracted data.")
            exit()

    train_paths, train_labels = split_data['train']
    val_paths, val_labels = split_data['validation']
    test_paths, test_labels = split_data['test']
    # Show how many pictures are in each set.
    print(f"\nSplit Sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # --- Create TensorFlow Datasets ---
    print("\nCreating TensorFlow datasets...")
    train_ds = create_tf_dataset(train_paths, train_labels, BATCH_SIZE, augment=True)
    val_ds = create_tf_dataset(val_paths, val_labels, BATCH_SIZE, augment=False)
    test_ds = create_tf_dataset(test_paths, test_labels, BATCH_SIZE, augment=False)
    if train_ds is None or val_ds is None or test_ds is None:
        print("Error creating datasets.")
        exit()
    print("TensorFlow datasets created.")

    # Build and train the model
    model = build_cnn_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), NUM_CLASSES)
    print("\n--- Model Summary ---")
    model.summary()

    print("\n--- Starting Model Training ---")
    # Define callbacks — monitor val_auc (more meaningful for imbalanced multi-label)
    callbacks_list = [
        # Stop training early if the model stops improving on the validation set.
        EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1),
        # Reduce the learning rate if the model gets stuck.
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        # Save the best model to disk so training progress isn't lost if interrupted.
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    ]
    # Start the training process
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )
    print("--- Training Finished ---")

    # Evaluate on test set
    print("\n--- Evaluating Model ---")
    test_loss, test_bin_acc, test_auc, test_prec, test_rec = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Binary Accuracy: {test_bin_acc:.4f}, "
          f"AUC: {test_auc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

    # Get predictions for classification report
    print("\n--- Generating Classification Report ---")
    try:
        # Get the model's predictions for the test set.
        y_pred_proba = model.predict(test_ds)
        # Use a fixed threshold for the report
        report_threshold = 0.5
        y_pred = (y_pred_proba > report_threshold).astype(int)

        # Use test_labels directly instead of re-iterating the dataset
        y_true = test_labels

        # Ensure shapes match before report (batching may pad the last batch)
        min_len = min(len(y_pred), len(y_true))
        print(classification_report(y_true[:min_len], y_pred[:min_len], target_names=CLASSES, zero_division=0))
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # ModelCheckpoint already saved the best model by val_auc during training.
    # No need to save again here — that would overwrite the best with the last epoch.

    # Predict on a single image
    if os.path.exists(TEST_IMAGE_PATH):
        detections = predict_multiple_classes(model, TEST_IMAGE_PATH, CLASSES, threshold=0.5)
        print("\nDetected Classes in Test Image:")
        for det in detections:
            print(f"- {det['class']} ({det['confidence']:.2f})")
    else:
        print(f"\nTest image not found, skipping prediction: {TEST_IMAGE_PATH}")

    print("\n--- Script Finished ---")