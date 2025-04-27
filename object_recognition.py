import os
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers, optimizers, Input
from keras.src import metrics
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
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
IMAGE_SIZE = (250, 250)  # Adjusted image size for the model input
BATCH_SIZE = 32  # Number of samples processed in each training iteration.
EPOCHS = 100  # Maximum number of times to iterate over the entire training dataset.
CLASSES = ['Man', 'Car', 'Wheel', 'Woman', 'Tree']  # List of specific object classes to detect.
NUM_CLASSES = len(CLASSES)  # The total number of classes based on the CLASSES list.


# --- Data Extraction: Multi-hot Labels ---
def extract_paths_and_multihot_labels(datasets, classes):
    """
    Finds picture files and creates labels for them.

    This function looks through the picture information.
    It only keeps pictures that show at least one of the items in CLASSES
    variable list and makes sure the picture file exists.
    For each picture, it creates a multi-hot label that includes info if the
    items from CLASSES variables exists into the picture.

    Args:
        datasets: Dictionary of fiftyone dataset splits.
        classes: List of target class names.
    Returns:
        image_paths: list of valid filepaths.
        labels_array: list of labels of the valid pictures
    """
    print("\nExtracting file paths and multi-hot labels...")
    image_paths = []  # An empty list to hold the file paths we find.
    labels_list = []  # An empty list to hold the labels we create.
    num_classes_local = len(classes)  # Use local variable for clarity

    # Check if datasets is empty or None, return empty lists
    if not datasets or all(v is None for v in datasets.values()):
        print("Warning: Dataset dictionary is empty or invalid. Cannot extract data.")
        return [], np.array([], dtype=np.float32)  # returns empty lists

    # Look at each of the set('train', 'validation', 'test')
    for split, dataset in datasets.items():
        # Skip if set is empty
        if dataset is None: print(f"Skipping empty split: {split}"); continue

        print(f"Extracting from {split}...")
        split_count = 0  # Counter for valid images

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
                    if hasattr(detection, 'label') and detection.label in classes:
                        # index class position
                        idx = classes.index(detection.label)
                        # 0 to 1 for this index position
                        label_vector[idx] = 1
                        # Change the flag
                        has_target_class = True

                # Check if we keep this picture and add info to the list variables
                if has_target_class and os.path.exists(sample.filepath):
                    image_paths.append(sample.filepath)
                    labels_list.append(label_vector)
                    split_count += 1
            # Handle error
            except Exception as e:
                print(f"Error processing sample {sample.id}: {e}")
        print(f"Extracted {split_count} valid samples from {split}.")

    # Check if we found any picture or return empty lists
    if not image_paths: print("Warning: No valid image paths extracted."); return [], np.array([])

    # Convert list of lists/arrays to a single NumPy array (float32 for TF)
    labels_array = np.array(labels_list, dtype=np.float32)
    print(f"Total extracted samples: {len(image_paths)}")
    return image_paths, labels_array


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
    def load_and_preprocess(path, label):
        try:
            # Read the picture file.
            img = tf.io.read_file(path)
            # Decode with error handling
            img = tf.io.decode_image(img, channels=3, expand_animations=False, dtype=tf.float32)
            img.set_shape([None, None, 3])  # Set shape after decoding
            img = tf.image.resize(img, IMAGE_SIZE)  # Resize the image
            img.set_shape([*IMAGE_SIZE, 3])  # Set shape after resizing

            # Apply basic augmentation if specified
            if augment:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=0.1)  # Reduced delta
                # Add more augmentation here if needed
                # img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            return img, label
        # handle errors
        except Exception as e:
            tf.print(f"Error processing image {path}: {e}")
            return tf.zeros([*IMAGE_SIZE, 3]), label

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

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
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
                  metrics=['accuracy', metrics.AUC(name='auc', multi_label=True)])
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
        # turn in into pixel data
        img = tf.io.decode_image(img, channels=3, expand_animations=False, dtype=tf.float32)
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
        preds = model.predict(img)[0]

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
        print(f"Error: Image not found at {image_path}"); return []
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}"); return []


# ==================
# Main Execution
# ==================
if __name__ == "__main__":
    # --- Configuration ---
    DATASET_DIR = "/Users/konstantinosevangelidis/fiftyone/open-images-v7"
    TEST_IMAGE_PATH = "/Users/konstantinosevangelidis/fiftyone/open-images-v7/train/data/00a0e0767835954f.jpg"
    MAX_SAMPLES = 10000  # Adjust number of samples to load per split
    MODEL_SAVE_PATH = "multilabel_cnn_tfdata_7_4.keras"
    # --- End Configuration ---

    # --- Basic Checks ---
    is_placeholder = "/path/to/" in DATASET_DIR
    if is_placeholder:
        print(
            "=" * 60 + "\n!!! WARNING: You MUST change the DATASET_DIR path in the script! Cannot run. !!!\n" + "=" * 60)
        exit()

    # --- Load Dataset Information ---
    # Download/load the dataset meta-info
    try:
        # Create the dataset tool object
        dataset_prep = OpenImagesDatasetPreparation(DATASET_DIR, CLASSES)
        datasets = dataset_prep.download_dataset(max_samples=MAX_SAMPLES)
        print("******************************************************************")
        dataset_prep.analyze_dataset(datasets)
        print("******************************************************************")
        if not datasets: raise ValueError("Dataset loading/download failed.")
    except Exception as e:
        print(f"Error initializing dataset: {e}");
        exit()

    # --- Get Picture Paths and Labels ---
    # Extract filepaths and multi-hot labels
    image_paths, labels_array = extract_paths_and_multihot_labels(datasets, CLASSES)
    if not image_paths: print("Error: No data extracted. Check dataset and classes."); exit()

    # --- Split Data into Sets ---
    # We need to split our data into three groups:
    # 1. Training set: Used to teach the model. (Largest group)
    # 2. Validation set: Used to check the model during training and tune settings.
    # 3. Test set: Used only at the very end to see how well the trained model performs on new data.
    indices = np.arange(len(image_paths))
    try:
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error during index split: {e}");
        exit()

    # Create path/label lists for each split
    train_paths = [image_paths[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    test_paths = [image_paths[i] for i in test_idx]

    # Get the labels corresponding to the paths.
    train_labels = labels_array[train_idx]
    val_labels = labels_array[val_idx]
    test_labels = labels_array[test_idx]
    # Show how many pictures are in each set.
    print(f"\nSplit Sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    if not train_paths or not val_paths or not test_paths: print("Error: Empty data splits."); exit()

    # --- Create TensorFlow Datasets ---
    print("\nCreating TensorFlow datasets...")
    train_ds = create_tf_dataset(train_paths, train_labels, BATCH_SIZE, augment=True)
    val_ds = create_tf_dataset(val_paths, val_labels, BATCH_SIZE, augment=False)
    test_ds = create_tf_dataset(test_paths, test_labels, BATCH_SIZE, augment=False)
    if train_ds is None or val_ds is None or test_ds is None: print("Error creating datasets."); exit()
    print("TensorFlow datasets created.")

    # Build and train the model
    model = build_cnn_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), NUM_CLASSES)
    print("\n--- Model Summary ---")
    model.summary()

    print("\n--- Starting Model Training ---")
    # Define callbacks
    callbacks_list = [
        # Stop training early if the model stops improving on the validation set.
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        # Reduce the learning rate if the model gets stuck (validation loss stops improving).
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
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
    test_loss, test_acc, test_auc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

    # Get predictions for classification report
    print("\n--- Generating Classification Report ---")
    try:
        # Get the model's predictions for the test set.
        y_pred_proba = model.predict(test_ds)
        # Use a fixed threshold for the report
        report_threshold = 0.5
        y_pred = (y_pred_proba > report_threshold).astype(int)

        # Need to get y_true by iterating through the test dataset again
        y_true = []
        for _, label_batch in test_ds:
            y_true.extend(label_batch.numpy())
        y_true = np.array(y_true)

        # Ensure shapes match before report
        min_len = min(len(y_pred), len(y_true))
        print(classification_report(y_true[:min_len], y_pred[:min_len], target_names=CLASSES, zero_division=0))
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Predict on a single image
    if os.path.exists(TEST_IMAGE_PATH):
        detections = predict_multiple_classes(model, TEST_IMAGE_PATH, CLASSES, threshold=0.5)
        # print("\nDetected Classes in Test Image:")
        # for det in detections:
        #     print(f"- {det['class']} ({det['confidence']:.2f})")
    else:
        print(f"\nTest image not found, skipping prediction: {TEST_IMAGE_PATH}")

    print("\n--- Script Finished ---")