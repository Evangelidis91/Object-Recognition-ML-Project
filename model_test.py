import tensorflow as tf
import numpy as np
import os # For path checking

def predict_image_multilabel(model_path, image_path, class_names, image_size=(300, 300), threshold=0.5):
    """
    Loads a trained Keras model and predicts for a single image.

    Returns:
        list: A list of dictionaries, each containing 'class', 'confidence' for detected classes,
              or None if an error occurred.
    """
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # --- Load Model ---
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        # --- Load and Preprocess Image ---
        img = tf.io.read_file(image_path)

        # Decode into uint8 then cast to float32 (keeps 0-255 range for Rescaling layer in model)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, image_size)
        img.set_shape([*image_size, 3])
        img = tf.expand_dims(img, axis=0)

        # --- Predict ---
        print("Running prediction...")
        preds = model.predict(img, verbose=0)[0]

        # --- Process Results ---
        detected_classes_list = []
        print("\n--- Predictions for Image ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"(Using threshold: {threshold})")
        print("\nScores:")
        for i, conf in enumerate(preds):
            if i < len(class_names):
                print(f"- {class_names[i]:<16}: {conf:.4f}")
            else:
                print(f"- Output index {i} (no class name): {conf:.4f}")

        print("\nDetected Classes above threshold:")
        found = False
        for i, conf in enumerate(preds):
             if i < len(class_names):
                 if conf > threshold:
                     class_name = class_names[i]
                     print(f"- {class_name} ({conf:.2f})")
                     detected_classes_list.append({'class': class_name, 'confidence': float(conf)})
                     found = True
        if not found:
            print("None")

        return detected_classes_list

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    MODEL_PATH = "multilabel_cnn_tfdata_7_6.keras"
    IMAGE_PATH = "./data/open-images-v7/test/data/sample.jpg"  # Set this to your test image path
    # For model <= 7.5
    #CLASSES = ['Man', 'Car', 'Wheel', 'Woman', 'Tree']
    # For model >= 7.6
    CLASSES = ['Man', 'Car', 'Wheel', 'Woman', 'Tree', 'Clothing', 'Mammal', 'Furniture']

    IMG_SIZE = (300, 300)
    CONFIDENCE_THRESHOLD = 0.5

    # Run the prediction
    detected = predict_image_multilabel(
        MODEL_PATH,
        IMAGE_PATH,
        CLASSES,
        image_size=IMG_SIZE,
        threshold=CONFIDENCE_THRESHOLD
    )

    if detected is not None:
        print(f"\nFunction returned {len(detected)} detected classes.")