import fiftyone as fo
import fiftyone.zoo as foz
import traceback

class OpenImagesDatasetPreparation:
    def __init__(self, dataset_dir, classes):
        self.dataset_dir = dataset_dir
        self.classes = classes

    def download_dataset(self, max_samples=10000):
        """
        Downloads the Open Images V7 dataset and returns the datasets.
        """
        try:
            # Download dataset
            print("Starting Open Images V7 download...")
            datasets = {}
            splits = ["train", "validation", "test"]

            for split in splits:
                print(f"\nDownloading {split} split...")
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split=split,
                    label_types=["detections"],  # Only get detections
                    classes=self.classes,
                    max_samples=max_samples,
                    #dataset_dir=self.dataset_dir  # Specify the download directory
                )
                # Specify dataset directory when creating the dataset
                #dataset.dataset_dir = self.dataset_dir

                # Load the dataset to trigger image download
                #dataset.load()
                datasets[split] = dataset

            if datasets:
                print("\nDataset preparation completed successfully!")
                return datasets
            else:
                print("Dataset download failed.")
                return None

        except Exception as e:
            print(f"Error in main execution: {e}")
            return None

    def analyze_dataset(self, dataset):
        """
        Analyzes a given FiftyOne dataset split and prints relevant information.
        """
        try:
            print("\nDataset Analysis:")
            print(f"  Number of samples: {len(dataset)}")

            # Count detections for each class
            class_counts = {}
            samples_with_detections = 0

            for sample in dataset:
                # Check if the sample has ground truth detections
                if sample.ground_truth and sample.ground_truth.detections:
                    samples_with_detections += 1
                    for detection in sample.ground_truth.detections:
                        label = detection.label
                        class_counts[label] = class_counts.get(label, 0) + 1

            print("\nDetection Statistics:")
            print(f"  Samples with detections: {samples_with_detections}")
            samples_without_detections = len(dataset) - samples_with_detections
            print(f"  Samples without detections: {samples_without_detections}")

            # # Print all unique labels found in the dataset:
            # unique_labels = set(class_counts.keys())
            # print("\nUnique labels found in the dataset:")
            # for label in unique_labels:
            #     print(f"    - {label}")

            if class_counts:

                sorted_class_items = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
                top_n_items = 10

                print("\nClass Distribution (Number of Detections per Class):")
                for label, count in sorted_class_items[:top_n_items]:
                    print(f"    {label}: {count}")

                # Calculate class distribution percentages
                total_detections = sum(class_counts.values())
                print("\nClass Distribution (Percentage of Detections per Class):")
                for label, count in sorted_class_items[:top_n_items]:
                    percentage = (count / total_detections) * 100
                    print(f"    {label}: {percentage:.2f}%")

            else:
                print("  No detections found in the dataset.")

        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            print(traceback.format_exc())

    def filter_dataset(self, dataset):
        """Filters the dataset to only include specified classes."""
        dataset.filter_labels("ground_truth", fo.ViewField("label").is_in(self.classes))
