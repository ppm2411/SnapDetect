import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 9
np.random.seed(SEED)

# Safety checks for required files
assert os.path.exists(
    "annotations/train_annotations.coco.json"
), "Train annotations file missing!"
assert os.path.exists(
    "annotations/valid_annotations.coco.json"
), "Validation annotations file missing!"


# Function to extract features manually
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys",
    )

    edges = cv2.Canny(gray, 50, 150)
    edge_features = edges.flatten() / 255.0
    gray_features = gray.flatten() / 255.0

    combined_features = np.concatenate([hog_features, edge_features, gray_features])
    return combined_features


def load_annotations(annotations_file):
    with open(annotations_file, "r") as f:
        data = json.load(f)
    annotations = []
    for item in data["images"]:
        anns = [ann for ann in data["annotations"] if ann["image_id"] == item["id"]]
        annotations.append({"file_name": item["file_name"], "annotations": anns})
    return annotations


def process_single_image(record, image_dir):
    img_path = os.path.join(image_dir, record["file_name"])
    image = cv2.imread(img_path)
    if image is None or len(record["annotations"]) == 0:
        return None, None
    bbox = record["annotations"][0]["bbox"]
    features = extract_features(image)
    return features, bbox


def preprocess_images_concurrently(annotations, image_dir):
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda record: process_single_image(record, image_dir), annotations
            )
        )
    X = [result[0] for result in results if result[0] is not None]
    y = [result[1] for result in results if result[1] is not None]
    return np.array(X), np.array(y)


# Load annotations with fixed paths
train_annotations = load_annotations("annotations/train_annotations.coco.json")
valid_annotations = load_annotations("annotations/valid_annotations.coco.json")

# Filter out images with no annotations
train_annotations = [ann for ann in train_annotations if len(ann["annotations"]) > 0]
valid_annotations = [ann for ann in valid_annotations if len(ann["annotations"]) > 0]

# Set image directories
train_images_dir = "BoneFractureYolo8/train/images"
valid_images_dir = "BoneFractureYolo8/valid/images"

# Prepare data
X_train, y_train = preprocess_images_concurrently(train_annotations, train_images_dir)
X_valid, y_valid = preprocess_images_concurrently(valid_annotations, valid_images_dir)

# Reduce feature dimensionality using PCA
pca = PCA(n_components=100)  # Choose an appropriate number of components
X_train_reduced = pca.fit_transform(X_train)
X_valid_reduced = pca.transform(X_valid)

# Use Random Forest instead of SVM
rf_models = []
for i in range(4):
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X_train_reduced, y_train[:, i])
    rf_models.append(rf)

# Predict and evaluate
y_pred = np.zeros_like(y_valid)
for i, rf in enumerate(rf_models):
    y_pred[:, i] = rf.predict(X_valid_reduced)

mse = mean_squared_error(y_valid, y_pred)
print(f"Validation MSE: {mse:.4f}")


# IoU calculation
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


ious = [calculate_iou(pred, true) for pred, true in zip(y_pred, y_valid)]
mean_iou = np.mean(ious)
print(f"Mean IoU: {mean_iou:.4f}")


# Save predictions instead of plotting
def save_predictions(
    X, y_true, y_pred, images_dir, annotations, output_dir="output_predictions"
):
    os.makedirs(output_dir, exist_ok=True)
    for i, ax in enumerate(X):
        img_path = os.path.join(images_dir, annotations[i]["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        true_box = y_true[i]
        cv2.rectangle(
            image,
            (int(true_box[0]), int(true_box[1])),
            (int(true_box[0] + true_box[2]), int(true_box[1] + true_box[3])),
            (0, 255, 0),
            2,
        )

        pred_box = y_pred[i]
        cv2.rectangle(
            image,
            (int(pred_box[0]), int(pred_box[1])),
            (int(pred_box[0] + pred_box[2]), int(pred_box[1] + pred_box[3])),
            (255, 0, 0),
            2,
        )

        output_path = os.path.join(output_dir, f"prediction_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# Save predictions instead of plotting
save_predictions(
    X_valid[:6], y_valid[:6], y_pred[:6], valid_images_dir, valid_annotations
)
