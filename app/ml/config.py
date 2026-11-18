"""Paths, model specs and limits. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _dirs(env_name, *defaults):
    override = [Path(os.environ[env_name])] if os.environ.get(env_name) else []
    return override + list(defaults)


# Search order: env override, this project, then a sibling folder (the notebooks use ../data).
DATA_DIRS = _dirs("DIGITS_DATA_DIR", ROOT / "data", ROOT.parent / "data")
MODEL_DIRS = _dirs("DIGITS_MODEL_DIR", ROOT / "models", ROOT / "notebooks", ROOT.parent / "notebooks")
RESULTS_DIR = Path(os.environ.get("DIGITS_RESULTS_DIR", ROOT / "results"))

IMG_SIDE = 28
N_PIXELS = IMG_SIDE * IMG_SIDE
N_CLASSES = 10

MAX_BATCH_ROWS = int(os.environ.get("DIGITS_MAX_BATCH_ROWS", 3000))
MAX_UPLOAD_MB = int(os.environ.get("DIGITS_MAX_UPLOAD_MB", 100))


def find(filename, dirs):
    """Return the first existing file called `filename` inside `dirs`, or None."""
    for d in dirs:
        candidate = Path(d) / filename
        if candidate.is_file():
            return candidate
    return None


# File names match the ones written by notebooks 02-05.
MODEL_SPECS = {
    "mlp": {
        "label": "MLP", "file": "mlp_digit_model.pkl", "kind": "sklearn", "notebook": "02_MLP",
        "summary": "Fully connected network (scikit-learn MLPClassifier): 784 inputs, hidden layers of 128 and "
                   "64 neurons with ReLU, 10 outputs, trained with Adam.",
    },
    "cnn": {
        "label": "CNN", "file": "cnn_digit_model.keras", "kind": "keras", "notebook": "03_CNN",
        "summary": "Keras convolutional network: two Conv2D and MaxPooling blocks (32 and 64 filters), "
                   "a 128-neuron dense layer and a 10-way softmax output.",
    },
    "svm": {
        "label": "SVM", "file": "svm_digit_model.pkl", "kind": "sklearn", "notebook": "04_SVM",
        "summary": "Support vector classifier (scikit-learn SVC) with an RBF kernel, C=10 and gamma='scale'.",
    },
    "knn": {
        "label": "kNN", "file": "knn_digit_model.pkl", "kind": "sklearn", "notebook": "05_kNN",
        "summary": "k-nearest neighbours (scikit-learn KNeighborsClassifier) using the 5 closest training "
                   "images, weighted by distance.",
    },
}

# Names used in results/model_comparison.csv (notebook 08) mapped to live model keys.
COMPARISON_NAME_TO_KEY = {"MLP": "mlp", "CNN": "cnn", "SVM": "svm", "KNN": "knn"}
