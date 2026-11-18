"""Optional: re-create the saved models exactly as notebooks 02-05 do.

Only needed when mlp_digit_model.pkl, svm_digit_model.pkl, knn_digit_model.pkl or
cnn_digit_model.keras are not available. If you already ran the notebooks, copy those files
into models/ instead and skip this script.

    python -m app.ml.train                 # train every model that is missing
    python -m app.ml.train --models svm    # only the SVM
    python -m app.ml.train --force         # retrain and overwrite

Split and preprocessing are identical to the notebooks: pixels / 255.0, then
train_test_split(test_size=0.2, random_state=42, stratify=y).
"""
import argparse
import sys
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, preprocessing


def load_split():
    path = config.find("train.csv", config.DATA_DIRS)
    if path is None:
        sys.exit("train.csv not found. Put it in the data/ folder (or set DIGITS_DATA_DIR).")
    df = pd.read_csv(path)
    y = df["label"].to_numpy()
    x = df.drop("label", axis=1).to_numpy(dtype="uint8")
    names = [c for c in df.columns if c != "label"]
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) + [names]


def train_mlp(x_train, y_train, names):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                          max_iter=50, random_state=42)
    model.fit(preprocessing.for_sklearn(x_train, names), y_train)
    return model


def train_svm(x_train, y_train, names):
    from sklearn.svm import SVC
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    model.fit(preprocessing.for_sklearn(x_train, names), y_train)
    return model


def train_knn(x_train, y_train, names):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    model.fit(preprocessing.for_sklearn(x_train, names), y_train)
    return model


def train_cnn(x_train, y_train, x_test, y_test):
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(preprocessing.for_keras(x_train), y_train, epochs=10, batch_size=128,
              validation_data=(preprocessing.for_keras(x_test), y_test), verbose=1)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(config.MODEL_SPECS), default=list(config.MODEL_SPECS))
    parser.add_argument("--force", action="store_true", help="retrain even if the model file exists")
    args = parser.parse_args()

    target_dir = config.MODEL_DIRS[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test, names = load_split()

    for key in args.models:
        spec = config.MODEL_SPECS[key]
        if config.find(spec["file"], config.MODEL_DIRS) and not args.force:
            print(f"[skip] {spec['file']} already exists (use --force to retrain)")
            continue
        print(f"[train] {spec['label']} ...")
        started = time.time()
        if key == "cnn":
            model = train_cnn(x_train, y_train, x_test, y_test)
            model.save(target_dir / spec["file"])
        else:
            model = {"mlp": train_mlp, "svm": train_svm, "knn": train_knn}[key](x_train, y_train, names)
            joblib.dump(model, target_dir / spec["file"])
        print(f"[done]  {spec['file']} saved in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
