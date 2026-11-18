"""Loads the models saved by notebooks 02-05 and runs predictions with them.

Nothing here changes a model: `predict` and `predict_proba` (sklearn) or `predict` (Keras)
are called exactly as in the notebooks. A model that is missing or fails to load is reported
as unavailable and the others keep working.
"""
import threading

import joblib
import numpy as np

from . import config, preprocessing


class ModelUnavailable(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self):
        self._models, self._paths, self._errors = {}, {}, {}
        self._locks = {key: threading.Lock() for key in config.MODEL_SPECS}

    # ---- loading -------------------------------------------------------------------------
    def load_all(self):
        for key, spec in config.MODEL_SPECS.items():
            self._load(key, spec)

    def _load(self, key, spec):
        path = config.find(spec["file"], config.MODEL_DIRS)
        if path is None:
            searched = ", ".join(str(d) for d in config.MODEL_DIRS[:2])
            self._errors[key] = f"{spec['file']} not found (looked in {searched})."
            return
        try:
            if spec["kind"] == "sklearn":
                model = joblib.load(path)
            else:
                try:
                    from tensorflow import keras
                except ImportError:
                    self._errors[key] = "TensorFlow is not installed (pip install tensorflow)."
                    return
                model = keras.models.load_model(path)
            self._models[key], self._paths[key] = model, path
        except Exception as exc:  # corrupt file, version mismatch, ...
            self._errors[key] = f"Could not load {path.name}: {exc}"

    # ---- queries -------------------------------------------------------------------------
    def keys(self):
        return [k for k in config.MODEL_SPECS if k in self._models]

    def status(self):
        out = []
        for key, spec in config.MODEL_SPECS.items():
            out.append({
                "key": key, "label": spec["label"], "available": key in self._models,
                "error": self._errors.get(key),
                "file": self._paths[key].name if key in self._paths else spec["file"],
            })
        return out

    def _get(self, key):
        if key not in self._models:
            raise ModelUnavailable(self._errors.get(key, f"Unknown model '{key}'."))
        return self._models[key]

    # ---- inference -----------------------------------------------------------------------
    def predict(self, key, pixels, with_proba=False):
        """pixels: (n, 784) uint8. Returns (labels int (n,), probabilities (n, 10) or None)."""
        model = self._get(key)
        with self._locks[key]:
            if config.MODEL_SPECS[key]["kind"] == "keras":
                probs = np.asarray(model.predict(preprocessing.for_keras(pixels), verbose=0, batch_size=256))
                return probs.argmax(axis=1).astype(int), (probs if with_proba else None)

            frame = preprocessing.for_sklearn(pixels, getattr(model, "feature_names_in_", None))
            labels = np.asarray(model.predict(frame)).astype(int)
            probs = None
            if with_proba and hasattr(model, "predict_proba"):  # SVC(probability=False) has none
                raw = model.predict_proba(frame)
                probs = np.zeros((len(frame), config.N_CLASSES))
                probs[:, np.asarray(model.classes_).astype(int)] = raw
            return labels, probs

    def predict_batched(self, key, pixels, chunk=500):
        parts = [self.predict(key, pixels[i: i + chunk])[0] for i in range(0, len(pixels), chunk)]
        return np.concatenate(parts)

    # ---- metadata for the Models page ---------------------------------------------------
    def describe(self, key):
        spec = config.MODEL_SPECS[key]
        info = {"key": key, "label": spec["label"], "summary": spec["summary"], "notebook": spec["notebook"],
                "available": key in self._models, "error": self._errors.get(key),
                "file": spec["file"], "size_mb": None, "params": {}, "layers": None}
        if key not in self._models:
            return info
        model, path = self._models[key], self._paths[key]
        info["file"] = path.name
        info["size_mb"] = round(path.stat().st_size / 1e6, 1)
        try:
            if spec["kind"] == "keras":
                info["params"] = {"Optimizer": "adam", "Loss": "sparse_categorical_crossentropy",
                                  "Total parameters": f"{model.count_params():,}"}
                info["layers"] = [_keras_layer(layer) for layer in model.layers]
            else:
                info["params"] = _sklearn_params(key, model)
        except Exception as exc:
            info["params"] = {"Note": f"Could not read parameters: {exc}"}
        return info


def _keras_layer(layer):
    try:
        shape = "x".join(str(d) for d in layer.output.shape[1:])
    except Exception:
        shape = "n/a"
    return {"type": layer.__class__.__name__, "output": shape, "params": int(layer.count_params())}


def _sklearn_params(key, model):
    p = model.get_params()
    if key == "mlp":
        hidden = list(model.hidden_layer_sizes) if hasattr(model.hidden_layer_sizes, "__iter__") \
            else [model.hidden_layer_sizes]
        return {"Layer sizes": " to ".join(str(n) for n in [model.n_features_in_, *hidden, model.n_outputs_]),
                "Activation": p["activation"], "Solver": p["solver"], "Max iterations": p["max_iter"],
                "Iterations run": getattr(model, "n_iter_", "n/a"), "Random state": p["random_state"]}
    if key == "svm":
        return {"Kernel": p["kernel"], "C": p["C"], "Gamma": p["gamma"], "Random state": p["random_state"],
                "Support vectors": f"{int(np.sum(model.n_support_)):,}"}
    if key == "knn":
        return {"Neighbours (k)": p["n_neighbors"], "Weights": p["weights"],
                "Training images stored": f"{model.n_samples_fit_:,}"}
    return {k: v for k, v in p.items()}
