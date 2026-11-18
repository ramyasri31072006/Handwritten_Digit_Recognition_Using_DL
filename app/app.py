"""Web server for the handwritten digit recognition project.

    python -m app.app            # then open http://127.0.0.1:5000

The server loads the models saved by notebooks 02-05 and serves the static UI in static/.
"""
import base64
import io
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from app.ml import config, data, preprocessing
from app.ml.registry import ModelRegistry, ModelUnavailable

app = Flask(__name__, static_folder=str(config.ROOT / "app" / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

registry = ModelRegistry()
registry.load_all()

GALLERY_ROWS = 48


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


@app.errorhandler(ApiError)
def _api_error(err):
    return jsonify(error=err.message), err.status


@app.errorhandler(413)
def _too_large(_):
    return jsonify(error=f"The upload is larger than the {config.MAX_UPLOAD_MB} MB limit."), 413


@app.errorhandler(404)
def _not_found(_):
    if request.path.startswith("/api/"):
        return jsonify(error="Unknown API endpoint."), 404
    return "Page not found.", 404


@app.errorhandler(Exception)
def _unexpected(err):
    if hasattr(err, "code") and isinstance(err.code, int):  # regular HTTP errors
        return jsonify(error=getattr(err, "description", "Request failed.")), err.code
    app.logger.exception("Unhandled error")
    return jsonify(error="Something went wrong on the server. Check the terminal for details."), 500


# ---- helpers ---------------------------------------------------------------------------------
def _choose_models(requested):
    available = registry.keys()
    if not available:
        raise ApiError("No trained models were found. Copy the model files into models/ (see the README).", 503)
    if not requested:
        return available
    unknown = [k for k in requested if k not in config.MODEL_SPECS]
    if unknown:
        raise ApiError(f"Unknown model: {', '.join(unknown)}.")
    missing = [k for k in requested if k not in available]
    if missing:
        info = next(s for s in registry.status() if s["key"] == missing[0])
        raise ApiError(f"Model not available: {info['label']}. {info['error']}", 503)
    return requested


def _decode_image(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ApiError("No image was received.")
    try:
        return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1], validate=False)))
    except (UnidentifiedImageError, ValueError, OSError):
        raise ApiError("That file is not a readable image (use PNG, JPG, GIF, WEBP or BMP).")


def _predict_one(vector, keys):
    pixels = vector.reshape(1, -1)
    results = []
    for key in keys:
        entry = {"model": key, "label": config.MODEL_SPECS[key]["label"]}
        try:
            started = time.perf_counter()
            labels, probs = registry.predict(key, pixels, with_proba=True)
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            entry["prediction"] = int(labels[0])
            if probs is not None:
                entry["probabilities"] = [round(float(p), 6) for p in probs[0]]
                entry["confidence"] = round(float(probs[0][labels[0]]), 6)
            else:
                entry["probabilities"], entry["confidence"] = None, None
        except Exception as exc:
            app.logger.exception("Prediction failed for %s", key)
            entry["error"] = f"Prediction failed: {exc}"
        results.append(entry)

    votes = Counter(r["prediction"] for r in results if "prediction" in r)
    consensus = None
    if votes:
        digit, count = votes.most_common(1)[0]
        consensus = {"digit": digit, "votes": count, "of": sum(votes.values())}
    return {"processed": vector.tolist(), "results": results, "consensus": consensus}


def _read_json():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("Expected a JSON body.")
    return payload


# ---- pages and status ------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(config.ROOT / "app" / "static", "index.html")


@app.get("/api/status")
def status():
    return jsonify(models=registry.status(), test_csv=data.test_csv_path() is not None,
                   max_batch_rows=config.MAX_BATCH_ROWS,
                   metrics_available=(config.RESULTS_DIR / "metrics.json").is_file())


@app.get("/api/models")
def models():
    return jsonify(models=[registry.describe(k) for k in config.MODEL_SPECS])


# ---- results and dataset ---------------------------------------------------------------------
@app.get("/api/results")
def results():
    csv_path = config.RESULTS_DIR / "model_comparison.csv"
    if not csv_path.is_file():
        raise ApiError("results/model_comparison.csv is missing. Run notebook 08_Model_Comparison first.", 404)
    df = pd.read_csv(csv_path)
    rows = []
    for rec in df.to_dict(orient="records"):
        key = config.COMPARISON_NAME_TO_KEY.get(str(rec["Algorithm"]).upper())
        rows.append({"algorithm": rec["Algorithm"], "accuracy": rec["Accuracy"], "precision": rec["Precision"],
                     "recall": rec["Recall"], "f1": rec["F1-Score"], "error_rate": rec["Error Rate"],
                     "key": key, "live": key in registry.keys()})
    detail = None
    metrics_path = config.RESULTS_DIR / "metrics.json"
    if metrics_path.is_file():
        try:
            detail = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            detail = None
    return jsonify(rows=rows, detail=detail)


@app.get("/api/dataset")
def dataset():
    info = {"test_csv": None, "mean_image": None}
    if data.test_csv_path() is not None:
        x = data.load_test_csv()
        info["test_csv"] = {"rows": int(x.shape[0]), "columns": int(x.shape[1]), "min": int(x.min()),
                            "max": int(x.max()), "mean_intensity": round(float(x.mean()), 2),
                            "blank_share": round(float((x.max(axis=1) == 0).mean()), 4)}
        info["mean_image"] = np.rint(x.mean(axis=0)).astype(int).tolist()
    metrics_path = config.RESULTS_DIR / "metrics.json"
    if metrics_path.is_file():
        try:
            saved = json.loads(metrics_path.read_text())
            info["split"], info["class_counts"] = saved.get("split"), saved.get("class_counts")
        except json.JSONDecodeError:
            pass
    return jsonify(info)


@app.get("/api/samples")
def samples():
    try:
        x = data.load_test_csv()
    except FileNotFoundError as exc:
        raise ApiError(str(exc), 404)
    except data.DataError as exc:
        raise ApiError(f"test.csv could not be read: {exc}", 422)
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(60, max(1, int(request.args.get("limit", 24))))
    except ValueError:
        raise ApiError("offset and limit must be whole numbers.")
    chunk = x[offset: offset + limit]
    return jsonify(total=int(len(x)), offset=offset,
                   samples=[{"index": offset + i, "pixels": row.tolist()} for i, row in enumerate(chunk)])


# ---- single predictions ----------------------------------------------------------------------
@app.post("/api/predict/image")
def predict_image():
    payload = _read_json()
    keys = _choose_models(payload.get("models"))
    invert = payload.get("invert")
    if invert not in (None, True, False):
        raise ApiError("invert must be true, false or null.")
    try:
        vector = preprocessing.image_to_mnist_vector(_decode_image(payload.get("image")), invert)
    except ValueError as exc:
        raise ApiError(str(exc), 422)
    return jsonify(_predict_one(vector, keys))


@app.post("/api/predict/pixels")
def predict_pixels():
    payload = _read_json()
    keys = _choose_models(payload.get("models"))
    pixels = payload.get("pixels")
    if not isinstance(pixels, list) or len(pixels) != config.N_PIXELS:
        raise ApiError(f"pixels must be a list of {config.N_PIXELS} numbers.")
    try:
        vector = np.asarray(pixels, dtype=float)
    except (TypeError, ValueError):
        raise ApiError("pixels must contain only numbers.")
    if np.isnan(vector).any() or vector.min() < 0 or vector.max() > 255:
        raise ApiError("Pixel values must be between 0 and 255.")
    return jsonify(_predict_one(np.rint(vector).astype(np.uint8), keys))


# ---- batch predictions -----------------------------------------------------------------------
@app.post("/api/batch")
def batch():
    form = request.form
    keys = _choose_models([k for k in form.get("models", "").split(",") if k])
    source = form.get("source", "bundled")
    cap = config.MAX_BATCH_ROWS
    skipped, truncated = 0, False

    try:
        if source == "upload":
            upload = request.files.get("file")
            if upload is None or not upload.filename:
                raise ApiError("Choose a CSV file to upload.")
            x, y, ids, skipped, truncated = data.parse_upload(upload.stream, cap)
        elif source == "bundled":
            try:
                start = max(0, int(form.get("start", 0)))
                count = min(cap, max(1, int(form.get("count", 500))))
            except ValueError:
                raise ApiError("Start row and row count must be whole numbers.")
            all_x = data.load_test_csv()
            if start >= len(all_x):
                raise ApiError(f"Start row must be below {len(all_x)}.")
            x, y = all_x[start: start + count], None
            ids = np.arange(start + 1, start + 1 + len(x))
        else:
            raise ApiError("Unknown data source.")
    except FileNotFoundError as exc:
        raise ApiError(str(exc), 404)
    except data.DataError as exc:
        raise ApiError(str(exc), 422)

    predictions, timing, distribution, metrics = {}, {}, {}, {}
    for key in keys:
        started = time.perf_counter()
        predictions[key] = registry.predict_batched(key, x)
        timing[key] = round(time.perf_counter() - started, 2)
        distribution[key] = np.bincount(predictions[key], minlength=config.N_CLASSES).tolist()
        if y is not None:
            correct = int((predictions[key] == y).sum())
            metrics[key] = {"correct": correct, "total": int(len(y)), "accuracy": correct / len(y)}

    agreement = None
    if len(keys) > 1:
        stacked = np.stack([predictions[k] for k in keys])
        agreement = float((stacked == stacked[0]).all(axis=0).mean())

    return jsonify(
        n=int(len(x)), skipped=skipped, truncated=truncated, cap=cap, models=keys,
        ids=ids.tolist(), labels=(y.tolist() if y is not None else None),
        predictions={k: v.tolist() for k, v in predictions.items()},
        timing=timing, distribution=distribution, metrics=metrics or None, agreement=agreement,
        gallery=[row.tolist() for row in x[:GALLERY_ROWS]],
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    loaded = ", ".join(registry.keys()) or "none"
    print(f"Models loaded: {loaded}")
    for s in registry.status():
        if not s["available"]:
            print(f"  - {s['label']} unavailable: {s['error']}")
    app.run(host=host, port=port, debug=False, threaded=True)
