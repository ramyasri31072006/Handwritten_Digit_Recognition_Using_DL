"""CSV loading and validation (Kaggle layout: optional `label` column plus pixel0 ... pixel783)."""
import threading

import numpy as np
import pandas as pd

from . import config


class DataError(ValueError):
    """Raised when a CSV cannot be turned into valid digit images."""


def frame_to_pixels(df):
    """Validate a DataFrame. Returns (pixels uint8 (n,784), labels int (n,) or None, row_ids, skipped)."""
    df = df.copy()
    labels = None
    label_col = next((c for c in df.columns if str(c).strip().lower() == "label"), None)
    if label_col is not None:
        labels = pd.to_numeric(df.pop(label_col), errors="coerce")

    if df.shape[1] != config.N_PIXELS:
        raise DataError(
            f"Expected {config.N_PIXELS} pixel columns (pixel0 to pixel783), found {df.shape[1]}. "
            "Use the same layout as train.csv or test.csv."
        )

    pixels = df.apply(pd.to_numeric, errors="coerce")
    bad = pixels.isna().any(axis=1)
    if labels is not None:
        bad |= ~labels.isin(range(config.N_CLASSES))
    bad = bad.to_numpy()
    skipped = int(bad.sum())
    row_ids = np.flatnonzero(~bad) + 1  # 1-based, matches the Kaggle ImageId
    pixels = pixels[~bad]
    if pixels.empty:
        raise DataError("No valid rows: every row has missing or non-numeric values.")

    arr = pixels.to_numpy(dtype=np.float64)
    if arr.min() < 0 or arr.max() > 255:
        raise DataError("Pixel values must be between 0 and 255.")
    if arr.max() <= 1.0 and not np.all(arr == np.floor(arr)):
        raise DataError("Pixel values look normalised (0 to 1). Upload the raw 0 to 255 values.")

    y = labels[~bad].to_numpy(dtype=int) if labels is not None else None
    return np.rint(arr).astype(np.uint8), y, row_ids, skipped


def parse_upload(stream, max_rows):
    """Read at most `max_rows` rows from an uploaded CSV stream."""
    try:
        df = pd.read_csv(stream, nrows=max_rows + 1)
    except pd.errors.EmptyDataError:
        raise DataError("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"The file is not a valid CSV: {exc}")
    if df.empty:
        raise DataError("The file has a header but no rows.")
    truncated = len(df) > max_rows
    x, y, ids, skipped = frame_to_pixels(df.iloc[:max_rows])
    return x, y, ids, skipped, truncated


_lock = threading.Lock()
_test_cache = {}


def test_csv_path():
    return config.find("test.csv", config.DATA_DIRS)


def load_test_csv():
    """Load test.csv once and keep it in memory as uint8 (about 22 MB)."""
    with _lock:
        if "x" not in _test_cache:
            path = test_csv_path()
            if path is None:
                raise FileNotFoundError("test.csv was not found. Put it in the data/ folder.")
            x, _, _, _ = frame_to_pixels(pd.read_csv(path))
            _test_cache["x"] = x
        return _test_cache["x"]
