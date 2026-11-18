# Handwritten Digit Recognition: web interface

A browser interface for your MNIST digit project. It loads the models saved by your notebooks
(`02_MLP`, `03_CNN`, `04_SVM`, `05_kNN`) and runs them on drawn digits, uploaded images and CSV rows.
Nothing is mocked: every prediction on screen comes from your saved model files.

## What the interface does

| Page | What it is for |
| --- | --- |
| **Predict** | Draw a digit on the slate, upload an image, or pick an image from your `test.csv`. Every loaded model predicts it; you see each model's answer, confidence, latency, and a per-digit probability chart. |
| **Batch** | Run the models on up to 3,000 rows of the bundled `test.csv` or an uploaded CSV. Shows agreement between models, predicted-digit counts, a gallery, a filterable table, and CSV export. If the file has a `label` column (like `train.csv`), accuracy is computed too. |
| **Performance** | Your notebook 08 comparison (6 algorithms, 5 metrics, sortable). After running `python -m app.ml.evaluate` it also shows per-digit precision/recall/F1 and confusion matrices. |
| **Models** | Loaded status, file, size and hyperparameters read from the actual model objects (CNN layer table included). |
| **Dataset** | Data format, the preprocessing pipeline from your notebooks, live statistics and an average image from `test.csv`. |

## Project layout

```
digit-recognition-ui/
  .git/
  .gitattributes
  .gitignore
  app/
    app.py             Flask server and JSON API
    ml/                configuration, preprocessing, training and evaluation
    static/            index.html, css/styles.css, js/*.js
  notebooks/           put your training notebooks here
  requirements.txt
  data/                put test.csv (and train.csv for evaluate/train) here
  models/              put the four model files here
  results/             model_comparison.csv (your notebook 08 numbers) + metrics.json (generated)
```

## Setup

You need Python 3.10 to 3.12.

```bash
cd digit-recognition-ui
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow is only needed for the CNN. If it will not install on your machine, delete the last line
of `requirements.txt`; the other three models still work and the UI says the CNN is not loaded.

### Put your files in place

1. Copy the four files your notebooks created into `models/`:
   `mlp_digit_model.pkl`, `svm_digit_model.pkl`, `knn_digit_model.pkl`, `cnn_digit_model.keras`
   (the app also looks in a `notebooks/` folder next to it, where the notebooks save them by default).
2. Copy `test.csv` into `data/`. Copy `train.csv` there too if you want the confusion matrices.

If you no longer have the model files, `python -m app.ml.train` re-creates them from `data/train.csv` with the same
hyperparameters, split and preprocessing as the notebooks (the SVM takes a few minutes).

## Run

```bash
python -m app.app
```

Open http://127.0.0.1:5000. The terminal lists which models loaded and why any did not.

Optional, once `train.csv` is in `data/`:

```bash
python -m app.ml.evaluate      # writes results/metrics.json; reload the Performance page afterwards
```

## Settings (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` / `HOST` | `5000` / `127.0.0.1` | Server address. Use `HOST=0.0.0.0` to reach it from another device. |
| `DIGITS_DATA_DIR` | `data/` | Folder with `test.csv` / `train.csv` |
| `DIGITS_MODEL_DIR` | `models/` | Folder with the model files |
| `DIGITS_RESULTS_DIR` | `results/` | Folder with `model_comparison.csv` / `metrics.json` |
| `DIGITS_MAX_BATCH_ROWS` | `3000` | Rows per batch run (SVM and kNN are slow on large batches) |

## How inputs reach the models

* **Dataset rows and uploaded CSVs** are used as-is: pixels are divided by 255.0, as in every notebook. The CNN input is
  reshaped to 28 x 28 x 1. Rows with missing values are skipped and reported; wrong column counts or values outside 0 to 255 are rejected with a message.
* **Drawn and uploaded pictures** are first converted to look like the MNIST training images: white digit on black,
  scaled into a 20 x 20 box and centred by centre of mass in a 28 x 28 frame. The "28 x 28 input" thumbnail on the Predict page
  shows exactly what the models received. Background colour is auto-detected for uploads and can be overridden.
* The SVM was trained with `SVC(probability=False)`, so it returns a label only. The UI shows that instead of inventing a confidence.

## What was and was not changed

Your notebooks, datasets, hyperparameters and preprocessing are untouched. Added: the web app, `ml/train.py` (a copy of
the notebook training code, only used if model files are missing), `ml/evaluate.py`, and `results/model_comparison.csv`,
seeded with the exact numbers printed by notebook 08 (it is the file that notebook writes; yours replaces it).

Random Forest and Logistic Regression appear in notebook 08's table but no notebook or model file for them was provided,
so they are shown as recorded results only and cannot be used for prediction.

## API (for reference)

`GET /api/status`, `/api/models`, `/api/results`, `/api/dataset`, `/api/samples?offset=&limit=` and
`POST /api/predict/image`, `/api/predict/pixels`, `/api/batch`. Errors return `{"error": "..."}` with a suitable status code.

## Troubleshooting

* **A model shows "Not loaded"**: the Models page prints the reason (file missing, TensorFlow missing, or the file could not be read).
* **scikit-learn version warnings / load errors**: `.pkl` files should be opened with the scikit-learn version that saved them. Run `pip install scikit-learn==<version you used>`.
* **Batch is slow**: lower the row count, or untick SVM and kNN.
* **Port in use**: `PORT=5050 python -m app.app`.
