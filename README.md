# Handwritten Digit Recognition Using Deep Learning

Train and compare six classifiers for recognizing handwritten digits **0–9** from 28 × 28 MNIST images. A local Flask dashboard lets you draw a digit, upload an image, or select a dataset sample and compare predictions from the loaded models.

The **convolutional neural network (CNN)** achieved the highest recorded accuracy at **98.31%** in the notebook comparison.

[Dashboard](#dashboard) · [Quick Start](#quick-start) · [Dataset](#dataset) · [Results](#results) · [Notebooks](#notebooks)

<a name="dashboard"></a>

## 🖊️ Interactive Dashboard

| Page | What you can do |
| --- | --- |
| **Predict** | Draw a digit, upload an image, or select a test sample. Compare predicted labels, inference times, and class probabilities where available. |
| **Batch** | Classify up to 3,000 CSV rows per run by default. Inspect model agreement, filter and export predictions, and see accuracy when the CSV includes labels. |
| **Performance** | Compare the six recorded model results. Generate per-digit metrics and confusion matrices with the evaluation script. |
| **Models** | Check which models loaded and inspect their files and configuration. |
| **Dataset** | Explore sample images, pixel statistics, and the average image from `test.csv`. |

<a name="quick-start"></a>

## 🚀 Quick Start

Use **Python 3.11** for the setup below. Open a terminal in the extracted project folder, where `requirements.txt` is located.

```bash
python -m venv .venv
```

<details>
<summary><strong>Activation commands for Windows, macOS, and Linux</strong></summary>

Activate the environment using the command for your shell.

| Shell | Command |
| --- | --- |
| Windows PowerShell | `& .\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS or Linux | `source .venv/bin/activate` |

</details>

```bash
python -m pip install -r requirements.txt
```

<details open>
<summary><strong>Place the dataset and model files</strong></summary>

The project ZIP already includes the CSVs and trained models. The Git ignore rules exclude those files, so for a fresh clone, copy them from the ZIP into these locations before running a demo.

| Path | Used for |
| --- | --- |
| `data/train.csv` | Labeled data for training and evaluation |
| `data/test.csv` | Unlabeled samples for the dashboard |
| `models/cnn_digit_model.keras` | CNN predictions |
| `models/mlp_digit_model.pkl` | MLP predictions |
| `models/svm_digit_model.pkl` | SVM predictions |
| `models/knn_digit_model.pkl` | kNN predictions |

</details>

Start the server and open [http://127.0.0.1:5000](http://127.0.0.1:5000).

```bash
python -m app.app
```

Try drawing a single digit in **Predict**. The 28 × 28 preview shows the input passed to the models. Check **Models** for load errors. If a saved scikit-learn model fails to load because of a version mismatch, use the scikit-learn version that saved it or retrain it in the current environment.

<a name="dataset"></a>

## 📂 Dataset and Preprocessing

**Dataset source** · [Handwritten Digit Recognition on Kaggle by Bhavik Jikadara](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition)

Each MNIST image has **784 grayscale pixels** with values from **0 to 255**. The training CSV includes a `label` column. The test CSV provides unlabeled examples for prediction. The training and evaluation scripts create a stratified **80/20 split** from the labeled training CSV with `random_state=42`.

CSV pixels are scaled to `[0, 1]`. Drawings and uploaded images are converted to a light digit on a dark background, scaled into a 20 × 20 box, and centered in a 28 × 28 frame. The CNN receives a 28 × 28 × 1 tensor per image. The scikit-learn models receive 784 flattened pixel features.

<a name="results"></a>

## 📊 Model Comparison

These percentages come from the project's [model comparison CSV](results/model_comparison.csv). The dashboard uses the CNN, MLP, SVM, and kNN for live predictions. Random Forest and Logistic Regression are included in the notebook comparison. The saved SVM returns a predicted label.

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| CNN | **98.31%** | **98.35%** | **98.31%** | **98.31%** |
| SVM | 97.92% | 97.92% | 97.92% | 97.92% |
| MLP | 97.43% | 97.43% | 97.43% | 97.43% |
| kNN | 96.77% | 96.82% | 96.77% | 96.77% |
| Random Forest | 96.39% | 96.39% | 96.39% | 96.39% |
| Logistic Regression | 91.29% | 91.26% | 91.29% | 91.26% |

<details>
<summary><strong>Train models and generate confusion matrices</strong></summary>

### Train and evaluate

With `data/train.csv` in place, run these commands from the project root.

```bash
python -m app.ml.train
python -m app.ml.evaluate
```

The training script generates missing model files for the four dashboard classifiers. The evaluation script writes `results/metrics.json`. Refresh **Performance** to view per-digit scores and confusion matrices.

</details>

<a name="notebooks"></a>

## 📓 Explore the Notebooks

<details>
<summary><strong>Open the notebook index</strong></summary>

| Notebook | Topic |
| --- | --- |
| [01 · Data preprocessing](notebooks/01_data_preprocessing.ipynb) | Explore the data, normalize pixels, and split the labeled examples |
| [02 · MLP](notebooks/02_MLP.ipynb) | Multilayer perceptron |
| [03 · CNN](notebooks/03_CNN.ipynb) | Convolutional neural network |
| [04 · SVM](notebooks/04_SVM.ipynb) | Support vector machine |
| [05 · kNN](notebooks/05_kNN.ipynb) | k-nearest neighbors |
| [06 · Random Forest](notebooks/06_Random_Forest.ipynb) | Random forest classifier |
| [07 · Logistic Regression](notebooks/07_Logistic_Regression.ipynb) | Logistic regression classifier |
| [08 · Model Comparison](notebooks/08_Model_Comparison.ipynb) | Compare recorded metrics across all six models |

</details>

## 🗂️ Project Structure

| Path | Contents |
| --- | --- |
| [`app/app.py`](app/app.py) | Flask server and prediction API |
| [`app/ml/`](app/ml/) | Model loading, preprocessing, training, and evaluation |
| [`app/static/`](app/static/) | Dashboard built with HTML, CSS, and JavaScript |
| [`notebooks/`](notebooks/) | Jupyter notebooks for the full model comparison |
| `data/` and `models/` | Local dataset CSVs and trained model files |
| [`results/`](results/) | Recorded comparison CSV and locally generated evaluation metrics |
| [`requirements.txt`](requirements.txt) | Python dependencies, including TensorFlow/Keras and scikit-learn |
