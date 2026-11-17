# Intelligent IT Support Ticket Triage — Full VS Code Project

This project compares classical machine-learning baselines, semantic embeddings,
fine-tuned multilingual Transformers, and a hybrid Transformer + semantic retrieval model.

## Dataset selected for this project

**Customer Support Tickets / Multilingual Customer Support Tickets**

- Hugging Face source: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
- Original Kaggle dataset: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
- Main target: `queue` (support-team routing)
- Input text: `subject` + `body`
- Additional available labels can include `priority`, `type`, and `language`.
- Dataset languages: English and German.
- The dataset source labels its license CC BY-NC 4.0. Keep attribution in reports and do not use the dataset for commercial deployment without checking the license conditions.
- The dataset is synthetically generated, so mention that limitation in the report.

## The dataset downloads automatically

You do not need to manually download a CSV in advance.

The project first tries to load the public Hugging Face dataset. If that fails, it attempts
the main multilingual CSV, then falls back to KaggleHub.

The main data file will be stored at:

`data/raw/customer_support_tickets.csv`

After preparing it, you will also see:

`data/cleaned_tickets.csv`  
`data/train.csv`  
`data/test.csv`

The training and testing CSVs are generated once and reused by all models.

## Model comparison

1. TF-IDF + Logistic Regression
2. TF-IDF + XGBoost
3. Sentence-BERT embeddings + XGBoost
4. DistilBERT fine-tuning
5. BERT fine-tuning
6. Proposed hybrid: DistilBERT probabilities + Sentence-BERT retrieval probabilities

The main comparison table contains Accuracy, Precision, Recall, F1, Top-3 Accuracy,
Training Time, Prediction Time, Embedding Time, and Retrieval Hit@3.
The `F1` column means weighted F1. Macro F1 is retained only as an extra diagnostic
inside each result JSON file; it is not a separate column in the main table.

No accuracy or F1 values are guessed. Each result is measured by the scripts.

## Measured results (held-out test set)

Test set: 10,692 tickets, 52 queue classes (80/20 split, `random_state=42`, 42,764 training tickets after removing 8,309 duplicate texts).

| Model | Accuracy | Weighted F1 | Top-3 Accuracy | Training time |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 52.5% | 0.504 | 81.4% | 102 s |
| TF-IDF + XGBoost | 49.7% | 0.458 | 77.7% | 1,083 s |
| Sentence-BERT + XGBoost | 55.3% | 0.531 | 82.9% | 197 s (+120 s embeddings) |
| DistilBERT (1 epoch) | 46.6% | 0.416 | 77.1% | 3,573 s |
| BERT (1 epoch) | 51.1% | 0.472 | 79.9% | 2,440 s |
| **Hybrid DistilBERT + SBERT** | **74.2%** | **0.734** | **93.6%** | DistilBERT training + 96 s embeddings |

Full tables (precision, recall, ROC-AUC, latency, Retrieval Hit@3 = 90.6%) are in `results/comprehensive_model_comparison.csv`.

### How to read these results

- **The Transformers are under-trained.** DistilBERT and BERT were fine-tuned for **1 epoch** on CPU (`max_length=256`), which is why they trail the TF-IDF baselines. Their numbers are not the ceiling for these models.
- **Most of the hybrid's gain comes from semantic retrieval.** It adds a probability distribution from the 3 most similar historical (training) tickets to the DistilBERT output, with fixed weights (0.75 Transformer / 0.25 retrieval) and temperature 0.07. The dataset is synthetic, so retrieval may work better here than it would on real tickets.
- **One run, one split.** There are no repeated runs or confidence intervals. Training times come from a single CPU run and are not a controlled speed comparison.
- **Headline metric:** weighted F1 and accuracy. Macro F1 is stored in each result JSON as an extra diagnostic only (it is higher than weighted F1 for every model here, so it should not be quoted as the main score).

## Folder layout

```text
IT_Ticket_Triage_Complete_Project/
├── data/
│   ├── raw/
│   │   ├── customer_support_tickets.csv
│   │   └── source_info.json
│   ├── cleaned_tickets.csv
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── tf-idf_plus_logistic_regression/
│   ├── tf-idf_plus_xgboost/
│   ├── sbert_xgboost/
│   ├── distilbert/
│   ├── bert/
│   └── hybrid/
├── results/
├── download_dataset.py
├── prepare_data.py
├── train_classical.py
├── train_sbert_xgboost.py
├── transformer_training.py
├── train_transformers.py
├── train_hybrid.py
├── predict_hybrid.py
├── make_comparison.py
├── run_all.py
├── common_metrics.py
├── hybrid_utils.py
├── project_utils.py
└── requirements.txt
```

## Install in VS Code on Windows

Open the extracted project folder in VS Code, then open Terminal.

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```cmd
.venv\Scripts\activate.bat
```

## Recommended run order

Run one script at a time so errors are easier to locate:

```powershell
python prepare_data.py
python train_classical.py
python train_sbert_xgboost.py
python train_transformers.py
python train_hybrid.py
python make_comparison.py
```

Or run the complete experiment in one command:

```powershell
python run_all.py
```

The first run downloads both the dataset and pretrained models, so the computer needs an internet connection. Transformer fine-tuning can take a long time on CPU. A compatible GPU is recommended. You can set fewer epochs for a preliminary test in PowerShell:

```powershell
$env:TICKET_TRAIN_EPOCHS="1"
python train_transformers.py
```

Use the same epoch setting for both Transformer models and record it when reporting results. For final reported experiments, use the chosen settings consistently.

## Try a sample ticket after training

```powershell
python predict_hybrid.py
```

## Results

Outputs appear in `results/`, including:

- `dataset_info.json`
- individual `*_results.json` files
- per-model confusion matrices
- `hybrid_report.txt`
- `model_comparison.csv`

Open `results/model_comparison.csv` in Excel or include the measured table in the project report.

## Methodological note

The hybrid model reuses the fine-tuned DistilBERT baseline and adds a probability distribution
derived from the nearest historical tickets. This makes it possible to compare the Transformer
alone with the fused hybrid output on the same held-out test split. The retrieval component is
also evaluated independently using Retrieval Hit@3.

## Web demo (`server.py`)

```powershell
python server.py
```

Then open http://localhost:8000 (change the port with the `PORT` environment variable). The page shows the comparison charts and lets you paste a ticket to see a suggested queue.

**Important:** live predictions in the demo are computed with the saved **TF-IDF + Logistic Regression** model (and TF-IDF + XGBoost when available). The per-model scores shown for BERT, DistilBERT, Sentence-BERT + XGBoost and Hybrid in the demo are approximations derived from the Logistic Regression probabilities, not outputs of those trained models. For real hybrid inference, use `python predict_hybrid.py` after training. The accuracy numbers above come from `train_hybrid.py` and the other training scripts, not from the web demo.

## Model weights are not stored in this repository

The small classical models (`.joblib`) are included. The large Transformer and Sentence-BERT weights (`*.safetensors`) and the hybrid embeddings are excluded because of GitHub's file-size limits (see `.gitignore`). Regenerate them with `python run_all.py` (or `train_transformers.py` then `train_hybrid.py`) before using `predict_hybrid.py`. The dataset is downloaded automatically by `download_dataset.py`.

## Data license and attribution

Dataset: Tobi-Bueck / Tobias Bueck, *Customer Support Tickets* (Hugging Face and Kaggle), licensed CC BY-NC 4.0 and synthetically generated. This project is for learning and portfolio use, not commercial deployment. See `DATASET_SOURCE.txt`.
