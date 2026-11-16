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
│   ├── tf_idf_plus_logistic_regression/
│   ├── tf_idf_plus_xgboost/
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
