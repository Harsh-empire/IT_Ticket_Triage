import os
import time

import joblib
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common_metrics import calculate_metrics, save_confusion_matrix, save_results
from project_utils import load_data, project_path

MAX_LENGTH = 256
RANDOM_STATE = 42
EPOCHS = int(os.environ.get("TICKET_TRAIN_EPOCHS", "2"))
TRAIN_BATCH_SIZE = int(os.environ.get("TICKET_TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("TICKET_EVAL_BATCH_SIZE", "8"))


def logits_to_probabilities(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def train_and_evaluate(model_name, save_name):
    train_df, test_df = load_data()

    # Both transformer experiments use the same deterministic class ordering.
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["queue"])
    y_test = encoder.transform(test_df["queue"])

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict({
        "text": train_df["text"].astype(str).tolist(),
        "labels": y_train.tolist(),
    })
    test_dataset = Dataset.from_dict({
        "text": test_df["text"].astype(str).tolist(),
        "labels": y_test.tolist(),
    })

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    label_names = [str(item) for item in encoder.classes_]
    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    model_dir = project_path("models", save_name)
    results_dir = project_path("results")
    model_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        logging_dir=str(results_dir / f"{save_name}_logs"),
        learning_rate=2e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
        logging_steps=50,
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
    )

    print(f"Fine-tuning {save_name}...")
    start = time.time()
    trainer.train()
    training_time = time.time() - start

    start = time.time()
    output = trainer.predict(test_dataset)
    prediction_time = time.time() - start
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    probabilities = logits_to_probabilities(logits)
    predictions = probabilities.argmax(axis=1)

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
        len(encoder.classes_),
    )
    metrics.update({
        "model": save_name,
        "base_model": model_name,
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "embedding_time_seconds": None,
    })

    save_results(project_path("results", f"{save_name}_results.json"), metrics)
    save_confusion_matrix(
        project_path("results", f"{save_name}_confusion_matrix.csv"),
        y_test,
        predictions,
    )

    final_dir = model_dir / "final"
    try:
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        joblib.dump(encoder, model_dir / "label_encoder.joblib")
    except Exception as exc:
        print(f"Notice during model serialization for {save_name}: {exc}")

    print(f"\n{save_name} results")
    for key in ("accuracy", "precision", "recall", "f1", "top_3_accuracy"):
        print(f"{key}: {metrics.get(key)}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics
