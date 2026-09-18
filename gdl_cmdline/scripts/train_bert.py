"""
Text Classifiers for Malware Detection

Trains BERT, a Model2Vec static classifier, and/or a bidirectional LSTM on
raw command-line strings. Dataset splitting, metrics, plotting, and shared
utilities are imported from train_gnn.py so all baselines use identical data.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer

from gdl_cmdline.scripts.train_gnn import (
    FocalLoss,
    StaticEmbedder,
    _prepare_combined_csv,
    _write_fp_fn,
    compute_metrics,
    get_class_weights,
    plot_confusion_matrix,
    plot_training_curves,
    print_metrics,
    set_seed,
    split_and_save_data,
)


class CommandLineTextDataset(Dataset):
    """Dataset containing raw command-line strings and binary labels."""

    def __init__(self, data_csv: str):
        super().__init__()
        print(f"[Dataset] Loading data from {data_csv}...")
        df = pd.read_csv(data_csv)
        self.texts = [str(command) for command in df["command_line"].values]
        self.labels = df["label"].astype(int).tolist()

        counts = pd.Series(self.labels).value_counts().to_dict()
        print(
            f"[Dataset] Total samples: {len(self.texts)} "
            f"(clean={counts.get(0, 0)}, malware={counts.get(1, 0)})"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.texts[index], self.labels[index]


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


class BERTClassifier(nn.Module):
    """Frozen BERT encoder followed by a trainable MLP classifier."""

    name = "BERTClassifier"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.bert(**encoded)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


class LSTMClassifier(nn.Module):
    """Two-layer bidirectional LSTM classifier over tokenized command lines.

    The tokenizer vocabulary is shared with the configured BERT tokenizer,
    but the token embeddings and the LSTM are trained from scratch.
    """

    name = "LSTMClassifier"

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        num_classes: int = 2,
        embedding_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.padding_idx = self.tokenizer.pad_token_id

        self.embedding = nn.Embedding(
            num_embeddings=len(self.tokenizer),
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        lengths = encoded["attention_mask"].sum(dim=1).cpu()

        embeddings = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # Last layer: forward state followed by backward state.
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(representation)


class StaticModelClassifier(nn.Module):
    """Frozen Model2Vec embeddings followed by a trainable MLP classifier."""

    name = "StaticClassifier"

    def __init__(
        self,
        embedder_model: str = "minishlab/potion-base-2M",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedder = StaticEmbedder(embedder_model)
        hidden_size = self.embedder.embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        with torch.no_grad():
            embeddings = self.embedder.embed_texts(texts).to(device)
        return self.classifier(embeddings)


MODEL_CHOICES = ["bert", "lstm", "static", "all"]


def _device_type(device: Union[str, torch.device]) -> str:
    return device.type if isinstance(device, torch.device) else str(device).split(":")[0]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for texts, labels in tqdm(loader, desc="Training", leave=False):
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=_device_type(device), enabled=use_amp):
            logits = model(texts)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.device],
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions = []
    all_labels = []

    for texts, labels in tqdm(loader, desc="Evaluating", leave=False):
        labels = labels.to(device)
        with torch.amp.autocast(device_type=_device_type(device), enabled=use_amp):
            logits = model(texts)
        all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return np.asarray(all_predictions), np.asarray(all_labels)


def run_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: Union[str, torch.device],
    model_key: str = "bert",
    loss_type: str = "standard",
    class_weights: Optional[torch.Tensor] = None,
    num_epochs: int = 25,
    learning_rate: float = 2e-5,
    lstm_learning_rate: float = 1e-3,
    results_dir: str = "./results",
    early_stopping_patience: int = 5,
    bert_model_name: str = "bert-base-uncased",
    max_length: int = 512,
    lstm_embedding_size: int = 128,
    embedder_model: str = "minishlab/potion-base-2M",
    test_csv: Optional[str] = None,
    experiment_params: Optional[Dict] = None,
    heldout_test_loader: Optional[DataLoader] = None,
    heldout_test_csv: Optional[str] = None,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)

    if model_key == "static":
        model = StaticModelClassifier(embedder_model=embedder_model).to(device)
    elif model_key == "lstm":
        model = LSTMClassifier(
            tokenizer_name=bert_model_name,
            embedding_size=lstm_embedding_size,
            hidden_size=64,
            num_layers=2,
            max_length=max_length,
        ).to(device)
    else:
        model = BERTClassifier(
            model_name=bert_model_name, max_length=max_length
        ).to(device)

    model_name = model.name
    print(f"\n{'#' * 60}\n# Training {model_name} with {loss_type} loss\n{'#' * 60}")

    if loss_type == "weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "smoothed":
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    effective_lr = lstm_learning_rate if model_key == "lstm" else learning_rate
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=effective_lr, weight_decay=1e-4)
    use_amp = _device_type(device) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_losses = []
    validation_history = []
    best_f1 = -1.0
    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    with trange(1, num_epochs + 1, desc="Training") as progress:
        for epoch in progress:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            train_losses.append(train_loss)
            y_pred, y_true = evaluate(model, test_loader, device, use_amp)
            metrics = compute_metrics(y_true, y_pred)
            validation_history.append(metrics)
            scheduler.step(metrics["f1_macro"])

            if metrics["f1_malware"] > best_f1:
                best_f1 = metrics["f1_malware"]
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            progress.set_postfix(
                loss=round(train_loss, 3),
                f1_macro=round(metrics["f1_macro"], 3),
                f1_malware=round(metrics["f1_malware"], 3),
                best_f1=round(best_f1, 3),
                no_improv=epochs_without_improvement,
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[Early Stopping] Stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_model_state)
    model.to(device)
    y_pred, y_true = evaluate(model, test_loader, device, use_amp)
    final_metrics = compute_metrics(y_true, y_pred)
    experiment_name = f"{model_name}_{loss_type}"

    print_metrics(final_metrics, f"Final Results: {experiment_name}")
    print(classification_report(y_true, y_pred, target_names=["Clean", "Malware"]))

    if test_csv is not None:
        _write_fp_fn(
            y_true, y_pred, test_csv,
            os.path.join(results_dir, f"{experiment_name}_fp_fn.txt"),
        )

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(results_dir, f"{experiment_name}_confusion.png"),
        title=f"Confusion Matrix: {experiment_name}",
        metrics=final_metrics,
        experiment_params=experiment_params,
    )
    plot_training_curves(
        train_losses, validation_history,
        save_path=os.path.join(results_dir, f"{experiment_name}_curves.png"),
        title=f"Training Curves: {experiment_name}",
    )
    torch.save(best_model_state, os.path.join(results_dir, f"{experiment_name}_model.pt"))

    heldout_metrics = None
    if heldout_test_loader is not None:
        heldout_pred, heldout_true = evaluate(model, heldout_test_loader, device, use_amp)
        heldout_metrics = compute_metrics(heldout_true, heldout_pred)
        print_metrics(heldout_metrics, f"Held-out Test Results: {experiment_name}")
        print(classification_report(
            heldout_true, heldout_pred, target_names=["Clean", "Malware"]
        ))
        if heldout_test_csv is not None:
            _write_fp_fn(
                heldout_true, heldout_pred, heldout_test_csv,
                os.path.join(results_dir, f"{experiment_name}_heldout_test_fp_fn.txt"),
            )
        plot_confusion_matrix(
            heldout_true, heldout_pred,
            save_path=os.path.join(
                results_dir, f"{experiment_name}_heldout_test_confusion.png"
            ),
            title=f"Held-out Test Confusion Matrix: {experiment_name}",
            metrics=heldout_metrics,
            experiment_params=experiment_params,
        )

    return {
        "model_name": model_name,
        "loss_type": loss_type,
        "train_losses": train_losses,
        "val_metrics_history": validation_history,
        "final_metrics": final_metrics,
        "heldout_test_metrics": heldout_metrics,
    }


def run_full_experiment_suite(
    malware_csv: str,
    clean_csv: str,
    seed: int = 42,
    results_dir: str = "./results_text",
    num_epochs: int = 25,
    batch_size: int = 16,
    val_batch_size: int = 64,
    test_size: float = 0.3,
    loss_type: str = "all",
    model_types: Optional[List[str]] = None,
    bert_model_name: str = "bert-base-uncased",
    max_length: int = 512,
    learning_rate: float = 2e-5,
    lstm_learning_rate: float = 1e-3,
    lstm_embedding_size: int = 128,
    cache_dir: str = "./graph_cache",
    embedder_model: str = "minishlab/potion-base-2M",
    test_malware_csv: Optional[str] = None,
    test_clean_csv: Optional[str] = None,
    timestamp: Optional[str] = None,
):
    model_types = model_types or ["bert"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = "+".join(model_types)
    results_dir = os.path.join(
        results_dir, f"{timestamp}/{test_size}/seed{seed}/{model_label}"
    )
    os.makedirs(results_dir, exist_ok=True)

    train_csv, test_csv = split_and_save_data(
        malware_csv, clean_csv, seed=seed, test_size=test_size, cache_dir=cache_dir
    )
    heldout_test_csv = None
    if test_malware_csv and test_clean_csv:
        heldout_test_csv = _prepare_combined_csv(
            test_malware_csv, test_clean_csv, cache_dir
        )

    train_dataset = CommandLineTextDataset(train_csv)
    validation_dataset = CommandLineTextDataset(test_csv)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn
    )

    heldout_loader = None
    if heldout_test_csv is not None:
        heldout_dataset = CommandLineTextDataset(heldout_test_csv)
        heldout_loader = DataLoader(
            heldout_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn
        )

    class_weights = get_class_weights(train_dataset.labels, device)
    selected_models = ["bert", "lstm", "static"] if "all" in model_types else list(dict.fromkeys(model_types))
    selected_losses = (
        ["standard", "weighted", "smoothed", "focal"]
        if loss_type == "all" else [loss_type]
    )

    base_parameters = {
        "seed": seed,
        "test_size": test_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lstm_learning_rate": lstm_learning_rate,
        "bert_model_name": bert_model_name,
        "max_length": max_length,
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "lstm_bidirectional": True,
        "lstm_embedding_size": lstm_embedding_size,
        "embedder_model": embedder_model,
    }

    all_results = []
    for model_key in selected_models:
        for current_loss in selected_losses:
            run_parameters = {
                **base_parameters, "model": model_key, "loss": current_loss
            }
            all_results.append(run_experiment(
                train_loader=train_loader,
                test_loader=validation_loader,
                device=device,
                model_key=model_key,
                loss_type=current_loss,
                class_weights=class_weights,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                lstm_learning_rate=lstm_learning_rate,
                results_dir=results_dir,
                bert_model_name=bert_model_name,
                max_length=max_length,
                lstm_embedding_size=lstm_embedding_size,
                embedder_model=embedder_model,
                test_csv=test_csv,
                experiment_params=run_parameters,
                heldout_test_loader=heldout_loader,
                heldout_test_csv=heldout_test_csv,
            ))

    summary = []
    for result in all_results:
        entry = {
            "seed": seed,
            "model": result["model_name"],
            "loss_type": result["loss_type"],
            **result["final_metrics"],
        }
        if result["heldout_test_metrics"]:
            entry.update({
                f"test_{key}": value
                for key, value in result["heldout_test_metrics"].items()
            })
        summary.append(entry)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, "experiment_summary.csv"), index=False)
    print("\nEXPERIMENT SUMMARY")
    print(summary_df.to_string(index=False))
    return all_results, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text classifiers for malware detection")
    parser.add_argument("--malware-csv", default="./unique_data/malware_unique.csv")
    parser.add_argument("--clean-csv", default="./unique_data/clean_unique.csv")
    parser.add_argument("--results-dir", default="./new_results_text")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument(
        "--loss", default="smoothed",
        choices=["standard", "weighted", "smoothed", "focal", "all"],
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--model", nargs="+", default=["bert"], choices=MODEL_CHOICES
    )
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lstm-embedding-size", type=int, default=64)
    parser.add_argument("--embedder-model", default="minishlab/potion-base-2M")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--cache-dir", default="./graph_cache")
    parser.add_argument("--test-malware-csv", default=None)
    parser.add_argument("--test-clean-csv", default=None)
    args = parser.parse_args()

    if bool(args.test_malware_csv) != bool(args.test_clean_csv):
        parser.error("--test-malware-csv and --test-clean-csv must be provided together")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for seed in args.seed:
        set_seed(seed)
        run_full_experiment_suite(
            malware_csv=os.path.expanduser(args.malware_csv),
            clean_csv=os.path.expanduser(args.clean_csv),
            seed=seed,
            results_dir=args.results_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            test_size=args.test_size,
            loss_type=args.loss,
            model_types=args.model,
            bert_model_name=args.bert_model,
            max_length=args.max_length,
            learning_rate=args.lr,
            lstm_learning_rate=args.lstm_lr,
            lstm_embedding_size=args.lstm_embedding_size,
            cache_dir=args.cache_dir,
            embedder_model=args.embedder_model,
            test_malware_csv=(
                os.path.expanduser(args.test_malware_csv) if args.test_malware_csv else None
            ),
            test_clean_csv=(
                os.path.expanduser(args.test_clean_csv) if args.test_clean_csv else None
            ),
            timestamp=timestamp,
        )

    print("\n[Main] Experiment complete!")


if __name__ == "__main__":
    main()
