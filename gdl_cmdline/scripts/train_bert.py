"""
BERT Fine-Tuning for Malware Detection

This script fine-tunes a pretrained BERT model to classify raw command-line strings
as malware or clean. This serves as a comparison baseline against the GNN approach
in train_gnn.py.

Key Differences from GNN:
- Uses raw command-line strings directly (no graph construction, no eval())
- Fine-tunes the full BERT model end-to-end
- No preprocessing cache needed (tokenization is fast)

Dataset splitting and shared utilities (metrics, plotting, loss functions) are
imported from train_gnn.py so both scripts use the same cached train/test split.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from typing import List, Tuple, Dict
from datetime import datetime

from torch.utils.data import Dataset, DataLoader

import pandas as pd
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import classification_report

from gdl_cmdline.scripts.train_gnn import (
    split_and_save_data,
    set_seed,
    FocalLoss,
    get_class_weights,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    _write_fp_fn,
    _prepare_combined_csv,
    StaticEmbedder,
)


class CommandLineTextDataset(Dataset):
    """
    Simple text dataset for command-line strings.

    Accepts a single CSV with ``command_line`` and ``label`` columns
    (the output of :func:`train_gnn.split_and_save_data`).
    No eval(), no graph construction — just raw strings and labels.
    """

    def __init__(self, data_csv: str):
        super().__init__()

        print(f"[Dataset] Loading data from {data_csv}...")
        df = pd.read_csv(data_csv)

        self.texts = [str(cmd) for cmd in df['command_line'].values]
        self.labels = df['label'].values.tolist()

        del df

        label_counts = {}
        for l in self.labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        print(f"[Dataset] Total samples: {len(self.texts)} "
              f"(clean={label_counts.get(0, 0)}, malware={label_counts.get(1, 0)})")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    """
    Custom collate function that keeps texts as a list of strings
    and labels as a tensor.
    """
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


class BERTClassifier(nn.Module):
    """
    BERT-based classifier for malware detection.
    
    Architecture:
    - Pretrained BERT encoder (fine-tuned end-to-end)
    - Dropout for regularization
    - Linear classification head
    
    Uses [CLS] token embedding as the sequence representation.
    """
    name = "BERTClassifier"
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2,
        dropout: float = 0.3,
        max_length: int = 512,
    ):
        super().__init__()
        
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass: tokenize raw strings, encode with BERT, classify.
        
        Args:
            texts: List of raw command-line strings
            
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        device = next(self.parameters()).device
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)
        
        outputs = self.bert(**encoded)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_embedding)
        return logits


class StaticModelClassifier(nn.Module):
    """
    Static-embedding classifier for malware detection.

    Architecture:
    - Model2Vec StaticModel produces a fixed embedding per input string
      (subword averaging, no transformers, extremely fast).
    - Two-layer MLP classification head (the only trainable part).

    This serves as a lightweight baseline: the embedding is frozen and only
    the classifier is trained.  The forward signature is identical to
    BERTClassifier so both models are interchangeable in the training loop.
    """
    name = "StaticClassifier"

    def __init__(
        self,
        embedder_model: str = 'minishlab/potion-base-2M',
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


MODEL_CHOICES = ['bert', 'static', 'all']


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.amp.GradScaler = None,
) -> float:
    """Train the model for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    total_samples = 0
    use_amp = scaler is not None
    
    for texts, labels in tqdm(loader, desc="Training", leave=False):
        labels = labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type if isinstance(device, torch.device) else device, enabled=use_amp):
            out = model(texts)
            loss = criterion(out, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    
    return total_loss / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the model on a dataset with mixed precision support."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for texts, labels in tqdm(loader, desc="Evaluating", leave=False):
        labels = labels.to(device)
        
        with torch.amp.autocast(device_type=device.type if isinstance(device, torch.device) else device, enabled=use_amp):
            out = model(texts)
        
        preds = out.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)



def run_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    model_key: str = 'bert',
    loss_type: str = 'standard',
    class_weights: torch.Tensor = None,
    num_epochs: int = 25,
    learning_rate: float = 2e-5,
    results_dir: str = './results',
    early_stopping_patience: int = 5,
    bert_model_name: str = 'bert-base-uncased',
    max_length: int = 512,
    embedder_model: str = 'minishlab/potion-base-2M',
    test_csv: str = None,
    experiment_params: Dict = None,
    heldout_test_loader: DataLoader = None,
    heldout_test_csv: str = None,
) -> Dict:
    """
    Run a single training experiment (BERT or StaticModel).

    Mirrors run_experiment() from train_gnn.py with the same training loop,
    early stopping, metrics, plotting, FP/FN logging, and held-out test
    evaluation.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Initialize model
    if model_key == 'static':
        model = StaticModelClassifier(
            embedder_model=embedder_model,
            num_classes=2,
        ).to(device)
    else:
        model = BERTClassifier(
            model_name=bert_model_name,
            num_classes=2,
            max_length=max_length,
        ).to(device)

    model_name = model.name

    print(f"\n{'#'*60}")
    print(f"# Training {model_name} with {loss_type} loss")
    print(f"{'#'*60}")

    # Set up loss function based on type
    if loss_type == 'weighted':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'smoothed':
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    elif loss_type == 'focal':
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    print(f"[DEBUG] Using criterion: {criterion}")

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Mixed precision scaler (only for CUDA)
    device_type = device.type if isinstance(device, torch.device) else device
    use_amp = 'cuda' in str(device_type)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"[DEBUG] Mixed precision (autocast): {use_amp}")

    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Training history
    train_losses = []
    val_metrics_history = []
    best_f1 = -1.0
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    # Training loop
    with trange(1, num_epochs + 1, desc="Training") as tbar:
        for epoch in tbar:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
            train_losses.append(train_loss)

            y_pred, y_true = evaluate(model, test_loader, device, use_amp=use_amp)
            metrics = compute_metrics(y_true, y_pred)
            val_metrics_history.append(metrics)

            scheduler.step(metrics['f1_macro'])

            if metrics['f1_malware'] > best_f1:
                best_f1 = metrics['f1_malware']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            tbar.set_postfix(
                loss=round(train_loss, 3),
                f1_macro=round(metrics['f1_macro'], 3),
                f1_malware=round(metrics['f1_malware'], 3),
                best_f1=round(best_f1, 3),
                no_improv=epochs_without_improvement
            )

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[Early Stopping] No improvement in f1_malware for "
                      f"{early_stopping_patience} epochs. Stopping at epoch {epoch}.")
                break

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    model.to(device)
    y_pred, y_true = evaluate(model, test_loader, device, use_amp=use_amp)
    final_metrics = compute_metrics(y_true, y_pred)

    exp_name = f"{model_name}_{loss_type}"

    print_metrics(final_metrics, f"Final Results: {exp_name}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Clean', 'Malware']))

    if test_csv is not None:
        fp_fn_path = os.path.join(results_dir, f'{exp_name}_fp_fn.txt')
        _write_fp_fn(y_true, y_pred, test_csv, fp_fn_path)

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(results_dir, f'{exp_name}_confusion.png'),
        title=f'Confusion Matrix: {exp_name}',
        metrics=final_metrics,
        experiment_params=experiment_params,
    )

    plot_training_curves(
        train_losses, val_metrics_history,
        save_path=os.path.join(results_dir, f'{exp_name}_curves.png'),
        title=f'Training Curves: {exp_name}'
    )

    # Save model
    torch.save(best_model_state, os.path.join(results_dir, f'{exp_name}_model.pt'))

    # ---- Held-out test evaluation ----
    heldout_metrics = None
    if heldout_test_loader is not None:
        print(f"\n{'#'*60}")
        print(f"# Held-out Test Evaluation: {exp_name}")
        print(f"{'#'*60}")

        ht_pred, ht_true = evaluate(model, heldout_test_loader, device, use_amp=use_amp)
        heldout_metrics = compute_metrics(ht_true, ht_pred)

        print_metrics(heldout_metrics, f"Held-out Test Results: {exp_name}")
        print("\nHeld-out Test Classification Report:")
        print(classification_report(ht_true, ht_pred, target_names=['Clean', 'Malware']))

        if heldout_test_csv is not None:
            fp_fn_path = os.path.join(results_dir, f'{exp_name}_heldout_test_fp_fn.txt')
            _write_fp_fn(ht_true, ht_pred, heldout_test_csv, fp_fn_path)

        plot_confusion_matrix(
            ht_true, ht_pred,
            save_path=os.path.join(results_dir, f'{exp_name}_heldout_test_confusion.png'),
            title=f'Held-out Test Confusion Matrix: {exp_name}',
            metrics=heldout_metrics,
            experiment_params=experiment_params,
        )

    return {
        'model_name': model_name,
        'loss_type': loss_type,
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history,
        'final_metrics': final_metrics,
        'heldout_test_metrics': heldout_metrics,
    }



def run_full_experiment_suite(
    malware_csv: str,
    clean_csv: str,
    seed: int = 42,
    results_dir: str = './results_bert',
    num_epochs: int = 25,
    batch_size: int = 16,
    val_batch_size: int = 64,
    test_size: float = 0.3,
    loss_type: str = 'all',
    model_types: List[str] = None,
    bert_model_name: str = 'bert-base-uncased',
    max_length: int = 512,
    learning_rate: float = 2e-5,
    cache_dir: str = './graph_cache',
    embedder_model: str = 'minishlab/potion-base-2M',
    test_malware_csv: str = None,
    test_clean_csv: str = None,
    timestamp: str = None,
):
    """
    Run the text-classifier experiment suite (BERT and/or StaticModel).

    Uses :func:`train_gnn.split_and_save_data` for the stratified train/test
    split so that both GNN and BERT experiments share the exact same split
    (and benefit from caching).

    Mirrors the feature set of :func:`train_gnn.run_full_experiment_suite`:
    multiple model types, held-out test evaluation, FP/FN logging, and
    experiment-params recording.
    """
    if model_types is None:
        model_types = ['bert']

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
    print(f"[Main] Using device: {device}")
    print(f"[Main] Seed: {seed}")

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_label = "+".join(model_types)
    exp_name = f"{timestamp}/{test_size}/seed{seed}/{model_label}"
    results_dir = os.path.join(results_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[Main] Results directory: {results_dir}")

    # ---- Phase 1: Split data using the shared GNN caching logic ----
    train_csv, test_csv = split_and_save_data(
        malware_csv, clean_csv,
        seed=seed, test_size=test_size,
        cache_dir=cache_dir,
    )

    # ---- Prepare held-out test CSV ----
    heldout_test_csv = None
    if test_malware_csv and test_clean_csv:
        heldout_test_csv = _prepare_combined_csv(
            test_malware_csv, test_clean_csv, cache_dir,
        )

    # ---- Phase 2: Load pre-split CSVs as text datasets ----
    train_dataset = CommandLineTextDataset(train_csv)
    test_dataset = CommandLineTextDataset(test_csv)

    heldout_test_loader = None
    if heldout_test_csv is not None:
        heldout_dataset = CommandLineTextDataset(heldout_test_csv)
        heldout_test_loader = DataLoader(
            heldout_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn,
        )
        print(f"[Split] Held-out test: {len(heldout_dataset)}")

    class_weights = get_class_weights(train_dataset.labels, device)

    print(f"[Split] Train: {len(train_dataset)}, Val: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Model / loss selection ----
    if 'all' in model_types:
        selected_models = ['bert', 'static']
    else:
        selected_models = list(dict.fromkeys(model_types))
    print(f"[Main] Running {len(selected_models)} model(s): {selected_models}")

    all_loss_types = ['standard', 'weighted', 'smoothed', 'focal']
    loss_types = all_loss_types if loss_type == 'all' else [loss_type]
    print(f"[Main] Running {len(loss_types)} loss type(s): {loss_types}")

    # ---- Build reproducibility dict ----
    base_params = {
        'seed': seed,
        'test_size': test_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'bert_model_name': bert_model_name,
        'max_length': max_length,
        'embedder_model': embedder_model,
    }

    # ---- Phase 3: Train ----
    all_results = []

    for model_key in selected_models:
        for current_loss_type in loss_types:
            run_params = {
                **base_params,
                'model': model_key,
                'loss': current_loss_type,
            }
            result = run_experiment(
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                model_key=model_key,
                loss_type=current_loss_type,
                class_weights=class_weights,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                results_dir=results_dir,
                bert_model_name=bert_model_name,
                max_length=max_length,
                embedder_model=embedder_model,
                test_csv=test_csv,
                experiment_params=run_params,
                heldout_test_loader=heldout_test_loader,
                heldout_test_csv=heldout_test_csv,
            )
            all_results.append(result)

    # ---- Summary ----
    summary = []
    for r in all_results:
        entry = {
            'seed': seed,
            'model': r['model_name'],
            'loss_type': r['loss_type'],
            **r['final_metrics'],
        }
        if r.get('heldout_test_metrics'):
            for k, v in r['heldout_test_metrics'].items():
                entry[f'test_{k}'] = v
        summary.append(entry)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, 'experiment_summary.csv'), index=False)
    print(f"\n[Main] Results saved to {results_dir}")

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

    return all_results, summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train text classifiers (BERT / StaticModel) for malware detection')
    parser.add_argument('--malware-csv', type=str,
                        default='./unique_data/malware_unique.csv',
                        help='Path to malware CSV file')
    parser.add_argument('--clean-csv', type=str,
                        default='./unique_data/clean_unique.csv',
                        help='Path to clean CSV file')
    parser.add_argument('--results-dir', type=str, default='./new_results_bert',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--loss', type=str, default='smoothed',
                        choices=['standard', 'weighted', 'smoothed', 'focal', 'all'],
                        help='Loss function to use')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test size for train/test split')
    parser.add_argument('--model', type=str, nargs='+', default=['bert'],
                        choices=MODEL_CHOICES,
                        help='Model(s) to use (space-separated, or "all")')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model name (used when --model includes bert)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Max token length for BERT tokenizer')
    parser.add_argument('--embedder-model', type=str,
                        default='minishlab/potion-base-2M',
                        help='Model2Vec static model (used when --model includes static; '
                             'potion-base-2M=64d, 4M=128d, 8M=256d)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, nargs='+', default=[42],
                        help='Random seed(s) for reproducibility (space-separated, default: 42)')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache',
                        help='Directory to cache split CSVs (shared with train_gnn)')
    parser.add_argument('--test-malware-csv', type=str, default=None,
                        help='Path to held-out test malware CSV (requires --test-clean-csv)')
    parser.add_argument('--test-clean-csv', type=str, default=None,
                        help='Path to held-out test clean CSV (requires --test-malware-csv)')
    args = parser.parse_args()

    malware_csv = os.path.expanduser(args.malware_csv)
    clean_csv = os.path.expanduser(args.clean_csv)
    test_malware_csv = os.path.expanduser(args.test_malware_csv) if args.test_malware_csv else None
    test_clean_csv = os.path.expanduser(args.test_clean_csv) if args.test_clean_csv else None

    if bool(test_malware_csv) != bool(test_clean_csv):
        parser.error("--test-malware-csv and --test-clean-csv must be provided together")

    print("="*70)
    print("TEXT CLASSIFIER MALWARE DETECTION EXPERIMENT")
    print("="*70)
    print(f"Malware CSV: {malware_csv}")
    print(f"Clean CSV: {clean_csv}")
    print(f"Results Dir: {args.results_dir}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"Seed(s): {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model(s): {args.model}")
    print(f"Loss: {args.loss}")
    print(f"Test Size: {args.test_size}")
    print(f"BERT Model: {args.bert_model}")
    print(f"Max Length: {args.max_length}")
    print(f"Embedder Model: {args.embedder_model}")
    print(f"Learning Rate: {args.lr}")
    if test_malware_csv:
        print(f"Test Malware CSV: {test_malware_csv}")
        print(f"Test Clean CSV: {test_clean_csv}")
    print("="*70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for seed in args.seed:
        set_seed(seed)

        results, summary = run_full_experiment_suite(
            malware_csv=malware_csv,
            clean_csv=clean_csv,
            seed=seed,
            results_dir=args.results_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size,
            loss_type=args.loss,
            model_types=args.model,
            bert_model_name=args.bert_model,
            max_length=args.max_length,
            learning_rate=args.lr,
            cache_dir=args.cache_dir,
            embedder_model=args.embedder_model,
            test_malware_csv=test_malware_csv,
            test_clean_csv=test_clean_csv,
            timestamp=timestamp,
        )

    print("\n[Main] Experiment complete!")


