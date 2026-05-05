"""
TF-IDF Baseline for Malware Detection

This script provides a traditional Machine Learning baseline using TF-IDF and
linear classifiers (Logistic Regression, LinearSVC). 

It handles the nested tuple format of the command lines by flattening them 
into a single string. It supports both word-level and character-level n-grams.
"""

import os
import ast
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Import shared utilities from train_gnn.py
from gdl_cmdline.scripts.train_gnn import (
    split_and_save_data,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    _write_fp_fn,
    _prepare_combined_csv,
)

def flatten_tuple(t) -> List[str]:
    """
    Recursively flatten a nested tuple of strings into a flat list of strings.
    """
    if isinstance(t, str):
        return [t]
    
    res = []
    if isinstance(t, (tuple, list)):
        for item in t:
            res.extend(flatten_tuple(item))
    return res

def parse_and_flatten_command_line(cmd_str: str) -> str:
    """
    Parse the stringified tuple from the CSV and flatten it into a single space-separated string.
    """
    try:
        # Safely evaluate the stringified tuple
        cmd_tuple = ast.literal_eval(cmd_str)
        # Flatten and join
        flat_list = flatten_tuple(cmd_tuple)
        return " ".join(flat_list)
    except Exception as e:
        # Fallback if parsing fails: just return the raw string without punctuation
        return str(cmd_str).replace('(', ' ').replace(')', ' ').replace("'", ' ').replace(',', ' ')

def load_and_preprocess_data(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load CSV and convert stringified tuples to flattened strings.
    """
    print(f"[Data] Loading and flattening {csv_path}...")
    df = pd.read_csv(csv_path)
    
    texts = [parse_and_flatten_command_line(cmd) for cmd in tqdm(df['command_line'].values, desc=f"Flattening {os.path.basename(csv_path)}")]
    labels = df['label'].values
    
    return texts, labels

def run_experiment(
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    model_name: str = 'logreg',
    analyzer: str = 'char_wb',
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 50000,
    results_dir: str = './results_tfidf',
    test_csv: str = None,
    experiment_params: Dict = None,
    heldout_test_texts: List[str] = None,
    heldout_test_labels: np.ndarray = None,
    heldout_test_csv: str = None,
) -> Dict:
    
    os.makedirs(results_dir, exist_ok=True)
    exp_name = f"{model_name}_{analyzer}_{ngram_range[0]}_{ngram_range[1]}"
    
    print(f"\n{'#'*60}")
    print(f"# Training TF-IDF Baseline: {exp_name}")
    print(f"{'#'*60}")
    
    # 1. TF-IDF Vectorization
    print(f"[TF-IDF] Fitting vectorizer on train data (analyzer={analyzer}, ngram_range={ngram_range}, max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True
    )
    
    t0 = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    print(f"[TF-IDF] Fit & transform on train data complete in {time.time() - t0:.2f}s")
    
    # Calculate memory size of the sparse matrix
    # Sparse matrix memory = data array + indices array + indptr array
    train_mem_mb = (X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes) / (1024 * 1024)
    print(f"[TF-IDF] X_train sparse matrix memory: {train_mem_mb:.2f} MB")
    print(f"[TF-IDF] Non-zero elements: {X_train.nnz} (Sparsity: {100 * X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}%)")
    
    print(f"[TF-IDF] Transforming test data...")
    t0 = time.time()
    X_test = vectorizer.transform(test_texts)
    print(f"[TF-IDF] Transform on test data complete in {time.time() - t0:.2f}s")
    
    test_mem_mb = (X_test.data.nbytes + X_test.indices.nbytes + X_test.indptr.nbytes) / (1024 * 1024)
    print(f"[TF-IDF] X_test sparse matrix memory: {test_mem_mb:.2f} MB")
    
    print(f"[TF-IDF] Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # 2. Model Training
    print(f"[Model] Training {model_name}...")
    t0 = time.time()
    if model_name == 'logreg':
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    clf.fit(X_train, train_labels)
    print(f"[Model] Training complete in {time.time() - t0:.2f}s")
    
    # 3. Evaluation on Test Set
    print("[Model] Evaluating on test set...")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    print(f"[Model] Prediction complete in {time.time() - t0:.2f}s")
    
    final_metrics = compute_metrics(test_labels, y_pred)
    print_metrics(final_metrics, f"Final Results: {exp_name}")
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=['Clean', 'Malware']))
    
    if test_csv is not None:
        fp_fn_path = os.path.join(results_dir, f'{exp_name}_fp_fn.txt')
        _write_fp_fn(test_labels, y_pred, test_csv, fp_fn_path)
        
    plot_confusion_matrix(
        test_labels, y_pred,
        save_path=os.path.join(results_dir, f'{exp_name}_confusion.png'),
        title=f'Confusion Matrix: {exp_name}',
        metrics=final_metrics,
        experiment_params=experiment_params,
    )
    
    # 4. Held-out Test Evaluation
    heldout_metrics = None
    if heldout_test_texts is not None and heldout_test_labels is not None:
        print(f"\n{'#'*60}")
        print(f"# Held-out Test Evaluation: {exp_name}")
        print(f"{'#'*60}")
        
        print(f"[TF-IDF] Transforming held-out test data...")
        t0 = time.time()
        X_heldout = vectorizer.transform(heldout_test_texts)
        print(f"[TF-IDF] Transform on held-out test data complete in {time.time() - t0:.2f}s")
        print(f"[TF-IDF] Predicting on held-out test data...")
        t0 = time.time()
        ht_pred = clf.predict(X_heldout)
        print(f"[TF-IDF] Predict on held-out test data complete in {time.time() - t0:.2f}s")
        
        heldout_metrics = compute_metrics(heldout_test_labels, ht_pred)
        print_metrics(heldout_metrics, f"Held-out Test Results: {exp_name}")
        print("\nHeld-out Test Classification Report:")
        print(classification_report(heldout_test_labels, ht_pred, target_names=['Clean', 'Malware']))
        
        if heldout_test_csv is not None:
            fp_fn_path = os.path.join(results_dir, f'{exp_name}_heldout_test_fp_fn.txt')
            _write_fp_fn(heldout_test_labels, ht_pred, heldout_test_csv, fp_fn_path)
            
        plot_confusion_matrix(
            heldout_test_labels, ht_pred,
            save_path=os.path.join(results_dir, f'{exp_name}_heldout_test_confusion.png'),
            title=f'Held-out Test Confusion Matrix: {exp_name}',
            metrics=heldout_metrics,
            experiment_params=experiment_params,
        )
        
    return {
        'model_name': model_name,
        'analyzer': analyzer,
        'final_metrics': final_metrics,
        'heldout_test_metrics': heldout_metrics,
    }

def run_full_experiment_suite(
    malware_csv: str,
    clean_csv: str,
    seed: int = 42,
    results_dir: str = './results_tfidf',
    test_size: float = 0.3,
    cache_dir: str = './graph_cache',
    models: List[str] = None,
    analyzers: List[str] = None,
    max_features: int = 50000,
    test_malware_csv: str = None,
    test_clean_csv: str = None,
    timestamp: str = None,
):
    if models is None:
        models = ['logreg']
    if analyzers is None:
        analyzers = ['char_wb', 'word']
        
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    exp_name = f"{timestamp}/{test_size}/seed{seed}"
    results_dir = os.path.join(results_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Split data using the shared GNN caching logic
    train_csv, test_csv = split_and_save_data(
        malware_csv, clean_csv,
        seed=seed, test_size=test_size,
        cache_dir=cache_dir,
    )
    
    # 2. Prepare held-out test CSV
    heldout_test_csv = None
    if test_malware_csv and test_clean_csv:
        heldout_test_csv = _prepare_combined_csv(
            test_malware_csv, test_clean_csv, cache_dir,
        )
        
    # 3. Load and flatten texts
    train_texts, train_labels = load_and_preprocess_data(train_csv)
    test_texts, test_labels = load_and_preprocess_data(test_csv)
    
    heldout_test_texts, heldout_test_labels = None, None
    if heldout_test_csv:
        heldout_test_texts, heldout_test_labels = load_and_preprocess_data(heldout_test_csv)
        
    # 4. Run experiments
    all_results = []
    
    for model_name in models:
        for analyzer in analyzers:
            # Set appropriate n-gram ranges based on analyzer
            if analyzer == 'word':
                ngram_range = (1, 3) # Unigrams, bigrams, trigrams
            else:
                ngram_range = (3, 5) # 3 to 5 character n-grams
                
            run_params = {
                'seed': seed,
                'test_size': test_size,
                'model': model_name,
                'analyzer': analyzer,
                'ngram_range': ngram_range,
                'max_features': max_features,
            }
            
            result = run_experiment(
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                model_name=model_name,
                analyzer=analyzer,
                ngram_range=ngram_range,
                max_features=max_features,
                results_dir=os.path.join(results_dir, f"{model_name}_{analyzer}"),
                test_csv=test_csv,
                experiment_params=run_params,
                heldout_test_texts=heldout_test_texts,
                heldout_test_labels=heldout_test_labels,
                heldout_test_csv=heldout_test_csv,
            )
            all_results.append(result)
            
    # 5. Summary
    summary = []
    for r in all_results:
        entry = {
            'seed': seed,
            'model': r['model_name'],
            'analyzer': r['analyzer'],
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
    parser = argparse.ArgumentParser(description='Train TF-IDF baseline for malware detection')
    parser.add_argument('--malware-csv', type=str, required=True, help='Path to malware CSV file')
    parser.add_argument('--clean-csv', type=str, required=True, help='Path to clean CSV file')
    parser.add_argument('--results-dir', type=str, default='./results_tfidf', help='Directory to save results')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test size for train/test split')
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Random seed(s)')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache', help='Directory to cache split CSVs')
    parser.add_argument('--models', type=str, nargs='+', default=['logreg'], choices=['logreg'], help='Models to train')
    parser.add_argument('--analyzers', type=str, nargs='+', default=['char_wb', 'word'], choices=['word', 'char', 'char_wb'], help='TF-IDF analyzers to test')
    parser.add_argument('--max-features', type=int, default=50000, help='Max features for TF-IDF')
    parser.add_argument('--test-malware-csv', type=str, default=None, help='Path to held-out test malware CSV')
    parser.add_argument('--test-clean-csv', type=str, default=None, help='Path to held-out test clean CSV')
    
    args = parser.parse_args()
    
    malware_csv = os.path.expanduser(args.malware_csv)
    clean_csv = os.path.expanduser(args.clean_csv)
    test_malware_csv = os.path.expanduser(args.test_malware_csv) if args.test_malware_csv else None
    test_clean_csv = os.path.expanduser(args.test_clean_csv) if args.test_clean_csv else None

    if bool(test_malware_csv) != bool(test_clean_csv):
        parser.error("--test-malware-csv and --test-clean-csv must be provided together")

    print("="*70)
    print("TF-IDF BASELINE EXPERIMENT")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for seed in args.seed:
        # Set seed for numpy and sklearn (via numpy)
        np.random.seed(seed)
        
        run_full_experiment_suite(
            malware_csv=malware_csv,
            clean_csv=clean_csv,
            seed=seed,
            results_dir=args.results_dir,
            test_size=args.test_size,
            cache_dir=args.cache_dir,
            models=args.models,
            analyzers=args.analyzers,
            max_features=args.max_features,
            test_malware_csv=test_malware_csv,
            test_clean_csv=test_clean_csv,
            timestamp=timestamp,
        )
        
    print("\n[Main] TF-IDF Baseline experiments complete!")
