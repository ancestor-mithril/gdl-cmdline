"""
Manual Feature Engineering Baseline for Malware Detection

This script provides a baseline using manually engineered features extracted 
from the command line strings, followed by a Logistic Regression classifier.
It extracts over 40 heuristic features including lengths, character statistics, 
entropy, special character counts, and specific keyword/pattern matches.
"""

import os
import ast
import time
import math
import re
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
        cmd_tuple = ast.literal_eval(cmd_str)
        flat_list = flatten_tuple(cmd_tuple)
        return " ".join(flat_list)
    except Exception:
        return str(cmd_str).replace('(', ' ').replace(')', ' ').replace("'", ' ').replace(',', ' ')

def extract_features(text: str) -> np.ndarray:
    """
    Extract 40+ manual heuristic features from a command line string.
    """
    if not text:
        return np.zeros(43, dtype=np.float32)
    
    # 1. Lengths and Tokens (4 features)
    length = len(text)
    tokens = text.split()
    num_tokens = len(tokens)
    avg_token_len = length / num_tokens if num_tokens > 0 else 0
    max_token_len = max([len(t) for t in tokens]) if num_tokens > 0 else 0
    
    # 2. Character type counts (5 features)
    num_alpha = sum(c.isalpha() for c in text)
    num_digit = sum(c.isdigit() for c in text)
    num_space = sum(c.isspace() for c in text)
    num_upper = sum(c.isupper() for c in text)
    num_lower = sum(c.islower() for c in text)
    
    # 3. Ratios (3 features)
    upper_lower_ratio = num_upper / num_lower if num_lower > 0 else num_upper
    digit_alpha_ratio = num_digit / num_alpha if num_alpha > 0 else num_digit
    special_ratio = (length - num_alpha - num_digit - num_space) / length if length > 0 else 0
    
    # 4. Entropy (1 feature)
    char_counts = Counter(text)
    entropy = -sum((count/length) * math.log2(count/length) for count in char_counts.values())
    
    # 5. Special Character Counts (16 features)
    # These are highly indicative of obfuscation or complex scripting
    c_caret = text.count('^')
    c_dollar = text.count('$')
    c_percent = text.count('%')
    c_pipe = text.count('|')
    c_redir_r = text.count('>')
    c_redir_l = text.count('<')
    c_amp = text.count('&')
    c_semi = text.count(';')
    c_bslash = text.count('\\')
    c_fslash = text.count('/')
    c_dash = text.count('-')
    c_squote = text.count("'")
    c_dquote = text.count('"')
    c_at = text.count('@')
    c_curly = text.count('{') + text.count('}')
    c_bracket = text.count('[') + text.count(']')
    
    # 6. Keywords and Patterns (13 features)
    text_lower = text.lower()
    has_ps = 1.0 if 'powershell' in text_lower or 'pwsh' in text_lower else 0.0
    has_cmd = 1.0 if 'cmd' in text_lower else 0.0
    has_http = 1.0 if 'http' in text_lower else 0.0
    has_exe = 1.0 if '.exe' in text_lower else 0.0
    has_dll = 1.0 if '.dll' in text_lower else 0.0
    has_ps1 = 1.0 if '.ps1' in text_lower else 0.0
    has_bat = 1.0 if '.bat' in text_lower else 0.0
    
    # PowerShell specific evasion flags
    has_b64 = 1.0 if 'base64' in text_lower or '-enc' in text_lower else 0.0
    has_invoke = 1.0 if 'invoke-' in text_lower else 0.0
    has_hidden = 1.0 if '-w hidden' in text_lower or '-windowstyle hidden' in text_lower else 0.0
    has_bypass = 1.0 if '-ep bypass' in text_lower or '-executionpolicy bypass' in text_lower else 0.0
    has_noni = 1.0 if '-noni' in text_lower else 0.0
    has_nop = 1.0 if '-nop' in text_lower else 0.0
    
    # 7. Regex patterns (1 feature)
    has_ip = 1.0 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text) else 0.0
    
    features = [
        length, num_tokens, avg_token_len, max_token_len,
        num_alpha, num_digit, num_space, num_upper, num_lower,
        upper_lower_ratio, digit_alpha_ratio, special_ratio, entropy,
        c_caret, c_dollar, c_percent, c_pipe, c_redir_r, c_redir_l,
        c_amp, c_semi, c_bslash, c_fslash, c_dash, c_squote, c_dquote,
        c_at, c_curly, c_bracket,
        has_ps, has_cmd, has_http, has_exe, has_dll, has_ps1, has_bat,
        has_b64, has_invoke, has_hidden, has_bypass, has_noni, has_nop,
        has_ip
    ]
    
    return np.array(features, dtype=np.float32)

def load_and_preprocess_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load CSV, flatten strings, and extract manual features.
    Returns: X (features), y (labels), texts (raw flattened strings for logging)
    """
    print(f"[Data] Loading and processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    texts = []
    features_list = []
    
    for cmd in tqdm(df['command_line'].values, desc=f"Extracting features {os.path.basename(csv_path)}"):
        flat_text = parse_and_flatten_command_line(cmd)
        texts.append(flat_text)
        features_list.append(extract_features(flat_text))
        
    X = np.vstack(features_list)
    y = df['label'].values
    
    return X, y, texts

def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_texts: List[str],
    model_name: str = 'logreg_manual',
    results_dir: str = './results_manual',
    test_csv: str = None,
    experiment_params: Dict = None,
    X_heldout: np.ndarray = None,
    y_heldout: np.ndarray = None,
    heldout_test_csv: str = None,
) -> Dict:
    
    os.makedirs(results_dir, exist_ok=True)
    exp_name = model_name
    
    print(f"\n{'#'*60}")
    print(f"# Training Manual Features Baseline: {exp_name}")
    print(f"{'#'*60}")
    
    # 1. Scaling
    print(f"[Scaling] Fitting StandardScaler on {X_train.shape[1]} features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model Training
    print(f"[Model] Training Logistic Regression...")
    t0 = time.time()
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    print(f"[Model] Training complete in {time.time() - t0:.2f}s")
    
    # 3. Evaluation on Test Set
    print("[Model] Evaluating on test set...")
    t0 = time.time()
    y_pred = clf.predict(X_test_scaled)
    print(f"[Model] Prediction complete in {time.time() - t0:.2f}s")
    
    final_metrics = compute_metrics(y_test, y_pred)
    print_metrics(final_metrics, f"Final Results: {exp_name}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Clean', 'Malware']))
    
    if test_csv is not None:
        fp_fn_path = os.path.join(results_dir, f'{exp_name}_fp_fn.txt')
        _write_fp_fn(y_test, y_pred, test_csv, fp_fn_path)
        
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(results_dir, f'{exp_name}_confusion.png'),
        title=f'Confusion Matrix: {exp_name}',
        metrics=final_metrics,
        experiment_params=experiment_params,
    )
    
    # 4. Held-out Test Evaluation
    heldout_metrics = None
    if X_heldout is not None and y_heldout is not None:
        print(f"\n{'#'*60}")
        print(f"# Held-out Test Evaluation: {exp_name}")
        print(f"{'#'*60}")
        
        X_heldout_scaled = scaler.transform(X_heldout)
        ht_pred = clf.predict(X_heldout_scaled)
        
        heldout_metrics = compute_metrics(y_heldout, ht_pred)
        print_metrics(heldout_metrics, f"Held-out Test Results: {exp_name}")
        print("\nHeld-out Test Classification Report:")
        print(classification_report(y_heldout, ht_pred, target_names=['Clean', 'Malware']))
        
        if heldout_test_csv is not None:
            fp_fn_path = os.path.join(results_dir, f'{exp_name}_heldout_test_fp_fn.txt')
            _write_fp_fn(y_heldout, ht_pred, heldout_test_csv, fp_fn_path)
            
        plot_confusion_matrix(
            y_heldout, ht_pred,
            save_path=os.path.join(results_dir, f'{exp_name}_heldout_test_confusion.png'),
            title=f'Held-out Test Confusion Matrix: {exp_name}',
            metrics=heldout_metrics,
            experiment_params=experiment_params,
        )
        
    return {
        'model_name': model_name,
        'final_metrics': final_metrics,
        'heldout_test_metrics': heldout_metrics,
    }

def run_full_experiment_suite(
    malware_csv: str,
    clean_csv: str,
    seed: int = 42,
    results_dir: str = './results_manual',
    test_size: float = 0.3,
    cache_dir: str = './graph_cache',
    test_malware_csv: str = None,
    test_clean_csv: str = None,
    timestamp: str = None,
):
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
        
    # 3. Load, flatten, and extract features
    X_train, y_train, _ = load_and_preprocess_data(train_csv)
    X_test, y_test, test_texts = load_and_preprocess_data(test_csv)
    
    X_heldout, y_heldout = None, None
    if heldout_test_csv:
        X_heldout, y_heldout, _ = load_and_preprocess_data(heldout_test_csv)
        
    # 4. Run experiment
    run_params = {
        'seed': seed,
        'test_size': test_size,
        'model': 'logreg_manual',
        'num_features': X_train.shape[1]
    }
    
    result = run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        test_texts=test_texts,
        model_name='logreg_manual',
        results_dir=os.path.join(results_dir, "logreg_manual"),
        test_csv=test_csv,
        experiment_params=run_params,
        X_heldout=X_heldout,
        y_heldout=y_heldout,
        heldout_test_csv=heldout_test_csv,
    )
    
    # 5. Summary
    summary = []
    entry = {
        'seed': seed,
        'model': result['model_name'],
        **result['final_metrics'],
    }
    if result.get('heldout_test_metrics'):
        for k, v in result['heldout_test_metrics'].items():
            entry[f'test_{k}'] = v
    summary.append(entry)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, 'experiment_summary.csv'), index=False)
    print(f"\n[Main] Results saved to {results_dir}")

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

    return [result], summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Manual Features baseline for malware detection')
    parser.add_argument('--malware-csv', type=str, required=True, help='Path to malware CSV file')
    parser.add_argument('--clean-csv', type=str, required=True, help='Path to clean CSV file')
    parser.add_argument('--results-dir', type=str, default='./results_manual', help='Directory to save results')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test size for train/test split')
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Random seed(s)')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache', help='Directory to cache split CSVs')
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
    print("MANUAL FEATURES BASELINE EXPERIMENT")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for seed in args.seed:
        np.random.seed(seed)
        
        run_full_experiment_suite(
            malware_csv=malware_csv,
            clean_csv=clean_csv,
            seed=seed,
            results_dir=args.results_dir,
            test_size=args.test_size,
            cache_dir=args.cache_dir,
            test_malware_csv=test_malware_csv,
            test_clean_csv=test_clean_csv,
            timestamp=timestamp,
        )
        
    print("\n[Main] Manual Features Baseline experiments complete!")
