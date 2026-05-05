# gdl-cmdline

Command-line malware detection experiments (GNN + baselines) with a small preprocessing pipeline. The repository includes `dummy_data/` for quick end-to-end runs. The dummy dataset is synthetic and contains no malware; it is only for demonstration purposes.

## Data layout

`dummy_data/` contains four CSVs:
- `malware.csv`
- `clean.csv`
- `test_malware.csv`
- `test_clean.csv`

Each CSV must include a `command_line` column. Optional columns like `process_name` or `total_occurrences` are used by the preprocessing step if present. To use real data, replace the CSVs with your own files that follow the same schema and rerun the flow below.

## Experiment flow

Run commands from the repository root. Install the requirements. All experiments were run using python 3.13.


```
pip install -r requirements.txt
```

### 1) Preprocess and create unique command lines

This generates `*_unique.csv` files in the same folder.

```
python -m gdl_cmdline.preprocess.make_unique dummy_data
```

### 2) Training recipes (GNN + baselines)

All commands below write results into their `--results-dir` folders, including
`experiment_summary.csv` and confusion matrices. You can pass multiple seeds as
`--seed 3 7 42` to run all three seeds in one command.

All GNN models, semantic features, all edges:
```
python -m gdl_cmdline.scripts.train_gnn \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_gnn_semantic_all_edges \
  --model all \
  --feature-type semantic_only \
  --edge-ablation all \
  --seed 3 7 42
```

RGCN with semantic features, sweep all edge ablations:
```
python -m gdl_cmdline.scripts.train_gnn \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_rgcn_semantic_edges \
  --model rgcn \
  --feature-type semantic_only \
  --edge-ablation all self_loops_only self_loops_parent_child self_loops_sequential \
  --seed 3 7 42
```

RGCN with syntactic features, sweep all edge ablations:
```
python -m gdl_cmdline.scripts.train_gnn \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_rgcn_syntactic_edges \
  --model rgcn \
  --feature-type syntactic_only \
  --edge-ablation all self_loops_only self_loops_parent_child self_loops_sequential \
  --seed 3 7 42
```

RGCN with all node feature combinations, all edges:
```
python -m gdl_cmdline.scripts.train_gnn \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_rgcn_all_features \
  --model rgcn \
  --feature-type all \
  --edge-ablation all \
  --seed 3 7 42
```

BERT and StaticModel baselines:
```
python -m gdl_cmdline.scripts.train_bert \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_bert \
  --model bert static \
  --seed 3 7 42
```

Logistic Regression + manual features:
```
python -m gdl_cmdline.scripts.train_manual_features \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_manual \
  --seed 3 7 42
```

Logistic Regression + TF-IDF (char + word):
```
python -m gdl_cmdline.scripts.train_tfidf \
  --malware-csv dummy_data/malware_unique.csv \
  --clean-csv dummy_data/clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --results-dir results_tfidf \
  --analyzers char word \
  --seed 3 7 42
```

### 3) Count model parameters

```
python -m gdl_cmdline.scripts.count_params
```

### 4) Analyze results and find runs

Summarize metrics into tables (per-seed rows plus mean/std for `f1_malware` and
`fpr`; held-out test tables are included when available):
```
python -m gdl_cmdline.scripts.summarize_results \
  --pattern "results_gnn*/**/experiment_summary.csv" "results_rgcn*/**/experiment_summary.csv"
```
Use other patterns for baselines, for example:
```
python -m gdl_cmdline.scripts.summarize_results \
  --pattern "results_tfidf/**/experiment_summary.csv"
```

Optional: write tables to files:
```
python -m gdl_cmdline.scripts.summarize_results \
  --pattern "results_gnn/**/experiment_summary.csv" \
  --output-dir results_tables
```

Find completed GNN runs for a specific configuration (use the `results_dir`
output as the `--results-dir` input for `gdl_cmdline.scripts.inference`):
```
python -m gdl_cmdline.scripts.find_gnn_runs \
  --results-root results_gnn \
  --feature-config semantic_only \
  --edge-ablation-label all \
  --model rgcn
```
The last line printed is a model `.pt` path you can copy directly.

### 5) GNN inference timing (latency profiling)


Pick a model file from `results_gnn/.../<feature_config>/` and run:
```
python -m gdl_cmdline.scripts.gnn_inference \
  --model-path results_gnn/<timestamp>/<test_size>/seed<seed>/<ft>/<ea>/<embedder>/<feature_config>/RGCN_standard_model.pt \
  --csv dummy_data/test_clean_unique.csv
```
You can also run the same search filters as `find_gnn_runs` (use `--seed` to pick
a specific seed; otherwise the first match is used):
```
python -m gdl_cmdline.scripts.gnn_inference \
  --results-root results_gnn \
  --feature-config semantic_only \
  --edge-ablation-label all \
  --model rgcn \
  --csv dummy_data/test_clean_unique.csv
```

### 6) Evaluate a trained GNN on held-out test data

Point `--results-dir` to the folder that contains the `*_model.pt` files for a single feature configuration, or pass a single `--model-path` file (the script uses the parent directory to infer the config).
```
python -m gdl_cmdline.scripts.inference \
  --model-path "results_gnn/<timestamp>/<test_size>/seed<seed>/<ft>/<ea>/<embedder>/<feature_config>/RGCN_standard_model.pt" \
  --clean-csv dummy_data/clean_unique.csv \
  --malware-csv dummy_data/malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv
```
You can also reuse the search filters (same as `find_gnn_runs`) to pick a model:
```
python -m gdl_cmdline.scripts.inference \
  --results-root results_gnn \
  --feature-config semantic_only \
  --edge-ablation-label all \
  --model rgcn \
  --clean-csv dummy_data/clean_unique.csv \
  --malware-csv dummy_data/malware_unique.csv \
  --test-clean-csv dummy_data/test_clean_unique.csv \
  --test-malware-csv dummy_data/test_malware_unique.csv
```

The script writes confusion matrices and FP/FN reports into `<results-dir>/inference/`.

## Comparing results

- Each training run writes `experiment_summary.csv` in its results directory. Primary metrics are `f1_malware` and `fpr`; use other metrics as supporting context.
- GNN runs also write `feature_comparison.png` and `loss_comparison.png` to summarize the best feature/loss combinations.
- `gnn_inference.py` prints preprocessing vs model inference timing, which is useful for comparing latency across models.
