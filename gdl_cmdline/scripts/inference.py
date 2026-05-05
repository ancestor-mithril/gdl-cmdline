import glob
import os
import argparse
import torch
from torch_geometric.loader import DataLoader

# Import required components from train_gnn
from gdl_cmdline.scripts.train_gnn import (
    FeatureConfig,
    CommandLineGraphDataset,
    GCNModel,
    GraphSAGEModel,
    GINModel,
    GATModel,
    RGCNModel,
    StaticEmbedder,
    split_and_save_data,
    _prepare_combined_csv,
    evaluate,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    _write_fp_fn
)

def parse_results_dir(results_dir: str) -> dict:
    """
    Parses the directory path to extract experiment configuration.
    Expected structure: .../{timestamp}/{test_size}/seed{seed}/{ft_label}/{ea_label}/{embedder_tag}/{feature_config}
    """
    results_dir = os.path.normpath(results_dir)
    parts = results_dir.split(os.sep)
    
    seed_idx = -1
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            seed_idx = i
            break
            
    if seed_idx == -1:
        raise ValueError(f"Could not find 'seedXXX' in the path {results_dir} to determine configuration.")
        
    test_size = float(parts[seed_idx - 1])
    seed = int(parts[seed_idx][4:])
    ft_label = parts[seed_idx + 1]
    ea_label = parts[seed_idx + 2]
    embedder_tag = parts[seed_idx + 3]
    feature_config_name = parts[seed_idx + 4]
    
    # Reconstruct embedder model name (e.g. minishlab_potion-base-2M -> minishlab/potion-base-2M)
    if "_" in embedder_tag and not embedder_tag.startswith("minishlab/"):
        embedder_model = embedder_tag.replace('_', '/', 1)
    else:
        embedder_model = embedder_tag
        
    return {
        'test_size': test_size,
        'seed': seed,
        'ft_label': ft_label,
        'edge_ablation': ea_label,
        'embedder_model': embedder_model,
        'feature_config_name': feature_config_name
    }

def get_model_class(model_name: str):
    models = {
        'GCN': GCNModel,
        'GraphSAGE': GraphSAGEModel,
        'GIN': GINModel,
        'GAT': GATModel,
        'RGCN': RGCNModel,
    }
    return models[model_name]


def _split_path(path: str) -> list[str]:
    normalized = os.path.normpath(path).replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def _find_seed_index(parts: list[str]) -> int:
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            return i
    return -1


_MODEL_NAME_ORDER = ['GraphSAGE', 'RGCN', 'GCN', 'GIN', 'GAT']


def _parse_model_metadata(model_path: str) -> dict:
    parts = _split_path(model_path)
    seed_idx = _find_seed_index(parts)
    if seed_idx == -1 or seed_idx + 5 >= len(parts):
        raise ValueError(f"Could not parse metadata from path {model_path}")

    seed = int(parts[seed_idx][4:])
    try:
        test_size = float(parts[seed_idx - 1])
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse test_size from path {model_path}")

    ft_label = parts[seed_idx + 1]
    edge_ablation_label = parts[seed_idx + 2]
    embedder_tag = parts[seed_idx + 3]
    feature_config_name = parts[seed_idx + 4]
    model_filename = parts[seed_idx + 5]

    if "_" in embedder_tag and not embedder_tag.startswith("minishlab/"):
        embedder_model = embedder_tag.replace("_", "/", 1)
    else:
        embedder_model = embedder_tag

    base = model_filename.replace("_model.pt", "")
    model_name = next((n for n in _MODEL_NAME_ORDER if base.startswith(n)), None)
    if model_name is None:
        raise ValueError(f"Could not infer model architecture from {model_filename}")

    return {
        "seed": seed,
        "test_size": test_size,
        "ft_label": ft_label,
        "edge_ablation_label": edge_ablation_label,
        "embedder_model": embedder_model,
        "feature_config_name": feature_config_name,
        "model_name": model_name,
    }


def _select_model_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    model_glob = os.path.join(args.results_root, "**", "*_model.pt")
    candidates = sorted(glob.glob(model_glob, recursive=True))
    if not candidates:
        parser.error(f"No model files found under {args.results_root}.")

    matches: list[str] = []
    for path in candidates:
        try:
            meta = _parse_model_metadata(path)
        except ValueError:
            continue

        if args.seed is not None and meta["seed"] != args.seed:
            continue
        if args.test_size is not None and meta["test_size"] != args.test_size:
            continue
        if args.feature_type_label and meta["ft_label"] != args.feature_type_label:
            continue
        if args.edge_ablation_label and meta["edge_ablation_label"] != args.edge_ablation_label:
            continue
        if args.embedder and meta["embedder_model"] != args.embedder:
            continue
        if args.feature_config and meta["feature_config_name"] not in args.feature_config:
            continue
        if args.model:
            if not any(meta["model_name"].lower().startswith(m.lower()) for m in args.model):
                continue

        matches.append(path)

    if not matches:
        parser.error("No models found matching the requested filters.")

    matches.sort()
    selected = matches[0]
    if len(matches) > 1:
        print(f"[Inference] Multiple models matched ({len(matches)}). Using {selected}.")
    return selected

def main():
    parser = argparse.ArgumentParser(description='Inference script for GNN malware detection')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--results-dir', type=str,
                              help='Path to the directory containing the .pt model files (e.g., .../semantic_only/)')
    source_group.add_argument('--model-path', type=str,
                              help='Path (or glob) to a single *_model.pt file or a directory containing it')
    source_group.add_argument('--results-root', type=str,
                              help='Root directory of GNN results to search.')
    parser.add_argument('--feature-config', type=str, nargs='+', default=None)
    parser.add_argument('--edge-ablation-label', type=str, default=None)
    parser.add_argument('--feature-type-label', type=str, default=None)
    parser.add_argument('--embedder', type=str, default=None)
    parser.add_argument('--model', type=str, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test-size', type=float, default=None)
    parser.add_argument('--clean-csv', type=str, required=True,
                        help='Path to the original clean CSV (to reconstruct the validation/test split)')
    parser.add_argument('--malware-csv', type=str, required=True,
                        help='Path to the original malware CSV (to reconstruct the validation/test split)')
    parser.add_argument('--test-clean-csv', type=str, required=True,
                        help='Path to future clean CSV')
    parser.add_argument('--test-malware-csv', type=str, required=True,
                        help='Path to future malware CSV')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache',
                        help='Directory to cache preprocessed graphs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers (must match training)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (must match training)')
    parser.add_argument('--model-type', type=str, default='rgcn',
                        choices=['gcn', 'graphsage', 'gin', 'gat', 'rgcn', 'all'],
                        help='Model type (must match training)')
    
    args = parser.parse_args()
    
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results_dir = None
    model_files = None
    if args.results_root:
        model_path = _select_model_path(args, parser)
        results_dir = os.path.dirname(model_path)
        model_files = [os.path.basename(model_path)]
    elif args.model_path:
        expanded = glob.glob(args.model_path, recursive=True)
        if len(expanded) == 0:
            parser.error(f"--model-path matched no files: {args.model_path}")
        if len(expanded) > 1:
            parser.error(f"--model-path matched multiple files: {expanded}")
        model_path = expanded[0]
        if os.path.isdir(model_path):
            results_dir = model_path
        else:
            results_dir = os.path.dirname(model_path)
            model_files = [os.path.basename(model_path)]
    else:
        path = glob.glob(args.results_dir, recursive=True)
        if len(path) != 1:
            print(f"Multiple paths found for {args.results_dir}: {path}")
            return
        results_dir = path[0]
    
    # 1. Parse configuration from path
    config = parse_results_dir(results_dir)
    print("\nInferred configuration from path:")
    for k, v in config.items():
        print(f"  {k}: {v}")
        
    # 2. Reconstruct FeatureConfig
    use_semantic = "semantic" in config['feature_config_name']
    use_positional = "positional" in config['feature_config_name']
    use_token = "sintactic" in config['feature_config_name']
    
    feature_config = FeatureConfig(
        use_semantic=use_semantic,
        use_positional_features=use_positional,
        use_token_features=use_token,
        edge_ablation=config['edge_ablation']
    )
    
    print(f"Reconstructed FeatureConfig: {feature_config.get_name()}")
    
    # 3. Prepare datasets
    # Original test set (validation)
    _, test_csv = split_and_save_data(
        args.malware_csv, args.clean_csv,
        seed=config['seed'], test_size=config['test_size'],
        cache_dir=args.cache_dir
    )
    
    # Future test set
    future_test_csv = _prepare_combined_csv(
        args.test_malware_csv, args.test_clean_csv,
        cache_dir=args.cache_dir
    )
    
    embedder = StaticEmbedder(config['embedder_model']) if use_semantic else None
    
    print("\nLoading Original Validation/Test Dataset...")
    test_ds = CommandLineGraphDataset(
        data_csv=test_csv,
        embedder=embedder,
        feature_config=feature_config,
        cache_dir=args.cache_dir,
        semantic_model_name=config['embedder_model']
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    print("\nLoading Future Test Dataset...")
    future_ds = CommandLineGraphDataset(
        data_csv=future_test_csv,
        embedder=embedder,
        feature_config=feature_config,
        cache_dir=args.cache_dir,
        semantic_model_name=config['embedder_model']
    )
    future_loader = DataLoader(future_ds, batch_size=args.batch_size, shuffle=False)
    
    input_dim = test_ds.get(0).x.shape[1]
    num_edge_types = feature_config.get_num_edge_types()
    
    # 4. Find all models in results_dir
    if model_files is None:
        model_files = [f for f in os.listdir(results_dir) if f.endswith('_model.pt')]
        if args.model_type != 'all':
            model_files = [f for f in model_files if f.lower().startswith(args.model_type.lower())]
    elif args.model_type != 'all':
        if not model_files[0].lower().startswith(args.model_type.lower()):
            parser.error("--model-type does not match the provided --model-path")
    if not model_files:
        print(f"No model files found in {results_dir}")
        return
        
    print(f"\nFound {len(model_files)} models to evaluate.")
    
    for model_file in model_files:
        print(f"\n{'='*80}")
        print(f"Evaluating Model: {model_file}")
        print(f"{'='*80}")
        
        base_name = model_file.replace('_model.pt', '')
        
        model_name = None
        for known_model in ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'RGCN']:
            if base_name.startswith(known_model):
                model_name = known_model
                break
                
        if not model_name:
            print(f"Could not infer model architecture from filename {model_file}. Skipping.")
            continue
            
        model_class = get_model_class(model_name)
        kwargs = {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
        }
        if model_class == RGCNModel:
            kwargs["num_relations"] = num_edge_types
            
        model = model_class(**kwargs).to(device)
        model_path = os.path.join(results_dir, model_file)
        
        # Load weights safely
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Evaluate on Original Test Set
        print("\n--- Original Validation/Test Set ---")
        y_pred, y_true = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(metrics, title=f"Original Test Results: {base_name}")
        
        # Evaluate on Future Test Set
        print("\n--- Future Test Set ---")
        fy_pred, fy_true = evaluate(model, future_loader, device)
        fmetrics = compute_metrics(fy_true, fy_pred)
        print_metrics(fmetrics, title=f"Future Test Results: {base_name}")
        
        os.makedirs(os.path.join(results_dir, "inference"), exist_ok=True)
        # Save confusion matrices for future test set
        plot_confusion_matrix(
            fy_true, fy_pred,
            save_path=os.path.join(results_dir, "inference", f'{base_name}_future_confusion.png'),
            title=f'Future Confusion Matrix: {base_name}',
            metrics=fmetrics
        )
        
        # Save FP/FN for future test set
        fp_fn_path = os.path.join(results_dir, "inference", f'{base_name}_future_fp_fn.txt')
        _write_fp_fn(fy_true, fy_pred, future_test_csv, fp_fn_path)
        
    print("\nInference complete!")

if __name__ == "__main__":
    main()


# new_results/20260310_195000/0.3/seed3/semantic_only/all/minishlab_potion-base-2M/semantic_only