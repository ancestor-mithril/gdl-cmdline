"""
GNN Inference Script with Detailed Timing

Loads a trained GNN model and runs inference over a CSV file containing a
``command_line`` column (and optionally a ``label`` column). The CSV can be a
``malware_unique.csv`` style file (raw rows) or a ``train.csv`` style file
(rows with labels). Preprocessing is performed on the fly so we can measure
true raw-string-to-prediction latency.

The script measures, separately and combined:
  - Preprocessing time (raw command_line string -> PyG Data object)
  - Model inference time  (Data object -> prediction, including batching,
    transfer to device, forward pass, and rule-label override)
  - Combined end-to-end time

For each stage we report the total wall clock time and the mean per
command line.

Within a batch, preprocessing is done sequentially (one command line at a
time, as it would be when serving live traffic). Only the model forward is
batched.
"""

import argparse
import glob
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from gdl_cmdline.preprocess.tokenize import parse_windows_cmdline

from gdl_cmdline.scripts.train_gnn import (
    CommandLineGraphDataset,
    FeatureConfig,
    GATModel,
    GCNModel,
    GINModel,
    GraphSAGEModel,
    RGCNModel,
    StaticEmbedder,
    command_line_to_graph,
)

MODEL_CLASSES = {
    'GCN': GCNModel,
    'GraphSAGE': GraphSAGEModel,
    'GIN': GINModel,
    'GAT': GATModel,
    'RGCN': RGCNModel,
}

# Order matters: longer / more specific names first to avoid GraphSAGE matching "G..."
_MODEL_NAME_ORDER = ['GraphSAGE', 'RGCN', 'GCN', 'GIN', 'GAT']


# =============================================================================
# Configuration parsing from model path
# =============================================================================

def _split_path(path: str) -> list[str]:
    normalized = os.path.normpath(path).replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def _find_seed_index(parts: list[str]) -> int:
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            return i
    return -1


def parse_model_path(model_path: str) -> dict:
    """
    Parse a model path of the form
        .../{timestamp}/{test_size}/seed{seed}/{ft_label}/{ea_label}/{embedder_tag}/{feature_config_dir}/{MODEL}_{LOSS}_model.pt
    and return everything we need to rebuild the model and its preprocessing.
    """
    parts = _split_path(model_path)

    seed_idx = _find_seed_index(parts)
    if seed_idx == -1:
        raise ValueError(f"Could not find 'seedX' segment in path {model_path}")
    if seed_idx + 5 >= len(parts):
        raise ValueError(
            f"Path is too short. Expected "
            ".../seedX/<ft>/<ea>/<embedder>/<feature_config>/MODEL_loss_model.pt"
        )

    seed = int(parts[seed_idx][4:])
    try:
        test_size = float(parts[seed_idx - 1])
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse test_size from path {model_path}")
    ft_label = parts[seed_idx + 1]
    edge_ablation_label = parts[seed_idx + 2]
    embedder_tag = parts[seed_idx + 3]
    feature_config_dir = parts[seed_idx + 4]
    model_filename = parts[seed_idx + 5]

    # Embedder: minishlab_potion-base-2M -> minishlab/potion-base-2M
    if "_" in embedder_tag and not embedder_tag.startswith("minishlab/"):
        embedder_model = embedder_tag.replace('_', '/', 1)
    else:
        embedder_model = embedder_tag

    # The directory tells us BOTH the feature flags and the actual edge
    # ablation used for this specific model run.
    # Example: "semantic_only", "semantic_positional",
    # "semantic_only_edges_self_loops_only", "no_features_edges_self_loops_only".
    if "_edges_" in feature_config_dir:
        feat_part, edge_ablation = feature_config_dir.split("_edges_", 1)
    else:
        feat_part = feature_config_dir
        edge_ablation = 'all'

    use_semantic = "semantic" in feat_part
    use_positional = "positional" in feat_part
    use_token = "sintactic" in feat_part

    # Model file: {Model}_{LossType}_model.pt
    base = model_filename.replace('_model.pt', '')
    model_name = next((n for n in _MODEL_NAME_ORDER if base.startswith(n)), None)
    if model_name is None:
        raise ValueError(f"Could not infer model architecture from {model_filename}")
    loss_type = base[len(model_name) + 1:] if len(base) > len(model_name) else ''

    return {
        'seed': seed,
        'test_size': test_size,
        'ft_label': ft_label,
        'edge_ablation_label': edge_ablation_label,
        'embedder_model': embedder_model,
        'use_semantic': use_semantic,
        'use_positional': use_positional,
        'use_token': use_token,
        'edge_ablation': edge_ablation,
        'model_name': model_name,
        'loss_type': loss_type,
        'feature_config_dir': feature_config_dir,
    }


# =============================================================================
# On-the-fly preprocessor
# =============================================================================

class GraphPreprocessor:
    """
    Lightweight wrapper around the preprocessing methods of
    ``CommandLineGraphDataset`` so we can convert a raw string to a PyG
    ``Data`` object without instantiating the full dataset (which would
    parse / cache the entire CSV).
    """

    PATH_PATTERNS = CommandLineGraphDataset.PATH_PATTERNS
    HASH_PATTERNS = CommandLineGraphDataset.HASH_PATTERNS

    _compute_degree_features = CommandLineGraphDataset._compute_degree_features
    _compute_token_features = CommandLineGraphDataset._compute_token_features
    _compute_positional_features = CommandLineGraphDataset._compute_positional_features
    _graph_to_pyg = CommandLineGraphDataset._graph_to_pyg

    def __init__(self, feature_config: FeatureConfig,
                 embedder: Optional[StaticEmbedder]):
        self.feature_config = feature_config
        self.embedder = embedder

    def process_one(self, raw_cmd: str) -> Optional[Data]:
        """Convert a single raw command-line string to a PyG ``Data`` object.

        Returns ``None`` if the string cannot be parsed (matches the same
        filtering done during training).
        """
        try:
            tokens = parse_windows_cmdline(raw_cmd)
            if not tokens:
                return None
            cmd_tuple = tuple(tokens)
        except Exception:
            return None

        if isinstance(cmd_tuple, tuple) and len(cmd_tuple) == 1:
            cmd_tuple = cmd_tuple[0]

        graph = command_line_to_graph(cmd_tuple)
        node_values = graph.get_node_values()

        token_to_embedding = {}
        if self.feature_config.use_semantic:
            unique_tokens = list(set(node_values))
            if unique_tokens:
                embs = self.embedder.embed_texts(unique_tokens)
                for token, emb in zip(unique_tokens, embs):
                    token_to_embedding[token] = emb

        # Pass 0 as a dummy label since _graph_to_pyg expects one
        data = self._graph_to_pyg(graph, node_values, 0, token_to_embedding)
        return data


# =============================================================================
# Model construction helpers
# =============================================================================

def compute_input_dim(feature_config: FeatureConfig,
                      embedder: Optional[StaticEmbedder]) -> int:
    """Compute the input feature dimension based on the feature config."""
    dim = 0
    if feature_config.use_semantic:
        if embedder is None:
            raise ValueError("Semantic features requested but embedder is None")
        dim += embedder.embedding_dim
    if feature_config.use_token_features:
        dim += 15  # see CommandLineGraphDataset._compute_token_features
    if feature_config.use_positional_features:
        dim += 4 + 4  # positional (4) + degree (4)
    return max(dim, 1)


def build_model(model_name: str,
                feature_config: FeatureConfig,
                input_dim: int,
                hidden_dim: int,
                num_layers: int) -> torch.nn.Module:
    cls = MODEL_CLASSES[model_name]
    kwargs = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
    }
    if cls is RGCNModel:
        kwargs['num_relations'] = feature_config.get_num_edge_types()
    return cls(**kwargs)


# =============================================================================
# Timing helpers
# =============================================================================

def _sync(device: torch.device):
    """Synchronize the device so wall-clock measurements are accurate."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _format_row(name: str, total_s: float, cl_times: List[float], batch_times: List[float]) -> str:
    n_cl = len(cl_times)
    n_batches = len(batch_times)
    
    mean_cl_ms = (np.mean(cl_times) * 1000.0) if n_cl > 0 else 0.0
    std_cl_ms = (np.std(cl_times) * 1000.0) if n_cl > 0 else 0.0
    
    mean_batch_ms = (np.mean(batch_times) * 1000.0) if n_batches > 0 else 0.0
    std_batch_ms = (np.std(batch_times) * 1000.0) if n_batches > 0 else 0.0
    
    cl_str = f"{mean_cl_ms:.4f} ± {std_cl_ms:.4f}" if n_cl > 0 else "N/A"
    batch_str = f"{mean_batch_ms:.4f} ± {std_batch_ms:.4f}" if n_batches > 0 else "N/A"
    
    return f"{name:<18}{total_s:>14.4f}  {cl_str:>22}  {batch_str:>24}"


# =============================================================================
# Main
# =============================================================================

def _select_model_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    model_glob = os.path.join(args.results_root, "**", "*_model.pt")
    candidates = sorted(glob.glob(model_glob, recursive=True))
    if not candidates:
        parser.error(f"No model files found under {args.results_root}.")

    matches: list[str] = []
    for path in candidates:
        try:
            cfg = parse_model_path(path)
        except ValueError:
            continue

        if args.seed is not None and cfg["seed"] != args.seed:
            continue
        if args.test_size is not None and cfg["test_size"] != args.test_size:
            continue
        if args.feature_type_label and cfg["ft_label"] != args.feature_type_label:
            continue
        if args.edge_ablation_label and cfg["edge_ablation_label"] != args.edge_ablation_label:
            continue
        if args.embedder and cfg["embedder_model"] != args.embedder:
            continue
        if args.feature_config and cfg["feature_config_dir"] not in args.feature_config:
            continue
        if args.model:
            if not any(cfg["model_name"].lower().startswith(m.lower()) for m in args.model):
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
    parser = argparse.ArgumentParser(
        description='GNN inference with preprocessing/model/combined timing.'
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--model-path', type=str,
                              help='Path to a trained .pt model. Example: '
                                   'new_results/20260311_102939/0.3/seed7/semantic_only/all/'
                                   'minishlab_potion-base-2M/semantic_only/RGCN_standard_model.pt')
    source_group.add_argument('--results-root', type=str,
                              help='Root directory of GNN results to search.')
    parser.add_argument('--feature-config', type=str, nargs='+', default=None)
    parser.add_argument('--edge-ablation-label', type=str, default=None)
    parser.add_argument('--feature-type-label', type=str, default=None)
    parser.add_argument('--embedder', type=str, default=None)
    parser.add_argument('--model', type=str, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test-size', type=float, default=None)
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to a CSV with `command_line` and optional `label` column.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for model inference (1-64).')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (must match training).')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers (must match training).')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device override (auto-detect by default).')
    parser.add_argument('--limit', type=int, default=6400 * 2,
                        help='Optional: process only the first N rows.')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warm-up batches whose timing is discarded '
                             '(useful for stable CUDA measurements). Default: 2.')
    args = parser.parse_args()

    if not (1 <= args.batch_size <= 64):
        parser.error("--batch-size must be between 1 and 64")

    # ---- Device ----
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[Inference] Device: {device}")

    if args.results_root:
        args.model_path = _select_model_path(args, parser)

    expanded = glob.glob(args.model_path)
    if len(expanded) == 1:
        args.model_path = expanded[0]
    elif len(expanded) > 1:
        parser.error(f"--model-path matched multiple files: {expanded}")

    if os.path.isdir(args.model_path):
        model_files = [
            f for f in os.listdir(args.model_path) if f.endswith("_model.pt")
        ]
        if len(model_files) == 1:
            args.model_path = os.path.join(args.model_path, model_files[0])
        elif len(model_files) > 1:
            parser.error(
                f"--model-path points to a directory with multiple models: {model_files}"
            )
        else:
            parser.error("--model-path directory contains no *_model.pt files.")

    # ---- Parse model path ----
    cfg = parse_model_path(args.model_path)
    print("\n[Inference] Configuration parsed from model path:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    feature_config = FeatureConfig(
        use_semantic=cfg['use_semantic'],
        use_positional_features=cfg['use_positional'],
        use_token_features=cfg['use_token'],
        edge_ablation=cfg['edge_ablation'],
    )
    print(f"[Inference] Feature config: {feature_config.get_name()}  "
          f"({feature_config.get_num_edge_types()} edge types)")

    # ---- Embedder + model ----
    embedder = StaticEmbedder(cfg['embedder_model']) if cfg['use_semantic'] else None
    input_dim = compute_input_dim(feature_config, embedder)
    print(f"[Inference] Input feature dim: {input_dim}")

    model = build_model(
        cfg['model_name'], feature_config, input_dim,
        args.hidden_dim, args.num_layers,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Inference] Loaded {cfg['model_name']} ({n_params:,} params) "
          f"from {args.model_path}")

    # ---- Read CSV ----
    df = pd.read_csv(args.csv)
    if 'command_line' not in df.columns:
        raise ValueError(f"CSV {args.csv} must contain a 'command_line' column")
    if args.limit is not None:
        df = df.head(args.limit)
    raw_cmds: List[str] = df['command_line'].tolist()
    del df
    print(f"\n[Inference] Loaded {len(raw_cmds)} rows from {args.csv}")

    preproc = GraphPreprocessor(feature_config, embedder)

    # ---- Inference loop with detailed timing ----
    bs = args.batch_size
    n_batches = (len(raw_cmds) + bs - 1) // bs
    warmup_batches = min(args.warmup, max(0, n_batches - 1))

    total_preprocess_time = 0.0
    total_model_time = 0.0
    n_processed = 0
    n_skipped = 0
    n_warmup_done = 0
    n_batches_processed = 0

    pp_times_per_cl: List[float] = []
    pp_times_per_batch: List[float] = []
    md_times_per_cl: List[float] = []
    md_times_per_batch: List[float] = []

    pbar = tqdm(range(n_batches), desc=f"Inference (bs={bs})")

    with torch.inference_mode():
        for b in pbar:
            start = b * bs
            end = min(start + bs, len(raw_cmds))

            # ---------- Preprocessing (sequential, on the fly) ----------
            _sync(device)
            t_pp_start = time.perf_counter()

            data_list: List[Data] = []
            for i in range(start, end):
                t_cl_start = time.perf_counter()
                d = preproc.process_one(raw_cmds[i])
                t_cl_end = time.perf_counter()
                
                if d is None:
                    continue
                    
                data_list.append(d)
                
                if n_warmup_done >= warmup_batches:
                    pp_times_per_cl.append(t_cl_end - t_cl_start)

            t_pp_end = time.perf_counter()
            pp_elapsed = t_pp_end - t_pp_start
            n_skipped += (end - start) - len(data_list)

            if not data_list:
                continue

            # ---------- Model inference (batched) ----------
            _sync(device)
            t_md_start = time.perf_counter()

            batch = Batch.from_data_list(data_list).to(device)
            _ = model(batch).argmax(dim=1).cpu()

            _sync(device)
            t_md_end = time.perf_counter()
            md_elapsed = t_md_end - t_md_start

            # Discard timing for warm-up batches
            if n_warmup_done < warmup_batches:
                n_warmup_done += 1
            else:
                total_preprocess_time += pp_elapsed
                total_model_time += md_elapsed
                n_processed += len(data_list)
                n_batches_processed += 1
                
                pp_times_per_batch.append(pp_elapsed)
                md_times_per_batch.append(md_elapsed)
                md_times_per_cl.extend([md_elapsed / len(data_list)] * len(data_list))

            if n_processed > 0:
                pbar.set_postfix(
                    proc=n_processed,
                    pp_ms=f"{total_preprocess_time / n_processed * 1000:.2f}",
                    md_ms=f"{total_model_time / n_processed * 1000:.2f}",
                )

    # ---- Summary ----
    print("\n" + "=" * 82)
    print("INFERENCE TIMING SUMMARY")
    print("=" * 82)
    print(f"CSV file:                     {args.csv}")
    print(f"Model:                        {cfg['model_name']} ({cfg['loss_type']})")
    print(f"Batch size:                   {bs}")
    print(f"Warm-up batches discarded:    {n_warmup_done}")
    print(f"Command lines timed:          {n_processed}")
    print(f"Skipped (unparseable):        {n_skipped}")
    print()
    print(f"{'Stage':<18}{'Total (s)':>14}{'Per CL (ms) ± StdDev':>24}{'Per Batch (ms) ± StdDev':>26}")
    print("-" * 82)
    print(_format_row('Preprocessing', total_preprocess_time, pp_times_per_cl, pp_times_per_batch))
    print(_format_row('Model inference', total_model_time, md_times_per_cl, md_times_per_batch))
    
    comb_cl_times = [p + m for p, m in zip(pp_times_per_cl, md_times_per_cl)]
    comb_batch_times = [p + m for p, m in zip(pp_times_per_batch, md_times_per_batch)]
    print(_format_row('Combined',
                     total_preprocess_time + total_model_time, comb_cl_times, comb_batch_times))
    print("=" * 82)


if __name__ == "__main__":
    main()
