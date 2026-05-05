"""
Count learnable / frozen / total parameters for all model configurations.

Covers:
- BERTClassifier (from train_bert.py): frozen BERT backbone + trainable head
- GNN variants (GCN, GraphSAGE, GIN, GAT, RGCN) across all feature configs
  For semantic configs the StaticModel (model2vec) embedder params are included as frozen.

No data preprocessing required.

Usage:
    python graph/count_params.py
    python graph/count_params.py --hidden-dim 256 --num-layers 4
    python graph/count_params.py --model gin --feature-type semantic_only positional_only
"""

import argparse
from collections import OrderedDict

import numpy as np
import torch.nn as nn
from model2vec import StaticModel

from gdl_cmdline.scripts.train_gnn import (
    GCNModel, GraphSAGEModel, GINModel, GATModel, RGCNModel,
    FEATURE_TYPE_MAP, EDGE_ABLATION_MAP,
)
from gdl_cmdline.scripts.train_bert import BERTClassifier, StaticModelClassifier

SEMANTIC_DIM = 64   # potion-base-2M output dimension
POSITIONAL_DIM = 4  # depth, is_leaf, is_root, subtree_size
DEGREE_DIM = 4      # in_degree, out_degree, total_degree, normalized_degree
TOKEN_DIM = 15      # length, entropy, is_arg, is_extension, ...

STRUCT_DIM = POSITIONAL_DIM + DEGREE_DIM


def input_dim_for(use_semantic: bool, use_positional: bool, use_token: bool) -> int:
    dim = 0
    if use_semantic:
        dim += SEMANTIC_DIM
    if use_positional:
        dim += STRUCT_DIM
    if use_token:
        dim += TOKEN_DIM
    return dim if dim > 0 else 1


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, frozen) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def get_static_model_params(model_name: str = "minishlab/potion-base-2M") -> int:
    """Count parameters in the StaticModel used as the GNN embedder."""
    model = StaticModel.from_pretrained(model_name)
    # StaticModel is not an nn.Module -- its only parameter is the
    # static embedding matrix stored as a numpy array.
    embedding = model.embedding
    weights = model.weights
    total = int(np.prod(embedding.shape))
    if weights is not None:
        total += int(np.prod(weights.shape))
    del embedding, weights, model
    return total


def fmt(n: int) -> str:
    return f"{n:>14,}"


def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--static-model", type=str, default="minishlab/potion-base-2M")
    parser.add_argument(
        "--model", type=str, nargs="+", default=["all"],
        choices=["gcn", "graphsage", "gin", "gat", "rgcn", "bert", "static", "all"],
    )
    parser.add_argument(
        "--feature-type", type=str, nargs="+", default=["all"],
        choices=list(FEATURE_TYPE_MAP.keys()) + ["all"],
    )
    parser.add_argument(
        "--edge-ablation", type=str, default="all",
        choices=list(EDGE_ABLATION_MAP.keys()),
    )
    args = parser.parse_args()

    all_gnn_models = OrderedDict([
        ("gcn", GCNModel),
        ("graphsage", GraphSAGEModel),
        ("gin", GINModel),
        ("gat", GATModel),
        ("rgcn", RGCNModel),
    ])

    include_bert = "all" in args.model or "bert" in args.model
    include_static = "all" in args.model or "static" in args.model
    selected_gnns = (
        all_gnn_models
        if "all" in args.model
        else OrderedDict((k, all_gnn_models[k]) for k in args.model if k not in ("bert", "static"))
    )

    selected_features = (
        list(FEATURE_TYPE_MAP.keys())
        if "all" in args.feature_type
        else args.feature_type
    )

    num_relations = len(EDGE_ABLATION_MAP[args.edge_ablation])

    # Pre-compute StaticModel embedder param count (frozen overhead for semantic GNN configs)
    print(f"Loading {args.static_model} to count parameters...")
    static_embedder_params = get_static_model_params(args.static_model)
    print(f"StaticModel embedder: {static_embedder_params:,} params (embedding dim={SEMANTIC_DIM})")

    sep = "=" * 112
    thin_sep = "-" * 112
    hdr = (f"{'Model':<18} {'Feature Config':<35} {'input':>5}  "
           f"{'Trainable':>14}  {'Frozen':>14}  {'Total':>14}")

    print(f"\nSettings: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
          f"edge_ablation={args.edge_ablation} ({num_relations} rel), "
          f"embedder={args.static_model}")
    print(sep)
    print(hdr)
    print(sep)

    # --- BERTClassifier (standalone baseline, from train_bert.py) ---
    if include_bert:
        model = BERTClassifier(model_name=args.bert_model, num_classes=args.num_classes)
        trainable, frozen = count_parameters(model)
        total = trainable + frozen
        print(f"{model.name:<18} {'(full model, head trainable)':<35} {'768':>5}  "
              f"{fmt(trainable)}  {fmt(frozen)}  {fmt(total)}")
        del model
        print(thin_sep)

    # --- StaticModelClassifier (standalone baseline, from train_bert.py) ---
    if include_static:
        model = StaticModelClassifier(embedder_model=args.static_model, num_classes=args.num_classes)
        trainable, frozen = count_parameters(model)
        
        # The StaticModel embedder parameters are not registered as nn.Module parameters,
        # so we add them manually to the frozen count.
        frozen += static_embedder_params
        
        total = trainable + frozen
        print(f"{model.name:<18} {'(frozen embedder, head trainable)':<35} {str(SEMANTIC_DIM):>5}  "
              f"{fmt(trainable)}  {fmt(frozen)}  {fmt(total)}")
        del model
        print(thin_sep)

    # --- GNN models across feature configs ---
    for ft_name in selected_features:
        use_sem, use_pos, use_tok = FEATURE_TYPE_MAP[ft_name]
        in_dim = input_dim_for(use_sem, use_pos, use_tok)

        for model_key, model_cls in selected_gnns.items():
            kwargs = dict(
                input_dim=in_dim,
                hidden_dim=args.hidden_dim,
                num_classes=args.num_classes,
                dropout=args.dropout,
                num_layers=args.num_layers,
            )
            if model_cls is RGCNModel:
                kwargs["num_relations"] = num_relations

            model = model_cls(**kwargs)
            trainable, frozen = count_parameters(model)

            if use_sem:
                frozen += static_embedder_params

            total = trainable + frozen
            print(f"{model.name:<18} {ft_name:<35} {in_dim:>5}  "
                  f"{fmt(trainable)}  {fmt(frozen)}  {fmt(total)}")
            del model

        print(thin_sep)

    print()


if __name__ == "__main__":
    main()
