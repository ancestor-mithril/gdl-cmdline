"""
Count learnable / frozen / total parameters for all model configurations.

Covers:
- BERTClassifier: frozen BERT backbone + trainable head
- LSTMClassifier: trainable embedding + 2-layer bidirectional LSTM + head
- StaticModelClassifier: frozen Model2Vec embedder + trainable head
- GNN variants (GCN, GraphSAGE, GIN, GAT, RGCN) across feature configs

No data preprocessing required.

Usage:
    python graph/count_params.py
    python graph/count_params.py --hidden-dim 256 --num-layers 4
    python graph/count_params.py --model gin --feature-type semantic_only positional_only
    python graph/count_params.py --model lstm
    python graph/count_params.py --model bert lstm
"""

import argparse
from collections import OrderedDict

import numpy as np
import torch.nn as nn
from model2vec import StaticModel

from gdl_cmdline.scripts.train_gnn import (
    GCNModel,
    GraphSAGEModel,
    GINModel,
    GATModel,
    RGCNModel,
    FEATURE_TYPE_MAP,
    EDGE_ABLATION_MAP,
)
from gdl_cmdline.scripts.train_bert import (
    BERTClassifier,
    LSTMClassifier,
    StaticModelClassifier,
)


SEMANTIC_DIM = 64
POSITIONAL_DIM = 4
DEGREE_DIM = 4
TOKEN_DIM = 15
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
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lstm-embedding-size", type=int, default=64)
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["all"],
        choices=[
            "gcn",
            "graphsage",
            "gin",
            "gat",
            "rgcn",
            "bert",
            "lstm",
            "static",
            "all",
        ],
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        nargs="+",
        default=["all"],
        choices=list(FEATURE_TYPE_MAP.keys()) + ["all"],
    )
    parser.add_argument(
        "--edge-ablation",
        type=str,
        default="all",
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
    include_lstm = "all" in args.model or "lstm" in args.model
    include_static = "all" in args.model or "static" in args.model
    selected_gnns = (
        all_gnn_models
        if "all" in args.model
        else OrderedDict(
            (key, all_gnn_models[key])
            for key in args.model
            if key in all_gnn_models
        )
    )

    selected_features = (
        list(FEATURE_TYPE_MAP.keys())
        if "all" in args.feature_type
        else args.feature_type
    )
    num_relations = len(EDGE_ABLATION_MAP[args.edge_ablation])

    # Load Model2Vec only when a selected configuration requires it.
    semantic_gnn_selected = bool(selected_gnns) and any(
        FEATURE_TYPE_MAP[feature_type][0] for feature_type in selected_features
    )
    needs_static_embedder = include_static or semantic_gnn_selected
    static_embedder_params = 0
    if needs_static_embedder:
        print(f"Loading {args.static_model} to count parameters...")
        static_embedder_params = get_static_model_params(args.static_model)
        print(
            f"StaticModel embedder: {static_embedder_params:,} params "
            f"(embedding dim={SEMANTIC_DIM})"
        )

    sep = "=" * 112
    thin_sep = "-" * 112
    header = (
        f"{'Model':<18} {'Feature Config':<35} {'input':>5}  "
        f"{'Trainable':>14}  {'Frozen':>14}  {'Total':>14}"
    )

    print(
        f"\nSettings: hidden_dim={args.hidden_dim}, "
        f"num_layers={args.num_layers}, "
        f"edge_ablation={args.edge_ablation} ({num_relations} rel), "
        f"embedder={args.static_model}"
    )
    print(sep)
    print(header)
    print(sep)

    if include_bert:
        model = BERTClassifier(
            model_name=args.bert_model,
            num_classes=args.num_classes,
            dropout=args.dropout,
            max_length=args.max_length,
        )
        trainable, frozen = count_parameters(model)
        total = trainable + frozen
        bert_hidden_size = model.bert.config.hidden_size
        print(
            f"{model.name:<18} {'(frozen backbone, head trainable)':<35} "
            f"{bert_hidden_size:>5}  {fmt(trainable)}  {fmt(frozen)}  {fmt(total)}"
        )
        del model
        print(thin_sep)

    if include_lstm:
        model = LSTMClassifier(
            tokenizer_name=args.bert_model,
            num_classes=args.num_classes,
            embedding_size=args.lstm_embedding_size,
            hidden_size=64,
            num_layers=2,
            dropout=args.dropout,
            max_length=args.max_length,
        )
        trainable, frozen = count_parameters(model)
        total = trainable + frozen
        print(
            f"{model.name:<18} {'(2-layer bidirectional, all trainable)':<35} "
            f"{args.lstm_embedding_size:>5}  "
            f"{fmt(trainable)}  {fmt(frozen)}  {fmt(total)}"
        )
        del model
        print(thin_sep)

    if include_static:
        model = StaticModelClassifier(
            embedder_model=args.static_model,
            num_classes=args.num_classes,
            dropout=args.dropout,
        )
        trainable, frozen = count_parameters(model)
        frozen += static_embedder_params
        total = trainable + frozen
        print(
            f"{model.name:<18} {'(frozen embedder, head trainable)':<35} "
            f"{SEMANTIC_DIM:>5}  {fmt(trainable)}  {fmt(frozen)}  {fmt(total)}"
        )
        del model
        print(thin_sep)

    for feature_type in selected_features:
        use_semantic, use_positional, use_token = FEATURE_TYPE_MAP[feature_type]
        input_dim = input_dim_for(use_semantic, use_positional, use_token)

        for _, model_class in selected_gnns.items():
            kwargs = dict(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_classes=args.num_classes,
                dropout=args.dropout,
                num_layers=args.num_layers,
            )
            if model_class is RGCNModel:
                kwargs["num_relations"] = num_relations

            model = model_class(**kwargs)
            trainable, frozen = count_parameters(model)
            if use_semantic:
                frozen += static_embedder_params
            total = trainable + frozen
            print(
                f"{model.name:<18} {feature_type:<35} {input_dim:>5}  "
                f"{fmt(trainable)}  {fmt(frozen)}  {fmt(total)}"
            )
            del model

        if selected_gnns:
            print(thin_sep)

    print()


if __name__ == "__main__":
    main()
