"""
Graph Neural Network Training for Malware Detection

This script trains various GNN architectures to classify command-line graphs as 
malware or clean.
"""

import ast
import gc
import os
import math
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from typing import Callable, List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
# PyTorch Geometric imports
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv, RGCNConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.utils import degree

# Static embeddings (model2vec)
from model2vec import StaticModel

# Sklearn for evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def self_eval(x):
    try:
        return eval(x)
    except:
        try:
            return ast.literal_eval(x)
        except:
            return None

class Node:
    """
    Represents a node in the command-line tree structure.
    
    Each node contains:
    - value: The string token (command name, argument, etc.)
    - children: List of child nodes (sub-arguments, nested commands)
    """
    def __init__(self, value: str):
        self.value = value
        self.children: List['Node'] = []
        
    def add_children(self, children: List['Node']):
        """Add a list of child nodes to this node."""
        self.children.extend(children)

    def get_full_tree(self) -> List['Node']:
        """
        Recursively collect all nodes in the subtree rooted at this node.
        Returns a flat list of all nodes for graph construction.
        """
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_full_tree())
        return nodes


class Graph:
    """
    Represents a bidirectional graph built from a command-line tree.
    
    Five edge types capture different structural relationships:
    
    Edge Type 1 (Parent→Child): Hierarchical containment
    Edge Type 2 (Sequential):   Argument ordering (parent→first child, sibling→sibling)
    Edge Type 3 (Child→Parent): Reverse of type 1
    Edge Type 4 (Reverse-Sequential): Reverse of type 2
    Edge Type 5 (Self-loops):   Identity connections for every node
    """
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node
        self.full_tree = parent_node.get_full_tree()
        
        node_to_idx = {node: idx for idx, node in enumerate(self.full_tree)}
        
        self.node_depths: Dict[Node, int] = {}
        self.node_subtree_sizes: Dict[Node, int] = {}
        self._compute_positional_features(parent_node, depth=0)
        
        # Edge Type 1: Parent-Child (hierarchical containment)
        self.edges_type_1: List[Tuple[int, int]] = []
        # Edge Type 3: Child-Parent (reverse of type 1)
        self.edges_type_3: List[Tuple[int, int]] = []
        for node in self.full_tree:
            parent_idx = node_to_idx[node]
            for child in node.children:
                child_idx = node_to_idx[child]
                self.edges_type_1.append((parent_idx, child_idx))
                self.edges_type_3.append((child_idx, parent_idx))
        
        # Edge Type 2: Sequential (argument ordering)
        self.edges_type_2: List[Tuple[int, int]] = []
        # Edge Type 4: Reverse-sequential
        self.edges_type_4: List[Tuple[int, int]] = []
        for node in self.full_tree:
            if len(node.children) == 0:
                continue
            parent_idx = node_to_idx[node]
            first_child_idx = node_to_idx[node.children[0]]
            self.edges_type_2.append((parent_idx, first_child_idx))
            self.edges_type_4.append((first_child_idx, parent_idx))
            for i in range(1, len(node.children)):
                prev_idx = node_to_idx[node.children[i-1]]
                curr_idx = node_to_idx[node.children[i]]
                self.edges_type_2.append((prev_idx, curr_idx))
                self.edges_type_4.append((curr_idx, prev_idx))

        # Edge Type 5: Self-loops
        self.edges_type_5: List[Tuple[int, int]] = []
        for i in range(len(self.full_tree)):
            self.edges_type_5.append((i, i))
    
    def _compute_positional_features(self, node: Node, depth: int) -> int:
        """
        Recursively compute depth and subtree size for each node.
        
        Args:
            node: Current node
            depth: Current depth from root
            
        Returns:
            Subtree size (including this node)
        """
        self.node_depths[node] = depth
        
        if len(node.children) == 0:
            # Leaf node
            self.node_subtree_sizes[node] = 1
            return 1
        
        # Compute subtree sizes for all children
        subtree_size = 1  # Count this node
        for child in node.children:
            subtree_size += self._compute_positional_features(child, depth + 1)
        
        self.node_subtree_sizes[node] = subtree_size
        return subtree_size
    
    def get_node_values(self) -> List[str]:
        """Return the string values of all nodes."""
        return [node.value for node in self.full_tree]
    
    def get_positional_features(self) -> Tuple[List[int], List[float], List[float], List[int]]:
        """
        Return positional features for all nodes.
        
        Returns:
            Tuple of (depths, is_leaf, is_root, subtree_sizes)
            - depths: List of integer depths from root
            - is_leaf: List of floats (1.0 if leaf, 0.0 otherwise)
            - is_root: List of floats (1.0 if root, 0.0 otherwise)
            - subtree_sizes: List of integer subtree sizes
        """
        depths = []
        is_leaf = []
        is_root = []
        subtree_sizes = []
        for node in self.full_tree:
            depths.append(self.node_depths[node])
            is_leaf.append(1.0 if len(node.children) == 0 else 0.0)
            is_root.append(1.0 if node == self.parent_node else 0.0)
            subtree_sizes.append(self.node_subtree_sizes[node])
        return depths, is_leaf, is_root, subtree_sizes
    
    def get_all_edges(self) -> Tuple[List[Tuple[int, int]], ...]:
        """Return all 5 edge type lists: parent-child, sequential, child-parent, reverse-sequential, self-loops."""
        return self.edges_type_1, self.edges_type_2, self.edges_type_3, self.edges_type_4, self.edges_type_5


def create_node_with_children(node_value: str, children: List[Node]) -> Node:
    """
    Create a node with children.
    """
    parent = Node(node_value)
    parent.add_children(children)
    return parent

def nested_tuple_to_tree(tuple_elements) -> List[Node]:
    """
    Convert nested tuple elements into a list of tree nodes.
    
    Handles both string elements (leaf nodes) and nested tuples (subtrees).
    """
    nodes = []
    for element in tuple_elements:
        if isinstance(element, str):
            nodes.append(Node(element))
        elif isinstance(element, tuple):
            result = nested_tuple_to_tree(element)
            if len(result) == 1:
                nodes.append(result[0])
            elif len(result) > 1:
                nodes.append(create_node_with_children("<subcommand>", result))
        else:
            raise ValueError(f"Invalid element type: {type(element)}")
    return nodes


def command_line_to_tree(command_line: tuple) -> Node:
    """
    Convert a command-line tuple representation to a tree structure.
    
    Args:
        command_line: Tuple representation of the command line
        
    Returns:
        Root node of the tree
    """
    if len(command_line) == 0:
        return Node("<empty>")
    
    if isinstance(command_line[0], str):
        # First element is the command name
        parent = Node(command_line[0])
        parent.add_children(nested_tuple_to_tree(command_line[1:]))
        return parent
    
    # Handle case where first element is also a tuple
    parent = Node("<root_command>")
    parent.add_children(nested_tuple_to_tree(command_line))
    return parent


def command_line_to_graph(command_line: tuple) -> Graph:
    """
    Full pipeline: command-line tuple → tree → graph (always bidirectional).
    
    Args:
        command_line: Tuple representation from CSV
        
    Returns:
        Graph object with nodes and edges
    """
    return Graph(command_line_to_tree(command_line))



def get_rule_label(command_line: tuple) -> int:
    """
    Apply rule-based logic to a command-line tuple to generate a label.
    
    Args:
        command_line: Tuple representation from CSV
        
    Returns:
        1 for malware, 0 for clean, -1 to fall back to the model's prediction.
    """
    for element in command_line:
        if isinstance(element, tuple):
            rule_label = get_rule_label(element)
            if rule_label != -1:
                return rule_label
        if element == "<payload>" or ".ps" in element:
            return 0
    return -1


class StaticEmbedder:
    """
    Extracts semantic embeddings for node text using a Model2Vec StaticModel.
    
    StaticModel stores a fixed embedding per subword token and computes
    sentence/phrase embeddings by averaging.  It is orders of magnitude
    faster and smaller than transformer-based encoders.
    
    Supported models and their output dimensions:
        minishlab/potion-base-2M  → 64
        minishlab/potion-base-4M  → 128
        minishlab/potion-base-8M  → 256
    """
    def __init__(self, model_name: str = 'minishlab/potion-base-2M'):
        print(f"[StaticEmbedder] Loading {model_name}...")
        self.model_name = model_name
        self.model = StaticModel.from_pretrained(model_name)
        self.embedding_dim = self.model.dim
        print(f"[StaticEmbedder] Loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            torch.Tensor of shape (len(texts), embedding_dim) in float16
        """
        embeddings_np = self.model.encode(texts)
        return torch.from_numpy(embeddings_np).to(torch.float32)



@dataclass
class FeatureConfig:
    """
    Configuration for which features to use in the model.
    
    This enables feature ablation experiments:
    - use_semantic: Include static embeddings (semantic understanding of tokens)
    - use_positional_features: Include graph-derived features:
          tree position (depth, is_leaf, is_root, subtree_size)
          + degree features (in_degree, out_degree, total_degree, normalized_degree)
    - use_token_features: Include syntactic features (length, entropy, is_arg, is_path, …)
    
    Edges are always bidirectional (5 edge types).
    
    Named configurations (--feature-type):
        positional_only              – positional (tree position + degree)
        syntactic_only               – token / syntactic
        positional_syntactic         – positional + token
        semantic_only                – static embeddings only
        semantic_syntactic           – static embeddings + token
        semantic_positional          – static embeddings + positional
        semantic_positional_syntactic– static embeddings + positional + token
    """
    use_semantic: bool = False            # Static model embeddings of node text
    use_token_features: bool = False      # Token length, entropy, is_arg, …
    use_positional_features: bool = False # Depth, is_leaf, is_root, subtree_size + degree features
    edge_ablation: str = 'all'            # Which edge types to keep (for ablation study)
    
    def get_name(self) -> str:
        """Generate a descriptive name for this configuration."""
        parts = []
        if self.use_semantic:
            parts.append("semantic")
        if self.use_positional_features:
            parts.append("positional")
        if self.use_token_features:
            parts.append("syntactic")
        
        if not parts:
            base_name = "no_features"
        elif len(parts) == 1:
            base_name = parts[0] + "_only"
        else:
            base_name = "_".join(parts)
        
        if self.edge_ablation != 'all':
            base_name += f"_edges_{self.edge_ablation}"
        
        return base_name
    
    def get_num_edge_types(self) -> int:
        """Return the number of edge types after ablation filtering."""
        return len(EDGE_ABLATION_MAP[self.edge_ablation])



EDGE_ABLATION_MAP = {
    'all':                      [0, 1, 2, 3, 4],
    'self_loops_only':          [4],
    'self_loops_parent_child':  [0, 2, 4],
    'self_loops_sequential':    [1, 3, 4],
}

EDGE_ABLATION_CHOICES = list(EDGE_ABLATION_MAP.keys())


def split_and_save_data(
    malware_csv: str,
    clean_csv: str,
    seed: int,
    test_size: float,
    cache_dir: str = './graph_cache',
) -> Tuple[str, str]:
    """
    Split malware + clean CSVs into stratified train/test files and save them.

    If the split files already exist on disk, returns their paths immediately
    without reading the source CSVs.  The filenames encode seed, test_size, 
    and a hash of the source paths so different configurations
    never collide.

    Returns:
        (train_csv_path, test_csv_path)
    """
    import hashlib

    os.makedirs(cache_dir, exist_ok=True)
    csv_hash = hashlib.md5(f"{malware_csv}_{clean_csv}".encode()).hexdigest()[:8]
    base = f"seed{seed}_split{test_size}_{csv_hash}"
    train_path = os.path.join(cache_dir, f"train_{base}.csv")
    test_path = os.path.join(cache_dir, f"test_{base}.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"[Split] Found existing split files:\n  train: {train_path}\n  test:  {test_path}")
        return train_path, test_path

    print(f"[Split] Reading source CSVs to create train/test split (seed={seed}, test_size={test_size})...")
    malware_df = pd.read_csv(malware_csv)
    clean_df = pd.read_csv(clean_csv)

    malware_df.drop(columns=malware_df.columns.difference(['command_line']), inplace=True)
    malware_df['label'] = 1
    clean_df.drop(columns=clean_df.columns.difference(['command_line']), inplace=True)
    clean_df['label'] = 0

    combined = pd.concat([malware_df, clean_df], ignore_index=True)
    del malware_df, clean_df

    train_df, test_df = train_test_split(
        combined,
        test_size=test_size,
        stratify=combined['label'],
        random_state=seed,
    )
    del combined

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    del train_df, test_df
    gc.collect()

    print(f"[Split] Saved:\n  train: {train_path}\n  test:  {test_path}")
    return train_path, test_path


def _prepare_combined_csv(
    malware_csv: str,
    clean_csv: str,
    cache_dir: str = './graph_cache',
) -> str:
    """
    Combine separate malware and clean CSVs into a single labeled CSV.

    Useful for held-out test sets supplied as two files.  The combined file
    is cached on disk so it is only created once.
    """
    import hashlib

    os.makedirs(cache_dir, exist_ok=True)
    csv_hash = hashlib.md5(f"{malware_csv}_{clean_csv}".encode()).hexdigest()[:8]
    combined_path = os.path.join(cache_dir, f"combined_{csv_hash}.csv")

    if os.path.exists(combined_path):
        print(f"[Combine] Found existing combined CSV: {combined_path}")
        return combined_path

    print(f"[Combine] Building combined CSV from:\n  malware: {malware_csv}\n  clean:   {clean_csv}")
    malware_df = pd.read_csv(malware_csv)
    clean_df = pd.read_csv(clean_csv)

    malware_df.drop(columns=malware_df.columns.difference(['command_line']), inplace=True)
    malware_df['label'] = 1
    clean_df.drop(columns=clean_df.columns.difference(['command_line']), inplace=True)
    clean_df['label'] = 0

    combined = pd.concat([malware_df, clean_df], ignore_index=True)
    combined.to_csv(combined_path, index=False)
    del malware_df, clean_df, combined
    gc.collect()
    print(f"[Combine] Saved combined CSV: {combined_path}")
    return combined_path


def upcast_graph(graph: Data) -> Data:
    kwargs = {}
    if hasattr(graph, 'rule_label'): kwargs['rule_label'] = graph.rule_label
    return Data(
        x=graph.x.to(torch.float32),
        edge_index=graph.edge_index.to(torch.long),
        edge_type=graph.edge_type.to(torch.long),
        y=graph.y,
        num_nodes=graph.num_nodes,
        **kwargs
    )

def downcast_graph(graph: Data) -> Data:
    kwargs = {}
    if hasattr(graph, 'rule_label'): kwargs['rule_label'] = graph.rule_label
    return Data(
        x=graph.x.to(torch.float16),
        edge_index=graph.edge_index.to(torch.int16),
        edge_type=graph.edge_type.to(torch.int16),
        y=graph.y,
        num_nodes=graph.num_nodes,
        **kwargs
    )


class CommandLineGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for command-line graphs.

    Accepts a single CSV file with ``command_line`` and ``label`` columns.
    Converts rows into PyG Data objects and caches them on disk via the
    native InMemoryDataset collate / save / load mechanism.
    If in_memory_only is True, graphs are generated on-the-fly during training.
    """

    def __init__(
        self,
        data_csv: str,
        embedder: Optional[StaticEmbedder] = None,
        feature_config: FeatureConfig = None,
        cache_dir: str = './graph_cache',
        force_reprocess: bool = False,
        semantic_model_name: str = '',
        in_memory_only: bool = False,
    ):
        self.in_memory_only = in_memory_only
        super().__init__(root=None)

        self.feature_config = feature_config or FeatureConfig()
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.data_csv = data_csv
        self.semantic_model_name = semantic_model_name
        if not in_memory_only:
            os.makedirs(cache_dir, exist_ok=True)

        print(f"[Dataset] CSV: {os.path.basename(data_csv)}")
        print(f"[Dataset] Feature config: {self.feature_config.get_name()}")
        print(f"[Dataset] Edge ablation: {self.feature_config.edge_ablation} "
              f"({self.feature_config.get_num_edge_types()} edge types)")

        cache_path = self._get_cache_path()
        print(f"[Dataset] Cache file: {os.path.basename(cache_path)}")

        if not in_memory_only and not force_reprocess and os.path.exists(cache_path):
            print(f"[Dataset] Loading from cache: {cache_path}")
            self.load(cache_path)
            self.labels = self._data.y.view(-1).numpy()
            print(f"[Dataset] Loaded {len(self)} graphs from cache")
            
            # Build rule labels on the fly for cached data
            print(f"[Dataset] Computing rule labels on the fly from {data_csv}...")
            df = pd.read_csv(data_csv)
            raw_command_lines = df['command_line'].values
            rule_labels = []
            for cmd in tqdm(raw_command_lines, desc="Computing rule labels"):
                cmd_tuple = self_eval(cmd)
                if cmd_tuple is None:
                    continue
                if isinstance(cmd_tuple, tuple) and len(cmd_tuple) == 1:
                    cmd_tuple = cmd_tuple[0]
                rule_labels.append(get_rule_label(cmd_tuple))
            del df
            
            self._data.rule_label = torch.tensor(rule_labels, dtype=torch.long)
            self.slices['rule_label'] = torch.arange(0, len(rule_labels) + 1, dtype=torch.long)
        else:
            print(f"[Dataset] Loading data from {data_csv}...")
            df = pd.read_csv(data_csv)
            raw_command_lines = df['command_line'].values
            raw_labels = df['label'].values
            del df
            print(f"[Dataset] Total samples: {len(raw_command_lines)}")

            # Filter valid command lines and compute rule labels
            self.command_lines = []
            self.labels_list = []
            self.rule_labels = []
            
            for idx, cmd in enumerate(tqdm(raw_command_lines, desc="Filtering and computing rule labels")):
                cmd_tuple = self_eval(cmd)
                if cmd_tuple is None:
                    continue
                if isinstance(cmd_tuple, tuple) and len(cmd_tuple) == 1:
                    cmd_tuple = cmd_tuple[0]
                
                self.command_lines.append(cmd_tuple)
                self.labels_list.append(raw_labels[idx])
                self.rule_labels.append(get_rule_label(cmd_tuple))
                
            self.labels = np.array(self.labels_list)
            
            if in_memory_only:
                print(f"[Dataset] Ready for on-the-fly generation of {len(self.command_lines)} graphs")
                if self.feature_config.use_semantic and self.embedder is None:
                    print(f"[Dataset] Instantiating embedder {self.semantic_model_name}...")
                    self.embedder = StaticEmbedder(self.semantic_model_name)
            else:
                data_list = self._build_graphs_from_tuples(self.command_lines, self.labels)
                self.save(data_list, cache_path)
                self.load(cache_path)
                print(f"[Dataset] Cached {len(self)} graphs")
                
                self._data.rule_label = torch.tensor(self.rule_labels, dtype=torch.long)
                self.slices['rule_label'] = torch.arange(0, len(self.rule_labels) + 1, dtype=torch.long)

    def len(self) -> int:
        if getattr(self, 'in_memory_only', False):
            return len(self.command_lines)
        return super().len()

    def get(self, idx: int) -> Data:
        if getattr(self, 'in_memory_only', False):
            return self._generate_graph_on_the_fly(idx)
        return super().get(idx)

    def _generate_graph_on_the_fly(self, idx: int) -> Data:
        cmd_tuple = self.command_lines[idx]
        label = self.labels_list[idx]
        rule_label = self.rule_labels[idx]
        
        graph = command_line_to_graph(cmd_tuple)
        node_values = graph.get_node_values()
        
        token_to_embedding = {}
        if self.feature_config.use_semantic:
            unique_tokens = list(set(node_values))
            if unique_tokens:
                embs = self.embedder.embed_texts(unique_tokens)
                for token, emb in zip(unique_tokens, embs):
                    token_to_embedding[token] = emb
                    
        data = self._graph_to_pyg(graph, node_values, label, token_to_embedding)
        data.rule_label = torch.tensor([rule_label], dtype=torch.long)
        return data

    @staticmethod
    def cache_exists(
        data_csv: str,
        feature_config: 'FeatureConfig' = None,
        cache_dir: str = './graph_cache',
        semantic_model_name: str = '',
    ) -> bool:
        """Check whether the cache file for this config already exists on disk."""
        import hashlib
        feature_config = feature_config or FeatureConfig()
        config_name = feature_config.get_name()
        csv_hash = hashlib.md5(data_csv.encode()).hexdigest()[:8]
        if feature_config.use_semantic and semantic_model_name:
            model_tag = semantic_model_name.replace('/', '_')
            filename = f"graphs_{config_name}_{model_tag}_{csv_hash}.pt"
        else:
            filename = f"graphs_{config_name}_{csv_hash}.pt"
        path = os.path.join(cache_dir, filename)
        return os.path.exists(path)

    def _get_cache_path(self) -> str:
        """
        Unique cache path based on the feature config, data CSV, and
        (when semantic features are enabled) the embedder model name.
        """
        import hashlib
        config_name = self.feature_config.get_name()
        csv_hash = hashlib.md5(self.data_csv.encode()).hexdigest()[:8]
        if self.feature_config.use_semantic and self.semantic_model_name:
            model_tag = self.semantic_model_name.replace('/', '_')
            filename = f"graphs_{config_name}_{model_tag}_{csv_hash}.pt"
        else:
            filename = f"graphs_{config_name}_{csv_hash}.pt"
        return os.path.join(self.cache_dir, filename)
    
    def _build_graphs_from_tuples(self, cmd_tuples, labels) -> List[Data]:
        """
        Build PyG Data objects from raw command-line tuples.

        Single-pass approach: for each command line, parse it into a Graph,
        embed any unseen tokens on the fly, convert to a PyG Data object,
        and discard the intermediate Graph immediately.
        """
        print(f"[Dataset] Preprocessing {len(cmd_tuples)} graphs...")

        use_semantic = self.feature_config.use_semantic
        token_to_embedding: Dict[str, torch.Tensor] = {}
        owns_embedder = False

        if use_semantic and self.embedder is None:
            print(f"[Dataset] Instantiating embedder {self.semantic_model_name}...")
            self.embedder = StaticEmbedder(self.semantic_model_name)
            owns_embedder = True

        data_list = []
        embed_batch_size = 512

        for idx in trange(len(cmd_tuples), desc="Building graphs"):
            cmd_tuple = cmd_tuples[idx]
            graph = command_line_to_graph(cmd_tuple)
            node_values = graph.get_node_values()

            if use_semantic:
                new_tokens = [t for t in node_values if t not in token_to_embedding]
                for i in range(0, len(new_tokens), embed_batch_size):
                    batch = new_tokens[i:i + embed_batch_size]
                    batch_embs = self.embedder.embed_texts(batch)
                    for token, emb in zip(batch, batch_embs):
                        token_to_embedding[token] = emb

            data_list.append(self._graph_to_pyg(graph, node_values, labels[idx], token_to_embedding))

        if use_semantic:
            print(f"[Dataset] Embedded {len(token_to_embedding)} unique tokens")
        if owns_embedder:
            del self.embedder
            self.embedder = None

        del token_to_embedding
        gc.collect()
        return data_list
    
    def _graph_to_pyg(
        self,
        graph: Graph,
        node_values: List[str],
        label: int,
        token_to_embedding: Dict[str, torch.Tensor],
    ) -> Data:
        """Convert a Graph into a PyG Data object using pre-computed embeddings."""
        num_nodes = len(graph.full_tree)
        edges = graph.get_all_edges()

        if self.feature_config.edge_ablation != 'all':
            keep_indices = EDGE_ABLATION_MAP[self.feature_config.edge_ablation]
            edges = tuple(edges[i] for i in keep_indices)

        all_edges = []
        edge_type = []
        for type_idx, edge in enumerate(edges):
            all_edges.extend(edge)
            edge_type.extend([type_idx] * len(edge))

        if len(all_edges) > 0:
            all_edges = list(zip(*all_edges))
            edge_index = torch.tensor(all_edges, dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        
        # ===== Build Node Features =====
        features_list = []
        
        # Semantic features: Use pre-computed embeddings from lookup
        if self.feature_config.use_semantic:
            embeddings = torch.stack([token_to_embedding[token] for token in node_values])
            features_list.append(embeddings)
        
        # Token features: length and entropy
        if self.feature_config.use_token_features:
            token_features = self._compute_token_features(node_values)
            features_list.append(token_features)
        
        # Positional features: tree position + degree
        if self.feature_config.use_positional_features:
            positional_features = self._compute_positional_features(graph)
            features_list.append(positional_features)
            degree_features = self._compute_degree_features(num_nodes, edge_index)
            features_list.append(degree_features)
        
        if features_list:
            x = torch.cat(features_list, dim=1)
        else:
            x = torch.ones(num_nodes, 1, dtype=torch.float32)

        # ===== Create PyG Data Object =====
        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=num_nodes
        )
    
    def _compute_degree_features(
        self,
        num_nodes: int,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structural degree features for each node.
        
        Features computed:
        1. In-degree: Number of incoming edges
        2. Out-degree: Number of outgoing edges  
        3. Total degree: In + Out
        4. Normalized degree: Total degree / max_degree
        
        These features capture the structural importance of each node
        without any semantic information.
        """
        out_degree = degree(edge_index[0], num_nodes=num_nodes)
        in_degree = degree(edge_index[1], num_nodes=num_nodes)
        
        # Compute derived features
        total_degree = in_degree + out_degree
        max_degree = total_degree.max().item()
        if max_degree == 0:
            max_degree = 1
        normalized_degree = total_degree / max_degree
        
        # Stack into feature matrix [num_nodes, 4]
        features = torch.stack([
            in_degree,
            out_degree,
            total_degree,
            normalized_degree
        ], dim=1).to(torch.float32)
        
        return features
    
    # Patterns for token type detection (normalized strings)
    PATH_PATTERNS = {'<path>', '<program_files>', '<Users>', '<Windows>', '<C>', '<tmp>'}
    HASH_PATTERNS = {'<key>', '<token>', '<hex_id>', '<client_id>', '<pass>', '<arn>', '<code>', '<md5>', '<SID>'}
    
    def _compute_token_features(self, node_values: List[str]) -> torch.Tensor:
        """
        Compute token-based features for each node.
        
        Features computed:
        1. Token length: Length of the string value (log-scaled)
        2. Entropy: Shannon entropy of character distribution
        3. Is path: Contains path-related patterns
        4. Is IP: Contains IP pattern
        5. Is PID: Contains PID pattern
        6. Is hash: Contains hash/secret-related patterns
        7. Is URL: Contains URL pattern
        8. Is date: Contains date/timestamp pattern
        
        These features capture token complexity and type without semantic embeddings.
        """
        features = []
        
        for value in node_values:
            # Token length (log-scaled to handle long strings)
            token_length = math.log1p(len(value))  # log(1 + len) for numerical stability
            
            # Shannon entropy of character distribution
            if len(value) == 0:
                entropy = 0.0
            else:
                char_counts = Counter(value)
                total_chars = len(value)
                entropy = 0.0
                for count in char_counts.values():
                    if count > 0:
                        prob = count / total_chars
                        entropy -= prob * math.log2(prob)


            is_arg = 1.0 if value.startswith("-") or value.startswith("/") else 0.0
            is_extension = 1.0 if value.startswith(".") and len(value) < 6 else 0.0
            is_exe = 1.0 if ".exe" in value else 0.0
            has_redirect_or_pipe = 1.0 if "|" in value or "<" in value and ">" not in value or ">" in value and "<" not in value or ">>" in value or "<<" in value or "&" in value else 0.0
            strange_count = math.log1p(sum(1 for char in value if char in '^`%$+&;'))
            quote_count = math.log1p(sum(1 for char in value if char in "\"'"))

            # Is path: contains path-related patterns
            if "<" in value and ">" in value:
                was_replaced = 1.0

                is_path = 1.0 if any(p in value for p in self.PATH_PATTERNS) else 0.0

                # Is IP: contains IP pattern
                is_ip = 1.0 if '_ip>' in value else 0.0

                # Is PID: contains PID pattern
                is_pid = 1.0 if '<PID>' in value else 0.0

                # Is hash: contains hash/secret-related patterns
                is_hash = 1.0 if any(p in value for p in self.HASH_PATTERNS) else 0.0

                # Is URL: contains URL pattern
                is_url = 1.0 if '<url>' in value else 0.0

                # Is date: contains date or timestamp pattern
                is_date = 1.0 if ('<date>' in value or '<timestamp>' in value) else 0.0
            else:
                was_replaced = 0.0
                is_path = 0.0
                is_ip = 0.0
                is_pid = 0.0
                is_hash = 0.0
                is_url = 0.0
                is_date = 0.0
            features.append([token_length, entropy,
                             is_arg, is_extension, is_exe, has_redirect_or_pipe, strange_count, quote_count,
                             was_replaced, is_path, is_ip, is_pid, is_hash, is_url, is_date])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_positional_features(self, graph: Graph) -> torch.Tensor:
        """
        Compute tree position features for each node.
        
        Features computed:
        1. Depth: Distance from root (normalized)
        2. Is leaf: 1 if node has no children, 0 otherwise
        3. Is root: 1 if node is root, 0 otherwise
        4. Subtree size: Number of nodes in subtree (log-scaled)
        
        These features capture the node's position in the tree structure.
        """
        depths, is_leaf, is_root, subtree_sizes = graph.get_positional_features()
        
        # Normalize depth by max depth in tree
        max_depth = max(depths) if depths else 1
        if max_depth == 0:
            max_depth = 1
        normalized_depths = [d / max_depth for d in depths]
        
        # Log-scale subtree sizes
        log_subtree_sizes = [math.log1p(s) for s in subtree_sizes]
                
        # Stack into feature matrix [num_nodes, 4]
        features = torch.tensor([
            [normalized_depths[i], is_leaf[i], is_root[i], log_subtree_sizes[i]]
            for i in range(len(depths))
        ], dtype=torch.float32)
        
        return features
    

class GNNBlock(nn.Module):
    """Single GNN layer: conv -> BN -> activation -> optional dropout."""
    
    def __init__(self, conv: nn.Module, hidden_dim: int,
                 activation=F.relu, dropout: float = 0.0):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = activation
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.drop(self.activation(self.bn(self.conv(x, edge_index))))


class RGCNBlock(nn.Module):
    """Single R-GCN layer: conv -> BN -> activation -> optional dropout."""
    
    def __init__(self, conv: nn.Module, hidden_dim: int,
                 activation=F.relu, dropout: float = 0.0):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = activation
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        return self.drop(self.activation(self.bn(self.conv(x, edge_index, edge_type))))


class GNNStack(nn.Module):
    """Sequential stack of GNN blocks, passing edge_index through all layers."""
    
    def __init__(self, *blocks: GNNBlock):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index)
        return x


class RGCNStack(nn.Module):
    """Sequential stack of R-GCN blocks, passing edge_index and edge_type through."""
    
    def __init__(self, *blocks: RGCNBlock):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_type)
        return x


class GINJKStack(nn.Module):
    """
    Stack of GIN blocks with Jumping Knowledge.
    
    Returns the concatenation of sum-pooled outputs from every layer,
    preserving multi-scale structural information.
    """
    
    def __init__(self, *blocks: GNNBlock):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        layer_pools = []
        for block in self.blocks:
            x = block(x, edge_index)
            layer_pools.append(global_add_pool(x, batch))
        return torch.cat(layer_pools, dim=1)



def _make_classifier(hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def _pool_mean_max(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Concatenate global mean and max pooling."""
    return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)



class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) for graph classification.
    
    GCN uses spectral convolutions that aggregate information from 
    neighboring nodes. Good for capturing local graph structure.
    
    Architecture:
    - N GCN convolutional layers with ReLU and dropout (configurable via num_layers)
    - Global mean + max pooling to create graph-level representation
    - 2-layer MLP classifier
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2,
                 dropout: float = 0.5, num_layers: int = 3):
        super().__init__()
        self.name = "GCN"
        
        if num_layers == 1:
            blocks = [GNNBlock(GCNConv(input_dim, hidden_dim), hidden_dim, dropout=0.0)]
        else:
            blocks = [GNNBlock(GCNConv(input_dim, hidden_dim), hidden_dim, dropout=dropout)]
            for _ in range(num_layers - 2):
                blocks.append(GNNBlock(GCNConv(hidden_dim, hidden_dim), hidden_dim, dropout=dropout))
            blocks.append(GNNBlock(GCNConv(hidden_dim, hidden_dim), hidden_dim, dropout=0.0))
        
        self.gnn_layers = GNNStack(*blocks)
        self.classifier = _make_classifier(hidden_dim, num_classes, dropout)
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.classifier(_pool_mean_max(self.gnn_layers(data.x, data.edge_index), data.batch))


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE (Sample and Aggregate) for graph classification.
    
    GraphSAGE learns aggregation functions over neighborhoods,
    making it more flexible than GCN. It's particularly good at
    generalizing to unseen graph structures.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2,
                 dropout: float = 0.5, num_layers: int = 3):
        super().__init__()
        self.name = "GraphSAGE"
        
        if num_layers == 1:
            blocks = [GNNBlock(SAGEConv(input_dim, hidden_dim), hidden_dim, dropout=0.0)]
        else:
            blocks = [GNNBlock(SAGEConv(input_dim, hidden_dim), hidden_dim, dropout=dropout)]
            for _ in range(num_layers - 2):
                blocks.append(GNNBlock(SAGEConv(hidden_dim, hidden_dim), hidden_dim, dropout=dropout))
            blocks.append(GNNBlock(SAGEConv(hidden_dim, hidden_dim), hidden_dim, dropout=0.0))
        
        self.gnn_layers = GNNStack(*blocks)
        self.classifier = _make_classifier(hidden_dim, num_classes, dropout)
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.classifier(_pool_mean_max(self.gnn_layers(data.x, data.edge_index), data.batch))


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph classification.
    
    GIN is provably as powerful as the Weisfeiler-Lehman test for
    distinguishing graph structures. Uses MLPs for aggregation and
    "jumping knowledge" (concatenation of all layer outputs) to
    preserve multi-scale structural information.
    
    This is likely the most important model for testing the
    "structure-only" hypothesis.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2,
                 dropout: float = 0.5, num_layers: int = 3):
        super().__init__()
        self.name = "GIN"
        
        def _gin_mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        if num_layers == 1:
            blocks = [GNNBlock(GINConv(_gin_mlp(input_dim)), hidden_dim, dropout=0.0)]
        else:
            blocks = [GNNBlock(GINConv(_gin_mlp(input_dim)), hidden_dim, dropout=dropout)]
            for _ in range(num_layers - 2):
                blocks.append(GNNBlock(GINConv(_gin_mlp(hidden_dim)), hidden_dim, dropout=dropout))
            blocks.append(GNNBlock(GINConv(_gin_mlp(hidden_dim)), hidden_dim, dropout=0.0))
        
        self.gnn_layers = GINJKStack(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.classifier(self.gnn_layers(data.x, data.edge_index, data.batch))


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for graph classification.
    
    GAT uses attention mechanisms to weight neighbor contributions
    differently. Multi-head attention provides multiple "perspectives"
    on which edges are important.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2, 
                 dropout: float = 0.5, heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.name = "GAT"
        
        if num_layers == 1:
            blocks = [GNNBlock(
                GATConv(input_dim, hidden_dim, heads=1, concat=False, dropout=dropout),
                hidden_dim, activation=F.elu, dropout=0.0
            )]
        else:
            blocks = [GNNBlock(
                GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout),
                hidden_dim, activation=F.elu, dropout=dropout
            )]
            for _ in range(num_layers - 2):
                blocks.append(GNNBlock(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout),
                    hidden_dim, activation=F.elu, dropout=dropout
                ))
            blocks.append(GNNBlock(
                GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout),
                hidden_dim, activation=F.elu, dropout=0.0
            ))
        
        self.gnn_layers = GNNStack(*blocks)
        self.classifier = _make_classifier(hidden_dim, num_classes, dropout)
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.classifier(_pool_mean_max(self.gnn_layers(data.x, data.edge_index), data.batch))


class RGCNModel(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) for graph classification.
    
    R-GCN is designed for multi-relational graphs where edges have different types.
    Each relation type has its own learnable weight matrix, allowing the model
    to learn different message passing functions for different edge semantics.
    
    In our case:
    - Edge type 0: Parent-child edges (hierarchical structure)
    - Edge type 1: Sequential edges (argument ordering)
    
    Reference: "Modeling Relational Data with Graph Convolutional Networks" 
               (Schlichtkrull et al., 2018)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2, 
                 dropout: float = 0.5, num_relations: int = 2, num_layers: int = 3):
        super().__init__()
        self.name = "RGCN"
        self.num_relations = num_relations

        rgcn_conv_class = RGCNConv
        
        if num_layers == 1:
            blocks = [RGCNBlock(
                rgcn_conv_class(input_dim, hidden_dim, num_relations=num_relations),
                hidden_dim, dropout=0.0
            )]
        else:
            blocks = [RGCNBlock(
                rgcn_conv_class(input_dim, hidden_dim, num_relations=num_relations),
                hidden_dim, dropout=dropout
            )]
            for _ in range(num_layers - 2):
                blocks.append(RGCNBlock(
                    rgcn_conv_class(hidden_dim, hidden_dim, num_relations=num_relations),
                    hidden_dim, dropout=dropout
                ))
            blocks.append(RGCNBlock(
                rgcn_conv_class(hidden_dim, hidden_dim, num_relations=num_relations),
                hidden_dim, dropout=0.0
            ))
        
        self.gnn_layers = RGCNStack(*blocks)
        self.classifier = _make_classifier(hidden_dim, num_classes, dropout)
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.classifier(_pool_mean_max(self.gnn_layers(data.x, data.edge_index, data.edge_type), data.batch))


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard ones.
    This is particularly useful for imbalanced datasets where the
    model might otherwise focus too much on the majority class.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma controls the focusing: higher gamma = more focus on hard examples
    - alpha balances the classes (similar to weighted CE)
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor (can be a tensor for per-class weights)
            gamma: Focusing parameter (0 = standard CE, 2 = typical for imbalanced)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from the model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability for the correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Gather the probabilities for the true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weighting to cross entropy
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(labels: np.ndarray, device: str) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    
    This helps the model pay more attention to the minority class (malware).
    
    weight_c = N / (n_classes * count_c)
    
    Args:
        labels: List of all labels in the dataset
        device: Device to place the tensor on
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    n_classes = len(class_counts)
    
    # Inverse frequency weighting
    weights = total / (n_classes * class_counts)

    assert len(weights) == 2
    
    print(f"[Class Weights] Class 0 (clean): {class_counts[0]} samples, weight: {weights[0]:.4f}")
    print(f"[Class Weights] Class 1 (malware): {class_counts[1]} samples, weight: {weights[1]:.4f}")
    
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: GNN model
        loader: Training data loader
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function
        device: 'cuda' or 'cpu'
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        # break
    
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Trained GNN model
        loader: Data loader (test or validation)
        device: 'cuda' or 'cpu'
        criterion: Optional loss function. When provided, also returns avg loss.
        
    Returns:
        (predictions, true_labels) when criterion is None
        (predictions, true_labels, avg_loss) when criterion is provided
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        out = model(batch)
                
        preds = out.argmax(dim=1)
        
        r_label = batch.rule_label.view(-1)
        rule_mask = r_label != -1
        preds[rule_mask] = r_label[rule_mask].to(preds.device)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        
        if criterion is not None:
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    if criterion is not None:
        avg_loss = total_loss / len(loader.dataset)
        return preds_arr, labels_arr, avg_loss
    return preds_arr, labels_arr


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Metrics computed:
    - Accuracy: Overall correctness
    - F1 Score: Harmonic mean of precision and recall (important for imbalanced)
    - Precision: Of predicted malware, how many are actually malware?
    - Recall: Of actual malware, how many did we catch?
    - ROC-AUC: Area under ROC curve (ranking quality)
    - Average Precision: Area under precision-recall curve
    
    For malware detection, recall is often prioritized (don't miss malware),
    but precision also matters (avoid false alarms).
    """
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_malware': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision_malware': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_malware': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'fpr': float(fp / (fp + tn)) if tn + fp != 0 else 0.0,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print('='*50)
    for name, value in metrics.items():
        print(f"  {name:25s}: {value:.4f}")
    print('='*50)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    title: str = "Confusion Matrix",
    metrics: Dict[str, float] = None,
    experiment_params: Dict = None,
):
    """
    Plot and optionally save a confusion matrix.
    
    The confusion matrix shows:
    - True Negatives (clean correctly identified)
    - False Positives (clean misclassified as malware)
    - False Negatives (malware missed - very bad!)
    - True Positives (malware correctly detected)
    
    When *experiment_params* is provided the dict is appended to the text
    file so the saved artefact is self-contained and reproducible.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    cm_text_lines = [
        f"{title}",
        "-" * 50,
        "                    Predicted",
        "                    Clean    Malware",
        f"Actual  Clean      {tn:7d}   {fp:7d}",
        f"        Malware    {fn:7d}   {tp:7d}",
        "-" * 50,
        f"  True Negatives (TN):  {tn:7d}  (Clean correctly identified)",
        f"  False Positives (FP): {fp:7d}  (Clean misclassified as Malware)",
        f"  False Negatives (FN): {fn:7d}  (Malware missed)",
        f"  True Positives (TP):  {tp:7d}  (Malware correctly detected)",
        "-" * 50,
    ]
    
    if metrics:
        cm_text_lines.extend([
            "",
            "METRICS",
            "-" * 50,
            f"  Accuracy:           {metrics.get('accuracy', 0):.4f}",
            f"  F1 Score (Macro):   {metrics.get('f1_macro', 0):.4f}",
            f"  F1 Score (Malware): {metrics.get('f1_malware', 0):.4f}",
            f"  Precision (Malware):{metrics.get('precision_malware', 0):.4f}",
            f"  Recall (Malware):   {metrics.get('recall_malware', 0):.4f}",
            "-" * 50,
        ])

    if experiment_params:
        cm_text_lines.append(f"\nExperiment Parameters:\n{experiment_params}\n")
    
    cm_text = "\n".join(cm_text_lines)
    
    print(f"\n{cm_text}")
    
    if save_path:
        text_path = save_path.replace('.png', '.txt')
        with open(text_path, 'w') as f:
            f.write(cm_text)
        print(f"[Plot] Confusion matrix text saved to {text_path}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Clean', 'Malware'],
        yticklabels=['Clean', 'Malware']
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_metrics: List[Dict[str, float]],
    save_path: str = None,
    title: str = "Training Curves",
    test_losses: List[float] = None
):
    """
    Plot training/test loss and validation metrics over epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and test loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    if test_losses is not None:
        axes[0].plot(epochs, test_losses, 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train / Test Loss')
    axes[0].legend()
    
    # Plot F1 scores
    f1_macro = [m['f1_macro'] for m in val_metrics]
    f1_malware = [m['f1_malware'] for m in val_metrics]
    axes[1].plot(epochs, f1_macro, 'g-', label='F1 Macro')
    axes[1].plot(epochs, f1_malware, 'r-', label='F1 Malware')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Scores')
    axes[1].legend()
        
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Training curves saved to {save_path}")
    
    plt.close()



def _write_fp_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_csv: str,
    save_path: str,
):
    """
    Write False Positive and False Negative cases to a text file.

    Reads the test CSV lazily to look up only misclassified command lines.

    FP = clean sample predicted as malware  (true=0, pred=1)
    FN = malware sample predicted as clean   (true=1, pred=0)
    """
    fp_pos = [i for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
    fn_pos = [i for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]

    if not fp_pos and not fn_pos:
        with open(save_path, 'w') as f:
            f.write("No misclassifications (FP=0, FN=0).\n")
        print(f"[FP/FN] 0 FP + 0 FN — no misclassifications")
        return

    needed_idx = set(fp_pos) | set(fn_pos)
    test_df = pd.read_csv(test_csv)
    idx_to_cmd = {i: test_df.iloc[i]['command_line'] for i in needed_idx}
    del test_df

    with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FALSE POSITIVES (Clean predicted as Malware): {len(fp_pos)}\n")
        f.write(f"{'='*80}\n\n")
        for rank, pos in enumerate(fp_pos, 1):
            f.write(f"{rank}. {idx_to_cmd[pos]}\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"FALSE NEGATIVES (Malware predicted as Clean): {len(fn_pos)}\n")
        f.write(f"{'='*80}\n\n")
        for rank, pos in enumerate(fn_pos, 1):
            f.write(f"{rank}. {idx_to_cmd[pos]}\n")

    print(f"[FP/FN] {len(fp_pos)} FP + {len(fn_pos)} FN written to {save_path}")


def run_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_class: type,
    input_dim: int,
    device: str,
    loss_type: str = 'standard',
    class_weights: torch.Tensor = None,
    num_epochs: int = 25,
    learning_rate: float = 0.001,
    hidden_dim: int = 128,
    results_dir: str = './results',
    early_stopping_patience: int = 5,
    num_layers: int = 3,
    num_edge_types: int = None,
    test_csv: str = None,
    experiment_params: Dict = None,
    heldout_test_loader: DataLoader = None,
    heldout_test_csv: str = None,
) -> Dict:
    """
    Run a single training experiment.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        model_class: GNN model class (GCNModel, GINModel, etc.)
        input_dim: Input feature dimension
        device: 'cuda' or 'cpu'
        loss_type: 'standard', 'weighted', or 'focal'
        class_weights: Pre-computed class weights (for weighted loss)
        num_epochs: Number of training epochs
        learning_rate: Optimizer learning rate
        hidden_dim: Hidden dimension for GNN layers
        results_dir: Directory to save results
        early_stopping_patience: Stop training if no improvement in f1_malware for this many epochs
        test_csv: Path to test CSV (for FP/FN logging)
        
    Returns:
        Dictionary with training history and final metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize model
    kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    if model_class == RGCNModel:
        num_relations = num_edge_types or 5
        kwargs["num_relations"] = num_relations
    model = model_class(**kwargs).to(device)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    train_losses = []
    test_losses = []
    val_metrics_history = []
    best_f1 = -1.0
    best_model_state = model.state_dict().copy()
    epochs_without_improvement = 0
    
    # Training loop
    with trange(1, num_epochs + 1, desc="Training") as tbar:
        for epoch in tbar:
            # Train for one epoch
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            
            # Evaluate on test set (also compute test loss)
            y_pred, y_true, test_loss = evaluate(model, test_loader, device, criterion)
            test_losses.append(test_loss)
            metrics = compute_metrics(y_true, y_pred)
            val_metrics_history.append(metrics)
            
            # Update learning rate based on F1
            scheduler.step(metrics['f1_macro'])
            
            # Save best model and track early stopping
            if metrics['f1_malware'] > best_f1:
                best_f1 = metrics['f1_malware']
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            tbar.set_postfix(
                train_loss=round(train_loss, 3),
                test_loss=round(test_loss, 3),
                f1_macro=round(metrics['f1_macro'], 3),
                f1_malware=round(metrics['f1_malware'], 3),
                best_f1=round(best_f1, 3),
                no_improv=epochs_without_improvement
            )
            
            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[Early Stopping] No improvement in f1_malware for {early_stopping_patience} epochs. Stopping at epoch {epoch}.")
                break
            
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    y_pred, y_true = evaluate(model, test_loader, device)
    final_metrics = compute_metrics(y_true, y_pred)
    
    # Generate experiment name
    exp_name = f"{model_name}_{loss_type}"
    
    # Print final results
    print_metrics(final_metrics, f"Final Results: {exp_name}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Clean', 'Malware']))
    
    if test_csv is not None:
        fp_fn_path = os.path.join(results_dir, f'{exp_name}_fp_fn.txt')
        _write_fp_fn(y_true, y_pred, test_csv, fp_fn_path)
    
    # Save plots
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
        title=f'Training Curves: {exp_name}',
        test_losses=test_losses
    )
    
    # Save model
    torch.save(best_model_state, os.path.join(results_dir, f'{exp_name}_model.pt'))

    # ---- Held-out test evaluation ----
    heldout_metrics = None
    if heldout_test_loader is not None:
        print(f"\n{'#'*60}")
        print(f"# Held-out Test Evaluation: {exp_name}")
        print(f"{'#'*60}")

        ht_pred, ht_true = evaluate(model, heldout_test_loader, device)
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
        'test_losses': test_losses,
        'val_metrics_history': val_metrics_history,
        'final_metrics': final_metrics,
        'heldout_test_metrics': heldout_metrics,
    }


FEATURE_TYPE_MAP = {
    'positional_only':               (False, True,  False),
    'syntactic_only':                (False, False, True),
    'positional_syntactic':          (False, True,  True),
    'semantic_only':                 (True,  False, False),
    'semantic_syntactic':            (True,  False, True),
    'semantic_positional':           (True,  True,  False),
    'semantic_positional_syntactic': (True,  True,  True),
}

FEATURE_TYPE_CHOICES = list(FEATURE_TYPE_MAP.keys()) + ['all']


def run_full_experiment_suite(
    malware_csv: str,
    clean_csv: str,
    seed: int = 42,
    results_dir: str = './results',
    num_epochs: int = 25,
    batch_size: int = 64,
    val_batch_size: int = 256,
    test_size: float = 0.3,
    force_reprocess: bool = False,
    cache_dir: str = './graph_cache',
    feature_types: List[str] = None,
    model_types: List[str] = None,
    loss_type: str = 'all',
    num_layers: int = 3,
    edge_ablations: List[str] = None,
    embedder_model: str = 'minishlab/potion-base-2M',
    in_memory_only: bool = False,
    test_malware_csv: str = None,
    test_clean_csv: str = None,
    timestamp: str = None,
):
    """
    Run the complete experiment suite with memory-efficient 3-phase approach.

    Phase 1 – Split:  Stratified train/test split saved as CSVs (cached).
    Phase 2 – Cache:  Build graph caches for train and test *separately*
              so only one split's data is in memory at a time.
    Phase 3 – Train:  Load cached graphs, create loaders, train/evaluate.

    Accepts lists for ``feature_types`` and ``edge_ablations`` — the suite
    iterates over their cross-product.  Pass ``['all']`` (the default) to
    expand to every option.
    """
    if feature_types is None:
        feature_types = ['all']
    if edge_ablations is None:
        edge_ablations = ['all']

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
    print(f"[Main] Using device: {device}")
    print(f"[Main] GNN layers: {num_layers}")
    print(f"[Main] Seed: {seed}")

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ft_label = "+".join(feature_types)
    ea_label = "+".join(edge_ablations)
    embedder_tag = embedder_model.replace('/', '_')
    exp_name = f"{timestamp}/{test_size}/seed{seed}/{ft_label}/{ea_label}/{embedder_tag}"
    results_dir = os.path.join(results_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[Main] Results directory: {results_dir}")

    # ---- Phase 1: Split data into train/test CSVs ----
    train_csv, test_csv = split_and_save_data(
        malware_csv, clean_csv,
        seed=seed, test_size=test_size,
        cache_dir=cache_dir,
    )

    # ---- Expand selections ----
    # 'all' in feature_types is a shorthand for every feature configuration.
    # 'all' in edge_ablations is a real config name ("keep all edge types"),
    # NOT a shorthand — users list the specific ablations they want.
    if 'all' in feature_types:
        selected_ft = list(FEATURE_TYPE_MAP.keys())
    else:
        selected_ft = list(dict.fromkeys(feature_types))  # deduplicate, preserve order

    selected_ea = list(dict.fromkeys(edge_ablations))

    # ---- Build feature configs (cross-product of features × edge ablations) ----
    feature_type_tuples = [FEATURE_TYPE_MAP[ft] for ft in selected_ft]

    needs_semantic = any(cfg[0] for cfg in feature_type_tuples)
    embedder = StaticEmbedder(embedder_model) if needs_semantic else None
    if not needs_semantic:
        print("[Main] Skipping semantic embedder (not needed for non-semantic features)")

    feature_configs = []
    for use_semantic, use_positional, use_token in feature_type_tuples:
        for ea in selected_ea:
            feature_configs.append(FeatureConfig(
                use_semantic=use_semantic,
                use_token_features=use_token,
                use_positional_features=use_positional,
                edge_ablation=ea,
            ))

    print(f"[Main] Running {len(feature_configs)} feature configurations "
          f"({len(selected_ft)} feature types × {len(selected_ea)} edge ablations)")
    for fc in feature_configs:
        print(f"  - {fc.get_name()}")

    # ---- Model / loss selection ----
    all_model_classes = {
        'gcn': GCNModel,
        'graphsage': GraphSAGEModel,
        'gin': GINModel,
        'gat': GATModel,
        'rgcn': RGCNModel,
    }
    if model_types is None:
        model_types = ['all']
    if 'all' in model_types:
        model_classes = list(all_model_classes.values())
    else:
        model_classes = [all_model_classes[m] for m in dict.fromkeys(model_types)]
    print(f"[Main] Running {len(model_classes)} model(s): {[m.__name__ for m in model_classes]}")

    all_loss_types = ['standard', 'weighted', 'smoothed', 'focal']
    loss_types = all_loss_types if loss_type == 'all' else [loss_type]
    print(f"[Main] Running {len(loss_types)} loss type(s): {loss_types}")

    # ---- Prepare held-out test CSV (if separate test CSVs provided) ----
    heldout_test_csv = None
    if test_malware_csv and test_clean_csv:
        heldout_test_csv = _prepare_combined_csv(
            test_malware_csv, test_clean_csv, cache_dir,
        )

    if not in_memory_only:
        # ---- Pre-pass: ensure all graph caches exist ----
        csvs_to_cache = [train_csv, test_csv]
        if heldout_test_csv:
            csvs_to_cache.append(heldout_test_csv)

        for feature_config in feature_configs:
            cur_embedder = embedder if feature_config.use_semantic else None
            for csv_path in csvs_to_cache:
                if not force_reprocess and CommandLineGraphDataset.cache_exists(
                    data_csv=csv_path,
                    feature_config=feature_config,
                    cache_dir=cache_dir,
                    semantic_model_name=embedder_model,
                ):
                    print(f"[Pre-pass] Cache exists for {feature_config.get_name()} "
                          f"({os.path.basename(csv_path)}), skipping")
                    continue
                ds = CommandLineGraphDataset(
                    data_csv=csv_path,
                    embedder=cur_embedder,
                    feature_config=feature_config,
                    cache_dir=cache_dir,
                    force_reprocess=force_reprocess,
                    semantic_model_name=embedder_model,
                )
                del ds
                gc.collect()

        # Free embedder -- no longer needed after caches are built
        if embedder is not None:
            print("[Main] Freeing semantic embedder (all caches built)")
            del embedder
            embedder = None
            gc.collect()
    else:
        print("[Main] In-memory-only mode: skipping graph cache pre-pass")

    # ---- Build reproducibility dict (shared across all runs) ----
    base_params = {
        'seed': seed,
        'test_size': test_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_layers': num_layers,
        'embedder_model': embedder_model,
        'in_memory_only': in_memory_only,
    }

    # ---- Training loop: load (or build) graphs and train ----
    all_results = []

    for feature_config in feature_configs:
        print(f"\n{'='*70}")
        print(f"FEATURE CONFIGURATION: {feature_config.get_name().upper()}")
        print(f"{'='*70}")

        cur_embedder = embedder if (embedder and feature_config.use_semantic) else None

        train_ds = CommandLineGraphDataset(
            data_csv=train_csv,
            embedder=cur_embedder,
            feature_config=feature_config,
            cache_dir=cache_dir,
            semantic_model_name=embedder_model,
            in_memory_only=in_memory_only,
        )
        test_ds = CommandLineGraphDataset(
            data_csv=test_csv,
            embedder=cur_embedder,
            feature_config=feature_config,
            cache_dir=cache_dir,
            semantic_model_name=embedder_model,
            in_memory_only=in_memory_only,
        )

        heldout_test_loader = None
        heldout_ds = None
        if heldout_test_csv is not None:
            heldout_ds = CommandLineGraphDataset(
                data_csv=heldout_test_csv,
                embedder=cur_embedder,
                feature_config=feature_config,
                cache_dir=cache_dir,
                semantic_model_name=embedder_model,
                in_memory_only=in_memory_only,
            )
            heldout_test_loader = DataLoader(heldout_ds, batch_size=val_batch_size, shuffle=False)

        class_weights = get_class_weights(train_ds.labels, device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False)

        input_dim = train_ds.get(0).x.shape[1]
        print(f"[Features] Input dimension: {input_dim}")
        print(f"[Split] Train: {len(train_ds)}, Val: {len(test_ds)}")
        if heldout_ds is not None:
            print(f"[Split] Held-out test: {len(heldout_ds)}")

        for model_class in model_classes:
            for current_loss_type in loss_types:
                run_params = {
                    **base_params,
                    'feature_config': feature_config.get_name(),
                    'edge_ablation': feature_config.edge_ablation,
                    'model': model_class.__name__,
                    'loss': current_loss_type,
                    'input_dim': input_dim,
                }
                result = run_experiment(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model_class=model_class,
                    input_dim=input_dim,
                    device=device,
                    loss_type=current_loss_type,
                    class_weights=class_weights,
                    num_epochs=num_epochs,
                    results_dir=os.path.join(results_dir, feature_config.get_name()),
                    num_layers=num_layers,
                    num_edge_types=feature_config.get_num_edge_types(),
                    test_csv=test_csv,
                    experiment_params=run_params,
                    heldout_test_loader=heldout_test_loader,
                    heldout_test_csv=heldout_test_csv,
                )

                result['feature_config'] = feature_config.get_name()
                all_results.append(result)

        del train_ds, test_ds, train_loader, test_loader
        if heldout_ds is not None:
            del heldout_ds, heldout_test_loader
        gc.collect()

    # Free embedder if it survived (in-memory-only mode)
    if embedder is not None:
        print("[Main] Freeing semantic embedder")
        del embedder
        gc.collect()

    # ---- Summary ----
    summary = []
    for r in all_results:
        entry = {
            'seed': seed,
            'feature_config': r['feature_config'],
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

    generate_comparison_plots(summary_df, results_dir)

    return all_results, summary_df


def generate_comparison_plots(summary_df: pd.DataFrame, results_dir: str):
    """
    Generate comparison plots across all experiments.
    """
    # Plot 1: F1 scores by feature configuration
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by feature config and model
    for i, metric in enumerate(['f1_macro', 'recall_malware']):
        ax = axes[i]
        pivot = summary_df.pivot_table(
            values=metric,
            index='model',
            columns='feature_config',
            aggfunc='max'  # Best across loss types
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric.upper()} by Feature Configuration')
        ax.set_ylabel(metric)
        ax.legend(title='Features')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Loss function comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = summary_df.pivot_table(
        values='f1_macro',
        index='model',
        columns='loss_type',
        aggfunc='max'  # Best across feature configs
    )
    pivot.plot(kind='bar', ax=ax)
    ax.set_title('F1 Macro Score by Loss Function')
    ax.set_ylabel('F1 Macro')
    ax.legend(title='Loss Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plots] Comparison plots saved to {results_dir}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train GNN for malware detection')
    parser.add_argument('--malware-csv', type=str,
                        default='./unique_data/malware_unique.csv',
                        help='Path to malware CSV file')
    parser.add_argument('--clean-csv', type=str,
                        default='./unique_data/clean_unique.csv',
                        help='Path to clean CSV file')
    parser.add_argument('--results-dir', type=str, default='./new_results',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--cache-dir', type=str, default='./graph_cache',
                        help='Directory to cache preprocessed graphs and split CSVs')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of graphs (ignore cache)')
    parser.add_argument('--seed', type=int, nargs='+', default=[42],
                        help='Random seed(s) for reproducibility (space-separated, default: 42)')
    parser.add_argument('--feature-type', type=str, nargs='+',
                        default=['semantic_only'],
                        choices=FEATURE_TYPE_CHOICES,
                        help='Feature configuration(s) to use (space-separated, or "all")')
    parser.add_argument('--model', type=str, nargs='+', default=['rgcn'],
                        choices=['gcn', 'graphsage', 'gin', 'gat', 'rgcn', 'all'],
                        help='Model(s) to use (space-separated, or "all")')
    parser.add_argument('--loss', type=str, default='standard',
                        choices=['standard', 'weighted', 'smoothed', 'focal', 'all'],
                        help='Loss function to use: standard, weighted, smoothed, focal, or all')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test size for train/test split')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN message-passing layers (default: 3)')
    parser.add_argument('--edge-ablation', type=str, nargs='+',
                        default=['all'],
                        choices=EDGE_ABLATION_CHOICES,
                        help='Edge type ablation(s) (space-separated, or "all")')
    parser.add_argument('--embedder-model', type=str,
                        default='minishlab/potion-base-2M',
                        help='Model2Vec static model for semantic embeddings '
                             '(potion-base-2M=64d, 4M=128d, 8M=256d)')
    parser.add_argument('--in-memory-only', action='store_true',
                        help='Generate graphs on-the-fly during training instead of '
                             'preprocessing and caching them all in memory/disk. '
                             'Useful for very large datasets that exceed RAM.')
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
    print("GNN MALWARE DETECTION EXPERIMENT")
    print("="*70)
    print(f"Malware CSV: {malware_csv}")
    print(f"Clean CSV: {clean_csv}")
    print(f"Results Dir: {args.results_dir}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"Force Reprocess: {args.force_reprocess}")
    print(f"In-Memory Only: {args.in_memory_only}")
    print(f"Seed(s): {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Feature Type(s): {args.feature_type}")
    print(f"Model(s): {args.model}")
    print(f"Loss: {args.loss}")
    print(f"GNN Layers: {args.num_layers}")
    print(f"Edge Ablation(s): {args.edge_ablation}")
    print(f"Embedder Model: {args.embedder_model}")
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
            test_size=args.test_size,
            results_dir=args.results_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            force_reprocess=args.force_reprocess,
            feature_types=args.feature_type,
            model_types=args.model,
            loss_type=args.loss,
            num_layers=args.num_layers,
            edge_ablations=args.edge_ablation,
            embedder_model=args.embedder_model,
            in_memory_only=args.in_memory_only,
            test_malware_csv=test_malware_csv,
            test_clean_csv=test_clean_csv,
            timestamp=timestamp,
        )

    print("\n[Main] Experiment complete!")

