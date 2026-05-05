import argparse
import glob
import os
from typing import Optional


def parse_results_dir(results_dir: str) -> Optional[dict]:
    parts = os.path.normpath(results_dir).split(os.sep)
    seed_idx = -1
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            seed_idx = i
            break
    if seed_idx == -1 or seed_idx + 4 >= len(parts):
        return None

    try:
        test_size = float(parts[seed_idx - 1])
    except (ValueError, IndexError):
        return None

    seed = int(parts[seed_idx][4:])
    ft_label = parts[seed_idx + 1]
    ea_label = parts[seed_idx + 2]
    embedder_tag = parts[seed_idx + 3]
    feature_config_name = parts[seed_idx + 4]

    if "_" in embedder_tag and not embedder_tag.startswith("minishlab/"):
        embedder_model = embedder_tag.replace("_", "/", 1)
    else:
        embedder_model = embedder_tag

    return {
        "test_size": test_size,
        "seed": seed,
        "ft_label": ft_label,
        "edge_ablation": ea_label,
        "embedder_model": embedder_model,
        "feature_config_name": feature_config_name,
    }


def _to_posix(path: str) -> str:
    return path.replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find completed GNN runs by configuration.")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_gnn",
        help="Root directory containing GNN results (default: results_gnn).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--feature-type-label", type=str, default=None)
    parser.add_argument("--edge-ablation-label", type=str, default=None)
    parser.add_argument("--embedder", type=str, default=None)
    parser.add_argument("--feature-config", type=str, nargs="+", default=None)
    parser.add_argument("--model", type=str, nargs="+", default=None)
    args = parser.parse_args()

    model_glob = os.path.join(args.results_root, "**", "*_model.pt")
    model_paths = glob.glob(model_glob, recursive=True)
    if not model_paths:
        print(f"No model files found under {args.results_root}.")
        return

    matches: dict[str, dict] = {}
    for model_path in model_paths:
        model_dir = os.path.dirname(model_path)
        cfg = parse_results_dir(model_dir)
        if cfg is None:
            continue

        if args.seed is not None and cfg["seed"] != args.seed:
            continue
        if args.test_size is not None and cfg["test_size"] != args.test_size:
            continue
        if args.feature_type_label and cfg["ft_label"] != args.feature_type_label:
            continue
        if args.edge_ablation_label and cfg["edge_ablation"] != args.edge_ablation_label:
            continue
        if args.embedder and cfg["embedder_model"] != args.embedder:
            continue
        if args.feature_config and cfg["feature_config_name"] not in args.feature_config:
            continue

        model_name = os.path.basename(model_path).replace("_model.pt", "")
        if args.model:
            normalized = [m.lower() for m in args.model]
            if not any(model_name.lower().startswith(m) for m in normalized):
                continue

        entry = matches.setdefault(
            model_dir, {"cfg": cfg, "models": set(), "model_paths": set()}
        )
        entry["models"].add(model_name)
        entry["model_paths"].add(model_path)

    if not matches:
        print("No runs found matching the requested filters.")
        return

    for idx, (results_dir, entry) in enumerate(sorted(matches.items())):
        if idx > 0:
            print()
        cfg = entry["cfg"]
        models = ",".join(sorted(entry["models"]))
        print(f"feature_config={cfg['feature_config_name']}")
        print(f"seed={cfg['seed']}")
        print(f"test_size={cfg['test_size']}")
        print(f"feature_type_label={cfg['ft_label']}")
        print(f"edge_ablation={cfg['edge_ablation']}")
        print(f"embedder={cfg['embedder_model']}")
        print(f"models={models}")
        print(f"results_dir={_to_posix(results_dir)}")
        for model_path in sorted(entry["model_paths"]):
            print(_to_posix(model_path))


if __name__ == "__main__":
    main()
