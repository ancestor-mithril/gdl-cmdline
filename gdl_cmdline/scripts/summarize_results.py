import argparse
import glob
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable


PRIMARY_COLUMNS = [
    "seed",
    "feature_config",
    "model",
    "loss_type",
    "analyzer",
]

METRIC_COLUMNS = [
    "f1_malware",
    "fpr",
]


def _format_cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{value:.4f}"
    return str(value)


def _build_table(
    df: pd.DataFrame,
    base_cols: list[str],
    metric_cols: list[str],
    metric_prefix: str = "",
) -> PrettyTable:
    headers = base_cols + metric_cols
    table = PrettyTable(headers)
    table.align = "l"
    for _, row in df.iterrows():
        row_cells = []
        for col in base_cols:
            row_cells.append(_format_cell(row.get(col)))
        for col in metric_cols:
            row_cells.append(_format_cell(row.get(f"{metric_prefix}{col}")))
        table.add_row(row_cells)
    return table


def _build_summary_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    metric_prefix: str = "",
) -> PrettyTable:
    metric_headers = []
    for col in metric_cols:
        metric_headers.append(f"{metric_prefix}{col}_mean")
        metric_headers.append(f"{metric_prefix}{col}_std")

    if not group_cols:
        headers = metric_headers
        table = PrettyTable(headers)
        table.align = "l"
        row_cells = []
        for col in metric_cols:
            series = df[f"{metric_prefix}{col}"].dropna()
            row_cells.append(_format_cell(series.mean()))
            row_cells.append(_format_cell(series.std(ddof=0)))
        table.add_row(row_cells)
        return table

    metric_fields = [f"{metric_prefix}{col}" for col in metric_cols]
    stats = df.groupby(group_cols, dropna=False)[metric_fields].agg(
        ["mean", lambda s: s.std(ddof=0)]
    ).reset_index()

    flat_cols = []
    for col in stats.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat == "<lambda_0>":
                stat = "std"
            if stat:
                flat_cols.append(f"{base}_{stat}")
            else:
                flat_cols.append(base)
        else:
            flat_cols.append(col)

    stats.columns = flat_cols
    std_cols = [col for col in stats.columns if str(col).endswith("_std")]
    if std_cols:
        stats.loc[:, std_cols] = stats.loc[:, std_cols].fillna(0.0)

    headers = group_cols + metric_headers
    table = PrettyTable(headers)
    table.align = "l"
    for _, row in stats.iterrows():
        row_cells = []
        for col in group_cols:
            row_cells.append(_format_cell(row.get(col)))
        for col in metric_headers:
            row_cells.append(_format_cell(row.get(col)))
        table.add_row(row_cells)

    return table


def _write_tables(output_path: str, tables: list[str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(tables))


def _safe_output_name(summary_path: str) -> str:
    rel_path = os.path.relpath(summary_path, os.getcwd())
    safe_path = rel_path.replace(":", "").replace("\\", "__").replace("/", "__")
    return safe_path.replace(".csv", ".txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results into tables.")
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        required=True,
        help="Glob pattern(s) for experiment_summary.csv files (use quotes).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write table outputs as .txt files.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help="Optional column name to sort by (descending).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows to print per table.",
    )
    args = parser.parse_args()

    patterns: list[str] = []
    for item in args.pattern:
        patterns.extend([p for p in item.split(",") if p])

    summary_files: list[str] = []
    for pattern in patterns:
        summary_files.extend(glob.glob(pattern, recursive=True))
    summary_files = sorted(set(summary_files))
    if not summary_files:
        print("No experiment_summary.csv files found for the pattern.")
        return

    all_rows = []
    for summary_path in summary_files:
        df = pd.read_csv(summary_path)
        df["summary_path"] = summary_path
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)
    if args.sort_by and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False)
    if args.limit:
        df = df.head(args.limit)

    base_cols = [col for col in PRIMARY_COLUMNS if col in df.columns]
    metric_cols = [col for col in METRIC_COLUMNS if col in df.columns]
    test_metric_cols = [
        col for col in METRIC_COLUMNS if f"test_{col}" in df.columns
    ]
    group_cols = [col for col in base_cols if col != "seed"]

    output_sections = []
    header = f"Summary patterns: {', '.join(patterns)}"
    output_sections.append(header)
    output_sections.append("-" * len(header))

    if metric_cols:
        base_table = _build_table(df, base_cols, metric_cols)
        output_sections.append("Validation/Test Split Metrics (Per Seed)")
        output_sections.append(str(base_table))
        summary_table = _build_summary_table(df, group_cols, metric_cols)
        output_sections.append("Validation/Test Split Metrics (Mean/Std)")
        output_sections.append(str(summary_table))
    else:
        output_sections.append("No validation metrics found.")

    if test_metric_cols:
        test_table = _build_table(df, base_cols, test_metric_cols, metric_prefix="test_")
        output_sections.append("Held-out Test Metrics (Per Seed)")
        output_sections.append(str(test_table))
        test_summary = _build_summary_table(
            df, group_cols, test_metric_cols, metric_prefix="test_"
        )
        output_sections.append("Held-out Test Metrics (Mean/Std)")
        output_sections.append(str(test_summary))

    output_text = "\n".join(output_sections)
    print(output_text)
    print()

    if args.output_dir:
        output_path = os.path.join(args.output_dir, "summary_tables.txt")
        _write_tables(output_path, output_sections)


if __name__ == "__main__":
    main()
