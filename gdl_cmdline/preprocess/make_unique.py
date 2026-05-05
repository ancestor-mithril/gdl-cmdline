import gc


import argparse
import os

from gdl_cmdline.preprocess.tokenize import parse_windows_cmdline
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pandas as pd




def wrap_touple(l):
    return tuple(wrap_touple(item) if isinstance(item, (list, tuple)) else item for item in l)


def normalize_command_line(command_line, process_name, no_normalize):
    """
    Normalize command line by:
    1. Tokenizing with parse_windows_cmdline
    2. Replacing first token with process_name
    3. Joining back together with spaces
    """
    try:
        valid_chars = sum(1 for c in command_line if c.isascii() or c.isspace())
        ratio = valid_chars / len(command_line)
        if ratio < 0.8:
            return process_name
        tokens = parse_windows_cmdline(command_line, no_normalize=no_normalize)
        if isinstance(tokens, str):
            tokens = [tokens]

        if not tokens:
            return process_name

        if no_normalize:
            if isinstance(tokens, tuple) and len(tokens) > 1:
                while isinstance(tokens[0], tuple):
                    tokens = tokens[0] + (tokens[1:], )
            return tokens

        

        # Replace first token with process_name
        t = list(tokens)
        while True:
            t = tokens
            while isinstance(t[0], (list, tuple)):
                if isinstance(t[0], tuple):
                    t[0] = list(t[0])
                t = t[0]

            while t and t[0] == "\\\\":
                t.pop(0)
            if t:
                break
        


        if process_name is not None and process_name.split(".")[0] not in t[0].lower():
            if process_name == "powershell.exe" and "pwsh" in t[0].lower():
                t[0] = process_name
            else:
                t.insert(0, process_name)
        elif process_name is not None:
            t[0] = process_name

        return wrap_touple(tokens)
    except Exception as e:
        # If parsing fails, just use process_name + original
        print(f"Error parsing: {command_line}... - {e}")
        raise e


def unwrap_nested_tuples(command_line):
    result = []
    for item in command_line:
        if isinstance(item, tuple):
            result.extend(unwrap_nested_tuples(item))
        else:
            result.append(item)
    return result


def is_not_short(unwrapped_command_line):
    min_length_for_no_file = 4
    min_length_for_file = 6
    if len(unwrapped_command_line) < min_length_for_no_file:
        return False
    if len(unwrapped_command_line) < min_length_for_file:
        for token in unwrapped_command_line[1:]:
            if token.count(".") == 1:
                if token.startswith("."):
                    if len(token) < min_length_for_file:
                        return False
                else:
                    index = token.index(".")
                    if token[index-1] in "\\/":
                        return False
                    if len(token[index:]) < min_length_for_file:
                        return False
            if (
                token.endswith(".exe") or 
                token.endswith(".ps1") or 
                token.endswith(".bat") or 
                token.endswith(".cmd") or 
                token.endswith(".js") or 
                token.endswith(".py") or 
                token.endswith(".vbs") 
            ):
                return False
            if "<" in token and ">" in token:
                for tag in ["<path>", "<users>", "<programfiles>", "<windows>", "<tmp>", "<c>", "<programdata>"]:
                    if tag in token:
                        return False

    return True


def is_ip_or_host(token: str):
    token = token.strip("\\")
    if len(token) == 0:
        return False
    if token[0].isdigit():
        return True
    if token.startswith("<int_ip>"):
        return True
    if token.startswith("<ext_ip>"):
        return True
    if token.startswith("<host>"):
        return True
    return False

def is_strange(token: str):
    if token == "//":
        return True
    if token == "x":
        return True
    if token.startswith("//--"):
        return True
    return False

def is_env_assigned(unwrapped_command_line):
    if len(unwrapped_command_line) < 3:
        return False
    if unwrapped_command_line[1] != "=":
        return False
    env_start = "%$"
    for i in env_start:
        if unwrapped_command_line[0].startswith(i):
            return True
    return False

def is_valid_command_line(command_line):
    unwrapped_command_line = unwrap_nested_tuples(command_line)
    if not is_not_short(unwrapped_command_line):
        return False
    first_token = unwrapped_command_line[0]
    if is_ip_or_host(first_token):
        return False
    if is_strange(first_token):
        return False
    if is_env_assigned(unwrapped_command_line):
        return False
    return True


def process_csv(input_file, output_file, no_normalize=False, num_processes=os.cpu_count() // 4):
    """
    Process CSV file:
    1. Read with columns: command_line, process_name, total_occurrences
    2. Normalize command lines
    3. Group by unique normalized command_line and sum occurrences
    4. Save to output file
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"  Original rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    process_names = [None] * len(df)
    if "process_name" in df.columns:
        process_names = df["process_name"]
    no_normalize = [no_normalize] * len(df)

    print(f"  Normalizing command lines...")
    df["normalized_command_line"] = process_map(
        normalize_command_line,
        df["command_line"],
        process_names,
        no_normalize,
        max_workers=num_processes,
        chunksize=500,
    )
    df.dropna(inplace=True)

    tqdm.pandas()
    drop_mask = df["normalized_command_line"].progress_apply(is_valid_command_line)
    print(f"  Dropping {(~drop_mask).sum()} invalid command lines...")

    df.drop(df[~drop_mask].index, inplace=True)

    if not no_normalize:
        df.sort_values("normalized_command_line", ascending=False, inplace=True, key=lambda s: s.map(repr))
        df.to_csv(output_file.replace(".csv", "_transformed_valid.csv"), index=False)
        

    print(f"  Grouping and aggregating...")
    if "total_occurrences" in df.columns:
        result = df.groupby("normalized_command_line", as_index=False).agg(
            {"total_occurrences": "sum"}
        )
    else:
        result = df[["normalized_command_line"]].drop_duplicates()

    # Rename column back to command_line
    result.rename(columns={"normalized_command_line": "command_line"}, inplace=True)
    result.sort_values("command_line", ascending=False, inplace=True, key=lambda s: s.map(repr))

    # Sort by total_occurrences descending

    print(f"  Unique rows after normalization: {len(result)}")

    # Save to output
    print(f"  Saving to {output_file}...")
    result.to_csv(output_file, index=False)

    print(f"  Done!")

    return result


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser(description="Process a data folder")
    parser.add_argument("data_path", type=str)
    parser.add_argument("-njobs", type=int, default=os.cpu_count() // 4)
    args = parser.parse_args()

    data_path = args.data_path

    for type in [
        "malware", 
        "clean", 
        "test_malware",
        "test_clean",
    ]:
        output_type = type
        print(f"Processing {type}.csv")
        result = process_csv(
            os.path.join(data_path, f"{type}.csv"),
            os.path.join(data_path, f"{output_type}_unique.csv"),
            args.njobs,
        )
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{type} unique commands: {len(result)}")
        print("\n" + "=" * 60)
        del result
        gc.collect()
    