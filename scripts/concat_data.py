import sys
import pandas as pd
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        help="Provide path to output file")
    args = parser.parse_args()
    return args


args = _parse_args()
files = [f.rstrip() for f in sys.stdin.readlines()]
print(f'Found {len(files)} files')
dfs = [pd.read_csv(f) for f in files]
print(f'Created {len(dfs)} dataframes')
result = pd.DataFrame().append(dfs, ignore_index=True)
print(f'Concatenated dataframes: results has shape {result.shape}')
result.to_csv(args.output)
