"""Merge data/raw/lob/lob_part_*.parquet into one validated, time-sorted LOB parquet."""
import argparse
from market_maker.data.binance_live import merge_parts

ap = argparse.ArgumentParser()
ap.add_argument("--parts-dir", default="data/raw/lob")
ap.add_argument("--out", default="data/processed/lob.parquet")
a = ap.parse_args()
df, rep = merge_parts(a.parts_dir, a.out)
print(f"merged -> {a.out}: {rep}")
