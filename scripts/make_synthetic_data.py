"""Writes a fake dataset with the same columns as the real one (results are meaningless).

python scripts/make_synthetic_data.py --n-features 60
"""
import argparse
from pathlib import Path

from radiomics_protocol.utils.synthetic import make_synthetic_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=113)
    ap.add_argument("--n-complete", type=int, default=78, help="patients without missing radiomics")
    ap.add_argument("--n-features", type=int, default=535, help="535 = 107 features x 5 compartments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synthetic/radiomics_dataset.csv")
    a = ap.parse_args()

    df = make_synthetic_dataset(a.n_patients, a.n_complete, a.n_features, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep=";", index=False)
    complete = df.dropna()
    print(f"wrote {out}: {len(df)} patients ({len(complete)} complete), "
          f"{df.filter(like='Corr_T1_').shape[1]} features, {int(complete['exitus'].sum())} events")


if __name__ == "__main__":
    main()
