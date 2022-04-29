import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", type=str, help="Path to the whole dataset")
parser.add_argument("--nfolds", "-n", type=int, help="Number of folds in cross-validation")
parser.add_argument("--seed", type=int, help="Random seed for shuffling")
parser.add_argument("--output-dir", "-o", type=str, help="Directory path for saved data subsets")
args = parser.parse_args()

# python3 cross_validation.py -d data/whole_sanitized_us_patents.csv -n 6 -o data --seed 1525

data = pd.read_csv(args.data, header=None)

n_val_samples = data.shape[0] // args.nfolds
print("Dataset size:", data.shape[0])
print("Number of validation samples:", n_val_samples)
kf = KFold(n_splits=args.nfolds, random_state=args.seed, shuffle=True)
for fold_no, (train_index, val_index) in enumerate(kf.split(data)):
    tr = data.iloc[train_index]
    val = data.iloc[val_index]
    print(flush=True)
    print("Fold:", fold_no, flush=True)
    print("Train size:", tr.shape[0], flush=True)
    print("Val size:", val.shape[0], flush=True)
    train_save_to = Path(args.output_dir) / Path(f"train_fold_{fold_no + 1}.csv")
    val_save_to = Path(args.output_dir) / Path(f"val_fold_{fold_no + 1}.csv")
    tr.to_csv(train_save_to, index=False, header=False)
    val.to_csv(val_save_to, index=False, header=False)
    print(f"Train subset saved to {train_save_to}", flush=True)
    print(f"Val subset saved to {val_save_to}", flush=True)
