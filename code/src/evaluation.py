import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

from imblearn.over_sampling import SMOTE, ADASYN
import dendropy

from data_augmentation.tada_sv import TADASV
from data_augmentation.no_tree_sv import NoTreeSV

import numpy as np

def reduce_minority_class(X_train, y_train, r, seed=42):
    """
    Entfernt r Samples aus der Minority-Klasse (y == True) im Trainingsset.
    """
    rng = np.random.default_rng(seed)

    # Indizes der Minority-Klasse
    minority_idx = y_train[y_train == True].index.to_numpy()

    if r > len(minority_idx):
        raise ValueError(
            f"r={r} is larger than number of minority samples ({len(minority_idx)})"
        )

    # zufällige Auswahl der zu entfernenden Samples
    remove_idx = rng.choice(minority_idx, size=r, replace=False)

    # neue Trainingsdaten
    X_reduced = X_train.drop(index=remove_idx)
    y_reduced = y_train.drop(index=remove_idx)

    return X_reduced, y_reduced


# ----------------------------
# Config
# ----------------------------
DATA_PATH = "/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/filtered_data.csv"
TREE_PATH = "/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/phylogeny_pruned.tre"
OUT_DIR   = "/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/results"
os.makedirs(OUT_DIR, exist_ok=True)

N_REPS = 0
SAVE_EVERY = 10

BASE_SEED = 0
TEST_SIZE = 0.2

RF_N_ESTIMATORS = 2000


# ----------------------------
# Model helper
# ----------------------------
def train_predict_rf(X_train, y_train, X_test, seed: int):
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def train_predict_lr(X_train, y_train, X_test, seed: int):
    """
    Logistic Regression mit Standardisierung.
    - solver='liblinear' ist robust für kleinere Daten
    - class_weight='balanced' wegen Imbalance
    """
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False ist sicher bei sparsity/Counts
        ("lr", LogisticRegression(
            max_iter=5000,
            random_state=seed,
            class_weight="balanced",
            solver="liblinear",
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def save_checkpoint(df: pd.DataFrame, rep: int):
    csv_path = os.path.join(OUT_DIR, f"results_checkpoint_rep_{rep:02d}.csv")
    df.to_csv(csv_path, index=False)

    # Optional: schneller / kleiner als CSV (falls du willst)
    # parquet_path = os.path.join(OUT_DIR, f"results_checkpoint_rep_{rep:02d}.parquet")
    # df.to_parquet(parquet_path, index=False)

    print(f"[saved] {csv_path}")


# ----------------------------
# Data loading
# ----------------------------
df = pd.read_csv(DATA_PATH, index_col=0)
X = df.drop(columns=["diagnosis_binary"])
y = df["diagnosis_binary"]

tree = dendropy.Tree.get(
    path=TREE_PATH,
    schema="newick",
    preserve_underscores=True,
    rooting="force-rooted",
)

# ----------------------------
# Main evaluation loop
# ----------------------------
rows = []

for rep in range(N_REPS):
    print(f"Replicate {rep+1}/{N_REPS}")

    seed = BASE_SEED + rep  # pro replicate variieren

    # 1) Split (Seed variiert, sonst immer gleicher Split!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )

    # E2 Klassenbalance anpassen  
    X_train, y_train = reduce_minority_class(
    X_train, y_train, r=194, seed=42
    )

    print(y_train.value_counts().to_dict())

    # 2) Precompute transforms used by some methods
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)

    # 3) Build augmented datasets (auch Seeds variieren!)
    smote = SMOTE(random_state=seed)
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

    """
    smote_log = SMOTE(random_state=seed)
    X_train_SMOTE_log, y_train_SMOTE_log = smote_log.fit_resample(X_train_log, y_train)
    """

    adasyn = ADASYN(random_state=seed)
    X_train_ADASYN, y_train_ADASYN = adasyn.fit_resample(X_train, y_train)

    tada5 = TADASV(tree=tree, k=5, random_state=seed, sampling_strategy="auto")
    X_train_TADASV_5, y_train_TADASV_5 = tada5.fit_resample(X_train, y_train)

    tada50 = TADASV(tree=tree, k=5, random_state=seed, sampling_strategy="auto")
    X_train_TADASV_50, y_train_TADASV_50 = tada50.fit_resample(X_train, y_train)

    tada_bal = TADASV(tree=tree, random_state=seed, sampling_strategy="balance")
    X_train_TADASV_bal, y_train_TADASV_bal = tada_bal.fit_resample(X_train, y_train)

    tada_balpp = TADASV(tree=tree, q=2, random_state=seed, sampling_strategy="balance++")
    X_train_TADASV_balpp, y_train_TADASV_balpp = tada_balpp.fit_resample(X_train, y_train)

    nt5 = NoTreeSV(k=5, random_state=seed, sampling_strategy="auto", smoothing=1.0)
    X_train_NoTree_5, y_train_NoTree_5 = nt5.fit_resample(X_train, y_train)

    nt_bal = NoTreeSV(random_state=seed, sampling_strategy="balance", smoothing=1.0)
    X_train_NoTree_bal, y_train_NoTree_bal = nt_bal.fit_resample(X_train, y_train)

    nt_balpp = NoTreeSV(random_state=seed, sampling_strategy="balance++", q=3, smoothing=1.0)
    X_train_NoTree_balpp, y_train_NoTree_balpp = nt_balpp.fit_resample(X_train, y_train)

    # 4) Define methods cleanly (train/test must match transform!)
    methods = {
        "NO_AUG": (X_train, y_train, X_test, y_test),
        "SMOTE": (X_train_SMOTE, y_train_SMOTE, X_test, y_test),
        #"SMOTElog1p": (X_train_SMOTE_log, y_train_SMOTE_log, X_test_log, y_test),
        "ADASYN": (X_train_ADASYN, y_train_ADASYN, X_test, y_test),
        "TADA-SV-5": (X_train_TADASV_5, y_train_TADASV_5, X_test, y_test),
        "TADA-SV-50": (X_train_TADASV_50, y_train_TADASV_50, X_test, y_test),
        "TADA-SV-BALANCE": (X_train_TADASV_bal, y_train_TADASV_bal, X_test, y_test),
        "TADA-SV-BALANCE++": (X_train_TADASV_balpp, y_train_TADASV_balpp, X_test, y_test),
        "DBS-5": (X_train_NoTree_5, y_train_NoTree_5, X_test, y_test),
        "DBS-BALANCE": (X_train_NoTree_bal, y_train_NoTree_bal, X_test, y_test),
        "DBS-BALANCE++": (X_train_NoTree_balpp, y_train_NoTree_balpp, X_test, y_test),
    }

    # 5) Train + score
    for name, (Xtr, ytr, Xte, yte) in methods.items():
        #y_pred, y_proba = train_predict_rf(Xtr, ytr, Xte, seed=seed) # <-- RF
        y_pred, y_proba = train_predict_lr(Xtr, ytr, Xte, seed=seed)  # <-- LR
        metrics = compute_metrics(yte, y_pred, y_proba)

        rows.append({
            "rep": rep + BASE_SEED,
            "seed": seed,
            #"model": "RandomForest", 
            "model": "LogisticRegression", 
            "method": name,
            **metrics
        })

    # 6) Save every 2 reps
    if (rep + 1) % SAVE_EVERY == 0:
        results = pd.DataFrame(rows)
        save_checkpoint(results, rep + 1)

# final results
results = pd.DataFrame(rows)
final_csv = os.path.join(OUT_DIR, "results.csv")
results.to_csv(final_csv, index=False)
print(f"[saved] {final_csv}")

# Summary mean ± SE like in paper
"""summary = (
    results.groupby("method")[["accuracy", "auc", "mcc"]]
    .agg(["mean", "std", "count"])
)
for metric in ["accuracy", "auc", "mcc"]:
    summary[(metric, "se")] = summary[(metric, "std")] / np.sqrt(summary[(metric, "count")])

summary_csv = os.path.join(OUT_DIR, "summary_mean_se.csv")
summary.to_csv(summary_csv)
print(f"[saved] {summary_csv}")"""