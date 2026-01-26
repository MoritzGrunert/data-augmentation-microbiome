import os
import argparse
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


# ----------------------------
# Helpers
# ----------------------------
def train_predict_rf(X_train, y_train, X_test, seed: int, n_estimators: int = 2000):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def train_predict_lr(X_train, y_train, X_test, seed: int):
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
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


def save_checkpoint(df: pd.DataFrame, out_dir: str, rep: int):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"results_checkpoint_rep_{rep:02d}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")


def reduce_minority_class(X_train, y_train, r: int, seed: int = 42, minority_value=True):
    """
    Entfernt r Samples aus der Minority-Klasse (standard: y == True) im Trainingsset.
    """
    rng = np.random.default_rng(seed)

    minority_idx = y_train[y_train == minority_value].index.to_numpy()
    if r > len(minority_idx):
        raise ValueError(f"r={r} > #minority_samples={len(minority_idx)}")

    remove_idx = rng.choice(minority_idx, size=r, replace=False)
    X_reduced = X_train.drop(index=remove_idx)
    y_reduced = y_train.drop(index=remove_idx)
    return X_reduced, y_reduced


# ----------------------------
# Main run
# ----------------------------
def run_experiment(
    data_path: str,
    tree_path: str,
    out_dir: str,
    reps: int,
    base_seed: int,
    test_size: float,
    save_every: int,
    model: str,              # "lr" | "rf"
    experiment: str,         # "e1" | "e2"
    reduce_r: int,
    rf_n_estimators: int,
):
    os.makedirs(out_dir, exist_ok=True)

    # --- Data loading ---
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop(columns=["diagnosis_binary"])
    y = df["diagnosis_binary"]

    tree = dendropy.Tree.get(
        path=tree_path,
        schema="newick",
        preserve_underscores=True,
        rooting="force-rooted",
    )

    # model dispatcher
    if model == "rf":
        train_predict = lambda Xtr, ytr, Xte, seed: train_predict_rf(
            Xtr, ytr, Xte, seed=seed, n_estimators=rf_n_estimators
        )
        model_name = "RandomForest"
    elif model == "lr":
        train_predict = lambda Xtr, ytr, Xte, seed: train_predict_lr(
            Xtr, ytr, Xte, seed=seed
        )
        model_name = "LogisticRegression"
    else:
        raise ValueError("model must be 'rf' or 'lr'")

    rows = []

    for rep in range(reps):
        seed = base_seed + rep
        print(f"Replicate {rep+1}/{reps} | seed={seed} | model={model_name} | exp={experiment}")

        # 1) Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )

        # 2) Experiment 2: reduce minority class (optional)
        if experiment.lower() == "e2":
            if reduce_r <= 0:
                raise ValueError("For experiment e2, reduce_r must be > 0 (use --reduce-r).")
            X_train, y_train = reduce_minority_class(X_train, y_train, r=reduce_r, seed=seed)
            print("[after reduction]", y_train.value_counts().to_dict())
        else:
            print("[train distribution]", y_train.value_counts().to_dict())

        # 3) Augmentations
        smote = SMOTE(random_state=seed)
        X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

        adasyn = ADASYN(random_state=seed)
        X_train_ADASYN, y_train_ADASYN = adasyn.fit_resample(X_train, y_train)

        tada5 = TADASV(tree=tree, k=5, random_state=seed, sampling_strategy="auto")
        X_train_TADASV_5, y_train_TADASV_5 = tada5.fit_resample(X_train, y_train)

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

        methods = {
            "NO_AUG": (X_train, y_train, X_test, y_test),
            "SMOTE": (X_train_SMOTE, y_train_SMOTE, X_test, y_test),
            "ADASYN": (X_train_ADASYN, y_train_ADASYN, X_test, y_test),
            "TADA-SV-5": (X_train_TADASV_5, y_train_TADASV_5, X_test, y_test),
            "TADA-SV-BALANCE": (X_train_TADASV_bal, y_train_TADASV_bal, X_test, y_test),
            "TADA-SV-BALANCE++": (X_train_TADASV_balpp, y_train_TADASV_balpp, X_test, y_test),
            "DBS-5": (X_train_NoTree_5, y_train_NoTree_5, X_test, y_test),
            "DBS-BALANCE": (X_train_NoTree_bal, y_train_NoTree_bal, X_test, y_test),
            "DBS-BALANCE++": (X_train_NoTree_balpp, y_train_NoTree_balpp, X_test, y_test),
        }

        # 4) Train + score
        for method_name, (Xtr, ytr, Xte, yte) in methods.items():
            y_pred, y_proba = train_predict(Xtr, ytr, Xte, seed=seed)
            metrics = compute_metrics(yte, y_pred, y_proba)

            rows.append({
                "rep": rep,
                "seed": seed,
                "model": model_name,
                "experiment": experiment.upper(),
                "method": method_name,
                **metrics
            })

        # 5) checkpoints
        if save_every > 0 and (rep + 1) % save_every == 0:
            results_df = pd.DataFrame(rows)
            save_checkpoint(results_df, out_dir, rep + 1)

    # final
    results_df = pd.DataFrame(rows)
    final_csv = os.path.join(out_dir, f"results_{experiment.lower()}_{model}.csv")
    results_df.to_csv(final_csv, index=False)
    print(f"[saved] {final_csv}")

    return results_df


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DA methods on microbiome OTU data.")
    p.add_argument("--data", default="/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/data.csv")
    p.add_argument("--tree", default="/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/phylogeny_otu.tre")
    p.add_argument("--out",  default="/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/results")

    p.add_argument("--reps", type=int, default=20, help="Number of replicates (runs).")
    p.add_argument("--seed", type=int, default=0, help="Base seed (replicate i uses seed+ i).")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=10)

    p.add_argument("--model", choices=["lr", "rf"], default="rf", help="Classifier to use.")
    p.add_argument("--experiment", choices=["e1", "e2"], default="e1", help="e1=normal, e2=reduce minority in train.")
    p.add_argument("--reduce-r", type=int, default=0, help="Only for e2: remove r minority samples from train.")
    p.add_argument("--rf-trees", type=int, default=2000, help="RF only: number of trees.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_path=args.data,
        tree_path=args.tree,
        out_dir=args.out,
        reps=args.reps,
        base_seed=args.seed,
        test_size=args.test_size,
        save_every=args.save_every,
        model=args.model,
        experiment=args.experiment,
        reduce_r=args.reduce_r,
        rf_n_estimators=args.rf_trees,
    )