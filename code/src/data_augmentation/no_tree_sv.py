from typing import Tuple
import numpy as np
import pandas as pd

class NoTreeSV:
    """
    Tree-freie Baseline, die pro Sample Sequencing/Sampling-Noise simuliert.

    Idee:
      - Originalsample: Counts c (Summe N)
      - Schätze p_i = c_i / N
      - Ziehe synthetisch: c' ~ Multinomial(N, p)   (sum-to-N bleibt erhalten)
    
    Optional:
      - smoothing (Laplace / pseudocount) verhindert p_i=0
    """

    def __init__(
        self,
        k: int = 5,
        q: int = 2,
        random_state: int = 0,
        sampling_strategy: str = "auto",
        smoothing: float = 0.0,
    ) -> None:
        self.k = k
        self.q = q
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.smoothing = float(smoothing)
        self.meta = None

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if self.sampling_strategy == "auto":
            X_aug, meta = notree_sv_augment(
                X, k=self.k, seed=self.random_state,
                smoothing=self.smoothing
            )
            # Label wird vom Source-Sample geerbt
            y_aug = meta.set_index("sample_id")["source"].map(y)
            self.meta = meta
            return X_aug, y_aug

        if self.sampling_strategy == "balance":
            return notree_sv_balance(
                X, y, seed=self.random_state,
                smoothing=self.smoothing
            )

        if self.sampling_strategy == "balance++":
            return notree_sv_balance_plus(
                X, y, q=self.q, seed=self.random_state,
                smoothing=self.smoothing
            )

        raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")


def _sample_multinomial_counts(
    counts: np.ndarray,
    rng: np.random.Generator,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    counts: (d,) non-negative counts
    Returns: new_counts (d,) with same total N (if N>0)
    """
    counts = counts.astype(float)
    N = int(counts.sum())
    d = counts.shape[0]

    if N <= 0:
        return np.zeros(d, dtype=int)

    # p_i schätzen (mit optionaler Glättung)
    p = counts + smoothing
    p_sum = p.sum()
    if p_sum <= 0:
        # fallback gleichverteilung
        p = np.ones(d) / d
    else:
        p = p / p_sum

    return rng.multinomial(N, p).astype(int)


def notree_sv_augment(
    X: pd.DataFrame,
    k: int = 5,
    seed: int = 0,
    smoothing: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    otu_ids = list(X.columns)

    rows = []
    ids = []
    meta = []

    # Originale behalten
    for sid in X.index:
        rows.append(X.loc[sid].values.astype(int))
        ids.append(sid)
        meta.append({"sample_id": sid, "source": sid, "type": "original"})

    # Synthetische pro Originalsample
    for sid in X.index:
        base = X.loc[sid].values.astype(float)
        for j in range(k):
            syn = _sample_multinomial_counts(
                base, rng,
                smoothing=smoothing,
            )
            new_id = f"{sid}__noTreeSV{j}"
            rows.append(syn)
            ids.append(new_id)
            meta.append({"sample_id": new_id, "source": sid, "type": "synthetic"})

    X_aug = pd.DataFrame(rows, index=ids, columns=otu_ids).astype(int)
    meta = pd.DataFrame(meta)
    return X_aug, meta


def notree_sv_balance(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    smoothing: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = X.copy()
    y = y.copy()

    classes = sorted(pd.unique(y))
    counts = y.value_counts().to_dict()
    target = max(counts.values())
    otu_ids = list(X.columns)

    new_rows, new_ids, new_labels = [], [], []

    for c in classes:
        n_c = counts.get(c, 0)
        need = target - n_c
        if need <= 0:
            continue

        idx_c = y.index[y == c].to_list()
        if len(idx_c) == 0:
            raise ValueError(f"Class {c} has 0 samples; cannot oversample.")

        sources = rng.choice(idx_c, size=need, replace=True)

        for j, src_id in enumerate(sources):
            base = X.loc[src_id].values.astype(float)
            syn = _sample_multinomial_counts(
                base, rng,
                smoothing=smoothing
            )
            new_id = f"{src_id}__noTreeSV_bal_{j}"
            new_rows.append(syn)
            new_ids.append(new_id)
            new_labels.append(c)

    X_syn = pd.DataFrame(new_rows, index=new_ids, columns=otu_ids).astype(int) if new_rows else X.iloc[0:0].copy()
    y_syn = pd.Series(new_labels, index=new_ids, name=y.name)

    return pd.concat([X, X_syn], axis=0), pd.concat([y, y_syn], axis=0)


def notree_sv_balance_plus(
    X: pd.DataFrame,
    y: pd.Series,
    q: int = 2,
    seed: int = 0,
    smoothing: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = X.copy()
    y = y.copy()

    classes = sorted(pd.unique(y))
    orig_counts = y.value_counts().to_dict()
    max_n = max(orig_counts.values())
    target = int(q * max_n)
    otu_ids = list(X.columns)

    new_rows, new_ids, new_labels = [], [], []

    for c in classes:
        n_c = orig_counts.get(c, 0)
        need = target - n_c
        if need <= 0:
            continue

        idx_c = y.index[y == c].to_list()
        if len(idx_c) == 0:
            raise ValueError(f"Class {c} has 0 samples; cannot oversample.")

        sources = rng.choice(idx_c, size=need, replace=True)

        for j, src_id in enumerate(sources):
            base = X.loc[src_id].values.astype(float)
            syn = _sample_multinomial_counts(
                base, rng,
                smoothing=smoothing
            )
            new_id = f"{src_id}__noTreeSV_balpp_{j}"
            new_rows.append(syn)
            new_ids.append(new_id)
            new_labels.append(c)

    X_syn = pd.DataFrame(new_rows, index=new_ids, columns=otu_ids).astype(int) if new_rows else X.iloc[0:0].copy()
    y_syn = pd.Series(new_labels, index=new_ids, name=y.name)

    return pd.concat([X, X_syn], axis=0), pd.concat([y, y_syn], axis=0)