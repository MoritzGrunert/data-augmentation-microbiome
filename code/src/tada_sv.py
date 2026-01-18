from typing import Tuple
import numpy as np
import pandas as pd
import dendropy

class TADASV:

    def __init__(
        self,
        tree: dendropy.Tree,
        k: int = 5,
        q: int = 2,
        random_state: int = 0,
        sampling_strategy: str = "auto"
    ) -> None:

        self.tree = tree
        self.k = k
        self.q = q
        self.random_state = random_state 
        self.sampling_strategy = sampling_strategy
        self.meta = None

    def fit_resample(self, X, y) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.sampling_strategy == "auto":
            X_aug, meta = tada_sv_augment(X, self.tree, self.k, self.random_state)
            y_aug =  meta.set_index("sample_id")["source"].map(y)
            self.meta = meta
        elif self.sampling_strategy == "balance":
            X_aug, y_aug = tada_sv_balance(X, y, self.tree, seed=self.random_state)
        elif self.sampling_strategy == "balance++":
            X_aug, y_aug = tada_sv_balance_plus(X, y, self.tree, q=self.q, seed=self.random_state)

        return X_aug, y_aug 


def estimate_p_left(tree, counts_dict, default=0.5):
    """
    Schätzt p_left(u) für jeden inneren Knoten u aus EINEM Sample
    """
    p_left = {}
    internal_idx = 0

    def dfs(node):
        nonlocal internal_idx
        if node.is_leaf():
            return int(counts_dict.get(node.taxon.label, 0))

        children = node.child_nodes()
        if len(children) != 2:
            raise ValueError("Tree must be strictly binary")

        my_idx = internal_idx
        internal_idx += 1

        left_sum = dfs(children[0])
        right_sum = dfs(children[1])
        total = left_sum + right_sum

        p_left[my_idx] = left_sum / total if total > 0 else default
        return total

    dfs(tree.seed_node)
    return p_left

def generate_one_sample(tree, p_left, N, rng):
    """
    Generiert EIN synthetisches Sample (TADA-SV)
    """
    out = {}
    internal_idx = 0

    def recurse(node, c_u):
        nonlocal internal_idx
        if node.is_leaf():
            otu = node.taxon.label
            out[otu] = out.get(otu, 0) + int(c_u)
            return

        children = node.child_nodes()
        if len(children) != 2:
            raise ValueError("Tree must be strictly binary")

        my_idx = internal_idx
        internal_idx += 1

        p = float(p_left.get(my_idx, 0.5))
        p = min(max(p, 0.0), 1.0)

        c_left = rng.binomial(c_u, p)
        recurse(children[0], c_left)
        recurse(children[1], c_u - c_left)

    recurse(tree.seed_node, int(N))
    return out

def tada_sv_augment(X, tree, k=5, seed=0):
    rng = np.random.default_rng(seed)
    otu_ids = list(X.columns)

    rows = []
    ids = []
    meta = []

    # Originale
    for sid in X.index:
        rows.append(X.loc[sid].values)
        ids.append(sid)
        meta.append({"sample_id": sid, "source": sid, "type": "original"})

    # Synthetische
    for sid in X.index:
        counts = X.loc[sid].to_dict()
        N = int(sum(counts.values()))
        p_left = estimate_p_left(tree, counts)

        for j in range(k):
            syn = generate_one_sample(tree, p_left, N, rng)
            vec = [syn.get(otu, 0) for otu in otu_ids]

            new_id = f"{sid}__tadaSV{j}"
            rows.append(vec)
            ids.append(new_id)
            meta.append({"sample_id": new_id, "source": sid, "type": "synthetic"})

    X_aug = pd.DataFrame(rows, index=ids, columns=otu_ids)
    meta = pd.DataFrame(meta)
    return X_aug, meta

def estimate_p_left_dendropy(tree, counts_dict, default=0.5):
    p_left = {}
    internal_idx = 0

    def dfs(node):
        nonlocal internal_idx
        if node.is_leaf():
            return int(counts_dict.get(node.taxon.label, 0))

        children = node.child_nodes()
        if len(children) != 2:
            raise ValueError("Tree must be strictly binary for TADA-SV.")

        my_idx = internal_idx
        internal_idx += 1

        left_sum = dfs(children[0])
        right_sum = dfs(children[1])
        total = left_sum + right_sum

        p_left[my_idx] = left_sum / total if total > 0 else default
        return total

    dfs(tree.seed_node)
    return p_left


def generate_one_sample_dendropy(tree, p_left, N, rng):
    out = {}
    internal_idx = 0

    def recurse(node, c_u):
        nonlocal internal_idx
        if node.is_leaf():
            otu = node.taxon.label
            out[otu] = out.get(otu, 0) + int(c_u)
            return

        children = node.child_nodes()
        if len(children) != 2:
            raise ValueError("Tree must be strictly binary for TADA-SV.")

        my_idx = internal_idx
        internal_idx += 1

        p = float(p_left.get(my_idx, 0.5))
        p = min(max(p, 0.0), 1.0)

        c_left = int(rng.binomial(n=int(c_u), p=p))
        recurse(children[0], c_left)
        recurse(children[1], int(c_u) - c_left)

    recurse(tree.seed_node, int(N))
    return out

def tada_sv_balance(X, y, tree, seed=0):
    """
    Erzeugt genau so viele synthetische Samples, dass jede Klasse so viele Samples hat
    wie die größte Klasse. (Balance)
    """
    rng = np.random.default_rng(seed)

    X = X.copy()
    y = y.copy()

    classes = sorted(pd.unique(y))
    counts = y.value_counts().to_dict()
    target = max(counts.values())

    otu_ids = list(X.columns)

    new_rows = []
    new_ids = []
    new_labels = []

    # für jede Klasse fehlende Samples auffüllen
    for c in classes:
        n_c = counts.get(c, 0)
        need = target - n_c
        if need <= 0:
            continue

        idx_c = y.index[y == c].to_list()
        if len(idx_c) == 0:
            raise ValueError(f"Class {c} has 0 samples; cannot oversample with TADA-SV.")

        # wir erzeugen 'need' synthetische Samples durch Ziehen von source-samples (mit replacement)
        sources = rng.choice(idx_c, size=need, replace=True)

        for j, src_id in enumerate(sources):
            counts_dict = X.loc[src_id].to_dict()
            N = int(sum(counts_dict.values()))
            p_left = estimate_p_left_dendropy(tree, counts_dict)

            syn = generate_one_sample_dendropy(tree, p_left, N, rng)
            vec = [int(syn.get(otu, 0)) for otu in otu_ids]

            new_id = f"{src_id}__tadaSV_bal_{j}"
            new_rows.append(vec)
            new_ids.append(new_id)
            new_labels.append(c)  # Label wird geerbt

    X_syn = pd.DataFrame(new_rows, index=new_ids, columns=otu_ids) if new_rows else X.iloc[0:0].copy()
    y_syn = pd.Series(new_labels, index=new_ids, name=y.name)

    X_out = pd.concat([X, X_syn], axis=0)
    y_out = pd.concat([y, y_syn], axis=0)

    return X_out, y_out

def tada_sv_balance_plus(X, y, tree, q=2, seed=0):
    """
    Balance++: sorgt für Balance und erhöht dann die Gesamtmenge:
    jede Klasse endet bei q * (max class size im ORIGINAL).
    """
    rng = np.random.default_rng(seed)

    X = X.copy()
    y = y.copy()

    classes = sorted(pd.unique(y))
    orig_counts = y.value_counts().to_dict()
    max_n = max(orig_counts.values())
    target = int(q * max_n)

    otu_ids = list(X.columns)

    new_rows = []
    new_ids = []
    new_labels = []

    for c in classes:
        n_c = orig_counts.get(c, 0)
        need = target - n_c
        if need <= 0:
            continue

        idx_c = y.index[y == c].to_list()
        if len(idx_c) == 0:
            raise ValueError(f"Class {c} has 0 samples; cannot oversample with TADA-SV.")

        sources = rng.choice(idx_c, size=need, replace=True)

        for j, src_id in enumerate(sources):
            counts_dict = X.loc[src_id].to_dict()
            N = int(sum(counts_dict.values()))
            p_left = estimate_p_left_dendropy(tree, counts_dict)

            syn = generate_one_sample_dendropy(tree, p_left, N, rng)
            vec = [int(syn.get(otu, 0)) for otu in otu_ids]

            new_id = f"{src_id}__tadaSV_balpp_{j}"
            new_rows.append(vec)
            new_ids.append(new_id)
            new_labels.append(c)

    X_syn = pd.DataFrame(new_rows, index=new_ids, columns=otu_ids) if new_rows else X.iloc[0:0].copy()
    y_syn = pd.Series(new_labels, index=new_ids, name=y.name)

    X_out = pd.concat([X, X_syn], axis=0)
    y_out = pd.concat([y, y_syn], axis=0)

    return X_out, y_out
