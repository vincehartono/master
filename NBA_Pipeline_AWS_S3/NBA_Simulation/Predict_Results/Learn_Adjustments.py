import os
import sys
import pandas as pd
import numpy as np


def _ridge(X: np.ndarray, y: np.ndarray, lam: float, no_penalize: int = 1) -> np.ndarray:
    # no_penalize = number of leading columns not penalized (e.g., intercept)
    XT = X.T
    n_features = X.shape[1]
    I = np.eye(n_features, dtype=float)
    if no_penalize > 0:
        I[:no_penalize, :no_penalize] = 0.0
    A = XT @ X + lam * I
    b = XT @ y
    return np.linalg.pinv(A) @ b


def learn_team_adjustments(bt_path: str, out_dir: str, lam: float = 5.0) -> str:
    bt = pd.read_csv(bt_path)
    required = {"home", "away", "home_pts", "away_pts", "home_pred", "away_pred"}
    if not required.issubset(bt.columns):
        raise ValueError(f"backtest_results.csv missing columns. Need {required}")

    teams = sorted(set(bt["home"].astype(str)) | set(bt["away"].astype(str)))
    t2i = {t: i for i, t in enumerate(teams)}

    def fit_side(side: str):
        # side in {"home","away"}
        if side == "home":
            y = (bt["home_pts"] - bt["home_pred"]).astype(float).to_numpy()
            off_idx = bt["home"].astype(str).map(t2i).to_numpy()
            def_idx = bt["away"].astype(str).map(t2i).to_numpy()
            # home-advantage per-team one-hot indices (home team)
            homeadv_idx = off_idx.copy()
        else:
            y = (bt["away_pts"] - bt["away_pred"]).astype(float).to_numpy()
            off_idx = bt["away"].astype(str).map(t2i).to_numpy()
            def_idx = bt["home"].astype(str).map(t2i).to_numpy()
            homeadv_idx = None

        n = len(y)
        k = len(teams)
        # X = [intercept | off_onehot | (-def_onehot) | homeadv_onehot (home only)]
        has_homeadv = (side == "home")
        X = np.zeros((n, 1 + k + k + (k if has_homeadv else 0)), dtype=float)
        X[:, 0] = 1.0
        X[np.arange(n), 1 + off_idx] = 1.0
        X[np.arange(n), 1 + k + def_idx] = -1.0
        if has_homeadv:
            X[np.arange(n), 1 + k + k + homeadv_idx] = 1.0
        b = _ridge(X, y, lam=lam, no_penalize=1)
        bias = float(b[0])
        off_adj = b[1:1 + k]
        def_adj = b[1 + k: 1 + k + k]
        home_adv_vec = b[1 + k + k:] if has_homeadv else None
        return bias, off_adj, def_adj, home_adv_vec

    bias_h, off_h, def_h, home_adv_vec = fit_side("home")
    bias_a, off_a, def_a, _ = fit_side("away")

    # Average home/away fits to stabilize
    off_adj = (off_h + off_a) / 2.0
    def_adj = (def_h + def_a) / 2.0
    bias_home = bias_h
    bias_away = bias_a

    # Write outputs
    os.makedirs(out_dir, exist_ok=True)
    adj_path = os.path.join(out_dir, "team_adjustments.csv")
    df_adj = pd.DataFrame({
        "team": teams,
        "off_adj": off_adj,
        "def_adj": def_adj,
    })
    if home_adv_vec is not None and len(home_adv_vec) == len(teams):
        df_adj["home_adv_adj"] = home_adv_vec
    df_adj.to_csv(adj_path, index=False)

    params_path = os.path.join(out_dir, "model_adjustments.csv")
    pd.DataFrame([
        {"param": "bias_home", "value": bias_home},
        {"param": "bias_away", "value": bias_away},
        {"param": "lambda", "value": float(lam)},
    ]).to_csv(params_path, index=False)

    return adj_path


def main():
    # Expect to be run from Predict_Results folder
    here = os.path.dirname(__file__)
    bt_path = os.path.join(here, "backtest_results.csv")
    if not os.path.exists(bt_path):
        print("backtest_results.csv not found; run Predict_Results.py first to generate backtest.")
        return
    adj_path = learn_team_adjustments(bt_path, here)
    print(f"Wrote adjustments to {adj_path}")


if __name__ == "__main__":
    main()
