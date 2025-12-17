# Imports
import numpy as np
from src.capacity_models import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def mark_residual_anomalies(cycle_df, test_cycles, use_model="sqrt", z_threshold=3.0):
    """
    For each cell, use the fitted degradation model (exp or sqrt) to compute
    expected capacity, then flag cycles whose residuals are large.
    Adds columns:
      - 'Q_hat_model'
      - 'residual_Ah'
      - 'residual_zscore'
      - 'anomaly_trend' (True/False)
    Returns an updated copy of cycle_df.
    """
    df_out = cycle_df.copy()

    df_out["Q_hat_model"] = np.nan
    df_out["residual_Ah"] = np.nan
    df_out["residual_zscore"] = np.nan
    df_out["anomaly_trend"] = False

    for cell_id in df_out["cell_id"].unique():
        # Fit model on CLEANED data (test_cycles)
        phys_res = fit_capacity_models(test_cycles, cell_id)
        n_clean = phys_res["n"]
        # use parameters
        if use_model == "exp":
            popt = phys_res["popt_exp"]
            model_func = model_exp
        else:
            popt = phys_res["popt_sqrt"]
            model_func = model_sqrt

        # Apply model to *all* cycles for this cell in cycle_df
        mask = (df_out["cell_id"] == cell_id)
        n_all = df_out.loc[mask, "Cycle"].values.astype(float)
        Q_all = df_out.loc[mask, "Discharge_Ah"].values.astype(float)

        Q_hat = model_func(n_all, *popt)
        residuals = Q_all - Q_hat

        # Z-score of residuals
        res_mean = np.nanmean(residuals)
        res_std = np.nanstd(residuals) if np.nanstd(residuals) > 0 else 1.0
        z = (residuals - res_mean) / res_std

        df_out.loc[mask, "Q_hat_model"] = Q_hat
        df_out.loc[mask, "residual_Ah"] = residuals
        df_out.loc[mask, "residual_zscore"] = z
        df_out.loc[mask, "anomaly_trend"] = np.abs(z) > z_threshold

    return df_out


def dbscan_cycle_anomalies(cycle_df, eps=0.5, min_samples=5):
    """
    Use DBSCAN per cell to cluster cycles based on multivariate features and
    label outlier cycles (cluster = -1).
    Adds:
      - 'cluster_label'
      - 'anomaly_dbscan' (True if label == -1)
    """
    df_out = cycle_df.copy()
    df_out["cluster_label"] = np.nan
    df_out["anomaly_dbscan"] = False

    # Feature set to use (only keep columns that exist)
    candidate_features = [
        "Discharge_Ah",
        "Coulombic_Efficiency_%",
        "C_rate_mean",
        "Energy_Throughput_Wh",
        "Cycle_Duration_s",
    ]
    existing_features = [c for c in candidate_features if c in df_out.columns]

    for cell_id in df_out["cell_id"].unique():
        mask = (df_out["cell_id"] == cell_id)
        data_cell = df_out.loc[mask].copy()

        # Drop rows that don't have these features
        data_cell = data_cell.dropna(subset=existing_features)

        if len(data_cell) < min_samples:
            print(f"Skipping DBSCAN for {cell_id}: not enough valid rows ({len(data_cell)})")
            continue

        X = data_cell[existing_features].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)  # -1 = outliers

        # Assign back
        df_out.loc[data_cell.index, "cluster_label"] = labels
        df_out.loc[data_cell.index, "anomaly_dbscan"] = (labels == -1)

    return df_out
