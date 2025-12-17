# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function for exponential expression
def model_exp(n, Q_inf, Q0, k):
    return Q_inf + (Q0 - Q_inf) * np.exp(-k * n)

# Function for square root expression
def model_sqrt(n, Q0, a):
    return Q0 - a * np.sqrt(n)

# Function for capacity function
def fit_capacity_models(test_cycles, cell_id):
    
    # Filter data for the specific cell
    data = test_cycles[test_cycles["cell_id"] == cell_id].copy()

    # Cycle index (global cycle number)
    n = data["Cycle"].values.astype(float)
    Q = data["Discharge_Ah"].values.astype(float)

    # Sort by cycle order
    idx = np.argsort(n)
    n, Q = n[idx], Q[idx]

    # ----- Initial guesses -----
    Q0_guess = Q[0]
    Qinf_guess = Q[-1]
    k_guess = 1e-3
    a_guess = (Q[0] - Q[-1]) / np.sqrt(n[-1] + 1e-6)

    # ----- Fit exponential model -----
    try:
        popt_exp, _ = curve_fit(
            model_exp, n, Q,
            p0=[Qinf_guess, Q0_guess, k_guess],
            maxfev=10000
        )
    except RuntimeError:
        popt_exp = [np.nan, np.nan, np.nan]

    # ----- Fit sqrt model -----
    try:
        popt_sqrt, _ = curve_fit(
            model_sqrt, n, Q,
            p0=[Q0_guess, a_guess],
            maxfev=10000
        )
    except RuntimeError:
        popt_sqrt = [np.nan, np.nan]

    # ----- Predictions -----
    Q_exp_fit  = model_exp(n,  *popt_exp)
    Q_sqrt_fit = model_sqrt(n, *popt_sqrt)

    # ----- Compute RMSE -----
    rmse_exp  = np.sqrt(np.mean((Q - Q_exp_fit)**2))
    rmse_sqrt = np.sqrt(np.mean((Q - Q_sqrt_fit)**2))

    return {
        "n": n,
        "Q": Q,
        "popt_exp": popt_exp,
        "popt_sqrt": popt_sqrt,
        "Q_exp_fit": Q_exp_fit,
        "Q_sqrt_fit": Q_sqrt_fit,
        "rmse_exp": rmse_exp,
        "rmse_sqrt": rmse_sqrt,
    }


def select_equally_spaced_cycles(df, cell_id, cycle_col, n_cycles=6, random_seed=42):
    """
    Select n_cycles approximately equally spaced cycle indices for a given cell.
    """
    rng = np.random.default_rng(random_seed)

    available_cycles = (
        df[df["cell_id"] == cell_id][cycle_col]
        .dropna()
        .unique()
    )

    available_cycles = np.sort(available_cycles)

    if len(available_cycles) < n_cycles:
        return available_cycles.tolist()

    # equally spaced indices
    indices = np.linspace(0, len(available_cycles) - 1, n_cycles, dtype=int)

    selected_cycles = available_cycles[indices]

    return selected_cycles.tolist()



def compute_dvdq(df, voltage_col, capacity_col):
    """
    Compute dV/dQ using numerical differentiation.
    """
    df = df.sort_values(capacity_col)

    dV = np.gradient(df[voltage_col].values)
    dQ = np.gradient(df[capacity_col].values)

    dvdq = dV / dQ

    return df[capacity_col].values, dvdq


def plot_dvdq_two_cells_side_by_side(
    df,
    cell_ids,
    cycle_col,
    voltage_col,
    capacity_col,
    current_col,
    mode="discharge",
    n_cycles=6
):
    """
    Plot dV/dQ evolution for two cells side by side.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, cell_id in zip(axes, cell_ids):

        cycles_to_plot = select_equally_spaced_cycles(
            df=df,
            cell_id=cell_id,
            cycle_col=cycle_col,
            n_cycles=n_cycles
        )

        for cycle in cycles_to_plot:
            cycle_df = df[
                (df["cell_id"] == cell_id) &
                (df[cycle_col] == cycle)
            ].copy()

            # mode filtering
            if mode == "discharge":
                cycle_df = cycle_df[cycle_df[current_col] < 0]
            else:
                cycle_df = cycle_df[cycle_df[current_col] > 0]

            if len(cycle_df) < 10:
                continue

            Q, dvdq = compute_dvdq(
                cycle_df,
                voltage_col=voltage_col,
                capacity_col=capacity_col
            )

            ax.plot(Q, dvdq, label=f"Cycle {cycle}", alpha=0.8)

        ax.set_title(f"Cell {cell_id} – dV/dQ Evolution")
        ax.set_xlabel("Capacity (Ah)")
        ax.grid(True)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("dV/dQ (V/Ah)")
    plt.tight_layout()
    plt.show()



def fit_statistical_models(test_cycles, cell_id):
    data = test_cycles[test_cycles["cell_id"] == cell_id].copy()
    n = data["Cycle"].values.reshape(-1, 1)
    Q = data["Discharge_Ah"].values

    # Linear: Q = a + b*n
    lin = LinearRegression()
    lin.fit(n, Q)
    Q_lin = lin.predict(n)
    rmse_lin = np.sqrt(mean_squared_error(Q, Q_lin))

    # Polynomial: Q = a + b*n + c*n^2
    n_poly = np.hstack([n, n**2])
    poly = LinearRegression()
    poly.fit(n_poly, Q)
    Q_poly = poly.predict(n_poly)
    rmse_poly = np.sqrt(mean_squared_error(Q, Q_poly))

    return {
        "n": n.flatten(),
        "Q": Q,
        "Q_lin": Q_lin,
        "Q_poly": Q_poly,
        "rmse_lin": rmse_lin,
        "rmse_poly": rmse_poly,
        "lin_model": lin,
        "poly_model": poly,
    }


def fit_ml_regressor(cycle_df, cell_id, min_samples=10):
    """
    ML regression for capacity prediction.
    Automatically removes all-NaN or mostly-NaN features.
    Works even when temperature columns are fully NaN.
    """

    data = cycle_df[cycle_df["cell_id"] == cell_id].copy()

    # Candidate features
    candidate_features = [
        "Cycle",
        "C_rate_mean",
        "C_rate_max",
        "Energy_Throughput_Wh",
        "Cycle_Duration_s",
        # DO NOT include Temp_* here since they are all NaN
    ]

    # Step 1 — Keep only features that exist in data
    existing_features = [f for f in candidate_features if f in data.columns]

    # Step 2 — Remove features that are fully or mostly NaN
    good_features = []
    for col in existing_features:
        missing_ratio = data[col].isna().mean()
        if missing_ratio < 0.9:  # allow up to 90% missing
            good_features.append(col)

    if len(good_features) == 0:
        print(f"No usable features found for ML in {cell_id}")
        return None

    # Step 3 — Drop rows missing target or selected features
    data = data.dropna(subset=good_features + ["Discharge_Ah"])

    if len(data) < min_samples:
        print(f"Not enough samples for ML for {cell_id}: only {len(data)} usable rows after filtering")
        return None

    # Prepare X, y
    X = data[good_features].values
    y = data["Discharge_Ah"].values

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.80, random_state=42
    )

    # Train RF
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predict & RMSE
    y_pred = rf.predict(X_val)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred))

    return {
        "cell_id": cell_id,
        "rmse_rf": rmse_rf,
        "n_samples": len(data),
        "features_used": good_features,
        "model": rf
    }