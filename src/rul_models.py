# Imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def model_square_root(n, Q0, a):
    return Q0 - a*np.sqrt(n)

def fit_square_root(cell_df):
    n = cell_df["Cycle"].values
    Q = cell_df["Discharge_Ah"].values
    guess = [Q[0], (Q[0]-Q[-1])/np.sqrt(max(n))]
    popt, _ = curve_fit(model_square_root, n, Q, p0=guess)
    Q0, a = popt

    # EOL when capacity hits 80% of nominal
    # here nominal ≈ Q0_fit (initial fitted capacity)
    Q_EOL = 0.8 * Q0
    n_eol = ((Q0 - Q_EOL) / a)**2

    return Q0, a, n_eol


def fit_square_root_and_rul_for_cell(test_cycles, cell_id, use_mid_life=False):
    """Fit sqrt model for one cell and compute EOL (80% of measured Q0)."""
    cell_data = test_cycles[test_cycles["cell_id"] == cell_id].copy()
    cell_data = cell_data.sort_values("Cycle")

    if use_mid_life:
        # optionally ignore very early cycles if they distort the fit
        cell_data = cell_data[cell_data["Cycle"] >= 5].copy()

    n = cell_data["Cycle"].values.astype(float)
    Q = cell_data["Discharge_Ah"].values.astype(float)

    # measured initial capacity (for EOL threshold)
    Q0_meas = Q.max()

    # initial guess for curve_fit
    p0 = [Q0_meas, 0.01]

    popt, _ = curve_fit(
        model_square_root,
        n,
        Q,
        p0=p0,
        maxfev=10000
    )
    Q0_fit, a_fit = popt

    # EOL at 80% of measured Q0
    Q_EOL = 0.8 * Q0_meas
    term = (Q0_fit - Q_EOL) / a_fit
    n_eol = term**2

    return {
        "cell_id": cell_id,
        "n": n,
        "Q": Q,
        "Q0_meas": Q0_meas,
        "Q0_fit": Q0_fit,
        "a_fit": a_fit,
        "Q_EOL": Q_EOL,
        "n_eol": n_eol,
    }


def lstm_predict_eol_from_fraction(
    cycle_df,
    cell_id,
    model_lstm,
    scaler,
    feature_cols,
    seq_len,
    eol_true,
    Q0_meas,
    obs_frac=0.4,
    frac_EOL=0.8,
    max_extra_cycles=300,
):
    """
    Estimate EOL for one cell using LSTM starting from an early-life window.

    eol_true:   reference EOL (from sqrt model) to decide where 'obs_frac' lies.
    Q0_meas:    measured initial capacity (Ah) for that cell (max Discharge_Ah).
    obs_frac:   fraction of true EOL at which we start forecasting (e.g. 0.4 = 40% life).
    frac_EOL:   EOL threshold as fraction of Q0_meas (0.8 = 80%).
    """

    # Extract this cell's history
    g = cycle_df[cycle_df["cell_id"] == cell_id].copy()
    g = g.sort_values("Cycle")
    g = g.dropna(subset=feature_cols)

    if len(g) <= seq_len:
        print(f"Not enough data for LSTM RUL on {cell_id}")
        return None

    cycles = g["Cycle"].values

    # Scale features with same scaler (use DataFrame -> avoids warning)
    g_scaled = pd.DataFrame(
        scaler.transform(g[feature_cols]),
        columns=feature_cols
    )

    # Where do we cut life? (observation point)
    obs_cycle_target = obs_frac * eol_true

    idx_candidates = np.where(cycles <= obs_cycle_target)[0]
    if len(idx_candidates) == 0:
        # if too early, start from first window
        obs_idx = seq_len
    else:
        obs_idx = idx_candidates[-1]
        if obs_idx < seq_len:
            obs_idx = seq_len

    # Window = last seq_len rows up to obs_idx
    window = g_scaled.iloc[obs_idx - seq_len : obs_idx].values
    current_cycle = cycles[obs_idx - 1]

    cap_idx = feature_cols.index("Discharge_Ah")
    Q_EOL = frac_EOL * Q0_meas

    extra_cycles = []
    extra_caps = []

    for _ in range(max_extra_cycles):
        x_in = window[np.newaxis, :, :]  # (1, seq_len, n_features)
        next_cap_scaled = model_lstm.predict(x_in, verbose=0)[0, 0]

        # new feature row: copy last row, replace capacity
        new_row = window[-1].copy()
        new_row[cap_idx] = next_cap_scaled

        # roll the window
        window = np.vstack([window[1:], new_row])

        # unscale JUST the capacity from this new_row
        temp_df = pd.DataFrame([new_row], columns=feature_cols)
        unscaled = scaler.inverse_transform(temp_df)[0, cap_idx]

        current_cycle += 1
        extra_cycles.append(current_cycle)
        extra_caps.append(unscaled)

        if unscaled <= Q_EOL:
            break

    if not extra_cycles:
        eol_lstm = None
    else:
        eol_lstm = extra_cycles[-1]

    return {
        "obs_frac": obs_frac,
        "start_cycle": cycles[obs_idx - 1],
        "EOL_cycle_LSTM": eol_lstm,
        "extra_cycles": np.array(extra_cycles),
        "extra_caps": np.array(extra_caps),
        "Q_EOL": Q_EOL,
    }


def eol_from_early_data(cycle_df, cell_id, obs_frac):
    """
    Fit sqrt capacity-fade model using only the first obs_frac of life
    and return the predicted EOL (cycle where capacity hits 80% Q0_fit).
    """
    g_full = cycle_df[cycle_df["cell_id"] == cell_id].sort_values("Cycle")
    max_cycle = g_full["Cycle"].max()

    cutoff_cycle = obs_frac * max_cycle
    g_early = g_full[g_full["Cycle"] <= cutoff_cycle].copy()

    if len(g_early) < 5:
        return None  # not enough points

    Q0_fit, a_fit, n_eol_pred = fit_square_root(g_early)
    return n_eol_pred

