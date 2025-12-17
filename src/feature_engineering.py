import numpy as np
import pandas as pd

def compute_cycle_features(
    df,
    time_col,
    nominal_capacity_ah,
    current_col=None,
    voltage_col=None,
    capacity_col=None,
    temperature_col=None,
):
    """
    Compute per-cycle battery features from clean row-level data.

    Parameters
    ----------
    df : pd.DataFrame
        Clean, preprocessed row-level data.
    cycle_col : str
        Column name for cycle index (e.g., Cycle or Cycle_Index_Global).
    time_col : str
        Column name for time in seconds.
    nominal_capacity_ah : float
        Nominal cell capacity (Ah) for C-rate calculation.
    current_col : str or None
        Column name for current (A).
    voltage_col : str or None
        Column name for voltage (V).
    capacity_col : str or None
        Column name for cumulative capacity (Ah).
    temperature_col : str or None
        Column name for temperature (°C), optional.

    Returns
    -------
    pd.DataFrame
        Per-cycle feature table.
    """

    # ==============================
    # 7. PER-CYCLE FEATURE CALCULATION
    # ==============================

    cycle_features = []

    for (cell_id, cycle), g in df.groupby(["cell_id", "Cycle_Index_Global"]):

        g = g.sort_values(time_col).copy()

        # delta time
        g["dt"] = g[time_col].diff().fillna(0)

        # C-rate
        g["C_rate"] = g[current_col].abs() / nominal_capacity_ah
        C_rate_mean = g["C_rate"].mean()
        C_rate_max  = g["C_rate"].max()

        # Charge / discharge masks
        charge_mask = g[current_col] > 0
        discharge_mask = g[current_col] < 0

        charge_data = g[charge_mask]
        discharge_data = g[discharge_mask]

        # capacities (NOTE: if capacity_col is unreliable, integrate current instead)
        Q_charge = charge_data[capacity_col].max() if not charge_data.empty else 0.0
        Q_discharge = discharge_data[capacity_col].max() if not discharge_data.empty else 0.0

        # coulombic efficiency
        eta = (Q_discharge / Q_charge * 100.0) if Q_charge > 0 else np.nan

        # cycle duration
        duration_s = g[time_col].iloc[-1] - g[time_col].iloc[0]

        # energy throughput
        g["dE_Wh"] = (g[current_col] * g[voltage_col]).abs() * g["dt"] / 3600.0
        energy_throughput = g["dE_Wh"].sum()

        # temperature stats
        if temperature_col is not None and temperature_col in g.columns:
            temp_min = g[temperature_col].min()
            temp_mean = g[temperature_col].mean()
            temp_max = g[temperature_col].max()
        else:
            temp_min = temp_mean = temp_max = np.nan

        cycle_features.append({
            "cell_id": cell_id,
            "Cycle": int(cycle),
            "C_rate_mean": C_rate_mean,
            "C_rate_max": C_rate_max,
            "Charge_Ah": Q_charge,
            "Discharge_Ah": Q_discharge,
            "Coulombic_Efficiency_%": eta,
            "Cycle_Duration_s": duration_s,
            "Energy_Throughput_Wh": energy_throughput,
            "Temp_Min_C": temp_min,
            "Temp_Mean_C": temp_mean,
            "Temp_Max_C": temp_max,
            "Num_points": len(g),
        })

    cycle_df = pd.DataFrame(cycle_features)


    return cycle_df


def filter_cycle_features(
    cycle_df,
    *,
    min_coul_eff=80.0,
    max_coul_eff=100.0,
    max_capacity_increase=0.05,
    max_capacity_jump=0.2,
    max_c_rate=5.0,
    min_cycle_duration_s=60,
    min_num_points=20,
):
    """
    Apply sanity and anomaly filters to per-cycle battery features.

    Parameters
    ----------
    cycle_df : pd.DataFrame
        Per-cycle feature table.
    min_coul_eff : float
        Minimum acceptable coulombic efficiency (%).
    max_coul_eff : float
        Maximum acceptable coulombic efficiency (%).
    max_capacity_increase : float
        Maximum allowed increase in discharge capacity (Ah) between cycles.
    max_capacity_jump : float
        Maximum allowed absolute capacity jump (Ah) between cycles.
    max_c_rate : float
        Maximum allowed mean C-rate.
    min_cycle_duration_s : float
        Minimum allowed cycle duration (seconds).
    min_num_points : int
        Minimum number of raw data points per cycle.

    Returns
    -------
    pd.DataFrame
        Filtered per-cycle feature table with Delta_Q column retained.
    """

    df = cycle_df.copy()

    # -------------------------
    # Basic physical filters
    # -------------------------
    df = df[
        (df["Charge_Ah"] > 0) &
        (df["Discharge_Ah"] > 0) &
        (df["Coulombic_Efficiency_%"].between(min_coul_eff, max_coul_eff))
    ]

    # Sort for per-cell differencing
    df = df.sort_values(["cell_id", "Cycle"])

    # -------------------------
    # Capacity anomaly filters
    # -------------------------
    df["Delta_Q"] = df.groupby("cell_id")["Discharge_Ah"].diff()

    # Capacity should not increase significantly
    df = df[df["Delta_Q"] < max_capacity_increase]

    # Remove unrealistic jumps (both directions)
    df = df[df["Delta_Q"].abs() < max_capacity_jump]

    # -------------------------
    # Operational sanity filters
    # -------------------------
    df = df[df["C_rate_mean"] < max_c_rate]
    df = df[df["Cycle_Duration_s"] > min_cycle_duration_s]
    df = df[df["Num_points"] > min_num_points]

    df = df.reset_index(drop=True)

    return df
