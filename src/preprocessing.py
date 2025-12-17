# Required imports
import glob
import pandas as pd
import numpy as np
import datetime as dt

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def time_to_seconds(val):
    """Convert various time formats to seconds."""
    if pd.isna(val):
        return np.nan

    # Case 1: already a datetime.time (your original issue)
    if isinstance(val, dt.time):
        return (
            val.hour * 3600
            + val.minute * 60
            + val.second
            + val.microsecond / 1e6
        )

    # Case 2: already a timedelta
    if isinstance(val, dt.timedelta):
        return val.total_seconds()

    # Convert to string for further parsing
    s = str(val)

    # Case 3: special format "d-hh:mm:ss"  (e.g. "1-11:19:45")
    # split on the first '-' only
    if "-" in s and s.count(":") == 2:
        day_part, time_part = s.split("-", 1)
        if day_part.isdigit():
            days = int(day_part)
            # parse hh:mm:ss
            t = dt.datetime.strptime(time_part, "%H:%M:%S").time()
            return (
                days * 86400
                + t.hour * 3600
                + t.minute * 60
                + t.second
            )

    # Case 4: normal hh:mm:ss or other formats that to_timedelta understands
    td = pd.to_timedelta(s)
    return td.total_seconds()


def find_col(df, substr_options):
    """
    Find the first column whose name contains any of the substrings in substr_options.
    Case-insensitive.
    """
    for col in df.columns:
        for s in substr_options:
            if s.lower() in col.lower():
                return col
    return None


def load_and_preprocess(raw_path_pattern):
    """
    Load raw Excel cycling data and return a clean DataFrame.
    
    Parameters
    ----------
    raw_path_pattern : str
        Glob pattern to raw Excel files (e.g. "../data/raw/LR1865SZ*.xlsx")
    
    Returns
    -------
    pd.DataFrame
        Concatenated raw data with cell_id and Source_File added
    """
    file_list = sorted(glob.glob(raw_path_pattern))
    if not file_list:
            raise FileNotFoundError("No input files found")

    dfs = []
    for idx, file in enumerate(file_list, start=1):
        print("Loading:", file)
        df_temp = pd.read_excel(file)
        
        # Create cell ID automatically
        df_temp["cell_id"] = f"Cell_{idx}"
        
        # Track original file name (optional but useful)
        df_temp["Source_File"] = file
        
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # -------------------------
    # Column cleanup
    # -------------------------

    # Time columns
    if "Test_Time(s)" in df.columns:
        test_time_col = "Test_Time(s)"
    else:
        # fallback: try to guess
        test_time_col = find_col(df, ["Test_Time", "Time"])
        if test_time_col is None:
            raise KeyError("Could not find Test_Time(s) column")

    # Date_Time
    if "Date_Time" not in df.columns:
        raise KeyError("Expected a 'Date_Time' column in the data")

    # Measurement columns (auto-detect)
    current_col     = find_col(df, ["current"])
    capacity_col    = find_col(df, ["capacity"])
    voltage_col     = find_col(df, ["voltage"])
    energy_col      = find_col(df, ["energy"])
    temperature_col = find_col(df, ["temp", "temperature"])

    # -------------------------
    # Numeric conversion
    # -------------------------

    # Cycle index
    if "Cycle_Index" in df.columns:
        cycle_col_raw = "Cycle_Index"
    else:
        # sometimes different naming
        cycle_col_raw = find_col(["cycle_index", "cycle"])
        if cycle_col_raw is None:
            raise KeyError("Could not find Cycle_Index column")

    # Parse Date_Time
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

    # Parse Test_Time(s) -> seconds
    df["Test_Time_sec"] = df[test_time_col].map(time_to_seconds)

    # Convert measurements to numeric
    for col in [current_col, capacity_col, voltage_col, energy_col, temperature_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cycle index to int
    df[cycle_col_raw] = pd.to_numeric(df[cycle_col_raw], errors="coerce")

    # Drop rows where we couldn't parse time or cycle index
    df = df[df["Test_Time_sec"].notna()]
    df = df[df[cycle_col_raw].notna()]

    # 6) After dropping NaNs, cast cycle index to integer (nullable)
    df[cycle_col_raw] = df[cycle_col_raw].astype("Int64")

    # -------------------------
    # Sorting
    # -------------------------
   
    df = df.sort_values(["cell_id", "Date_Time", "Test_Time_sec"]).reset_index(drop=True)

    # Convert local cycle to int
    df[cycle_col_raw] = df[cycle_col_raw].astype(int)

    # Create a new global cycle index for each cell
    df["Cycle_Index_Global"] = (
        df.groupby("cell_id")[cycle_col_raw]
        .rank(method="dense")   # preserves unique cycle order
        .astype(int) - 1         # start from 0
    )

    # ==============================
    # 6. BASIC SANITY FILTERS
    # ==============================

    df = df.drop_duplicates()

    # Filter insane voltage
    if voltage_col is not None:
        df = df[(df[voltage_col] > 0.0) & (df[voltage_col] < 5.0)]


    # Re-sort by true time for analysis
    df = df.sort_values(["Date_Time", "Test_Time_sec"]).reset_index(drop=True)

    # 3) Re-sort by cell + true time for analysis
    df = (
        df
        .sort_values(["cell_id", "Date_Time", "Test_Time_sec"])
        .reset_index(drop=True)
    )

    return df