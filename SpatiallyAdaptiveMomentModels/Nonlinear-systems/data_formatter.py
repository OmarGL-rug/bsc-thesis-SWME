import numpy as np
import pandas as pd
from pathlib import Path
from os import getcwd
from itertools import product
import re

# ---------- configuration ----------
SAVE_DATA = 1
height_points = 100
z_values = np.linspace(0, 1, height_points)
reference_order = 6

data_dir = Path(getcwd()) / "Data-processing/Output/Omar/20260720-Final"
out_dir  = Path(getcwd()) / "Data-processing/Results/Omar/"
out_dir.mkdir(parents=True, exist_ok=True)


# ---------- helper functions ----------
def parse_filename(filename: str):
    """
    Extract metadata from filename.
    Expected format:
        [time-]initialCondition_<str>-density_<int>-height_<float_comma>-resolution_<int>-order_<int>.csv

    Returns dict with keys:
        is_time_file, initial_condition,
        density, height (float), resolution, order
    """
    stem = Path(filename).stem  # remove extension
    # Optional 'time-' prefix
    if stem.startswith('time-'):
        is_time = True
        stem = stem[5:]          # remove 'time-'
    else:
        is_time = False

    # Regex: initialCondition_<base>-density_<digits>-height_<digits,comma>-resolution_<digits>-order_<digits>
    pattern = r'^initialCondition_([A-Za-z0-9_]+)-density_(\d+)-height_([\d,]+)-resolution_(\d+)-order_(\d+)$'
    match = re.match(pattern, stem)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")

    init_condition = match.group(1)
    density = match.group(2)
    height = match.group(3).replace(',', '.')   # convert comma to dot
    resolution = match.group(4)
    order = match.group(5)

    return {
        'is_time_file': is_time,
        'initial_condition': init_condition,
        'density': int(density),
        'height': float(height),
        'resolution': int(resolution),
        'order': int(order),
    }

def insert_meta_data(df: pd.DataFrame, meta: dict):
    df.insert(0, "Initial Condition", meta['initial_condition'])
    df.insert(1, "Plant Density", meta['density'])
    df.insert(2, "Resolution", meta['resolution'])
    df.insert(3, "Plant Height", meta['height'])
    df.insert(4, "Order", meta['order'])

def compute_error(error_type: str, data: pd.DataFrame, reference: pd.DataFrame)->np.ndarray:
    """
    Compute the errors.
        L2
        LInf
    """
    data, reference = np.asarray(data), np.asarray(reference)
    if error_type == 'LInf':
        return np.max(np.absolute(data - reference), axis=0)
    elif error_type == "L2":
        return np.sqrt(np.sum(np.square(data - reference) / data.shape[0], axis=0)) 

def compute_vertical_velocity_profile(order: int, 
                                      values: pd.DataFrame,
                                      z: np.ndarray) -> np.ndarray:
        """
        reconstructs the vertical velocity profile from the moment values and evaluates the velocity profile pointwise

        Parameters
        ----------
        order: integer
            order of the model
        values: np.ndarray (2D)
            2D numpy array containing the values of the variables in each mesh cell
        z: 
            the locations in vertical direction in which the velocity is computed
        
        Returns
        -------
        velocity_profile: numpy 2D array
            lateral velocity evaluated in in each point in z_points in z-direction

        """
        ones = np.ones_like(z)

        # base term: repeat "Average Velocity" across all z columns
        avg_vel = values["Average Velocity"].to_numpy()[:, None]      # (n_cells, 1)
        velocity_profile = np.tile(avg_vel, (1, z.size))               # (n_cells, len(z))

        # Legendre-derived polynomial terms, one per order
        polynomials = {
            1: ones - 2*z,
            2: ones - 6*z + 6*z**2,
            3: ones - 12*z + 30*z**2 - 20*z**3,
            4: ones - 20*z + 90*z**2 - 140*z**3 + 70*z**4,
            5: ones - 30*z + 210*z**2 - 560*z**3 + 630*z**4 - 252*z**5,
            6: ones - 42*z + 420*z**2 - 1680*z**3 + 3150*z**4 - 2772*z**5 + 924*z**6,
        }

        for k in range(1, order + 1):
            moment_vals = values[f"Moment {k}"].to_numpy()[:, None]   # (n_cells, 1)
            velocity_profile += moment_vals * polynomials[k][None, :]  # broadcast -> (n_cells, len(z))

        velocity_profile = pd.DataFrame(data=velocity_profile,
                                        columns=[str(z_val) for z_val in z]
                                    )
        return velocity_profile


data_frames = []
time_frames = []
profile_frames = []

for filepath in sorted(data_dir.glob("*.csv")):
    meta = parse_filename(filepath.name)

    order = meta['order']

    if meta['is_time_file']:
        tf = pd.read_csv(filepath, header=None, names=["Time"])
        insert_meta_data(tf, meta)
        time_frames.append(tf)
        continue
    
    # Non-time file

    data_columns = ["x position", "Water Height", "Average Velocity"] + \
                   [f"Moment {o+1}" for o in range(order)]
    
    df = pd.read_csv(filepath, header=None, names=data_columns)

    # Compute velocity profile
    profiles = compute_vertical_velocity_profile(order, df, z_values)

    insert_meta_data(df=df, meta=meta) 
    insert_meta_data(df=profiles, meta=meta)

    data_frames.append(df)
    profile_frames.append(profiles)

data = pd.concat(data_frames, ignore_index=True)
time = pd.concat(time_frames, ignore_index=True)
profile = pd.concat(profile_frames, ignore_index=True, axis=0)

# Compute errors
KEY_COLS = ["Initial Condition", "Plant Density", "Resolution", "Plant Height"]
available_errors = ["L2", "LInf"]

error_records = []
for keys, group in data.groupby(KEY_COLS):
    ic, density, resolution, height = keys

    ref_mask = group["Order"] == reference_order
    ref_idx = group.loc[ref_mask].index

    reference_solution = data.loc[ref_idx, ["Water Height", "Average Velocity"]]
    reference_u_profile = profile.loc[ref_idx, "0.0":"1.0"]

    for order in sorted(group.loc[~ref_mask, "Order"].unique()):
        order_idx = group.loc[group["Order"] == order].index
        
        compared_solution = data.loc[order_idx, ["Water Height", "Average Velocity"]]
        compared_u_profile = profile.loc[order_idx, "0.0":"1.0"]

        for error_type in available_errors:
            sol_err = compute_error(error_type, compared_solution, reference_solution)
            u_err = compute_error(error_type, compared_u_profile, reference_u_profile)

            error_records.append({
                "Initial Condition": ic,
                "Plant Density": density,
                "Resolution": resolution,
                "Plant Height": height,
                "Order": order,
                "Error Type": error_type,
                "Water Height": sol_err[0],
                "Average Velocity": sol_err[1],
                "U profile": np.average(u_err),
                "std U profile": np.std(u_err) / np.sqrt(len(u_err)),
            })

error = pd.DataFrame(error_records)

print("Saving data to files")

if SAVE_DATA:
    for init_cond in time["Initial Condition"].unique():
        mask = lambda array: (array["Initial Condition"] == init_cond)

        data[mask(data)].to_csv(out_dir / f"01-data-{init_cond}.csv", index=False)
        time[mask(time)].to_csv(out_dir / f"02-time-{init_cond}.csv", index=False)
        profile[mask(profile)].to_csv(out_dir / f"03-u_profile-{init_cond}.csv", index=False)
        error[mask(error)].to_csv(out_dir / f"04-error-{init_cond}.csv", index=False)
