import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pathlib import Path
from os import getcwd
from itertools import product
import re

# ---------- configuration ----------
SAVE_DATA = 1
height_points = 100
z_values = np.linspace(0, 1, height_points)

REFERENCE_ORDER = 6
REFERENCE_RESOLUTION = 10000

x1_bound, x2_bound = -2.5, 2.5
DOMAIN_LENGTH = x2_bound - x1_bound

# Order REFERENCE_ORDER (6) is excluded entirely below -- it's the same order
# as the reference itself, so its "error" is really just grid/interpolation
# noise, not a meaningful model-convergence data point. Order 5 (one below the
# reference order) is the highest order that still appears in the data, and is
# flagged (not excluded) near the top of the sweep resolutions, where it sits
# close enough to the reference to be untrustworthy.
CONTAMINATED_NEAR_REFERENCE_ORDER = REFERENCE_ORDER - 1
CONTAMINATED_NEAR_REFERENCE_MIN_RESOLUTION = REFERENCE_RESOLUTION // 2 + 1

data_dir = Path(getcwd()) / "Data-processing/Output/Omar/20260720-Final-copy"
out_dir  = Path(getcwd()) / "Data-processing/Results/Omar/clanker_made"
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
        # max(abs(diff)) per column -- no summation.
        return np.max(np.absolute(data - reference), axis=0)
    elif error_type == "L2":
        # Grid-normalized RMS, 1/N inside the sqrt (Roy 2003, Eq. 26):
        # sqrt( sum((u - u_ref)^2) / N )
        return np.sqrt(np.sum(np.square(data - reference) / data.shape[0], axis=0))

def align_reference_to_test(values: np.ndarray, ref_resolution: int, test_resolution: int):
    """
    Puts `values`, defined on the REFERENCE_RESOLUTION-cell reference grid, onto
    a grid of `test_resolution` cells covering the same physical domain (both are
    uniform finite-volume grids from UniformRectangularMesh1D over the same
    x1boundary/x2boundary).

    Returns (aligned_values, method, interp_order):
      - "identity"     -- test grid IS the reference grid, values returned as-is.
      - "restriction"  -- ref_resolution is an integer multiple of test_resolution.
        Each coarse cell's control volume is exactly the union of `factor`
        consecutive fine cells, so averaging those fine cells reproduces the
        coarse cell average with zero interpolation error (no polynomial fit
        involved) -- this is the standard FV restriction operator, not point
        subsampling (cell centers only coincide when `factor` is odd, so
        subsampling nodes would silently fail for even refinement factors).
      - "interpolation" -- grids are not nested; cubic spline (order 3), which
        is above the ~2nd-order spatial accuracy of the PVM/PRICE finite-volume
        scheme, so interpolation error stays below discretization error.
    """
    if ref_resolution == test_resolution:
        return values, "identity", None

    if ref_resolution % test_resolution == 0:
        factor = ref_resolution // test_resolution
        assert values.shape[0] == ref_resolution, \
            f"expected {ref_resolution} reference rows, got {values.shape[0]}"
        reshaped = values.reshape((test_resolution, factor) + values.shape[1:])
        return reshaped.mean(axis=1), "restriction", None

    interp_order = 3
    ref_x = (np.arange(ref_resolution) + 0.5) / ref_resolution
    test_x = (np.arange(test_resolution) + 0.5) / test_resolution
    if values.ndim == 1:
        aligned = CubicSpline(ref_x, values)(test_x)
    else:
        aligned = np.column_stack([
            CubicSpline(ref_x, values[:, j])(test_x) for j in range(values.shape[1])
        ])
    return aligned, "interpolation", interp_order

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

    # Time file
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

# Compute errors against the single shared reference (REFERENCE_ORDER, REFERENCE_RESOLUTION).
# Mirror the lookup style used downstream in plots-script.py: start from the
# known reference resolution/order to find which (IC, density, height) cases
# actually have a reference run, then only compute errors for those cases --
# avoids looping over/warning about every unrelated case in `data`.
CASE_COLS = ["Initial Condition", "Plant Density", "Plant Height"]
available_errors = ["L2", "LInf"]

reference_mask = (data["Resolution"] == REFERENCE_RESOLUTION) & (data["Order"] == REFERENCE_ORDER)
reference_cases = data.loc[reference_mask, CASE_COLS].drop_duplicates()

error_records = []
alignment_notes_printed = set()

for _, (ic, density, height) in reference_cases.iterrows():
    case_mask = (
        (data["Initial Condition"] == ic) &
        (data["Plant Density"] == density) &
        (data["Plant Height"] == height)
    )
    case = data.loc[case_mask]

    ref_mask = (case["Order"] == REFERENCE_ORDER) & (case["Resolution"] == REFERENCE_RESOLUTION)
    ref_idx = case.loc[ref_mask].index

    reference_solution = data.loc[ref_idx, ["Water Height", "Average Velocity"]].to_numpy()
    reference_u_profile = profile.loc[ref_idx, "0.0":"1.0"].to_numpy()

    # Exclude the reference's own resolution AND anything finer (>=
    # REFERENCE_RESOLUTION), not just the reference order itself. Two reasons:
    #   - same-resolution: comparing any curve to a same-resolution reference
    #     can't separate model error from grid error (see
    #     plots-documentation.md).
    #   - finer-than-reference: align_reference_to_test would have to upsample
    #     the (coarser) reference onto a finer test grid via interpolation,
    #     fabricating detail the reference doesn't have -- not a real test.
    # This makes REFERENCE_RESOLUTION swappable (e.g. 10000 while the 12000
    # run is still in progress, currently only order 6) without needing to
    # special-case leftover/partial data at resolutions above it.
    test_groups = case.loc[case["Resolution"] < REFERENCE_RESOLUTION].groupby(["Order", "Resolution"])

    for (order, resolution), grp in test_groups:
        order_idx = grp.index

        compared_solution = data.loc[order_idx, ["Water Height", "Average Velocity"]].to_numpy()
        compared_u_profile = profile.loc[order_idx, "0.0":"1.0"].to_numpy()

        aligned_solution, method, interp_order = align_reference_to_test(
            reference_solution, REFERENCE_RESOLUTION, resolution
        )
        aligned_u_profile, method_u, interp_order_u = align_reference_to_test(
            reference_u_profile, REFERENCE_RESOLUTION, resolution
        )

        note_key = (ic, density, height, resolution)
        if note_key not in alignment_notes_printed:
            note = (f"[grid alignment] case=(IC={ic}, density={density}, height={height}) "
                    f"res={resolution} vs reference res={REFERENCE_RESOLUTION}: {method}")
            if method == "interpolation":
                note += f" (cubic spline, order {interp_order})"
            print(note)
            alignment_notes_printed.add(note_key)

        is_contaminated = (order == REFERENCE_ORDER) or (
            order == CONTAMINATED_NEAR_REFERENCE_ORDER
            and resolution >= CONTAMINATED_NEAR_REFERENCE_MIN_RESOLUTION
        )

        for error_type in available_errors:
            sol_err = compute_error(error_type, compared_solution, aligned_solution)
            u_err = compute_error(error_type, compared_u_profile, aligned_u_profile)

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
                "Reference Contaminated": is_contaminated,
                "dx": DOMAIN_LENGTH / resolution,
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
