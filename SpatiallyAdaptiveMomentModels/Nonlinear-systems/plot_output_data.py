import numpy as np
import pandas as pd
from pathlib import Path
from os import getcwd
from itertools import product
import re

# ---------- configuration ----------
SAVE_DATA = True
height_points = 100
z_values = np.linspace(0, 1, height_points)

data_dir = Path(getcwd()) / "Data-processing/Output/Omar/combined"
out_dir  = Path(getcwd()) / "Data-processing/Results/Omar"
out_dir.mkdir(parents=True, exist_ok=True)


# ---------- helper functions ----------
def parse_filename(filename: str):
    """
    Extract metadata from filename.
    Expected patterns:
        damBreak_noVelocity-height_0.0-resolution_100-order_2.csv
        time-damBreak_noVelocity-height_0.0-resolution_100-order_2.csv
        initialCondition_damBreakNoVelocity-density_512-height_0.0-resolution_100-order_2.csv
    Returns dict with keys: is_time, base, height, resolution, order, density
    """
    stem = Path(filename).stem
    meta = {}

    # optional time- prefix
    if stem.startswith('time-'):
        meta['is_time'] = True
        stem = stem[5:]          # remove 'time-'
    else:
        meta['is_time'] = False

    # Try to extract density if present (e.g., density_512)
    dens_match = re.search(r'density_(\d+)', stem)
    meta['density'] = int(dens_match.group(1)) if dens_match else 512  # default

    # Extract base, height, resolution, order
    # Pattern: (.*?)_noVelocity-height_([\d.]+)-resolution_(\d+)-order_(\d+)
    pattern = r'^(.*?)_noVelocity-height_([\d.]+)-resolution_(\d+)-order_(\d+)$'
    match = re.match(pattern, stem)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")

    meta['base'] = match.group(1)          # e.g., 'damBreak'
    meta['height'] = float(match.group(2)) # e.g., 0.0
    meta['resolution'] = int(match.group(3))
    meta['order'] = int(match.group(4))

    return meta

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
        return np.sqrt(np.sum(np.square(data - reference), axis=0)) / data.shape[0]

def compute_vertical_velocity_profile(order: int, 
                                      values: np.ndarray,
                                      z_points: np.ndarray) -> np.ndarray:
        """
        reconstructs the vertical velocity profile from the moment values and evaluates the velocity profile pointwise

        Parameters
        ----------
        order: integer
            order of the model
        values: np.ndarray (2D)
            2D numpy array containing the values of the variables in each mesh cell
        z_points: 
            the locations in vertical direction in which the velocity is computed
        
        Returns
        -------
        velocity_profile: numpy 2D array
            lateral velocity evaluated in in each point in z_points in z-direction

        """
        ones_array = np.ones(len(z_points))
        velocity_profile = np.zeros((len(values), len(z_points))) # (Grid-points, height-points)
        for i in range(len(values)):
            velocity_profile[i,:] += values[i,2]*ones_array

            if order >= 1:
                velocity_profile[i,:] += values[i,3]*(ones_array - 2*z_points)
            if order >= 2:
                velocity_profile[i,:] += values[i,4]*(ones_array - 6*z_points + 6*np.square(z_points))
            if order >= 3:
                velocity_profile[i,:] += values[i,5]*(ones_array - 12*z_points + 30*np.square(z_points) - 20*np.power(z_points,3))
            if order >= 4:
                velocity_profile[i,:] += values[i,6]*(ones_array - 20*z_points + 90*np.square(z_points) - \
                                                140*np.power(z_points,3) + 70*np.power(z_points,4))
            if order >= 5:
                velocity_profile[i,:] += values[i,7]*(ones_array - 30*z_points + 210*np.square(z_points) - \
                                                560*np.power(z_points,3) + 630*np.power(z_points,4) - 252*np.power(z_points,5))
            if order >= 6:
                velocity_profile[i,:] += values[i,8]*(ones_array - 42*z_points + 420*np.square(z_points) - \
                                                1680*np.power(z_points,3) + 3150*np.power(z_points,4) - \
                                                2772*np.power(z_points,5) + 924*np.power(z_points,6))
        return velocity_profile

def make_height_sepparator_coma(height: float) -> str:
    return ','.join(str(height).split('.'))

output_data_path = Path(getcwd()) / "Data-processing/Results/Omar"
output_data_path.mkdir(parents=True, exist_ok=True)

### start Creation arrays ###
rows_for_data = 7*sum(resolution_list)*len(height_list)
rows_for_error = len(resolution_list)*len(height_list)
cols_vp = height_points + 3

data = np.empty((rows_for_data, 12))                     # Total columns: 1 resolution + 1 order + 1 h_v + 3 non-moments + 6 moments max
velocity_profiles = np.empty((rows_for_data, cols_vp))   # Total columns: 1 resolution 1 order + 1 h_v + height points
data_time = np.empty((7*rows_for_error, 4))                 # Total columns: 1 resolution + 1 order + 1 h_v + 1 time
errors = np.empty((6*rows_for_error, 11))                   # Total columns: 1 resolution + 1 order + 1 h_v + 8 errors

data[:] = np.nan
data_time[:] = np.nan
velocity_profiles[:] = np.nan
errors[:] = np.nan

base_columns = ["Resolution", "Plant Height", "Order"]
data_columns = ["x position", "Water Height", "Average Velocity"] + \
               [f"Moment {order}" for order in range(1,7)]
u_profile_columns = [str(x) for x in np.linspace(0,1,height_points)]
errors_quantities = ["Water Height", "Average Velocity", "U profile", "std U profile"]
errors_types   = ["L2", "LInf"]
errors_columns = [error+" "+col for error, col in product(errors_types,errors_quantities)]

data = pd.DataFrame(data, columns=(base_columns+data_columns))
data_time = pd.DataFrame(data_time, columns=(base_columns+["Time"]))
velocity_profiles = pd.DataFrame(velocity_profiles, columns=(base_columns+u_profile_columns))
errors = pd.DataFrame(errors, columns=(base_columns+errors_columns))
### fi Creation arrays ###

z_values = np.linspace(0, 1, height_points)

low_bound, counter = 0, 0
for resolution, height, order in product(resolution_list, height_list, range(7)):
    
    resolution = int(resolution)
    height_str = make_height_sepparator_coma(height)

    file_name = file_name_func(height_str, resolution, order)

    try:
        loaded_data = np.loadtxt(file_path / file_name, delimiter=',')
        loaded_time_data = np.loadtxt(file_path / ("time-"+file_name), delimiter=',')

    except OSError:
        print(f"No file was found with name: {file_name}\n\n") # file_path / file_name
        continue

    top_bound = low_bound + resolution
    
    data.loc[low_bound:top_bound-1, base_columns] = [resolution, height, order]
    data.loc[low_bound:top_bound-1, data_columns[:3+order]] = loaded_data
    data_time.iloc[counter,0:3] = [resolution, height, order]
    data_time.iloc[counter,-1] = loaded_time_data
    velocity_profiles.loc[low_bound:top_bound-1, "0.0":"1.0"
                    ] = compute_vertical_velocity_profile(order, loaded_data, z_values)
    
    low_bound, counter = top_bound, counter+1

errors.loc[:,base_columns] = data_time.loc[data_time["Order"] != 6 ,base_columns].reset_index()
velocity_profiles.loc[:,base_columns] = data.loc[:,base_columns]

print("Arrays were created successfully")


# Compute the error
for resolution, height, order in product(resolution_list, height_list, range(6)):

    mask = (data["Order"] == order) & (data["Plant Height"] == height) & (data["Resolution"] == resolution)
    mask_reference = (data["Order"] == 6) & (data["Plant Height"] == height) & (data["Resolution"] == resolution)

    reference_solution = data.loc[mask_reference,["Water Height", "Average Velocity"]]
    reference_u_profile = velocity_profiles.loc[mask_reference,"0.0":"1.0"]
    compared_solution = data.loc[mask, ["Water Height", "Average Velocity"]]
    compared_u_profile = velocity_profiles.loc[mask,"0.0":"1.0"]

    mask = (errors["Order"] == order) & (errors["Plant Height"] == height) & (errors["Resolution"] == resolution)

    errors.loc[mask,["L2 Water Height", "L2 Average Velocity"]] = compute_error("L2", compared_solution, reference_solution)
    errors.loc[mask,["LInf Water Height", "LInf Average Velocity"]] = compute_error("LInf", compared_solution, reference_solution)

    # Velocity profiles
    l2_u_profile = compute_error("L2", compared_u_profile, reference_u_profile)    
    linf_u_profile = compute_error("LInf", compared_u_profile, reference_u_profile)

    errors.loc[mask,["L2 U profile", "L2 std U profile"]] = np.hstack((np.average(l2_u_profile), 
                                                                      np.std(l2_u_profile) / np.sqrt(len(l2_u_profile)))
                                                                    )
    errors.loc[mask,["LInf U profile", "LInf std U profile"]] = np.hstack((np.average(linf_u_profile), 
                                                                            np.std(linf_u_profile) / np.sqrt(len(linf_u_profile)))
                                                                    )

print("Errors were computed successfully")

if SAVE_DATA == True:
    data.to_csv(output_data_path / 'convergence-01-data_stacked.csv', sep=',')
    data_time.to_csv(output_data_path / 'convergence-02-time_stacked.csv', sep=',')
    errors.to_csv(output_data_path / 'convergence-03-errors.csv', sep=',')
    velocity_profiles.to_csv(output_data_path / 'convergence-04-velocity_profiles.csv', sep=',')

    print("Data was saved successfully")