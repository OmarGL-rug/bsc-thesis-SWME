import os
import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path(os.getcwd()) / "Results/Omar"

data = pd.read_csv(data_path / "01-data_stacked.csv")
errors = pd.read_csv(data_path / "03-errors.csv")

print("Data size: ", data.shape )
print("Errors size:", errors.shape)
print(errors)