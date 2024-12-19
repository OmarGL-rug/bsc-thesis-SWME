import Simulation
import numpy as np


# SPECIFY BOUNDARIES OF THE DOMAIN
x1 = -2
x2 = 2

# SPECIFY NUMBER OF GRID CELLS
n = 400
deltaX = (x2-x1)/n 
cellCentersX = np.linspace(x1, x2, n)

# SPECIFY INITIAL CONDITION:
### Implemented: ### 
#1 constantHeight_noVelocity 
#2 constantHeight_constantVelocity
#3 damBreak_noVelocity
#4 linearHeight_noVelocity
initialCondition = 'damBreak_noVelocity'

# SPECIFY WHETHER THE SIMULATION IS SPATIALLY ADAPTIVE (TRUE) OR CLASSICAL (FALSE)
spatiallyAdaptive = True

# SPECIFY DOMAIN DECOMPOSITION
orders = [0,1,0]                         # List of moments that is used in each subdomain
boundaryInterfaces = [-1,1]               # Physical position of the boundary interfaces
boundaryInterfaces_Discretized = []     # Initialization of the list of boundary interfaces in the discretized domain

# SPECTIFY BOUNDARY CONDITION
boundaryCondition = 'INFLOW_OUTFLOW'

# SPECIFY PARAMETER VALUES
slipLength = 1.0                        # slip length
viscosity = 1.0                         # dynamic viscosity
g = 1.0                                 # gravity

# VISCOSITY MODEL
# PRICE
# L-F (Lax-Friedrichs)
viscosityModel = 'PRICE'

def main():

    if spatiallyAdaptive:

    print("python main function")


if __name__ == '__main__':
    main()