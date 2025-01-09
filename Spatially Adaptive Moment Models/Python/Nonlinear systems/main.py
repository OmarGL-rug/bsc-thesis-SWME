import simulation
import pde
import mesh
import spatialDiscretization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser

# SPECIFY BOUNDARIES OF THE DOMAIN
x1 = -2
x2 = 2

# SPECIFY NUMBER OF GRID CELLS
n = 200

# SPECIFY INITIAL CONDITION:
### Implemented: ### 
#1 constantHeight_noVelocity 
#2 constantHeight_constantVelocity
#3 damBreak_noVelocity
#4 linearHeight_noVelocity
initialCondition = 'damBreak_noVelocity'

# SPECIFY WHETHER THE SIMULATION IS SPATIALLY ADAPTIVE (TRUE) OR CLASSICAL (FALSE)
spatiallyAdaptive = False

# SPECIFY WHETHER THE SIMULATION IS 1D OR 2D
OneDimensional = True

# SPECIFY WHETHER THE SIMULATION IS HSWME OR SWME
hyperbolic = False

# SPECIFY DOMAIN DECOMPOSITION
orders = [0,3,0]                         # List of moments that is used in each subdomain
order = 3                                # Order of classical simulation
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

    config = configparser.ConfigParser()
    
    tend = 0.25
    _pde = pde.SWME1D(initialCondition,viscosity,slipLength,hyperbolic)
    _mesh = mesh.CartesianUniformMesh1D([x1,x2],n)
    _spatialDiscretization = spatialDiscretization.PRICE()

    if spatiallyAdaptive:
        if OneDimensional:
            spatiallyAdaptiveSimulation1D = simulation.SpatiallyAdaptiveSimulation1D(boundaryInterfaces,
                                                                                    orders,
                                                                                    [x1,x2],
                                                                                    _pde,
                                                                                    _mesh,
                                                                                    boundaryCondition,
                                                                                    initialCondition,
                                                                                    _spatialDiscretization
                                                                                    )

            endValues = spatiallyAdaptiveSimulation1D.runSimulation(tend)

            relativeValuesLastMoment = spatiallyAdaptiveSimulation1D.computeBreakdownCriteria(endValues)

            maxOrder = max(orders)

            dataArray = np.zeros((n,maxOrder+3))
 
            for i in range(n):
                dataArray[i][0] = _mesh.cellCenterPositions[i]
                for j in range(len(endValues[i+1])):
                    dataArray[i][j+1] = endValues[i+1][j]

            dataFrame = pd.DataFrame(dataArray)
            dataFrame.to_csv('data.csv', index=False)

            plt.plot(_mesh.cellCenterPositions, dataArray[:,1])
            #plt.plot(_mesh.cellCenterPositions,relativeValuesLastMoment)
            plt.show()
        else:
            print("2D not implemented yet")

    else:
        if OneDimensional:
            classicalSimulation1D = simulation.ClassicalSimulation1D(order,
                                                                     [x1,x2],
                                                                     _pde,
                                                                     _mesh,
                                                                     boundaryCondition,
                                                                     initialCondition,
                                                                     _spatialDiscretization)

            endValues = classicalSimulation1D.runSimulation(tend)

            dataArray = np.zeros((n,order+3))
 
            for i in range(n):
                dataArray[i][0] = _mesh.cellCenterPositions[i]
                for j in range(len(endValues[i+1])):
                    dataArray[i][j+1] = endValues[i+1][j]

            dataFrame = pd.DataFrame(dataArray)
            dataFrame.to_csv('data.csv', index=False)

            plt.plot(_mesh.cellCenterPositions, dataArray[:,1])
            #plt.plot(_mesh.cellCenterPositions,relativeValuesLastMoment)
            plt.show()
        else:
            print("not implemented yet")



if __name__ == '__main__':
    main()