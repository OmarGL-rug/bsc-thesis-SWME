import simulation
import pde
import mesh
import spatialDiscretization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser

def main():

    config = configparser.ConfigParser()
    config.read('config.txt')
    pde_information = config['pde_information']
    grid_information = config['grid_information']
    numerical_method_information = config['numerical_method_information']


    OneDimensional = pde_information.getboolean('1D')
    pde_type = pde_information['pde_type']
    initialCondition = pde_information['initialCondition']

    spatiallyAdaptive = numerical_method_information.getboolean('spatiallyAdaptive')
    fvm_type = numerical_method_information['fvm_type']
    boundaryCondition = numerical_method_information['boundaryCondition']


    if pde_type == 'SWME1D':
        viscosity = pde_information.getfloat('viscosity')
        slipLength = pde_information.getfloat('slipLength')
        _pde = pde.SWME1D(initialCondition,viscosity,slipLength,hyperbolic=False)
    elif pde_type == 'HSWME1D':
        viscosity = pde_information.getfloat('viscosity')
        slipLength = pde_information.getfloat('slipLength')
        _pde = pde.SWME1D(initialCondition,viscosity,slipLength,hyperbolic=True)
    else:
        print('PDE_type is not implemented yet')
    
    ##########################################################################

    if fvm_type == 'PVM':
        pvm = numerical_method_information['pvm']
        if pvm == 'PRICE':
            _spatialDiscretization = spatialDiscretization.PRICE()
        else:
            print('this pvm method is not implemented yet')
    else:
        print('this finite volume type is not implemented yet')


    #########################################################################

    t_end = numerical_method_information.getfloat('t_end')

    if OneDimensional:
        x1 = grid_information.getfloat('x1boundary')
        x2 = grid_information.getfloat('x2boundary')

        n = grid_information.getint('resolutionX')

        _mesh = mesh.CartesianUniformMesh1D([x1,x2],n) #TODO: Implement different grids

        if spatiallyAdaptive:
            boundaryInterfaces = numerical_method_information['boundaryInterfaces']
            boundaryInterfaces = [float(boundaryInterface) for boundaryInterface in boundaryInterfaces.split(',')]
            orders = numerical_method_information['orders']
            orders = [int(order) for order in orders.split(',')]

            _simulation = simulation.SpatiallyAdaptiveSimulation1D(
                boundaryInterfaces,
                orders,
                [x1,x2],
                _pde,
                _mesh,
                boundaryCondition,
                initialCondition,
                _spatialDiscretization
            )

            endValues = _simulation.runSimulation(tend)
            
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
            order = numerical_method_information.getint('order')
            _simulation = simulation.ClassicalSimulation1D(
                order,
                [x1,x2],
                _pde,
                _mesh,
                boundaryCondition,
                initialCondition,
                _spatialDiscretization)
            
            tend = 0.25
            endValues = _simulation.runSimulation(tend)

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
        print('2D not implemented yet')



if __name__ == '__main__':
    main()