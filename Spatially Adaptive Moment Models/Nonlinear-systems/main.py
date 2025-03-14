import simulation
import pde
import mesh
import spatialDiscretization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import timeit

def main():

    config = configparser.ConfigParser()
    config.read('Config-files/config.txt')
    pde_information = config['pde_information']
    grid_information = config['grid_information']
    numerical_method_information = config['numerical_method_information']

    if pde_information['pde_type'] == 'SWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                          pde_information.getfloat('viscosity'),
                          pde_information.getfloat('slipLength'),
                          hyperbolic=False)
    elif pde_information['pde_type'] == 'HSWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                          pde_information.getfloat('viscosity'),
                          pde_information.getfloat('slipLength'),
                          hyperbolic=True)
    elif pde_information['pde_type'] == 'VegetationSWME1D':
        _pde = pde.VegetationSWME1D(pde_information['initialCondition'],
                                    pde_information.getfloat('viscosity'),
                                    pde_information.getfloat('slipLength'),
                                    False,
                                    1,
                                    1,
                                    1)
    else:
        print('PDE_type is not implemented yet')
    
    ##########################################################################

    if numerical_method_information['fvm_type'] == 'PVM':
        if numerical_method_information['pvm'] == 'PRICE':
            _spatialDiscretization = spatialDiscretization.PRICE()
        elif numerical_method_information['pvm'] == 'LF':
            _spatialDiscretization = spatialDiscretization.LF()
        else:
            print('this pvm method is not implemented yet')
    else:
        print('this finite volume type is not implemented yet')


    #########################################################################

    if pde_information.getboolean('1D'):

        _mesh = mesh.UniformRectangularMesh1D([grid_information.getfloat('x1boundary'),grid_information.getfloat('x2boundary')],
                                               grid_information.getint('resolutionX')) #TODO: Implement different grids

        if numerical_method_information.getboolean('spatiallyAdaptive'):
            boundaryInterfaces = numerical_method_information['boundaryInterfaces']
            boundaryInterfaces = [float(boundaryInterface) for boundaryInterface in boundaryInterfaces.split(',')]
            orders = numerical_method_information['orders']
            orders = [int(order) for order in orders.split(',')]

            _simulation = simulation.SpatiallyAdaptiveSimulation1D(
                [float(boundaryInterface) for boundaryInterface in numerical_method_information['boundaryInterfaces'].split(',')],
                [int(order) for order in numerical_method_information['orders'].split(',')],
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                _spatialDiscretization
            )
        
        else:

            _simulation = simulation.ClassicalSimulation1D(
                numerical_method_information.getint('order'),
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                _spatialDiscretization)

        start = timeit.default_timer()
        data_array = _simulation.run_simulation(numerical_method_information.getfloat('t_end'))
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        data_frame = pd.DataFrame(data_array)
        #data_frame.to_csv('Data-processing/Results/test_LF.csv', index=False,header=False)

        z = np.linspace(0,1,100)
        if numerical_method_information.getboolean('spatiallyAdaptive'):
            velocity_profile = _pde.compute_vertical_velocity_profile(np.max([int(order) for order in numerical_method_information['orders'].split(',')]),
                                                                      data_array,
                                                                      z)
        else: 
            velocity_profile = _pde.compute_vertical_velocity_profile(numerical_method_information.getint('order'),
                                                                      data_array,
                                                                      z)

        #plt.plot(velocity_profile[200,:], z)

        plt.plot(_mesh.cell_center_positions, data_array[:,1])
       # plt.plot(_mesh.cell_center_positions,_pde.compute_all_breakdown_criteria(data_array,_mesh.resolution,max(_simulation.orders))[:,0])
        plt.show()
    else:
        print('2D not implemented yet')

if __name__ == '__main__':
    main()