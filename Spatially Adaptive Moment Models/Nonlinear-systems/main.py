import simulation
import pde
import mesh
import spatialDiscretization
import timeIntegration
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

    linear_source = pde_information['linear_source']
    time_integrator = numerical_method_information['timeIntegrator']
    linear_source_implicit = linear_source and time_integrator == 'ImplicitEuler'

    if pde_information['pde_type'] == 'SWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                        pde_information.getfloat('viscosity'),
                        pde_information.getfloat('slipLength'),
                        hyperbolic=False,
                        linear_source=linear_source_implicit)
    elif pde_information['pde_type'] == 'HSWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                        pde_information.getfloat('viscosity'),
                        pde_information.getfloat('slipLength'),
                        hyperbolic=True,
                        linear_source=linear_source_implicit)
        
    elif pde_information['pde_type'] == 'VegetationSWME1D':
        _pde = pde.VegetationSWME1D(pde_information['initialCondition'],
                                pde_information.getfloat('viscosity'),
                                pde_information.getfloat('slipLength'),
                                False,
                                linear_source_implicit,
                                0.008,
                                1,
                                264)
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

    if numerical_method_information['timeIntegrator'] == 'ImplicitEuler':
        _time_integration = timeIntegration.ImplicitEuler(linear_source)
    elif numerical_method_information['timeIntegrator'] == 'ExplicitEuler':
        _time_integration = timeIntegration.ExplicitEuler()

    #########################################################################

    if pde_information.getboolean('1D'):

        _mesh = mesh.UniformRectangularMesh1D([grid_information.getfloat('x1boundary'),grid_information.getfloat('x2boundary')],
                                               grid_information.getint('resolutionX')) #TODO: Implement different grids

        if numerical_method_information['method'] == 'spatially_adaptive':
            start_order = int(numerical_method_information['start_order'])

            if numerical_method_information['coupling'] == 'nonconservative':
                _simulation = simulation.NonConservativeAdaptiveSimulation1D(
                    start_order,
                    _pde,
                    _mesh,
                    numerical_method_information['boundaryCondition'],
                    pde_information['initialCondition'],
                    pde_information['breakdown_criterion'],
                    _spatialDiscretization,
                    _time_integration
                )
            elif numerical_method_information['coupling'] == 'conservative':
                _simulation = simulation.ConservativeAdaptiveSimulation1D(
                    start_order,
                    _pde,
                    _mesh,
                    numerical_method_information['boundaryCondition'],
                    pde_information['initialCondition'],
                    pde_information['breakdown_criterion'],
                    _spatialDiscretization,
                    _time_integration
                )

        elif numerical_method_information['method'] == 'classical':
            _simulation = simulation.ClassicalSimulation1D(
                numerical_method_information.getint('order'),
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                _spatialDiscretization,
                _time_integration)
            
        elif numerical_method_information['method'] == 'micro_macro':
            _simulation = simulation.Micro_macro(
                [int(order) for order in numerical_method_information['orders'].split(',')],
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                _spatialDiscretization,
                _time_integration)

        start = timeit.default_timer()
        data_array = _simulation.run_simulation(numerical_method_information.getfloat('t_end'))
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        data_frame = pd.DataFrame(data_array)
        # data_frame.to_csv('Data-processing/Results/Adaptive_SWME_paper/damBreak-and-smooth_linear_lambda0.1_nu0.1_adaptiveConservativeTest_t5.csv', index=False,header=False)
        # data_frame.to_csv('Data-processing/Results/Adaptive_SWME_paper/test.csv', index=False,header=False)

        z = np.linspace(0,1,100)
        if numerical_method_information['method'] == 'spatially_adaptive':
            velocity_profile = _pde.compute_vertical_velocity_profile(_simulation.max_order,
                                                                      data_array,
                                                                      z)
            orders = _simulation.orders_cellwise
            number_of_variables = _simulation.numbers_of_variables_cellwise
        else: 
            velocity_profile = _pde.compute_vertical_velocity_profile(numerical_method_information.getint('order'),
                                                                      data_array,
                                                                      z)
            orders = _simulation.order
            number_of_variables = _simulation.number_of_variables

        print('total mass = ',np.sum(data_array[:,1]*data_array[:,2]))

        plt.figure()
        plt.subplot(4,4,1)
        plt.plot(velocity_profile[np.floor_divide(_mesh.resolution,2),:], z)
        plt.title('Velocity profile')

        plt.subplot(4,4,2)
        plt.plot(_mesh.cell_center_positions, data_array[:,1])
        plt.title('Height')

        plt.subplot(4,4,3)
        plt.plot(_mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        # plt.subplot(4,4,4)
        # plt.plot(_mesh.cell_center_positions, data_array[:,3])
        # plt.title('alpha_1')

        # plt.subplot(4,4,5)
        # plt.plot(_mesh.cell_center_positions, data_array[:,4])
        # plt.title('alpha_2')

        # plt.subplot(4,4,6)
        # plt.plot(_mesh.cell_center_positions, data_array[:,5])
        # plt.title('alpha_3')

        # plt.subplot(4,4,7)
        # plt.plot(_mesh.cell_center_positions, data_array[:,6])
        # plt.title('alpha_4')

        # plt.subplot(4,4,8)
        # plt.plot(_mesh.cell_center_positions, data_array[:,7])
        # plt.scatter(_mesh.cell_center_positions,(data_array[:,-1]*np.max(data_array[:,7])+(5-data_array[:,-1])*np.min(data_array[:,7]))/5,s=5,color = 'hotpink')
        # plt.title('alpha_5')

        # plt.subplot(4,4,9)
        # plt.plot(_mesh.cell_center_positions, data_array[:,7])
        # plt.scatter(_mesh.cell_center_positions,(data_array[:,-1]*np.max(data_array[:,8])+(5-data_array[:,-1])*np.min(data_array[:,8]))/5,s=5,color = 'hotpink')
        # plt.title('alpha_5')

        if numerical_method_information['method'] == 'spatially_adaptive':

            plt.subplot(4,4,10)
            # plt.plot(_mesh.cell_center_positions[:-1],height_gradient)
            plt.plot(_mesh.cell_center_positions,_simulation.breakdown_estimators[:,2])
            plt.title('Height gradient')

            plt.subplot(4,4,11)
            # plt.plot(_mesh.cell_center_positions[:-1],momentum_gradient)
            plt.plot(_mesh.cell_center_positions,_simulation.breakdown_estimators[:,3])
            plt.title('Velocity gradient')

            plt.subplot(4,4,12)
            plt.plot(_mesh.cell_center_positions,_simulation.dom_decomp_val_res1)
            plt.title('domain_decomposition_values')

            plt.subplot(4,4,13)
            plt.plot(_mesh.cell_center_positions,_simulation.breakdown_estimators[:,2])
            plt.scatter(_mesh.cell_center_positions,(data_array[:,-1]*np.max(_simulation.breakdown_estimators[:,2])+(5-data_array[:,-1])*np.min(_simulation.breakdown_estimators[:,2]))/5,s=5,color = 'hotpink')
            plt.title('orders vs height-gradient')

            plt.subplot(4,4,14)
            # plt.plot(_mesh.cell_center_positions[:-1],_pde.compute_breakdown_criterion(data_array[:,1:],orders,number_of_variables,'last_moment',_mesh.resolution-1,delta_x))
            plt.plot(_mesh.cell_center_positions,_simulation.breakdown_estimators[:,1])
            plt.title('Absolute value last moment')

            plt.subplot(4,4,15)
            # plt.plot(_mesh.cell_center_positions[:-1],_pde.compute_breakdown_criterion(data_array[:,1:],orders,number_of_variables,'source_term',_mesh.resolution-1,delta_x))
            plt.plot(_mesh.cell_center_positions,_simulation.breakdown_estimators[:,0])
            plt.title('source term')

        plt.show()

    else:
        print('2D not implemented yet')

if __name__ == '__main__':
    main()