from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pde
import mesh
import simulation

class Plotting(ABC):

    """
    This abstract class represents a plotting object (for the plotting of the simulation results).

    ...

    Attributes
    ----------
    pde_type : the specific PDE model
        True if the spatial discretization is of the conservative type, false if non-conservative
    
    Class method
    -------
    def __init__(self,pde_type):
        initializes the plotting object

    Abstract methods
    -------
    def plot(self):
        creates a plotting object and plots the simulation results
    """
    def __init__(self,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 adaptive: bool,
                 simulation: simulation.Simulation):
        self.pde_type = pde_type #implement it as SWME
        self.mesh = mesh
        self.adaptive = adaptive
        self.simulation = simulation 

    @abstractmethod
    def plot(self,data_array):

        """
        Documented in the child classes

        """
        pass


class SWMEP1DPlotClassical(Plotting):

    """
    This class represents a plotting object for the plotting of numerical results of the 1D SWME of a classical simulation.

    ...

    Attributes
    ----------
    None

    Implemented methods from parent class 'Plotting'
    ------------------------------------------------
    def plot(self):
        computes the fluctuation between two cells with values value_left and value_right
            
    Abstract methods    
    -----------------
    None

    """

    def __init__(self,
                 pde: pde.SWME1D,
                 mesh: mesh.RectangularMesh,
                 simulation: simulation.ClassicalSimulation1D):
        self.pde = pde 
        self.mesh = mesh
        self.simulation = simulation

    def plot(self,data_array):
        
        """
        Creates a plot

        Parameters
        ----------
        
        Returns
        -------

        """
        z = np.linspace(0,1,100)

        velocity_profile = self.pde.compute_vertical_velocity_profile(self.simulation.order,
                                                                    data_array,
                                                                    z)
        order = self.simulation.order

        print('total mass = ',np.sum(data_array[:,1]*data_array[:,2]))

        plt.figure()
        plt.subplot(3,3,1)
        plt.plot(velocity_profile[np.floor_divide(self.mesh.resolution,2),:], z)
        plt.title('Velocity profile')

        plt.subplot(3,3,2)
        plt.plot(self.mesh.cell_center_positions, data_array[:,1])
        plt.title('Height')

        plt.subplot(3,3,3)
        plt.plot(self.mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        k = 4
        for i in range(order):
            plt.subplot(3,3,k)
            plt.plot(self.mesh.cell_center_positions, data_array[:,3+i])
            plt.title('alpha_'+str(i))
            k += 1

        plt.show()

class SWMEP1DPlotAdaptive(Plotting):

    """
    This class represents a plotting object for the plotting of numerical results of the 1D SWME of a classical simulation.

    ...

    Attributes
    ----------
    None

    Implemented methods from parent class 'Plotting'
    ------------------------------------------------
    def plot(self):
        computes the fluctuation between two cells with values value_left and value_right
            
    Abstract methods    
    -----------------
    None

    """

    def __init__(self,
                 pde: pde.SWME1D,
                 mesh: mesh.RectangularMesh,
                 simulation: simulation.SpatiallyAdaptiveSimulation1D):
        self.pde = pde 
        self.mesh = mesh
        self.simulation = simulation

    def plot(self,data_array):
        
        """
        Creates a plot

        Parameters
        ----------
        
        Returns
        -------

        """
        z = np.linspace(0,1,100)

        velocity_profile = self.pde.compute_vertical_velocity_profile(self.simulation.max_order,
                                                                    data_array,
                                                                    z)
        orders = self.simulation.orders_cellwise
        number_of_variables = self.simulation.numbers_of_variables_cellwise
        order = self.simulation.max_order

        print('total mass = ',np.sum(data_array[:,1]*data_array[:,2]))

        plt.figure()
        plt.subplot(4,4,1)
        plt.plot(velocity_profile[np.floor_divide(self.mesh.resolution,2),:], z)
        plt.title('Velocity profile')

        plt.subplot(4,4,2)
        plt.plot(self.mesh.cell_center_positions, data_array[:,1])
        plt.title('Height')

        plt.subplot(4,4,3)
        plt.plot(self.mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        k = 4
        for i in range(order):
            plt.subplot(4,4,k)
            plt.plot(self.mesh.cell_center_positions, data_array[:,3+i])
            plt.title('alpha_'+str(i))
            k += 1

        plt.subplot(4,4,k)
        # plt.plot(self.mesh.cell_center_positions[:-1],height_gradient)
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,2])
        plt.title('Height gradient')

        plt.subplot(4,4,k+1)
        # plt.plot(self.mesh.cell_center_positions[:-1],momentum_gradient)
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,3])
        plt.title('Velocity gradient')

        plt.subplot(4,4,k+2)
        plt.plot(self.mesh.cell_center_positions,self.simulation.dom_decomp_val_res1)
        plt.title('domain_decomposition_values 1')

        plt.subplot(4,4,k+3)
        plt.plot(self.mesh.cell_center_positions,self.simulation.dom_decomp_val_res2)
        plt.title('domain_decomposition_values 2')

        plt.subplot(4,4,k+4)
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,2])
        plt.scatter(self.mesh.cell_center_positions,(data_array[:,-1]*np.max(self.simulation.breakdown_estimators[:,2])+(5-data_array[:,-1])*np.min(self.simulation.breakdown_estimators[:,2]))/5,s=5,color = 'hotpink')
        plt.title('orders vs height-gradient')

        plt.subplot(4,4,k+5)
        # plt.plot(self.mesh.cell_center_positions[:-1],_pde.compute_breakdown_criterion(data_array[:,1:],orders,number_of_variables,'last_moment',self.mesh.resolution-1,delta_x))
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,1])
        plt.title('Absolute value last moment')

        plt.subplot(4,4,k+6)
        # plt.plot(self.mesh.cell_center_positions[:-1],_pde.compute_breakdown_criterion(data_array[:,1:],orders,number_of_variables,'source_term',self.mesh.resolution-1,delta_x))
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,0])
        plt.title('source term')

        plt.show()


