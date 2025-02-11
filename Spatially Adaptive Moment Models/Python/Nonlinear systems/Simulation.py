from abc import ABC, abstractmethod
import numpy as np
import pde
import mesh
import copy
import spatialDiscretization

class Simulation(ABC):

    """
    This interface represents a simulation.

    ...

    Attributes
    ----------
    pde_type : str
        the partial differential equations that is simulated
    mesh : RectangularMesh
        the used mesh
    boundary_condition: str
        the used boundary condition
    initial_condition: str
        the initial condition for the simulation
    spatial_discretization: spatial_discretization
        the numerical method for the spatial discretization

    
    Abstract methods
    -------
    def run_simulation(t_end):
        runs the simulation and outputs the end values
    def _get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def _update_boundary_conditions(self,values_boundary):
        updates the boundary conditions
    """

    @abstractmethod
    def __init__(self):
        """
        Implemented in the child classes.
        """
        pass

    @abstractmethod
    def run_simulation(self,
                       t_end: float) -> np.array:
        """
        Runs the simulation until the end time t_end and returns the end values of the variables

        Parameters
        ----------
        t_end : float
            end time of the simulation
        
        Returns
        -------
        values: numpy arrays
            data array containing values of the variables at the end of the simulation

        """
        pass

    @abstractmethod
    def _get_initial_conditions(self,
                               cell_centers):
        """
        Implemented and documented in the child classes. 
        """
        pass

    @abstractmethod
    def _update_boundary_conditions(self,
                                   values_boundary):
        """
        Implemented and documented in the child classes.
        """
        pass

    @abstractmethod
    def _post_processing(self,
                         end_values):
        """
        Implemented and documented in the child classes.
        """
        pass

class ClassicalSimulation1D(Simulation):

    """
    This interface represents a classical (not spatially adaptive) simulation in 1D.

    ...

    Attributes
    ----------
    order: int
        order of the moment model
    pde_type : str
        the partial differential equations that is simulated
    number_of_variables : int
        number of state variables
    mesh : RectangularMesh
        the used mesh
    boundary_condition: str
        the used boundary condition
    initial_condition: str
        the initial condition for the simulation
    spatial_discretization: spatial_discretization
        the numerical method for the spatial discretization

    
    Implemented methods from interface Simulation
    -------
    def run_simulation(t_end):
        runs the simulation and outputs the end values
    def _get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def _update_boundary_conditions(self,values_boundary):
        updates the boundary conditions
    """

    def __init__(self,
                 order: int,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 boundary_condition: str,
                 initial_condition: str,
                 spatial_discretization: spatialDiscretization.SpatialDiscretization):
        """
        Constructs all the necessary attributes for the ClassicalSimulation1D object.

        Parameters
        ----------
        order: int
            order of the moment model
        pde_type : str
            the partial differential equations that is simulated
        number_of_variables : int
            number of state variables
        mesh : RectangularMesh
            the used mesh
        boundary_condition: str
            the used boundary condition
        initial_condition: str
            the initial condition for the simulation
        spatial_discretization: spatial_discretization
            the numerical method for the spatial discretization

        """
        self.order = order
        self.pde_type = pde_type
        self.number_of_variables = pde_type.compute_number_of_variables(self.order)
        self.mesh = mesh
        self.boundary_condition = boundary_condition
        self.initial_condition = initial_condition
        self.spatial_discretization = spatial_discretization

    def run_simulation(self,
                       t_end: float,
                       g = 9.81) -> np.array:

        delta_x = (self.mesh.boundaries[1] - self.mesh.boundaries[0])/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        values = self._get_initial_conditions(self.mesh.cell_center_positions)

        CFL = 0.5
        t = 0

        def system_matrix(cell_values):
            return self.pde_type.compute_system_matrix(self.order,cell_values)

        def source_term(cell_values):
            return self.pde_type.compute_source_term(self.order,cell_values)


        while t < t_end:

            # update boundary conditions
            values[0,:] = self._update_boundary_conditions(values[1,:])
            values[self.mesh.resolution+1,:] = self._update_boundary_conditions(values[self.mesh.resolution,:])
            
            wave_speed_sqrt = values[:,0]*int(g)
            for i in range(self.order):
                wave_speed_sqrt += np.divide(values[:,i+2]*values[:,i+2],values[:,0]*values[:,0])
            max_wave_speed_plus = np.max(np.abs(np.divide(values[:,1],values[:,0])+wave_speed_sqrt))
            max_wave_speed_min = np.max(np.abs(np.divide(values[:,1],values[:,0])-wave_speed_sqrt))
            max_speed = max(max_wave_speed_plus,max_wave_speed_min)

            delta_t = CFL*delta_x/max_speed #TODO implement CFL condition

            for i in range(1,self.mesh.resolution+1):
                fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                    values[i-1,:],
                    values[i,:],
                    system_matrix,
                    'positive',
                    delta_t,
                    delta_x) 
                fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                    values[i,:],
                    values[i+1,:],
                    system_matrix,
                    'negative',
                    delta_t,
                    delta_x) 
                source_term_value = source_term(values[i,:]) 
                values[i,:] = values[i,:] - delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value # solve FVM equations

            t+=delta_t
        simulation_data = self._post_processing(values)
        return simulation_data

    def _get_initial_conditions(self,
                               cell_centers_x: np.array) -> np.array:

        """
        construct the initial values for the variables

        Parameters
        ----------
        cell_centers_x : numpy 1D array
            the centers of the cells
        
        Returns
        -------
        initial_values: numpy 2D array
            initial values of the variables in each grid cell

        """
        
        initial_values = np.zeros((self.mesh.resolution+2,self.number_of_variables))

        for i in range(0,self.mesh.resolution):
            initial_values[i+1,:] = self.pde_type.get_initial_values(self.order,self.initial_condition,cell_centers_x[i])            
        
        return initial_values
    
    def _update_boundary_conditions(self,
                                   values_boundary: np.array) -> np.array:
        """
        update the boundary conditions

        Parameters
        ----------
        values_boundary : numpy 1D array #TODO: implement boundary conditions that include more cells
            the values of the variables in the boundary cell
        
        Returns
        -------
        values_ghost: numpy 1D array
            the values of the variables in the ghost cell

        """

        if self.boundary_condition == 'INFLOW_OUTFLOW':
            values_ghost = values_boundary
        return values_ghost 
    
    def _post_processing(self,
                         values) -> list:

        data_array = np.zeros((self.mesh.resolution,self.number_of_variables+1)) # rewrite this such that it can be generalized to other PDE models

        for i in range(self.mesh.resolution):
            data_array[i,0] = self.mesh.cell_center_positions[i]
        data_array[:,1] = values[1:-1,0]
        data_array[:,2] = np.divide(values[1:-1,1],data_array[:,1])
        for j in range(self.order): #TODO: this is unnecessary routine here
            data_array[:,j+3] = np.divide(values[1:-1,j+2],data_array[:,1])

        return data_array

class SpatiallyAdaptiveSimulation1D(Simulation):

    """
    This interface represents a spatially adaptive simulation in 1D.

    ...

    Attributes
    ----------
    boundary_interfaces: list of floats
        list of the physical positions of the interfaces that separate the domain into subdomains
    orders: list of integers
        list of the order of the moment model in each subdomain
    pde_type : str
        the partial differential equations that is simulated
    numbers_of_variables
        list of the number of state variables in each subdomain
    mesh : RectangularMesh
        the used mesh
    boundary_condition: str
        the used boundary condition
    initial_condition: str
        the initial condition for the simulation
    spatial_discretization: spatial_discretization
        the numerical method for the spatial discretization

    
    Implemented methods from interface Simulation
    -------
    def run_simulation(t_end):
        runs the simulation and outputs the end values
    def _get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def _update_boundary_conditions(self,values_boundary):
        updates the boundary conditions


    Instance methods
    ----------------
    def compute_all_breakdown_criteria(self,values): 
        compute all the breakdown criteria values for increasing/decreasing the order in a subdomain
    def compute_breakdown_criterion(self,values,breakdown_criterion): 
        compute breakdown criterion value for increasing/decreasing the order in a subdomain for a given breakdwon criterion
    """

    def __init__(self,
                 boundary_interfaces: list,
                 orders: list,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 boundary_condition: str,
                 initial_condition: str,
                 spatial_discretization: spatialDiscretization.SpatialDiscretization):

        """
        Constructs all the necessary attributes for the SpatiallyAdaptiveSimulation1D object.

        Parameters
        ----------
        boundary_interfaces: list of floats
            list of the physical positions of the boundary interfaces between the different subdomains
        orders: list of integers
            list of the orders of the moment model in each subdomain
        pde_type : str
            the partial differential equations that is simulated
        numbers_of_variables
            list of the number of state variables in each subdomain
        mesh : RectangularMesh
            the used mesh
        boundary_condition: str
            the used boundary condition
        initial_condition: str
            the initial condition for the simulation
        spatial_discretization: spatial_discretization
            the numerical method for the spatial discretization

        """

        self.boundary_interfaces = boundary_interfaces
        self.orders = orders
        self.pde_type = pde_type
        self.numbers_of_variables = np.zeros(len(self.orders))
        for i in range(len(self.orders)):
            self.numbers_of_variables[i] = pde_type.compute_number_of_variables(self.orders[i])
        self.mesh = mesh
        self.boundary_condition = boundary_condition
        self.initial_condition = initial_condition
        self.spatial_discretization = spatial_discretization

        self.boundary_interfaces_discretized = []
        self.max_order = 5

    
    def run_simulation(self,
                       t_end: float,
                       g = 9.81) -> np.array:
        
        delta_x = (self.mesh.boundaries[1] - self.mesh.boundaries[0])/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        self._calculate_boundary_interfaces()

        values = self._get_initial_conditions(self.mesh.cell_center_positions)

        CFL = 0.7
        
        t = 0

        while t < t_end:

            # update boundary conditions
            previous_values[0] = self._update_boundary_conditions(previous_values[1])
            previous_values[self.mesh.resolution+1] = self._update_boundary_conditions(previous_values[self.mesh.resolution])
            values[0] = self._update_boundary_conditions(previous_values[1])
            values[self.mesh.resolution+1] = self._update_boundary_conditions(previous_values[self.mesh.resolution])

            min_order = min(self.orders)
            #TODO: add method to PDE class that computes the wave speed (approximately)
            if min_order == 0:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g))) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g))) 
                                    for value in previous_values])
            elif min_order == 1:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                    for value in previous_values])
            elif min_order == 2:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                    for value in previous_values]) 
            elif min_order >= 3: #TODO: add higher orders
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                    for value in previous_values])         
            max_speed = max(max_speed_plus,max_speed_min)
            delta_t = CFL*delta_x/max_speed 

            right_boundary_subdomain = 0

            for m in range(len(self.boundary_interfaces_discretized)):
                order_left = self.orders[m]
                order_right = self.orders[m+1]

                def system_matrix_left(cell_values):
                    return self.pde_type.compute_system_matrix(order_left,cell_values)

                def source_term_left(cell_values):
                    return self.pde_type.compute_source_term(order_left,cell_values)

                def system_matrix_right(cell_values):
                    return self.pde_type.compute_system_matrix(order_right,cell_values)

                def source_term_right(cell_values):
                    return self.pde_type.compute_source_term(order_right,cell_values)

                left_boundary_subdomain = right_boundary_subdomain+1
                right_boundary_subdomain = self.boundary_interfaces_discretized[m]
                
                if order_right > order_left:
                    previous_values[right_boundary_subdomain-1][order_left+2:] = previous_values[right_boundary_subdomain][order_left+2:] # update boundary interface boundary condition 
                    for i in range(left_boundary_subdomain,right_boundary_subdomain-2):
                        fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                            previous_values[i-1],
                            previous_values[i],
                            system_matrix_left,
                            'positive',
                            delta_t,
                            delta_x) 
                        fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                            previous_values[i],
                            previous_values[i+1],
                            system_matrix_left,
                            'negative',
                            delta_t,
                            delta_x) 
                        source_term_value = source_term_left(previous_values[i]) 
                        values[i] = previous_values[i] - delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value # solve FVM equations
                    
                    # Evolution equation for the cell with index right_boundary_subdomain-2
                    fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain-3],
                        previous_values[right_boundary_subdomain-2],
                        system_matrix_left,
                        'positive',
                        delta_t,
                        delta_x) 
                    fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain-2],
                        previous_values[right_boundary_subdomain-1][:order_left+2],
                        system_matrix_left,
                        'negative',
                        delta_t,
                        delta_x) 
                    source_term_value = source_term_left(previous_values[right_boundary_subdomain-2]) 
                    values[right_boundary_subdomain-2] = (previous_values[right_boundary_subdomain-2]
                    -delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value) # solve FVM equations
                    
                    # Evolution equation for the cell with index right_boundary_subdomain-1
                    fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain-2],
                        previous_values[right_boundary_subdomain-1][:order_left+2],
                        system_matrix_left,
                        'positive',
                        delta_t,
                        delta_x) 
                    fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain-1][:order_left+2],
                        previous_values[right_boundary_subdomain][:order_left+2],
                        system_matrix_left,
                        'negative',
                        delta_t,
                        delta_x) 
                    source_term_value = source_term_left(previous_values[right_boundary_subdomain-1][:order_left+2]) 
                    values[right_boundary_subdomain-1][:order_left+2] = (previous_values[right_boundary_subdomain-1][:order_left+2]
                    -delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value) # solve FVM equations


                    # Evolution equation for the cell with index right_boundary_subdomain
                    fluctuation_plus_Full = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain-1],
                        previous_values[right_boundary_subdomain],
                        system_matrix_right,
                        'positive',
                        delta_t,
                        delta_x) 
                    fluctuation_plus_Restricted = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain-1][:order_left+2],
                        previous_values[right_boundary_subdomain][:order_left+2],
                        system_matrix_left,'positive',
                        delta_t,
                        delta_x) 
                    fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain],
                        previous_values[right_boundary_subdomain+1],
                        system_matrix_right,
                        'negative',
                        delta_t,
                        delta_x) 
                    source_term_value = source_term_right(previous_values[right_boundary_subdomain]) 
                    
                    values[right_boundary_subdomain][:order_left+2] = (previous_values[right_boundary_subdomain][:order_left+2]
                    -delta_t/delta_x*(fluctuation_plus_Restricted+fluctuation_minus[:order_left+2])
                    +delta_t*source_term_value[:order_left+2]) # solve FVM equations for first moments
                    values[right_boundary_subdomain][order_left+2:] = (previous_values[right_boundary_subdomain][order_left+2:]
                    -delta_t/delta_x*(fluctuation_plus_Full[order_left+2:]+fluctuation_minus[order_left+2:])
                    +delta_t*source_term_value[order_left+2:]) # solve FVM equations for last moment
                else:
                    previous_values[right_boundary_subdomain+2][order_right+2:] = previous_values[right_boundary_subdomain+1][order_right+2:] # update boundary interface boundary condition
                    for i in range(left_boundary_subdomain,right_boundary_subdomain+1):
                        fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                            previous_values[i-1],
                            previous_values[i],
                            system_matrix_left,
                            'positive',
                            delta_t,
                            delta_x) 
                        fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                            previous_values[i],
                            previous_values[i+1],
                            system_matrix_left,
                            'negative',
                            delta_t,
                            delta_x) 
                        source_term_value = source_term_left(previous_values[i]) 
                        values[i] = previous_values[i] - delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value # solve FVM equations
                    
                    # Evolution equation for the cell with index right_boundary_subdomain+1
                    fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain],
                        previous_values[right_boundary_subdomain+1],
                        system_matrix_left,
                        'positive',
                        delta_t,
                        delta_x)             
                    fluctuation_minus_Full = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain+1],
                        previous_values[right_boundary_subdomain+2],
                        system_matrix_left,'negative',
                        delta_t,
                        delta_x) 
                    fluctuation_minus_Restricted = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain+1][:order_right+2],
                        previous_values[right_boundary_subdomain+2][:order_right+2],
                        system_matrix_right,'negative',
                        delta_t,
                        delta_x) 
                    source_term_value = source_term_left(previous_values[right_boundary_subdomain+1]) 
                    values[right_boundary_subdomain+1][:order_right+2] = (previous_values[right_boundary_subdomain+1][:order_right+2] 
                    -delta_t/delta_x*(fluctuation_plus[:order_right+2]+fluctuation_minus_Restricted)
                    +delta_t*source_term_value[:order_right+2]) # solve FVM equations for first moments
                    values[right_boundary_subdomain+1][order_right+2:] = (previous_values[right_boundary_subdomain+1][order_right+2:] 
                    -delta_t/delta_x*(fluctuation_plus[order_right+2:]+fluctuation_minus_Full[order_right+2:])
                    +delta_t*source_term_value[order_right+2:]) # solve FVM equations for last moments
                    
                    # Evolution equation for the cell with index right_boundary_subdomain+2
                    fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain+1][:order_right+2],
                        previous_values[right_boundary_subdomain+2][:order_right+2],
                        system_matrix_right,'positive',
                        delta_t,
                        delta_x) 
                    fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain+2][:order_right+2],
                        previous_values[right_boundary_subdomain+3],
                        system_matrix_right,
                        'negative',
                        delta_t,
                        delta_x) 
                    source_term_value = source_term_right(previous_values[right_boundary_subdomain+2][:order_right+2]) 
                    values[right_boundary_subdomain+2][:order_right+2] = (previous_values[right_boundary_subdomain+2][:order_right+2]
                    -delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value) # solve FVM equations

                    # Evolution equation for the cell with index right_boundary_subdomain+3
                    fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain+2][:order_right+2],
                        previous_values[right_boundary_subdomain+3],
                        system_matrix_right,'positive',
                        delta_t,
                        delta_x) 
                    fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                        previous_values[right_boundary_subdomain+3],
                        previous_values[right_boundary_subdomain+4],
                        system_matrix_right,
                        'negative',
                        delta_t,
                        delta_x) 
                    source_term_value = source_term_right(previous_values[right_boundary_subdomain+3]) 
                    values[right_boundary_subdomain+3] = (previous_values[right_boundary_subdomain+3]
                    -delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value) # solve FVM equations

                    right_boundary_subdomain += 3
            
            for i in range(right_boundary_subdomain+1,self.mesh.resolution+1):
                fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                    previous_values[i-1],
                    previous_values[i],
                    system_matrix_right,
                    'positive',
                    delta_t,
                    delta_x) 
                fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                    previous_values[i],
                    previous_values[i+1],
                    system_matrix_right,
                    'negative',
                    delta_t,
                    delta_x) 
                source_term_value = source_term_right(previous_values[i]) 
                values[i] = previous_values[i] - delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value # solve FVM equations
            t+=delta_t
            previous_values = copy.deepcopy(values)

            self._update_domain_decomposition(values, 0.01, 0.001)

        simulation_data = self._post_processing(values)
        return simulation_data
    
    def _calculate_boundary_interfaces(self):

        """
        Calculates the interfaces in the grid that correspond to the physical positions of the boundary interfaces

        Parameters
        ----------
        None
        
        
        Returns
        -------
        None
        """

        self.boundary_interfaces_discretized.append(round((self.boundary_interfaces[0] - self.mesh.boundaries[0])/(self.mesh.boundaries[1] - self.mesh.boundaries[0])*self.mesh.resolution))
        for i in range(1,len(self.boundary_interfaces)):
            self.boundary_interfaces_discretized.append(self.boundary_interfaces_discretized[i-1]
                + round((self.boundary_interfaces[i]-self.boundary_interfaces[i-1])/(self.mesh.boundaries[1]-self.mesh.boundaries[0])*self.mesh.resolution))
            
    def _update_boundary_conditions(self,
                                    values_boundary: np.array) -> np.array:

        """
        update the boundary conditions

        Parameters
        ----------
        values_boundary : numpy 1D array #TODO: implement boundary conditions that include more cells
            the values of the variables in the boundary cell
        
        Returns
        -------
        values_ghost: numpy 1D array
            the values of the variables in the ghost cell

        """

        if self.boundary_condition == 'INFLOW_OUTFLOW':
            values_ghost = values_boundary
        return values_ghost 

    def _get_initial_conditions(self,
                                cell_centers_x: np.array) -> np.array:
            
        """
        construct the initial values for the variables

        Parameters
        ----------
        cell_centers_x : numpy 1D array
            the centers of the cells
        
        Returns
        -------
        initial_values: list of numpy arrays
            initial values of the variables in each grid cell

        """

        initial_values = np.zeros((self.mesh.resolution+2,max(self.orders) + 2))      

        right_boundary_subdomain = 0
        for m in range(len(self.boundary_interfaces_discretized)):
            left_boundary_subdomain = right_boundary_subdomain
            if self.orders[m+1] > self.orders[m]: 
                right_boundary_subdomain = self.boundary_interfaces_discretized[m]-2
            else:
                right_boundary_subdomain = self.boundary_interfaces_discretized[m]+2
            for i in range(left_boundary_subdomain,right_boundary_subdomain):
                initial_values[i+1,:2+self.orders[m]] = self.pde_type.get_initial_values(self.orders[m],self.initial_condition,cell_centers_x[i])  
        for i in range(right_boundary_subdomain,self.mesh.resolution):
            initial_values[i+1,:2+self.orders[-1]] = self.pde_type.get_initial_values(self.orders[-1],self.initial_condition,cell_centers_x[i])
            
        return initial_values
    
    def compute_all_breakdown_criteria(self,
                                   values: np.array) -> np.array:

        """
        Compute ALL breakdown criteria for quantifying the required modelling complexity

        Parameters
        ----------
        values : list of numpy 1D arrays
            the values of the variables in each mesh cell
        
        Returns
        -------
        relative_value_last_moment: numpy 2D array
            modelling complexity quantities in each mesh cell

        """

        relative_value_last_moment = np.zeros(self.mesh.resolution)
        for i in range(self.mesh.resolution):
            relative_value_last_moment[i] = np.abs(values[i,-1])/np.sum(np.abs(values[i,0:]))

        gradients = np.zeros((self.mesh.resolution,max(self.orders ) + 2)) #TODO: rewrite this such that it is generalizable

        for j in range(max(self.orders) + 2):
            for i in range(self.mesh.resolution-1):
                if values[i,j+1] < 0.01:
                    gradients[i,j] = np.abs((values[i+1,j+1] - values[i,j+1])/0.01)
                else:
                    gradients[i,j] = np.abs((values[i+1,j+1] - values[i,j+1])/values[i,j+1])
            if values[i,j+1] < 0.01:
                gradients[i,j] = np.abs((values[i+1,j+1] - values[i,j+1])/0.01)
            else:
                gradients[i,j] = np.abs((values[i+1,j+1] - values[i,j+1])/values[i,j+1])  

        return gradients
    
    def compute_breakdown_criterion(self,
                                   values: list,
                                   breakdown_criterion: str) -> np.array:

        """
        Compute ALL breakdown criteria for quantifying the required modelling complexity

        Parameters
        ----------
        values : list of numpy 1D arrays
            the values of the variables in each mesh cell
        breakdown_criterion : str
            the breakdwon criterion that is considered
        
        Returns
        -------
        relative_value_last_moment: np.array
            value for the breakdown criterion in each mesh cell

        """

        breakdown_criterion_values = np.zeros(self.mesh.resolution)
        
        if breakdown_criterion == 'height_gradient':
            for i in range(self.mesh.resolution-1):
                if values[i][0] < 0.01:
                    breakdown_criterion_values[i] = np.abs((values[i+1][0] - values[i][0])/0.01)
                else:
                    breakdown_criterion_values[i] = np.abs((values[i+1][0] - values[i][0])/values[i][0])
        else:
            print('this criterion is not implemented yet')  

        return breakdown_criterion_values

    def _update_domain_decomposition(self,
                                     values,
                                     tolerance_up = 0.01,
                                     tolerance_down = 0.001):
        
        breakdown_criteria = self.compute_breakdown_criterion(values, 'height_gradient')

        interface_left = 0
        for i in range(len(self.boundary_interfaces_discretized)):
            interface_right = self.boundary_interfaces_discretized[i]
            if np.max(breakdown_criteria[interface_left:interface_right]) > tolerance_up:
                self.orders[i] = max(self.orders[i]+1, 5)
            elif np.max(breakdown_criteria[interface_left:interface_right]) < tolerance_down:
                self.orders[i] = min(self.orders[i]-1, 0)
            interface_left = interface_right + 1
        
        if np.max(breakdown_criteria[interface_left:self.mesh.resolution]) > tolerance_up:
            self.orders[-1] = max(self.orders[-1]+1, 5)
        elif np.max(breakdown_criteria[interface_left:self.mesh.resolution]) < tolerance_down:
            self.orders[-1] = min(self.orders[-1]-1, 0)
    
    def _post_processing(self,
                         values: list) -> np.array:
        """
        TODO: write docstring
        """
        max_order = max(self.orders)

        data_array = np.zeros((self.mesh.resolution,max_order+3)) # rewrite this such that it can be generalized to other PDE models
 
        for i in range(self.mesh.resolution):
            data_array[i,0] = self.mesh.cell_center_positions[i]
            data_array[i,1] = values[i+1][0]
            for j in range(1,len(values[i+1])):
                data_array[i,j+1] = values[i+1][j]/data_array[i,1]

        return data_array