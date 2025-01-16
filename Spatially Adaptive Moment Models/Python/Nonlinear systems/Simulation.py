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
    physical_domain : list of floats (if 1D) or numpy 2D array of floats (if 2D)  
        the boundaries of the physical domain
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
                       t_end: float) -> list:
        """
        Runs the simulation until the end time t_end and returns the end values of the variables

        Parameters
        ----------
        t_end : float
            end time of the simulation
        
        Returns
        -------
        values: list of numpy arrays
            values of the variables at the end of the simulation

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
    physical_domain : list of floats (if 1D) or numpy 2D array of floats (if 2D)  
        the boundaries of the physical domain
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
                 physical_domain: list,
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
        physical_domain : list of floats  
            the boundaries of the physical domain
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

        """
        self.order = order
        self.physical_domain = physical_domain
        self.x1 = physical_domain[0]
        self.x2 = physical_domain[1]
        self.pde_type = pde_type
        self.mesh = mesh
        self.boundary_condition = boundary_condition
        self.initial_condition = initial_condition
        self.spatial_discretization = spatial_discretization

    def run_simulation(self,
                       t_end: float) -> list:

        delta_x = (self.x2 - self.x1)/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        values = self._get_initial_conditions(self.mesh.cell_center_positions)
        previous_values = copy.deepcopy(values)

        CFL = 0.25
        t = 0

        def system_matrix(cell_values):
            return self.pde_type.compute_system_matrix(self.order,cell_values)

        def source_term(cell_values):
            return self.pde_type.compute_source_term(self.order,cell_values)


        while t < t_end:
            g = 1

            # update boundary conditions
            previous_values[0] = self._update_boundary_conditions(previous_values[1])
            previous_values[self.mesh.resolution+1] = self._update_boundary_conditions(previous_values[self.mesh.resolution])

            if self.order == 0:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g))) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g))) 
                                    for value in previous_values])
            elif self.order == 1:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                    for value in previous_values])
            elif self.order == 2:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                    for value in previous_values]) 
            elif self.order == 3:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                    for value in previous_values])         
            max_speed = max(max_speed_plus,max_speed_min)
            delta_t = CFL*delta_x*max_speed #TODO implement CFL condition

            for i in range(1,self.mesh.resolution+1):
                fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                    previous_values[i-1],
                    previous_values[i],
                    system_matrix,
                    'positive',
                    delta_t,
                    delta_x) 
                fluctuation_minus = self.spatial_discretization.compute_fluctuation(
                    previous_values[i],
                    previous_values[i+1],
                    system_matrix,
                    'negative',
                    delta_t,
                    delta_x) 
                source_term_value = source_term(previous_values[i]) 
                values[i] = previous_values[i] - delta_t/delta_x*(fluctuation_plus+fluctuation_minus) + delta_t*source_term_value # solve FVM equations

            t+=delta_t
            previous_values = copy.deepcopy(values)
        return values

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
        
        initial_values = []

        initial_values.append(np.zeros(2+self.order)) # Initialize ghost cell, this value is overriden before the start of the simulation

        for i in range(0,self.mesh.resolution):
            initial_values.append(self.pde_type.get_initial_values(self.order,self.initial_condition,cell_centers_x[i]))

        initial_values.append(np.zeros(2+self.order)) # Initialize ghost cell, this value is overriden before the start of the simulation
        
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
                         end_values):
        """
        TO BE IMPLEMENTED.
        """
        pass

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
    physical_domain : list of floats (if 1D) or numpy 2D array of floats (if 2D)  
        the boundaries of the physical domain
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
    def compute_breakdown_criteria(self,values): 
        compute the breakdown criteria for increasing/decreasing the order in a subdomain
    """

    def __init__(self,
                 boundary_interfaces: list,
                 orders: list,
                 physical_domain: list,
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
        physical_domain : list of floats 
            the boundaries of the physical domain
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

        """

        self.boundary_interfaces = boundary_interfaces
        self.orders = orders
        self.physical_domain = physical_domain
        self.x1 = physical_domain[0]
        self.x2 = physical_domain[1]
        self.pde_type = pde_type
        self.mesh = mesh
        self.boundary_condition = boundary_condition
        self.initial_condition = initial_condition
        self.spatial_discretization = spatial_discretization

        self.boundary_interfaces_discretized = []

    
    def run_simulation(self,
                       t_end: float) -> np.array:
        g = 1
        delta_x = (self.x2 - self.x1)/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        self._calculate_boundary_interfaces()

        values = self._get_initial_conditions(self.mesh.cell_center_positions)
        previous_values = copy.deepcopy(values)

        CFL = 0.25
        
        t = 0

        while t < t_end:

            # update boundary conditions
            previous_values[0] = self._update_boundary_conditions(previous_values[1])
            previous_values[self.mesh.resolution+1] = self._update_boundary_conditions(previous_values[self.mesh.resolution])
            values[0] = self._update_boundary_conditions(previous_values[1])
            values[self.mesh.resolution+1] = self._update_boundary_conditions(previous_values[self.mesh.resolution])

            min_order = min(self.orders)
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
            elif min_order == 3:
                max_speed_plus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                        for value in previous_values])
                max_speed_min = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                    for value in previous_values])         
            max_speed = max(max_speed_plus,max_speed_min)
            delta_t = CFL*delta_x*max_speed #TODO implement CFL condition

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

        self.boundary_interfaces_discretized.append(round((self.boundary_interfaces[0] - self.x1)/(self.x2 - self.x1)*self.mesh.resolution))
        for i in range(1,len(self.boundary_interfaces)):
            self.boundary_interfaces_discretized.append(self.boundary_interfaces_discretized[i-1]
                + round((self.boundary_interfaces[i]-self.boundary_interfaces[i-1])/(self.x2-self.x1)*self.mesh.resolution))
            
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
                                cell_centers_x: np.array) -> list:
            
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
            
        initial_values = []

        initial_values.append(np.zeros(2+self.orders[0])) # Initialize ghost cell, this value is overriden before the start of the simulation

        right_boundary_subdomain = 0
        for m in range(len(self.boundary_interfaces_discretized)):
            left_boundary_subdomain = right_boundary_subdomain
            if self.orders[m+1] > self.orders[m]: 
                right_boundary_subdomain = self.boundary_interfaces_discretized[m]-2
            else:
                right_boundary_subdomain = self.boundary_interfaces_discretized[m]+2
            for i in range(left_boundary_subdomain,right_boundary_subdomain):
                initial_values.append(self.pde_type.get_initial_values(self.orders[m],self.initial_condition,cell_centers_x[i])) 
        for i in range(right_boundary_subdomain,self.mesh.resolution):
            initial_values.append(self.pde_type.get_initial_values(self.orders[-1],self.initial_condition,cell_centers_x[i]))

        initial_values.append(np.zeros(2+self.orders[-1])) # Initialize ghost cell, this value is overriden before the start of the simulation
            
        return initial_values
    
    def compute_breakdown_criteria(self,
                                   values: np.array) -> list:

        """
        Compute breakdown criteria for quantifying the required modelling complexity

        Parameters
        ----------
        values : list of numpy 1D arrays
            the values of the variables in each mesh cell
        
        Returns
        -------
        relative_value_last_moment: list of floats
            modelling complexity quantity in each mesh cell

        """

        relative_value_last_moment = np.zeros(self.mesh.resolution)
        for i in range(self.mesh.resolution):
            relative_value_last_moment[i] = np.abs(values[i,-1])/np.sum(np.abs(values[i,0:]))

        gradients = np.zeros((self.mesh.resolution,max(self.orders ) + 2)) #TODO: rewrite this such that it is generalizable

        height_gradient = np.zeros(self.mesh.resolution)
        velocity_gradient = np.zeros(self.mesh.resolution)

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
            #gradients[:,j] = gradients[:,j]/gradients[:,j].max()   

        return gradients[:,1]
    
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