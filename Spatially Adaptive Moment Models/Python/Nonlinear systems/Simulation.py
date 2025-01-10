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
    def get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def update_boundary_conditions(self,values_boundary):
        updates the boundary conditions
    """

    @abstractmethod
    def __init__(self):
        """
        Implemented in the child classes.
        """
        pass

    @abstractmethod
    def run_simulation(self,t_end):
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
    def get_initial_conditions(self,cell_centers):
        """
        Implemented and documented in the child classes. 
        """
        pass

    @abstractmethod
    def update_boundary_conditions(self,values_boundary):
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
    def get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def update_boundary_conditions(self,values_boundary):
        updates the boundary conditions
    """

    def __init__(self,
                 order,
                 physical_domain,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 boundary_condition,
                 initial_condition,
                 spatial_discretization: spatialDiscretization.SpatialDiscretization):
        """
        Constructs all the necessary attributes for the ClassicalSimulation1D object.

        Parameters
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

    def run_simulation(self,t_end):

        delta_x = (self.x2 - self.x1)/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        values = self.get_initial_conditions(self.mesh.cell_center_positions)
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
            previous_values[0] = self.update_boundary_conditions(previous_values[1])
            previous_values[self.mesh.resolution+1] = self.update_boundary_conditions(previous_values[self.mesh.resolution])

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

    def get_initial_conditions(self,cell_centers_x):

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
    
    def update_boundary_conditions(self,values_boundary):
        """
        update the boundary conditions

        Parameters
        ----------
        values_boundary : numpy 1D array #TODO: implement boundary conditions that include more cells
            the values of the variables in the boundary cell
        
        Returns
        -------
        values_boundary: numpy 1D array
            the values of the variables in the ghost cell

        """

        if self.boundary_condition == 'INFLOW_OUTFLOW':
            return values_boundary 

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
    def get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def update_boundary_conditions(self,values_boundary):
        updates the boundary conditions


    Instance methods
    ----------------
    def compute_breakdown_criteria(self,values): 
        compute the breakdown criteria for increasing/decreasing the order in a subdomain
    """

    def __init__(self,
                 boundary_interfaces,
                 orders,
                 physical_domain,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 boundary_condition,
                 initial_condition,
                 spatial_discretization: spatialDiscretization.SpatialDiscretization):
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

    
    def run_simulation(self,t_end):
        g = 1
        delta_x = (self.x2 - self.x1)/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        self.calculate_boundary_interfaces()

        values = self.get_initial_conditions(self.mesh.cell_center_positions)
        previous_values = copy.deepcopy(values)

        CFL = 0.25
        
        t = 0

        while t < t_end:

            # update boundary conditions
            previous_values[0] = self.update_boundary_conditions(previous_values[1])
            previous_values[self.mesh.resolution+1] = self.update_boundary_conditions(previous_values[self.mesh.resolution])

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
        return values
    
    def calculate_boundary_interfaces(self):
        self.boundary_interfaces_discretized.append(round((self.boundary_interfaces[0] - self.x1)/(self.x2 - self.x1)*self.mesh.resolution))
        for i in range(1,len(self.boundary_interfaces)):
            self.boundary_interfaces_discretized.append(self.boundary_interfaces_discretized[i-1]
                + round((self.boundary_interfaces[i]-self.boundary_interfaces[i-1])/(self.x2-self.x1)*self.mesh.resolution))
            
    def update_boundary_conditions(self,values_boundary):
        if self.boundary_condition == 'INFLOW_OUTFLOW':
            return values_boundary 

    def get_initial_conditions(self,cell_centers_x):
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
    
    def compute_breakdown_criteria(self,values):
        relative_value_last_moment = np.zeros(self.mesh.resolution)
        for i in range(self.mesh.resolution):
            #relativeValueLastMoment[i] = np.abs(values[i+1][-1])/np.sum(np.abs(values[i+1][0:]))
            relative_value_last_moment[i] = values[i+1][-1]

        return relative_value_last_moment