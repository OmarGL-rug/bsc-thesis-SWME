from abc import ABC, abstractmethod
import numpy as np

#TODO: implement MomentModel as a subclass of PDE and include the possibility of simulating PDEs that are not moment models (and don't have an order)
class PDE(ABC):
    """
    This interface represents a partial differential equation.

    ...

    Attributes
    ----------
    initial_condition : str
        initial condition for the partial differential equation

    
    Abstract methods
    -------
    def compute_system_matrix(order,values):
        computes the system matrix of the partial differential equation evaluated in the given values, for the given order.
    def compute_source_term(self,order,values):
        computes the system matrix of the partial differential equation evaluated in the given values, for the given order.
    def get_initial_values(self,order,initial_condition,position):
        calculates the initial values for one specific physical position
    """

    @abstractmethod
    def __init__(self, initial_condition):
        """
        Constructs all the necessary attributes for the PDE object.

        Parameters
        ----------
        initial_condition : str
            initial condition of the PDE
        """

        pass

    @abstractmethod
    def compute_system_matrix(self,order,values):
        """
        Computes the system matrix with a given order of the PDE evaluated in the given values.

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        values : numpy 1D array
            values of the variables
        
        
        Returns
        -------
        A: numpy 2D array
            System matrix

        """


        pass
    
    @abstractmethod
    def compute_source_term(self,order,values):
        """
        Computes the source term with a given order of the PDE evaluated in the given values.

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        values : numpy 1D array
            values of the variables
        
        
        Returns
        -------
        S: numpy 1D array
            source term vector

        """

        pass

    @abstractmethod
    def get_initial_values(self,order,initial_condition,position):

        """
        calculates the initial values for one specific physical position

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        initial condition : str
            name of the initial condition
        position : float (if 1D) or numpy 1D array of floats (2D)
            the physical position in which the initial values are computed
        
        
        Returns
        -------
        initial_values: numpy 1D array
            initial values for the given initial condition evaluated in the phyiscal position

        """

        pass

class SWME1D(PDE):

    """
    This class represents the one-dimensional Shallow Water Moment Equations (SWME1D).

    ...

    Attributes
    ----------
    initial_condition : str
        initial condition for the SWME1D

    
    Implemented methods from interface PDE
    ---------------------------------
    def compute_system_matrix(order,values):
        computes the system matrix of the SWME1D evaluated in the given values, for the given order. 
    def compute_source_term(self,order,values):
        computes the system matrix of the SWME1D evaluated in the given values, for the given order.
    def get_initial_values(self,order,initial_condition,position):
        calculates the initial values for one specific physical position

    Instance methods
    ----------------
    None
    
    """

    def __init__(self, initial_condition,viscosity,slip_length,hyperbolic):
        self.initial_condition = initial_condition
        self.viscosity = viscosity
        self.slip_length = slip_length
        self.hyperbolic = hyperbolic

    def compute_system_matrix(self, order, values):
            g = 1
            A=np.zeros((order+2,order+2)) 
            h = values[0]
            um = values[1]/values[0]
            if order == 0:
                A[0][0] = 0
                A[0][1] = 1
                A[1][0] = g*h - um*um
                A[1][1] = 2*um
            if order == 1:
                alpha1 = values[2]/values[0]

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3
                A[1][1] = 2*um
                A[1][2] = 2*alpha1/3
                A[2][0] = -2*um*alpha1
                A[2][1] = 2*alpha1
                A[2][2] = um
            if order == 2:
                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]

                if self.hyperbolic:
                    alpha2 = 0

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[0][3] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3 - alpha2*alpha2/5
                A[1][1] = 2*um
                A[1][2] = 2*alpha1/3
                A[1][3] = 2*alpha2/5
                A[2][0] = -2/5*alpha1*(5*um + 2*alpha2)
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = 3*alpha1/5
                A[3][0] = -2/21*(7*alpha1*alpha1 + 3*alpha2*(7*um + alpha2))
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3
                A[3][3] = um + 3/7*alpha2
            if order == 3:
                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]
                alpha3 = values[4]/values[0]

                if self.hyperbolic:
                    alpha3 = 0

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[0][3] = 0
                A[0][4] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3 - alpha2*alpha2/5 - alpha3*alpha3/7
                A[1][1] = 2*um
                A[1][2] = 2*alpha1/3
                A[1][3] = 2*alpha2/5
                A[1][4] = 2*alpha3/7
                A[2][0] = -2/35*(7*alpha1*(5*um + 2*alpha2) + 9*alpha2*alpha3)
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = 3*(alpha1 + alpha3)/5
                A[2][4] = 3*alpha2/7
                A[3][0] = -2/21*(3*alpha2*(7*um + alpha2)+(alpha1 + alpha3)*(7*alpha1 + 2*alpha3))
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3 + 9*alpha3/7
                A[3][3] = um + 3/7*alpha2
                A[3][4] = 4*alpha1/7 + alpha3/3
                A[4][0] = -2*um*alpha3 - 2*alpha2*(9*alpha1 + 4*alpha3)/15 
                A[4][1] = 2*alpha3
                A[4][2] = 0
                A[4][3] = 2*(alpha1 + alpha3)/5
                A[4][4] = um + alpha2/3
            if order == 4:
                A=[0][0]=1 #fill (not implemented yet)
            if order == 5:
                A=[0][0]=1 #fill (not implemented yet)
            if order == 6:
                A=[0][0]=1 #fill (not implemented yet)
            return A

    def compute_source_term(self, order, values):
        g = 1 #TODO: set g = 1 somewhere else
        
        S = np.zeros(order+2) 
        h = values[0]
        um = values[1]/values[0]
        if order == 0:
            S[0] = 0
            S[1] = -self.viscosity/self.slip_length*um
        if order == 1:
            alpha1 = values[2]/values[0]

            S[0] = 0
            S[1] = -self.viscosity/self.slip_length*(um + alpha1)
            S[2] = -3*self.viscosity/self.slip_length*(um + (1 + 4*self.slip_length/h)*alpha1)
        if order == 2:
            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]

            S[0] = 0
            S[1] = -self.viscosity/self.slip_length*(um + alpha1 + alpha2)
            S[2] = -3*self.viscosity/self.slip_length*(um + (1 + 4*self.slip_length/h)*alpha1 + alpha2)
            S[3] = -5*self.viscosity/self.slip_length*(um + alpha1 + (1 + 12*self.slip_length/h)*alpha2)
        if order == 3:
            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]
            alpha3 = values[4]/values[0]

            S[0] = 0
            S[1] = -self.viscosity/self.slip_length*(um + alpha1 + alpha2 + alpha3)
            S[2] = -3*self.viscosity/self.slip_length*((h + 4*self.slip_length)*alpha1 + h*(um + alpha2) + (h + 4*self.slip_length)*alpha3)/h
            S[3] = -5*self.viscosity/self.slip_length*(um + alpha1 + (1 + 12*self.slip_length/h)*alpha2 + alpha3)
            S[4] = -7*self.viscosity/self.slip_length*((h + 4*self.slip_length)*alpha1 + h*(um + alpha2) + (h + 24*self.slip_length)*alpha3)/h
        if order == 4:
            S[0]=1 #fill (not implemented yet)
        if order == 5:
            S[0]=1 #fill (not implemented yet)
        if order == 6:
            S[0]=1 #fill (not implemented yet)
        return S
    
    def get_initial_values(self,order,initial_condition,position):
            initial_values = np.zeros(2+order)
            if initial_condition == 'constantHeight_noVelocity':
                initial_values[0] = 1
                initial_values[1] = 0
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
            elif initial_condition == 'constantHeight_constantVelocity':
                initial_values[0] = 1
                initial_values[1] = 1*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
            elif initial_condition == 'damBreak_noVelocity':
                x0 = 0
                if position < x0:
                    initial_values[0] = 2
                    initial_values[1] = 0*initial_values[0]
                    if order > 0:
                        initial_values[2] = 0 
                    if order > 1:
                        initial_values[3] = 0 
                    if order > 2:
                        initial_values[4] = 0 
                    if order > 3:
                        initial_values[5] = 0 
                    if order > 4:
                        initial_values[6] = 0 
                else:
                    initial_values[0] = 1
                    initial_values[1] = 0*initial_values[0]
                    if order > 0:
                        initial_values[2] = 0 
                    if order > 1:
                        initial_values[3] = 0 
                    if order > 2:
                        initial_values[4] = 0 
                    if order > 3:
                        initial_values[5] = 0 
                    if order > 4:
                        initial_values[6] = 0 
            elif initial_condition == 'linearHeight_noVelocity':
                initial_values[0] = 1 + 0.1*position
                initial_values[1] = 0*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
            return initial_values