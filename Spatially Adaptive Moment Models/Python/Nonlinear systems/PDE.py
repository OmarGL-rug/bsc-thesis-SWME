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
    def __init__(self, 
                initial_condition: str):
        """
        Constructs all the necessary attributes for the PDE object.

        Parameters
        ----------
        initial_condition : str
            initial condition of the PDE
        """

        pass

    @abstractmethod
    def compute_system_matrix(self,
                              order: int,
                              values: np.array) -> np.array:
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
    def compute_source_term(self,
                            order: int,
                            values: np.array) -> np.array:
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
    def get_initial_values(self,
                           order: int,
                           initial_condition: str,
                           position) -> np.array:

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
    def compute_vertical_velocity_profile(values):
        reconstruct the vertical velocity profiles from the moment values
    
    """

    def __init__(self, 
                initial_condition: str,
                viscosity: float,
                slip_length: float,
                hyperbolic: bool):
        self.initial_condition = initial_condition
        self.viscosity = viscosity
        self.slip_length = slip_length
        self.hyperbolic = hyperbolic

    def compute_system_matrix(self,
                              order: int,
                              values: np.array) -> np.array:
            g = 1
            A=np.zeros((order+2,order+2)) 
            h = values[0]
            um = values[1]/values[0]
            if order == 0:
                A[0][0] = 0
                A[0][1] = 1
                A[1][0] = g*h - um*um
                A[1][1] = 2.*um
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
                    alpha2 = 0
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

                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]
                alpha3 = values[4]/values[0]
                alpha4 = values[5]/values[0]

                if self.hyperbolic:
                    alpha2 = 0
                    alpha3 = 0
                    alpha4 = 0

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[0][3] = 0
                A[0][4] = 0
                A[0][5] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3. - alpha2*alpha2/5. - \
                alpha3*alpha3/7. - alpha4*alpha4/9.
                A[1][1] = 2*um
                A[1][2] = (2*alpha1)/3.
                A[1][3] = (2*alpha2)/5.
                A[1][4] = (2*alpha3)/7.
                A[1][5] = (2*alpha4)/9.
                A[2][0] = (-2*(21*alpha1*(5*um + 2*alpha2) + alpha3*(27*alpha2 + \
                20*alpha4)))/105.
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = (3*(alpha1 + alpha3))/5.
                A[2][4] = (3*(alpha2 + alpha4))/7.
                A[2][5] = alpha3/3.
                A[3][0] = (-2*(99*alpha2*alpha2 + 33*(alpha1 + alpha3)*(7*alpha1 + \
                2*alpha3) + 50*alpha4*alpha4 + 99*alpha2*(7*um + 2*alpha4)))/693.
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3. + (9*alpha3)/7.
                A[3][3] = um + (3*alpha2)/7. + (16*alpha4)/21.
                A[3][4] = (4*alpha1)/7. + alpha3/3.
                A[3][5] = (3*alpha2)/7. + (185*alpha4)/693.
                A[4][0] = alpha1*((-6*alpha2)/5. - (8*alpha4)/9.) - (2*alpha3*(165*um \
                + 44*alpha2 + 30*alpha4))/165.
                A[4][1] = 2*alpha3
                A[4][2] = (14*alpha4)/9.
                A[4][3] = (2*(alpha1 + alpha3))/5.
                A[4][4] = um + (alpha2 + alpha4)/3.
                A[4][5] = (5*alpha1)/9. + (3*alpha3)/11.
                A[5][0] = -2*um*alpha4 - (2*(1287*alpha2*alpha2 + \
                65*alpha3*(44*alpha1 + 9*alpha3) + 1300*alpha2*alpha4 + \
                405*alpha4*alpha4))/5005.
                A[5][1] = 2*alpha4
                A[5][2] = (-2*alpha3)/7.
                A[5][3] = (6*alpha2)/35. + (30*alpha4)/77.
                A[5][4] = (3*alpha1)/7. + (3*alpha3)/11.
                A[5][5] = um + (23*alpha2)/77. + (243*alpha4)/1001.
            if order == 5:

                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]
                alpha3 = values[4]/values[0]
                alpha4 = values[5]/values[0]
                alpha5 = values[6]/values[0]

                if self.hyperbolic:
                    alpha2 = 0
                    alpha3 = 0
                    alpha4 = 0
                    alpha5 = 0

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[0][3] = 0
                A[0][4] = 0
                A[0][5] = 0
                A[0][6] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3. - alpha2*alpha2/5. - \
                alpha3*alpha3/7. - alpha4*alpha4/9. - alpha5*alpha5/11.
                A[1][1] = 2*um
                A[1][2] = (2*alpha1)/3.
                A[1][3] = (2*alpha2)/5.
                A[1][4] = (2*alpha3)/7.
                A[1][5] = (2*alpha4)/9.
                A[1][6] = (2*alpha5)/11.
                A[2][0] = (-2*alpha1*(5*um + 2*alpha2))/5. - (18*alpha2*alpha3)/35. - \
                (2*alpha4*(44*alpha3 + 35*alpha5))/231.
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = (3*(alpha1 + alpha3))/5.
                A[2][4] = (3*(alpha2 + alpha4))/7.
                A[2][5] = (alpha3 + alpha5)/3.
                A[2][6] = (3*alpha4)/11.
                A[3][0] = (-2*(429*(alpha1 + alpha3)*(7*alpha1 + 2*alpha3) + \
                13*(99*alpha2*alpha2 + 50*alpha4*alpha4 + 99*alpha2*(7*um + \
                2*alpha4)) + 1950*alpha3*alpha5 + 525*alpha5*alpha5))/9009.
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3. + (9*alpha3)/7.
                A[3][3] = um + (3*alpha2)/7. + (16*alpha4)/21.
                A[3][4] = (4*alpha1)/7. + alpha3/3. + (125*alpha5)/231.
                A[3][5] = (3*alpha2)/7. + (185*alpha4)/693.
                A[3][6] = (80*alpha3)/231. + (95*alpha5)/429.
                A[4][0] = -2*um*alpha3 + alpha1*((-6*alpha2)/5. - (8*alpha4)/9.) - \
                (4*alpha4*(13*alpha3 + 10*alpha5))/143. - (4*alpha2*(22*alpha3 + \
                25*alpha5))/165.
                A[4][1] = 2*alpha3
                A[4][2] = (14*alpha4)/9.
                A[4][3] = (2*(alpha1 + alpha3))/5. + (10*alpha5)/11.
                A[4][4] = um + (alpha2 + alpha4)/3.
                A[4][5] = (5*alpha1)/9. + (3*(alpha3 + alpha5))/11.
                A[4][6] = (14*(13*alpha2 + 7*alpha4))/429.
                A[5][0] = (-2*(1287*alpha2*alpha2 + 5005*um*alpha4 + \
                1300*alpha2*alpha4 + 405*alpha4*alpha4 + 45*(alpha3 + \
                alpha5)*(13*alpha3 + 7*alpha5) + 65*alpha1*(44*alpha3 + \
                35*alpha5)))/5005.
                A[5][1] = 2*alpha4
                A[5][2] = (-2*alpha3)/7. + (20*alpha5)/11.
                A[5][3] = (6*alpha2)/35. + (30*alpha4)/77.
                A[5][4] = (3*(143*alpha1 + 91*alpha3 + 115*alpha5))/1001.
                A[5][5] = um + (23*alpha2)/77. + (243*alpha4)/1001.
                A[5][6] = (6*(91*alpha1 + 41*alpha3 + 35*alpha5))/1001.
                A[6][0] = (-2*(819*um*alpha5 + 30*alpha2*(13*alpha3 + 7*alpha5) + \
                alpha4*(455*alpha1 + 180*alpha3 + 126*alpha5)))/819.
                A[6][1] = 2*alpha5
                A[6][2] = (-5*alpha4)/9.
                A[6][3] = (5*alpha5)/13.
                A[6][4] = (5*(alpha2 + alpha4))/21.
                A[6][5] = (4*alpha1)/9. + (3*(alpha3 + alpha5))/13.
                A[6][6] = um + (11*alpha2)/39. + (8*alpha4)/39.
            if order == 6:

                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]
                alpha3 = values[4]/values[0]
                alpha4 = values[5]/values[0]
                alpha5 = values[6]/values[0]
                alpha6 = values[6]/values[0]

                if self.hyperbolic:
                    alpha2 = 0
                    alpha3 = 0
                    alpha4 = 0
                    alpha5 = 0
                    alpha6 = 0

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[0][3] = 0
                A[0][4] = 0
                A[0][5] = 0
                A[0][6] = 0
                A[0][7] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3. - alpha2*alpha2/5. - \
                alpha3*alpha3/7. - alpha4*alpha4/9. - alpha5*alpha5/11. - \
                alpha6*alpha6/13.
                A[1][1] = 2*um
                A[1][2] = (2*alpha1)/3.
                A[1][3] = (2*alpha2)/5.
                A[1][4] = (2*alpha3)/7.
                A[1][5] = (2*alpha4)/9.
                A[1][6] = (2*alpha5)/11.
                A[1][7] = (2*alpha5)/13.
                A[2][0] = (-2*alpha1*(5*um + 2*alpha2))/5. - (18*alpha2*alpha3)/35. - \
                (8*alpha3*alpha4)/21. - (2*alpha5*(65*alpha4 + 54*alpha5))/429.
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = (3*(alpha1 + alpha3))/5.
                A[2][4] = (3*(alpha2 + alpha4))/7.
                A[2][5] = (alpha3 + alpha5)/3.
                A[2][6] = (3*(alpha4 + alpha5))/11.
                A[2][7] = (3*alpha5)/13.
                A[3][0] = (-2*(1287*alpha2*alpha2 + 429*(alpha1 + alpha3)*(7*alpha1 + \
                2*alpha3) + 1287*alpha2*(7*um + 2*alpha4) + 1950*alpha3*alpha5 + \
                525*alpha5*alpha5 + (10*alpha4 + 21*alpha5)*(65*alpha4 + \
                21*alpha5)))/9009.
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3. + (9*alpha3)/7.
                A[3][3] = um + (3*alpha2)/7. + (16*alpha4)/21.
                A[3][4] = (4*alpha1)/7. + alpha3/3. + (125*alpha5)/231.
                A[3][5] = (3*alpha2)/7. + (185*alpha4)/693. + (60*alpha5)/143.
                A[3][6] = (80*alpha3)/231. + (95*alpha5)/429.
                A[3][7] = (125*alpha4 + 81*alpha5)/429.
                A[4][0] = (-2*(143*alpha1*(27*alpha2 + 20*alpha4) + \
                78*alpha2*(22*alpha3 + 25*alpha5) + 15*alpha5*(60*alpha4 + 49*alpha5) \
                + 15*alpha3*(429*um + 78*alpha4 + 100*alpha5)))/6435.
                A[4][1] = 2*alpha3
                A[4][2] = (14*alpha4)/9.
                A[4][3] = (2*(alpha1 + alpha3))/5. + (10*alpha5)/11.
                A[4][4] = um + alpha2/3. + alpha4/3. + (25*alpha5)/39.
                A[4][5] = (5*alpha1)/9. + (3*(alpha3 + alpha5))/11.
                A[4][6] = (14*(13*alpha2 + 7*(alpha4 + alpha5)))/429.
                A[4][7] = (2*(25*alpha3 + 14*alpha5))/143.
                A[5][0] = (-2*(21879*alpha2*alpha2 + 6885*alpha4*alpha4 + 765*(alpha3 \
                + alpha5)*(13*alpha3 + 7*alpha5) + 1105*alpha1*(44*alpha3 + \
                35*alpha5) + 4410*alpha6*alpha6 + 595*alpha4*(143*um + 20*alpha5) + \
                425*alpha2*(52*alpha4 + 63*alpha5)))/85085.
                A[5][1] = 2*alpha4
                A[5][2] = (-2*alpha3)/7. + (20*alpha5)/11.
                A[5][3] = (6*(143*alpha2 + 325*alpha4 + 875*alpha5))/5005.
                A[5][4] = (3*(143*alpha1 + 91*alpha3 + 115*alpha5))/1001.
                A[5][5] = um + (299*alpha2 + 243*alpha4 + 287*alpha5)/1001.
                A[5][6] = (6*(91*alpha1 + 41*alpha3 + 35*alpha5))/1001.
                A[5][7] = (6*(85*(2*alpha2 + alpha4) + 74*alpha5))/2431.
                A[6][0] = (-2*(510*alpha2*(13*alpha3 + 7*alpha5) + \
                51*alpha3*(60*alpha4 + 49*alpha5) + 119*alpha1*(65*alpha4 + \
                54*alpha5) + 21*alpha5*(663*um + 102*alpha4 + 80*alpha5)))/13923.
                A[6][1] = 2*alpha5
                A[6][2] = (-5*alpha4)/9. + (27*alpha5)/13.
                A[6][3] = (5*alpha5)/13.
                A[6][4] = (5*(alpha2 + alpha4))/21. + (14*alpha5)/39.
                A[6][5] = (4*alpha1)/9. + (3*(alpha3 + alpha5))/13.
                A[6][6] = um + (11*alpha2)/39. + (8*(alpha4 + alpha5))/39.
                A[6][7] = (119*alpha1 + 51*alpha3 + 40*alpha5)/221.
                A[7][0] = (-100*alpha3*alpha3)/231. - (14*alpha3*alpha5)/33. - \
                2*um*alpha5 - (2*alpha2*(25*alpha4 + 14*alpha5))/55. - \
                (4*(19*(85*alpha4*alpha4 + 459*alpha1*alpha5 + 60*alpha5*alpha5) + \
                2394*alpha4*alpha5 + 900*alpha6*alpha6))/31977.
                A[7][1] = 2*alpha5
                A[7][2] = (-9*alpha5)/11.
                A[7][3] = (-5*alpha4)/33. + (21*alpha5)/55.
                A[7][4] = (25*alpha3 + 49*alpha5)/231.
                A[7][5] = (3*alpha2)/11. + (19*alpha4)/99. + (42*alpha5)/187.
                A[7][6] = (255*alpha1 + 119*alpha3 + 104*alpha5)/561.
                A[7][7] = um + (3*alpha2)/11. + (104*alpha4)/561. + \
                (600*alpha5)/3553.

            return A

    def compute_source_term(self,
                            order: int,
                            values: np.array) -> np.array:
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

            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]
            alpha3 = values[4]/values[0]
            alpha4 = values[5]/values[0]

            S[0] = 0
            S[1] = -((self.viscosity*(um + alpha1 + alpha2 + alpha3 + \
            alpha4))/self.slip_length)
            S[2] = (-3*self.viscosity*(um + alpha2 + alpha3 + ((h + \
            4*self.slip_length)*alpha1 + 4*self.slip_length*alpha3)/h + \
            alpha4))/self.slip_length
            S[3] = (-5*self.viscosity*(um + alpha1 + alpha3 + alpha4 + ((h + \
            12*self.slip_length)*alpha2 + \
            12*self.slip_length*alpha4)/h))/self.slip_length
            S[4] = (-7*self.viscosity*(um + alpha2 + alpha3 + ((h + \
            4*self.slip_length)*alpha1 + 24*self.slip_length*alpha3)/h + \
            alpha4))/self.slip_length
            S[5] = (-9*self.viscosity*(um + alpha1 + alpha3 + alpha4 + ((h + \
            12*self.slip_length)*alpha2 + \
            40*self.slip_length*alpha4)/h))/self.slip_length

        if order == 5:

            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]
            alpha3 = values[4]/values[0]
            alpha4 = values[5]/values[0]
            alpha5 = values[6]/values[0]

            S[0] = 0
            S[1] = -((self.viscosity*(um + alpha1 + alpha2 + alpha3 + alpha4 + \
            alpha5))/self.slip_length)
            S[2] = (-3*self.viscosity*(h*um + (h + 4*self.slip_length)*alpha1 + \
            h*alpha2 + (h + 4*self.slip_length)*alpha3 + h*alpha4 + (h + \
            4*self.slip_length)*alpha5))/(h*self.slip_length)
            S[3] = (-5*self.viscosity*(um + alpha1 + alpha3 + alpha4 + ((h + \
            12*self.slip_length)*alpha2 + 12*self.slip_length*alpha4)/h + \
            alpha5))/self.slip_length
            S[4] = (-7*self.viscosity*(h*um + (h + 4*self.slip_length)*alpha1 + \
            h*alpha2 + (h + 24*self.slip_length)*alpha3 + h*alpha4 + (h + \
            24*self.slip_length)*alpha5))/(h*self.slip_length)
            S[5] = (-9*self.viscosity*(um + alpha1 + alpha3 + alpha4 + ((h + \
            12*self.slip_length)*alpha2 + 40*self.slip_length*alpha4)/h + \
            alpha5))/self.slip_length
            S[6] = (-11*self.viscosity*(h*um + (h + 4*self.slip_length)*alpha1 + \
            h*alpha2 + (h + 24*self.slip_length)*alpha3 + h*alpha4 + (h + \
            60*self.slip_length)*alpha5))/(h*self.slip_length)

        if order == 6:

            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]
            alpha3 = values[4]/values[0]
            alpha4 = values[5]/values[0]
            alpha5 = values[6]/values[0]
            alpha6 = values[7]/values[0]

            S[0] = 0
            S[1] = -((self.viscosity*(um + alpha1 + alpha2 + alpha3 + alpha4 + \
            alpha5 + alpha6))/self.slip_length)
            S[2] = (-3*self.viscosity*(um + alpha2 + alpha3 + alpha4 + alpha5 + \
            ((h + 4*self.slip_length)*alpha1 + 4*self.slip_length*(alpha3 + \
            alpha5))/h + alpha6))/self.slip_length
            S[3] = (-5*self.viscosity*(um + alpha1 + alpha3 + alpha4 + alpha5 + \
            alpha6 + ((h + 12*self.slip_length)*alpha2 + \
            12*self.slip_length*(alpha4 + alpha6))/h))/self.slip_length
            S[4] = (-7*self.viscosity*(um + alpha2 + alpha3 + alpha4 + alpha5 + \
            ((h + 4*self.slip_length)*alpha1 + 24*self.slip_length*(alpha3 + \
            alpha5))/h + alpha6))/self.slip_length
            S[5] = (-9*self.viscosity*(um + alpha1 + alpha3 + alpha4 + alpha5 + \
            alpha6 + ((h + 12*self.slip_length)*alpha2 + \
            40*self.slip_length*(alpha4 + alpha6))/h))/self.slip_length
            S[6] = (-11*self.viscosity*(um + alpha2 + alpha3 + alpha4 + alpha5 + \
            ((h + 4*self.slip_length)*alpha1 + 12*self.slip_length*(2*alpha3 + \
            5*alpha5))/h + alpha6))/self.slip_length
            S[7] = (-13*self.viscosity*(um + alpha1 + alpha3 + alpha4 + alpha5 + \
            alpha6 + ((h + 12*self.slip_length)*alpha2 + \
            40*self.slip_length*alpha4 + \
            84*self.slip_length*alpha6)/h))/self.slip_length

        return S
    
    def get_initial_values(self,
                           order: int,
                           initial_condition: str,
                           position: float) -> np.array:
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
            if order > 5:
                initial_values[7] = 0
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
            if order > 5:
                initial_values[7] = 0
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
                if order > 5:
                    initial_values[7] = 0 
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
                if order > 5:
                    initial_values[7] = 0
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
            if order > 5:
                initial_values[7] = 0
        return initial_values
    
    def compute_vertical_velocity_profile(self,
                                          order: int, 
                                          values: np.array,
                                          z_points: np.array) -> np.array:
        """
        reconstructs the vertical velocity profile from the moment values and evaluates the velocity profile pointwise

        Parameters
        ----------
        order: integer
            order of the model
        values: np.array (2D)
            2D numpy array containing the values of the variables in each mesh cell
        z_points: 
            the locations in vertical direction in which the velocity is computed
        
        Returns
        -------
        velocity_profile: numpy 2D array
            lateral velocity evaluated in in each point in z_points in z-direction

        """
        velocity_profile = np.zeros((len(values), len(z_points)))
        if order >= 0:
            for i in range(len(values)):
                velocity_profile[i,:] += values[i,2]*(np.ones(len(z_points)))
        if order >= 1:
            for i in range(len(values)):
                velocity_profile[i,:] += values[i,3]*(np.ones(len(z_points)) - 2*z_points)
        if order >= 2:
            for i in range(len(values)):
                velocity_profile[i,:] += values[i,4]*(np.ones(len(z_points)) - 6*z_points + 6*np.square(z_points))
        if order >= 3:
            for i in range(len(values)):
                velocity_profile[i,:] += values[i,5]*(np.ones(len(z_points)) - 12*z_points + 30*np.square(z_points) - 20*np.power(z_points,3))
        if order >= 4:
            for i in range(len(values)):
                velocity_profile[i,:] += values[i,6]*(np.ones(len(z_points)) - 20*z_points + 90*np.square(z_points) - \
                                                      140*np.power(z_points,3) + 70*np.power(z_points,4))
        if order >= 5:
            for i in range(len(values)):
                velocity_profile[i,:] += values[i,7]*(np.ones(len(z_points)) - 30*z_points + 210*np.square(z_points) - \
                                                      560*np.power(z_points,3) + 630*np.power(z_points,4) - 252*np.power(z_points,5)) 
               
        return velocity_profile