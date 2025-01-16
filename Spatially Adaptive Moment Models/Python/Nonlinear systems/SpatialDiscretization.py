from abc import ABC, abstractmethod
import numpy as np

#TODO: use duck typing
class SpatialDiscretization(ABC):

    """
    This interface represents a spatial discretization.

    ...

    Attributes
    ----------
    None

    
    Abstract methods
    -------
    def compute_fluctuation(self):
        computes a fluctuation between two cells
    """

    @abstractmethod
    def compute_fluctuation(self):
        pass

class PVM(SpatialDiscretization):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute_fluctuation(self):
        pass

    @abstractmethod
    def computeViscosity(self,roeMatrix,deltaT,deltaX):
        pass

class PRICE(PVM):

    def __init__(self):
        pass

    def compute_fluctuation(self,valueLeft,valueRight,systemMatrix,direction,deltaT,deltaX):
        generalizedRoe = systemMatrix((valueLeft+valueRight)/2)
        viscosity = self.computeViscosity(generalizedRoe,deltaT,deltaX)
        if direction == 'negative':
            viscosity *= -1
        
        fluctuation = (generalizedRoe.dot(valueRight-valueLeft) + viscosity.dot(valueRight-valueLeft))/2

        return fluctuation

    def computeViscosity(self,roeMatrix,deltaT,deltaX):
        viscosity = deltaX/(2*deltaT)*np.identity(roeMatrix.shape[0])+deltaT/(2*deltaX)*roeMatrix 
        return viscosity
    

