from abc import ABC, abstractmethod
import numpy as np

#TODO: use duck typing
class SpatialDiscretization(ABC):

    @abstractmethod
    def computeFluctuation(self):
        pass

class PVM(SpatialDiscretization):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def computeFluctuation(self):
        pass

    @abstractmethod
    def computeViscosity(self,roeMatrix,deltaT,deltaX):
        pass

class PRICE(PVM):

    def __init__(self):
        pass

    def computeFluctuation(self,valueLeft,valueRight,systemMatrix,direction,deltaT,deltaX):
        generalizedRoe = systemMatrix((valueLeft+valueRight)/2)
        viscosity = self.computeViscosity(generalizedRoe,deltaT,deltaX)
        if direction == 'negative':
            viscosity *= -1
        
        fluctuation = (generalizedRoe.dot(valueRight-valueLeft) + viscosity.dot(valueRight-valueLeft))/2

        return fluctuation

    def computeViscosity(self,roeMatrix,deltaT,deltaX):
        viscosity = deltaX/(2*deltaT)*np.identity(roeMatrix.shape[0])+deltaT/(2*deltaX)*roeMatrix 
        return viscosity
