from abc import ABC, abstractmethod

#TODO: use duck typing
class SpatialDiscretization(ABC):

    @abstractmethod
    def computeFluctuation(self):
        pass

class PVM(SpatialDiscretization,metaclass=ABC):
    @abstractmethod
    def __init__(self,viscosity):
        pass

    @abstractmethod
    def computeFluctuation(self):
        pass

class PRICE(PVM):

    def __init__(self,viscosityType):
        self.viscosityType = viscosityType

    def computeFluctuation(self):
        a=1

    def computeRoe(self):
        a=1
    def compute