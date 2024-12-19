from abc import ABC, abstractmethod
import numpy as np

class Simulation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def runSimulation(self):
        pass

class ClassicalSimulation1D(Simulation):
    def __init__(self):
        pass

    def runSimulation(self):
        pass

class SpatiallyAdaptiveSimulation1D(Simulation):
    def __init__(self,boundaryInterfaces,physicalDomain,resolution):
        self.boundaryInterfaces = boundaryInterfaces
        self.physicalDomain = physicalDomain
        self.resolution = resolution
        self.x1 = physicalDomain[0]
        self.x2 = physicalDomain[1]

        self.boundaryInterfaces_Discretized = []
    
    def runSimulation(self):
        pass
    
    def calculateBoundaryInterfaces(self):
        self.selfboundaryInterfaces_Discretized.append(round((self.boundaryInterfaces[0] - self.x1)/(self.x2 - self.x1)*self.resolution))
        for i in range(1,len(self.boundaryInterfaces)):
            self.boundaryInterfaces_Discretized.append(self.boundaryInterfaces_Discretized[i-1]
                + round((self.boundaryInterfaces[i]-self.boundaryInterfaces[i-1])/(self.x2-self.x1)*self.resolution))
            
    def updateBoundaryConditions(self):
        pass
