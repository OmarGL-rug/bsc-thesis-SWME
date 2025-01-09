from abc import ABC, abstractmethod
import numpy as np

class Mesh(ABC):
    @abstractmethod
    def __init__(self, boundaries, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cellCenterPositions = self.computeCellCenters()

    @abstractmethod
    def computeCellCenters(self):
        pass

class CartesianUniformMesh1D(Mesh):
    def __init__(self, boundaries, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cellCenterPositions = self.computeCellCenters()

    def computeCellCenters(self):
        cellCentersX = np.linspace(self.boundaries[0], self.boundaries[1], self.resolution)
        return cellCentersX
