from abc import ABC, abstractmethod
import numpy as np

#TODO: only rectangular meshes are considered here. Extend to other types of mesh.
class RectangularMesh(ABC):
    @abstractmethod
    def __init__(self, boundaries: list, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cell_center_positions = self.compute_cell_centers()

    @abstractmethod
    def compute_cell_centers(self):
        pass

class UniformRectangularMesh1D(RectangularMesh):
    def __init__(self, boundaries, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cell_center_positions = self.compute_cell_centers()

    def compute_cell_centers(self):
        cell_centers_x = np.linspace(self.boundaries[0], self.boundaries[1], self.resolution)
        return cell_centers_x
    
class UniformRectangularMesh2D(RectangularMesh):
    def __init__(self, boundaries, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cell_center_positions = self.compute_cell_centers()

    def compute_cell_centers(self):
        cell_centers_x = np.linspace(self.boundaries[0,0], self.boundaries[0,1], self.resolution[0])
        cell_centers_y = np.linspace(self.boundaries[1,0], self.boundaries[1,1], self.resolution[1])
        return [cell_centers_x,cell_centers_y]
