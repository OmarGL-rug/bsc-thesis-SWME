from abc import ABC, abstractmethod
import numpy as np
import pde
import mesh
import copy
import spatialDiscretization

class Simulation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def runSimulation(self,tend):
        pass

class ClassicalSimulation1D(Simulation):
    def __init__(self,
                 order,
                 physicalDomain,
                 PDEType: pde.PDE,
                 mesh: mesh.Mesh,
                 boundaryCondition,
                 initialCondition,
                 spatialDiscretization: spatialDiscretization.SpatialDiscretization):
        self.order = order
        self.physicalDomain = physicalDomain
        self.x1 = physicalDomain[0]
        self.x2 = physicalDomain[1]
        self.PDEType = PDEType
        self.mesh = mesh
        self.boundaryCondition = boundaryCondition
        self.initialCondition = initialCondition
        self.spatialDiscretization = spatialDiscretization

    def runSimulation(self,tend):

        deltaX = (self.x2 - self.x1)/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        values = self.getInitialConditions(self.mesh.cellCenterPositions)
        previousValues = copy.deepcopy(values)

        CFL = 0.25
        t = 0

        def systemMatrix(cellValues):
            return self.PDEType.computeSystemMatrix(self.order,cellValues)

        def sourceTerm(cellValues):
            return self.PDEType.computeSourceTerm(self.order,cellValues)


        while t < tend:
            g = 1

            # update boundary conditions
            previousValues[0] = self.updateBoundaryConditions(previousValues[1])
            previousValues[self.mesh.resolution+1] = self.updateBoundaryConditions(previousValues[self.mesh.resolution])

            if self.order == 0:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g))) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g))) 
                                    for value in previousValues])
            elif self.order == 1:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                    for value in previousValues])
            elif self.order == 2:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                    for value in previousValues]) 
            elif self.order == 3:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                    for value in previousValues])         
            maxSpeed = max(maxSpeedPlus,maxSpeedMin)
            deltaT = CFL*deltaX*maxSpeed #TODO implement CFL condition

            for i in range(1,self.mesh.resolution+1):
                fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                    previousValues[i-1],
                    previousValues[i],
                    systemMatrix,
                    'positive',
                    deltaT,
                    deltaX) 
                fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                    previousValues[i],
                    previousValues[i+1],
                    systemMatrix,'negative',
                    deltaT,
                    deltaX) 
                sourceTermValue = sourceTerm(previousValues[i]) 
                values[i] = previousValues[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue # solve FVM equations

            t+=deltaT
            previousValues = copy.deepcopy(values)
        return values

    def getInitialConditions(self,cellCentersX):
        initialValues = []

        initialValues.append(np.zeros(2+self.order)) # Initialize ghost cell, this value is overriden before the start of the simulation

        for i in range(0,self.mesh.resolution):
            initialValues.append(self.PDEType.getInitialValues(self.order,self.initialCondition,cellCentersX[i]))

        initialValues.append(np.zeros(2+self.order)) # Initialize ghost cell, this value is overriden before the start of the simulation
        
        return initialValues
    
    def updateBoundaryConditions(self,valuesBoundary):
        if self.boundaryCondition == 'INFLOW_OUTFLOW':
            return valuesBoundary 

class SpatiallyAdaptiveSimulation1D(Simulation):
    def __init__(self,
                 boundaryInterfaces,
                 orders,
                 physicalDomain,
                 PDEType: pde.PDE,
                 mesh: mesh.Mesh,
                 boundaryCondition,
                 initialCondition,
                 spatialDiscretization: spatialDiscretization.SpatialDiscretization):
        self.boundaryInterfaces = boundaryInterfaces
        self.orders = orders
        self.physicalDomain = physicalDomain
        self.x1 = physicalDomain[0]
        self.x2 = physicalDomain[1]
        self.PDEType = PDEType
        self.mesh = mesh
        self.boundaryCondition = boundaryCondition
        self.initialCondition = initialCondition
        self.spatialDiscretization = spatialDiscretization

        self.boundaryInterfaces_Discretized = []

    
    def runSimulation(self,tend):
        g = 1
        deltaX = (self.x2 - self.x1)/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        self.calculateBoundaryInterfaces()

        values = self.getInitialConditions(self.mesh.cellCenterPositions)
        previousValues = copy.deepcopy(values)

        CFL = 0.25
        
        t = 0

        while t < tend:

            # update boundary conditions
            previousValues[0] = self.updateBoundaryConditions(previousValues[1])
            previousValues[self.mesh.resolution+1] = self.updateBoundaryConditions(previousValues[self.mesh.resolution])

            minOrder = min(self.orders)
            if minOrder == 0:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g))) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g))) 
                                    for value in previousValues])
            elif minOrder == 1:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0])) 
                                    for value in previousValues])
            elif minOrder == 2:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0])) 
                                    for value in previousValues]) 
            elif minOrder == 3:
                maxSpeedPlus = max([abs(value[1]/value[0]
                                        +np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                        for value in previousValues])
                maxSpeedMin = max([abs(value[1]/value[0]
                                    -np.sqrt(value[0]*int(g)+value[2]/value[0]*value[2]/value[0]+value[3]/value[0]*value[3]/value[0]+value[4]/value[0]*value[4]/value[0])) 
                                    for value in previousValues])         
            maxSpeed = max(maxSpeedPlus,maxSpeedMin)
            deltaT = CFL*deltaX*maxSpeed #TODO implement CFL condition

            rightBoundary_subDomain = 0

            for m in range(len(self.boundaryInterfaces_Discretized)):
                orderLeft = self.orders[m]
                orderRight = self.orders[m+1]

                def systemMatrixLeft(cellValues):
                    return self.PDEType.computeSystemMatrix(orderLeft,cellValues)

                def sourceTermLeft(cellValues):
                    return self.PDEType.computeSourceTerm(orderLeft,cellValues)

                def systemMatrixRight(cellValues):
                    return self.PDEType.computeSystemMatrix(orderRight,cellValues)

                def sourceTermRight(cellValues):
                    return self.PDEType.computeSourceTerm(orderRight,cellValues)

                leftBoundary_subDomain = rightBoundary_subDomain+1
                rightBoundary_subDomain = self.boundaryInterfaces_Discretized[m]
                
                if orderRight > orderLeft:
                    previousValues[rightBoundary_subDomain-1][orderLeft+2:] = previousValues[rightBoundary_subDomain][orderLeft+2:] # update boundary interface boundary condition 
                    for i in range(leftBoundary_subDomain,rightBoundary_subDomain-2):
                        fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                            previousValues[i-1],
                            previousValues[i],
                            systemMatrixLeft,
                            'positive',
                            deltaT,
                            deltaX) 
                        fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                            previousValues[i],
                            previousValues[i+1],
                            systemMatrixLeft,
                            'negative',
                            deltaT,
                            deltaX) 
                        sourceTermValue = sourceTermLeft(previousValues[i]) 
                        values[i] = previousValues[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue # solve FVM equations
                    
                    # Evolution equation for the cell with index rightBoundary_subDomain-2
                    fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain-3],
                        previousValues[rightBoundary_subDomain-2],
                        systemMatrixLeft,
                        'positive',
                        deltaT,
                        deltaX) 
                    fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain-2],
                        previousValues[rightBoundary_subDomain-1][:orderLeft+2],
                        systemMatrixLeft,
                        'negative',
                        deltaT,
                        deltaX) 
                    sourceTermValue = sourceTermLeft(previousValues[rightBoundary_subDomain-2]) 
                    values[rightBoundary_subDomain-2] = (previousValues[rightBoundary_subDomain-2]
                    -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue) # solve FVM equations
                    
                    # Evolution equation for the cell with index rightBoundary_subDomain-1
                    fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain-2],
                        previousValues[rightBoundary_subDomain-1][:orderLeft+2],
                        systemMatrixLeft,
                        'positive',
                        deltaT,
                        deltaX) 
                    fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain-1][:orderLeft+2],
                        previousValues[rightBoundary_subDomain][:orderLeft+2],
                        systemMatrixLeft,
                        'negative',
                        deltaT,
                        deltaX) 
                    sourceTermValue = sourceTermLeft(previousValues[rightBoundary_subDomain-1][:orderLeft+2]) 
                    values[rightBoundary_subDomain-1][:orderLeft+2] = (previousValues[rightBoundary_subDomain-1][:orderLeft+2]
                    -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue) # solve FVM equations


                    # Evolution equation for the cell with index rightBoundary_subDomain
                    fluctuationPlus_Full = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain-1],
                        previousValues[rightBoundary_subDomain],
                        systemMatrixRight,
                        'positive',
                        deltaT,
                        deltaX) 
                    fluctuationPlus_Restricted = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain-1][:orderLeft+2],
                        previousValues[rightBoundary_subDomain][:orderLeft+2],
                        systemMatrixLeft,'positive',
                        deltaT,
                        deltaX) 
                    fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain],
                        previousValues[rightBoundary_subDomain+1],
                        systemMatrixRight,
                        'negative',
                        deltaT,
                        deltaX) 
                    sourceTermValue = sourceTermRight(previousValues[rightBoundary_subDomain]) 
                    
                    values[rightBoundary_subDomain][:orderLeft+2] = (previousValues[rightBoundary_subDomain][:orderLeft+2]
                    -deltaT/deltaX*(fluctuationPlus_Restricted+fluctuationMinus[:orderLeft+2])
                    +deltaT*sourceTermValue[:orderLeft+2]) # solve FVM equations for first moments
                    values[rightBoundary_subDomain][orderLeft+2:] = (previousValues[rightBoundary_subDomain][orderLeft+2:]
                    -deltaT/deltaX*(fluctuationPlus_Full[orderLeft+2:]+fluctuationMinus[orderLeft+2:])
                    +deltaT*sourceTermValue[orderLeft+2:]) # solve FVM equations for last moment
                else:
                    previousValues[rightBoundary_subDomain+2][orderRight+2:] = previousValues[rightBoundary_subDomain+1][orderRight+2:] # update boundary interface boundary condition
                    for i in range(leftBoundary_subDomain,rightBoundary_subDomain+1):
                        fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                            previousValues[i-1],
                            previousValues[i],
                            systemMatrixLeft,
                            'positive',
                            deltaT,
                            deltaX) 
                        fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                            previousValues[i],
                            previousValues[i+1],
                            systemMatrixLeft,
                            'negative',
                            deltaT,
                            deltaX) 
                        sourceTermValue = sourceTermLeft(previousValues[i]) 
                        values[i] = previousValues[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue # solve FVM equations
                    
                    # Evolution equation for the cell with index rightBoundary_subDomain+1
                    fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain],
                        previousValues[rightBoundary_subDomain+1],
                        systemMatrixLeft,
                        'positive',
                        deltaT,
                        deltaX)             
                    fluctuationMinus_Full = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain+1],
                        previousValues[rightBoundary_subDomain+2],
                        systemMatrixLeft,'negative',
                        deltaT,
                        deltaX) 
                    fluctuationMinus_Restricted = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain+1][:orderRight+2],
                        previousValues[rightBoundary_subDomain+2][:orderRight+2],
                        systemMatrixRight,'negative',
                        deltaT,
                        deltaX) 
                    sourceTermValue = sourceTermLeft(previousValues[rightBoundary_subDomain+1]) 
                    values[rightBoundary_subDomain+1][:orderRight+2] = (previousValues[rightBoundary_subDomain+1][:orderRight+2] 
                    -deltaT/deltaX*(fluctuationPlus[:orderRight+2]+fluctuationMinus_Restricted)
                    +deltaT*sourceTermValue[:orderRight+2]) # solve FVM equations for first moments
                    values[rightBoundary_subDomain+1][orderRight+2:] = (previousValues[rightBoundary_subDomain+1][orderRight+2:] 
                    -deltaT/deltaX*(fluctuationPlus[orderRight+2:]+fluctuationMinus_Full[orderRight+2:])
                    +deltaT*sourceTermValue[orderRight+2:]) # solve FVM equations for last moments
                    
                    # Evolution equation for the cell with index rightBoundary_subDomain+2
                    fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain+1][:orderRight+2],
                        previousValues[rightBoundary_subDomain+2][:orderRight+2],
                        systemMatrixRight,'positive',
                        deltaT,
                        deltaX) 
                    fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain+2][:orderRight+2],
                        previousValues[rightBoundary_subDomain+3],
                        systemMatrixRight,
                        'negative',
                        deltaT,
                        deltaX) 
                    sourceTermValue = sourceTermRight(previousValues[rightBoundary_subDomain+2][:orderRight+2]) 
                    values[rightBoundary_subDomain+2][:orderRight+2] = (previousValues[rightBoundary_subDomain+2][:orderRight+2]
                    -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue) # solve FVM equations

                    # Evolution equation for the cell with index rightBoundary_subDomain+3
                    fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain+2][:orderRight+2],
                        previousValues[rightBoundary_subDomain+3],
                        systemMatrixRight,'positive',
                        deltaT,
                        deltaX) 
                    fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                        previousValues[rightBoundary_subDomain+3],
                        previousValues[rightBoundary_subDomain+4],
                        systemMatrixRight,
                        'negative',
                        deltaT,
                        deltaX) 
                    sourceTermValue = sourceTermRight(previousValues[rightBoundary_subDomain+3]) 
                    values[rightBoundary_subDomain+3] = (previousValues[rightBoundary_subDomain+3]
                    -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue) # solve FVM equations

                    rightBoundary_subDomain += 3
            
            for i in range(rightBoundary_subDomain+1,self.mesh.resolution+1):
                fluctuationPlus = self.spatialDiscretization.computeFluctuation(
                    previousValues[i-1],
                    previousValues[i],
                    systemMatrixRight,
                    'positive',
                    deltaT,
                    deltaX) 
                fluctuationMinus = self.spatialDiscretization.computeFluctuation(
                    previousValues[i],
                    previousValues[i+1],
                    systemMatrixRight,
                    'negative',
                    deltaT,
                    deltaX) 
                sourceTermValue = sourceTermRight(previousValues[i]) 
                values[i] = previousValues[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + deltaT*sourceTermValue # solve FVM equations
            t+=deltaT
            previousValues = copy.deepcopy(values)
        return values
    
    def calculateBoundaryInterfaces(self):
        self.boundaryInterfaces_Discretized.append(round((self.boundaryInterfaces[0] - self.x1)/(self.x2 - self.x1)*self.mesh.resolution))
        for i in range(1,len(self.boundaryInterfaces)):
            self.boundaryInterfaces_Discretized.append(self.boundaryInterfaces_Discretized[i-1]
                + round((self.boundaryInterfaces[i]-self.boundaryInterfaces[i-1])/(self.x2-self.x1)*self.mesh.resolution))
            
    def updateBoundaryConditions(self,valuesBoundary):
        if self.boundaryCondition == 'INFLOW_OUTFLOW':
            return valuesBoundary 

    def getInitialConditions(self,cellCentersX):
            initialValues = []

            initialValues.append(np.zeros(2+self.orders[0])) # Initialize ghost cell, this value is overriden before the start of the simulation

            rightBoundary_subDomain = 0
            for m in range(len(self.boundaryInterfaces_Discretized)):
                leftBoundary_subDomain = rightBoundary_subDomain
                if self.orders[m+1] > self.orders[m]: 
                    rightBoundary_subDomain = self.boundaryInterfaces_Discretized[m]-2
                else:
                    rightBoundary_subDomain = self.boundaryInterfaces_Discretized[m]+2
                for i in range(leftBoundary_subDomain,rightBoundary_subDomain):
                    initialValues.append(self.PDEType.getInitialValues(self.orders[m],self.initialCondition,cellCentersX[i])) 
            for i in range(rightBoundary_subDomain,self.mesh.resolution):
                initialValues.append(self.PDEType.getInitialValues(self.orders[-1],self.initialCondition,cellCentersX[i]))

            initialValues.append(np.zeros(2+self.orders[-1])) # Initialize ghost cell, this value is overriden before the start of the simulation
            
            return initialValues
    
    def computeBreakdownCriteria(self,values):
        relativeValueLastMoment = np.zeros(self.mesh.resolution)
        for i in range(self.mesh.resolution):
            #relativeValueLastMoment[i] = np.abs(values[i+1][-1])/np.sum(np.abs(values[i+1][0:]))
            relativeValueLastMoment[i] = values[i+1][-1]

        return relativeValueLastMoment