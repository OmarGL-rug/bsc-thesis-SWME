
import numpy as np
import pandas as pd

##############################################################
################ Parameter specifications ####################
##############################################################

# SPECIFY BOUNDARIES OF THE DOMAIN
x1 = -2
x2 = 1

# SPECIFY NUMBER OF GRID CELLS
n = 30
deltaX = (x2-x1)/n 
cellCentersX = np.linspace(x1, x2, n)

# SPECIFY INITIAL CONDITION
initialCondition = 'constantHeight_noVelocity'

# SPECIFY DOMAIN DECOMPOSITION
orders = [0,1]                         # List of moments that is used in each subdomain
boundaryInterfaces = [-1]               # Physical position of the boundary interfaces
boundaryInterfaces_Discretized = []     # Initialization of the list of boundary interfaces in the discretized domain

# SPECTIFY BOUNDARY CONDITION
boundaryCondition = 'INFLOW_OUTFLOW'

# SPECIFY PARAMETER VALUES
slipLength = 1.0                        # slip length
viscosity = 1.0                         # dynamic viscosity
g = 1.0                                 # gravity

# VISCOSITY MODEL
viscosityModel = 'PRICE'

# This function converts the physical boundary interface positions to the boundary interface position in the discretized domain
def calculateBoundaryInterfaces():
    boundaryInterfaces_Discretized.append(round((boundaryInterfaces[0] - x1)/(x2 - x1)*n))
    for i in range(1,len(boundaryInterfaces)):
        boundaryInterfaces_Discretized.append(boundaryInterfaces_Discretized[i-1]
                                              +round((boundaryInterfaces[i]-boundaryInterfaces[i-1])/(x2-x1)*n))

# This function defines initial conditions
def getInitialValues(order,initialCondition,x):
    vector = np.zeros(2+order)
    if initialCondition=='constantVelocity':
        vector[0] = 1
        vector[1] = 1*vector[0]
        if order > 0:
            vector[2] = 0 
        if order > 1:
            vector[3] = 0 
        if order > 2:
            vector[4] = 0 
        if order > 3:
            vector[5] = 0 
        if order > 4:
            vector[6] = 0 
    elif initialCondition=='constantHeight_noVelocity':
        vector[0] = 1
        vector[1] = 0
        if order > 0:
            vector[2] = 0 
        if order > 1:
            vector[3] = 0 
        if order > 2:
            vector[4] = 0 
        if order > 3:
            vector[5] = 0 
        if order > 4:
            vector[6] = 0 

    return vector

# This function preprocesses the given initial conditions and creates a list of intitial values in the grid cells
def getInitialConditions(initialCondition,x):
    initialValues = []

    initialValues.append(np.zeros(2+orders[0])) # Initialize ghost cell, this value is overriden before the start of the simulation

    rightBoundary_subDomain = 0
    for m in range(len(boundaryInterfaces_Discretized)):
        leftBoundary_subDomain = rightBoundary_subDomain
        if orders[m+1] > orders[m]: 
            rightBoundary_subDomain = boundaryInterfaces_Discretized[m]-2
        else:
            rightBoundary_subDomain = boundaryInterfaces_Discretized[m]+2
        for i in range(leftBoundary_subDomain,rightBoundary_subDomain):
            initialValues.append(getInitialValues(orders[m],initialCondition,x[i])) 
    for i in range(rightBoundary_subDomain,n):
         initialValues.append(getInitialValues(orders[-1],initialCondition,x[i]))

    initialValues.append(np.zeros(2+orders[-1])) # Initialize ghost cell, this value is overriden before the start of the simulation
    
    return initialValues

# This function computes the system matrix
def computeSystemMatrix(order,values):
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
        A[0][0]=1 #fill (not implemented yet)
    if order == 3:
        A=[0][0]=1 #fill (not implemented yet)
    if order == 4:
        A=[0][0]=1 #fill (not implemented yet)
    if order == 5:
        A=[0][0]=1 #fill (not implemented yet)
    if order == 6:
        A=[0][0]=1 #fill (not implemented yet)
    return A

# Tnis function computes the source term
def computeSourceTerm(order,values):
    S=np.zeros(order+2) 
    h = values[0]
    um = values[1]/values[0]
    if order == 0:
        S[0]=0
        S[1]=-viscosity/slipLength*um
    if order == 1:
        alpha1 = values[2]/values[0]

        S[0]=0
        S[1]=-viscosity/slipLength*um
        S[2]=-3*viscosity/slipLength*(um+(1+4*slipLength/h)*alpha1)
    if order == 2:
        S[0]=1 #fill (not implemented yet)
    if order == 3:
        S[0]=1 #fill (not implemented yet)
    if order == 4:
        S[0]=1 #fill (not implemented yet)
    if order == 5:
        S[0]=1 #fill (not implemented yet)
    if order == 6:
        S[0]=1 #fill (not implemented yet)
    return S

# This function computes the Roe linearization
def computeRoe(valueLeft,valueRight,systemMatrix,direction,deltaT):
    generalizedRoe = systemMatrix((valueLeft+valueRight)/2) 
    viscosity = computeViscosity(generalizedRoe,deltaT)
    if direction == 'negative':
        viscosity *= -1
    
    fluctuation = generalizedRoe.dot(valueRight-valueLeft) + viscosity.dot(valueRight-valueLeft)

    return fluctuation

# This function computes the numerical viscosity matrix
def computeViscosity(roeMatrix,deltaT):
    if viscosityModel == 'PRICE':
        viscosity = deltaX/(2*deltaT)*np.identity(roeMatrix.shape[0])+deltaT/(2*deltaX)*roeMatrix 
    return viscosity

def updateBoundaryConditions(valuesBoundary):
    if boundaryCondition == 'INFLOW_OUTFLOW':
        return valuesBoundary #do something

# This function runs the simulation and outputs the values at the end of the simulation
def runSimulation(tend):
    calculateBoundaryInterfaces()
    
    CFL = 0.7
    deltaT = CFL*deltaX #TODO implement CFL condition

    values = getInitialConditions(initialCondition,cellCentersX)
    print(len(values))

    rightBoundary_subDomain = 0

    # update boundary conditions
    values[0] = updateBoundaryConditions(values[1])
    values[n+1] = updateBoundaryConditions(values[n])

    for m in range(len(boundaryInterfaces_Discretized)):
        orderLeft = orders[m]
        orderRight = orders[m+1]

        def systemMatrixLeft(cellValues):
            return computeSystemMatrix(orderLeft,cellValues)

        def sourceTermLeft(cellValues):
            return computeSourceTerm(orderLeft,cellValues)

        def systemMatrixRight(cellValues):
            return computeSystemMatrix(orderRight,cellValues)

        def sourceTermRight(cellValues):
            return computeSourceTerm(orderRight,cellValues)

        leftBoundary_subDomain = rightBoundary_subDomain+1
        rightBoundary_subDomain = boundaryInterfaces_Discretized[m]
        if orderRight > orderLeft:
            values[rightBoundary_subDomain-1][orderLeft+2:] = values[rightBoundary_subDomain][orderLeft+2:] # update boundary interface boundary condition 
            for i in range(leftBoundary_subDomain,rightBoundary_subDomain-2):
                fluctuationPlus = computeRoe(
                    values[i-1],
                    values[i],
                    systemMatrixLeft,
                    'positive',
                    deltaT) 
                fluctuationMinus = computeRoe(
                    values[i],
                    values[i+1],
                    systemMatrixLeft,
                    'negative',
                    deltaT) 
                sourceTerm = sourceTermLeft(values[i]) 
                values[i] = values[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations
            
            # Evolution equation for the cell with index rightBoundary_subDomain-2
            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain-3],
                values[rightBoundary_subDomain-2],
                systemMatrixLeft,'positive',
                deltaT) 
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain-2],
                values[rightBoundary_subDomain-1][:orderLeft+2],
                systemMatrixLeft,
                'negative',
                deltaT) 
            sourceTerm = sourceTermLeft(values[rightBoundary_subDomain-2]) 
            values[rightBoundary_subDomain-2] = values[rightBoundary_subDomain-2] 
            -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations
            
            # Evolution equation for the cell with index rightBoundary_subDomain-1
            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain-2],
                values[rightBoundary_subDomain-1][:orderLeft+2],
                systemMatrixLeft,'positive',
                deltaT) 
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain-1][:orderLeft+2],
                values[rightBoundary_subDomain][:orderLeft+2],
                systemMatrixLeft,
                'negative',
                deltaT) 
            sourceTerm = sourceTermLeft(values[rightBoundary_subDomain-1][:orderLeft+2]) 
            values[rightBoundary_subDomain-1][:orderLeft+2] = values[rightBoundary_subDomain-1][:orderLeft+2] 
            -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations

            # Evolution equation for the cell with index rightBoundary_subDomain
            fluctuationPlus_Full = computeRoe(
                values[rightBoundary_subDomain-1],
                values[rightBoundary_subDomain],
                systemMatrixRight,
                'positive',
                deltaT) 
            fluctuationPlus_Restricted = computeRoe(
                values[rightBoundary_subDomain-1][:orderLeft+2],
                values[rightBoundary_subDomain][:orderLeft+2],
                systemMatrixLeft,'positive',
                deltaT) 
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain],
                values[rightBoundary_subDomain+1],
                systemMatrixRight,
                'negative',
                deltaT) 
            sourceTerm = sourceTermRight(values[rightBoundary_subDomain]) 
            values[rightBoundary_subDomain][:orderLeft+2] = values[rightBoundary_subDomain][:orderLeft+2] 
            -deltaT/deltaX*(fluctuationPlus_Restricted+fluctuationMinus[:orderLeft+2]) + sourceTerm[:orderLeft+2] # solve FVM equations for first moments
            values[rightBoundary_subDomain][orderLeft+2:] = values[rightBoundary_subDomain][orderLeft+2:] 
            -deltaT/deltaX*(fluctuationPlus_Full[orderLeft+2:]+fluctuationMinus[orderLeft+2:]) + sourceTerm[orderLeft+2:] # solve FVM equations for last moment
        else:
            values[rightBoundary_subDomain+2][orderRight+2:] = values[rightBoundary_subDomain+1][orderRight+2:] # update boundary interface boundary condition
            for i in range(leftBoundary_subDomain,rightBoundary_subDomain+1):
                fluctuationPlus = computeRoe(
                    values[i-1],
                    values[i],
                    systemMatrixLeft,
                    'positive',
                    deltaT) 
                fluctuationMinus = computeRoe(
                    values[i],
                    values[i+1],
                    systemMatrixLeft,
                    'negative',
                    deltaT) 
                sourceTerm = sourceTermLeft(values[i]) 
                values[i] = values[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations
            
            # Evolution equation for the cell with index rightBoundary_subDomain+1
            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain],
                values[rightBoundary_subDomain+1],
                systemMatrixLeft,
                'positive',
                deltaT)             
            fluctuationMinus_Full = computeRoe(
                values[rightBoundary_subDomain+1],
                values[rightBoundary_subDomain+2],
                systemMatrixLeft,'negative',
                deltaT) 
            fluctuationMinus_Restricted = computeRoe(
                values[rightBoundary_subDomain+1][:orderRight+2],
                values[rightBoundary_subDomain+2][:orderRight+2],
                systemMatrixRight,'negative',
                deltaT) 
            sourceTerm = sourceTermLeft(values[rightBoundary_subDomain+1]) 
            values[rightBoundary_subDomain][:orderRight+2] = values[rightBoundary_subDomain][:orderRight+2] 
            -deltaT/deltaX*(fluctuationPlus+fluctuationMinus_Restricted[:orderRight+2]) + sourceTerm[:orderRight+2] # solve FVM equations for first moments
            values[rightBoundary_subDomain][orderRight+2:] = values[rightBoundary_subDomain][orderRight+2:] 
            -deltaT/deltaX*(fluctuationPlus[orderRight+2:]+fluctuationMinus_Full[orderRight+2:]) + sourceTerm[orderRight+2:] # solve FVM equations for last m
            
            # Evolution equation for the cell with index rightBoundary_subDomain+2
            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain+1][:orderRight+2],
                values[rightBoundary_subDomain+2][:orderRight+2],
                systemMatrixRight,'positive',
                deltaT) 
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain+2][:orderRight+2],
                values[rightBoundary_subDomain+3],
                systemMatrixRight,
                'negative',
                deltaT) 
            sourceTerm = sourceTermRight(values[rightBoundary_subDomain+2][:orderRight+2]) 
            values[rightBoundary_subDomain+2][:orderRight+2] = values[rightBoundary_subDomain+2][:orderRight+2]
            -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations

            # Evolution equation for the cell with index rightBoundary_subDomain+3
            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain+2][:orderRight+2],
                values[rightBoundary_subDomain+3],
                systemMatrixRight,'positive',
                deltaT) 
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain+3],
                values[rightBoundary_subDomain+4],
                systemMatrixRight,
                'negative',
                deltaT) 
            sourceTerm = sourceTermRight(values[rightBoundary_subDomain+3]) 
            values[rightBoundary_subDomain+3] = values[rightBoundary_subDomain+3]
            -deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations

            rightBoundary_subDomain += 3
    
    for i in range(rightBoundary_subDomain+1,n+1):
        fluctuationPlus = computeRoe(
            values[i-1],
            values[i],
            systemMatrixRight,
            'positive',
            deltaT) 
        fluctuationMinus = computeRoe(
            values[i],
            values[i+1],
            systemMatrixRight,
            'negative',
            deltaT) 
        sourceTerm = sourceTermRight(values[i]) 
        values[i] = values[i] - deltaT/deltaX*(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations

    return values

# This function postprocesses the output data of the simulation and transforms it in data that can be plotted
def postProcessing(endValues):
    maxOrder = max(orders)
    minOrder = min(orders)

    dataArray = np.zeros((n,minOrder+3))

    for i in range(n):
        dataArray[i][0] = cellCentersX[i]
        for j in range(minOrder+2):
            dataArray[i][j+1] = endValues[i+1][j]

    dataFrame = pd.DataFrame(dataArray)
    dataFrame.to_csv('data.csv', index=False)

def main(tend):
    endValues = runSimulation(tend)
    postProcessing(endValues)

main(0.5)