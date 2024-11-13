
import numpy as np

##############################################################
################ Parameter specifications ####################
##############################################################

# SPECIFY BOUNDARIES OF THE DOMAIN
x1 = -2
x2 = 1

# SPECIFY NUMBER OF GRID CELLS
n = 30
deltaX = (x2-x1)/n #

# SPECIFY INITIAL CONDITION
initialCondition = 'constantVelocity'

# SPECIFY DOMAIN DECOMPOSITION
moments = [4,6]                         # List of moments that is used in each subdomain
boundaryInterfaces = [-1]               # Physical position of the boundary interfaces
boundaryInterfaces_Discretized = []     # Initialization of the list of boundary interfaces in the discretized domain

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
def getInitialValues(numberOfMoments,initialCondition,x):
    vector = np.zeros(numberOfMoments)
    if initialCondition=='constantVelocity':
        vector[0] = 1
        vector[1] = 1*vector[0]
        if numberOfMoments > 2:
            vector[2] = 0 
        if numberOfMoments > 3:
            vector[3] = 0 
        if numberOfMoments > 4:
            vector[4] = 0 
        if numberOfMoments > 5:
            vector[5] = 0 
        if numberOfMoments > 6:
            vector[6] = 0 
    return vector

# This function preprocesses the given initial conditions and creates a list of intitial values in the grid cells
def getInitialConditions(initialCondition,x):
    initialValues = []
    rightBoundary_subDomain = -1
    for m in range(len(boundaryInterfaces_Discretized)):
        leftBoundary_subDomain = rightBoundary_subDomain+1
        if moments[m+1] > moments[m]: 
            rightBoundary_subDomain = boundaryInterfaces_Discretized[m]-2
        else:
            rightBoundary_subDomain = boundaryInterfaces_Discretized[m]+2
        for i in range(leftBoundary_subDomain,rightBoundary_subDomain):
            initialValues.append(getInitialValues(moments[m],initialCondition,x[i])) 
    for i in range(rightBoundary_subDomain+1,n):
         initialValues.append(getInitialValues(moments[m],initialCondition,x[i]))
    return initialValues

# This function computes the system matrix
def computeSystemMatrix(order,values):
    A=np.zeros((order,order)) 
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
    S=np.zeros(order) 
    h = values[0]
    um = values[1]/values[0]
    if order == 0:
        S[0]=0
        S[1]=1
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
    viscosity = computeViscosity(valueLeft,valueRight,generalizedRoe,deltaT)
    if direction == 'negative':
        viscosity *= -1
    
    fluctuation = generalizedRoe.dot(valueRight-valueLeft) + viscosity.dot(valueRight-valueLeft)

    return fluctuation

# This function computes the numerical viscosity matrix
def computeViscosity(roeMatrix,deltaT,):
    if viscosityModel == 'PRICE':
        viscosity = deltaX/(2*deltaT)*np.identity(roeMatrix.shape(0))+deltaT/(2*deltaX)*roeMatrix 
    return viscosity

# This function runs the simulation and outputs the values at the end of the simulation
def runSimulation(tend):
    calculateBoundaryInterfaces()
    
    deltaT = 1 #TODO implement CFL condition

    cellCentersX = np.linspace(x1, x2, n)

    values = getInitialConditions(initialCondition,cellCentersX)

    leftBoundary_subDomain = -1
    for m in range(len(boundaryInterfaces_Discretized)):
        def systemMatrix(cellValues):
            return computeSystemMatrix(moments[m],cellValues)
        def sourceTerm(cellValues):
            return computeSourceTerm(moments[m],cellValues)
        if moments[m+1] > moments[m]: 
            for i in range(leftBoundary_subDomain,rightBoundary_subDomain-3):
                fluctuationPlus = computeRoe(
                    values[i-1],
                    values[i],
                    systemMatrix,
                    'positive',
                    deltaT) #TODO: fill in arguments
                fluctuationMinus = computeRoe(
                    values[i],
                    values[i+1],
                    systemMatrix,
                    'negative',
                    deltaT) #TODO: fill in arguments
                sourceTerm = sourceTerm(values[i]) #TODO: fill in arguments
                values[i] = values[i] - deltaT/deltaX(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations
            

            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain-3],
                values[rightBoundary_subDomain-2],
                systemMatrix,'positive',
                deltaT) #TODO: fill in arguments
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain-2],
                values[rightBoundary_subDomain-1][:moments[m]],
                systemMatrix,
                'negative',
                deltaT) #TODO: fill in arguments
            sourceTerm = sourceTerm(values[i]) #TODO: fill in arguments
            values[rightBoundary_subDomain-2] = values[rightBoundary_subDomain-2] 
            -deltaT/deltaX(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations
            
            fluctuationPlus = computeRoe(
                values[rightBoundary_subDomain-3],
                values[rightBoundary_subDomain-2],
                systemMatrix,'positive',
                deltaT) #TODO: fill in arguments
            fluctuationMinus = computeRoe(
                values[rightBoundary_subDomain-2],
                values[rightBoundary_subDomain-1][:moments[m]],
                systemMatrix,
                'negative',
                deltaT) #TODO: fill in arguments
            sourceTerm = sourceTerm(values[i]) #TODO: fill in arguments
            values[rightBoundary_subDomain-2] = values[rightBoundary_subDomain-2] 
            -deltaT/deltaX(fluctuationPlus+fluctuationMinus) + sourceTerm # solve FVM equations

            #TODO: treat special equations
        else:
            rightBoundary_subDomain = boundaryInterfaces_Discretized[m]+2
    for i in range(rightBoundary_subDomain+1,n):
         solve=1# solve FVM equations for the last subdomain

    return values

# This function postprocesses the output data of the simulation and transforms it in data that can be plotted
def postProcessing(endValues):
    processedValues=1 #postprocess the data
    return processedValues
