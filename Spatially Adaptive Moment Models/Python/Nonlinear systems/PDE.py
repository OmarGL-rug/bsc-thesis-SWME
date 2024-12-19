from abc import ABC, abstractmethod
import numpy as np

#TODO: use duck typing
class PDE(ABC):

    @abstractmethod
    def __init__(self, initialCondition):
        pass

    @abstractmethod
    def computeSystemMatrix(self,order,values):
        pass
    
    @abstractmethod
    def computeSourceTerm(self,order,values):
        pass


class SWME1D(PDE):

    def __init__(self, initialCondition):
        self.initialCondition = initialCondition

    def systemMatrix(self, order, values):
            g = 1
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
                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]

                A[0][0] = 0
                A[0][1] = 1
                A[0][2] = 0
                A[0][3] = 0
                A[1][0] = g*h - um*um - alpha1*alpha1/3 - alpha2*alpha2/5
                A[1][1] = 2*um
                A[1][2] = 2*alpha1/3
                A[1][3] = 2*alpha2/5
                A[2][0] = -2/5*alpha1*(5*um+2*alpha2)
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = 3*alpha1/5
                A[3][0] = -2/21*(7*alpha1*alpha1+3*alpha2*(7*um+alpha2))
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3
                A[3][3] = um + 3/7*alpha2
            if order == 3:
                alpha1 = values[2]/values[0]
                alpha2 = values[3]/values[0]
                alpha3 = values[4]/values[0]

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
                A[2][0] = -2/35*(7*alpha1*(5*um+2*alpha2)+9*alpha2*alpha3)
                A[2][1] = 2*alpha1
                A[2][2] = um + alpha2
                A[2][3] = 3*(alpha1+alpha3)/5
                A[2][4] = 3*alpha2/7
                A[3][0] = -2/21*(3*alpha2*(7*um+alpha2)+(alpha1+alpha3)*(7*alpha1+2*alpha3))
                A[3][1] = 2*alpha2
                A[3][2] = alpha1/3+9*alpha3/7
                A[3][3] = um + 3/7*alpha2
                A[3][4] = 4*alpha1/7 + alpha3/3
                A[4][0] = -2*um*alpha3 - 2*alpha2*(9*alpha1+4*alpha3)/15 
                A[4][1] = 2*alpha3
                A[4][2] = 0
                A[4][3] = 2*(alpha1+alpha3)/5
                A[4][4] = um+alpha2/3
            if order == 4:
                A=[0][0]=1 #fill (not implemented yet)
            if order == 5:
                A=[0][0]=1 #fill (not implemented yet)
            if order == 6:
                A=[0][0]=1 #fill (not implemented yet)
            return A

    def computeSourceTerm(self, order, values):
        viscosity = 1
        g = 1
        slipLength = 1
        
        S=np.zeros(order+2) 
        h = values[0]
        um = values[1]/values[0]
        if order == 0:
            S[0]=0
            S[1]=-viscosity/slipLength*um
        if order == 1:
            alpha1 = values[2]/values[0]

            S[0]=0
            S[1]=-viscosity/slipLength*(um+alpha1)
            S[2]=-3*viscosity/slipLength*(um+(1+4*slipLength/h)*alpha1)
        if order == 2:
            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]

            S[0]=0
            S[1]=-viscosity/slipLength*(um+alpha1+alpha2)
            S[2]=-3*viscosity/slipLength*(um+(1+4*slipLength/h)*alpha1+alpha2)
            S[3]=-5*viscosity/slipLength*(um+alpha1+(1+12*slipLength/h)*alpha2)
        if order == 3:
            alpha1 = values[2]/values[0]
            alpha2 = values[3]/values[0]
            alpha3 = values[4]/values[0]

            S[0]=0
            S[1]=-viscosity/slipLength*(um+alpha1+alpha2+alpha3)
            S[2]=-3*viscosity/slipLength*((h+4*slipLength)*alpha1+h*(um+alpha2)+(h+4*slipLength)*alpha3)/h
            S[3]=-5*viscosity/slipLength*(um+alpha1+(1+12*slipLength/h)*alpha2+alpha3)
            S[4]=-7*viscosity/slipLength*((h+4*slipLength)*alpha1+h*(um+alpha2)+(h+24*slipLength)*alpha3)/h
        if order == 4:
            S[0]=1 #fill (not implemented yet)
        if order == 5:
            S[0]=1 #fill (not implemented yet)
        if order == 6:
            S[0]=1 #fill (not implemented yet)
        return S
    
    def getInitialValues(self,order,initialCondition,position):
            vector = np.zeros(2+order)
            if initialCondition=='constantHeight_noVelocity':
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
            elif initialCondition=='constantHeight_constantVelocity':
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
            elif initialCondition=='damBreak_noVelocity':
                x0 = 0
                if position < x0:
                    vector[0] = 2
                    vector[1] = 0*vector[0]
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
                else:
                    vector[0] = 1
                    vector[1] = 0*vector[0]
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
            elif initialCondition == 'linearHeight_noVelocity':
                vector[0] = 1+0.1*position
                vector[1] = 0*vector[0]
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