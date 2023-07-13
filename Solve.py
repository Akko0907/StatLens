import numpy as np


#------------------------------------------
# LINEAR SOLVES
#------------------------------------------


def GaussE(A: np.ndarray, bvec: np.ndarray) -> np.ndarray:
    ''' Linear equation solve with Gauss-Elimination method.
    Receives a matrix with the equations coefficients and
    a vector of the independent terms '''

    # Gen the augmented matrix
    A = np.column_stack((A,bvec))
    dim1,dim2 = np.shape(A)

    # Triangulization of the matrix
    for i in range(dim1-1):
        pivot = A[i][i]
        count = 0
        while (pivot==0) and (count<=dim1-i):
            B = A[i]
            A[i] = A[i+1]
            A[i+1] = B
            pivot = A[i][i]
        if pivot==0:
            break

        m_ij = (A.T[i]/A[i][i])
        A[i+1:] = np.outer( m_ij, A[i] )[i+1:] - A[i+1:]

    # Solve for the variables
    s = np.zeros(dim1)
    for i in range(0,dim1):
        if np.any(A[dim1-1-i]!=0):
            b = A.T[dim2-1][dim1-1-i]
            a = A[dim1-1-i][dim2-2-i]
            k = np.dot(A[dim1-1-i][:-1],s[::-1])

            s[i] = (b-k)/a

    s = s[::-1]
    return s

#============================================================================

def Converge(A: np.ndarray) -> bool :
    k = np.sum(A,axis=1)-np.diag(A)
    converge = np.all(k<=np.diag(A))

    return converge
def GaussJ(A: np.ndarray, bvec: np.ndarray, 
            x0: np.ndarray, error: float=0.001) -> np.ndarray:
    ''' Linear equation solve with Gauss-Jacobi method.
    Receives a matrix with the equations coefficients,
    a vector of the independent terms and a first kick value
    for the solution '''

    # Check convergence
    s =cvg.Converge(A)
    if s:
        counter = 0
        loop = True
        while loop:
            counter+=1
            #Iterate using GJ method
            xn = (bvec-(np.dot(A,x0)-np.diag(A)*x0))/np.diag(A)
            x0 = xn

            # Check condition to keep looping
            check = abs(np.dot(A,xn)-bvec)
            if np.all(check<error):
                loop = False

        xf = xn
        print(f'\n{counter} steps taken to reach an error of {error}')
        print(f'x = {xf}\n')
        return xf
   
    else:
        print("sorry, but it doesn't seems to converge :c ")

#============================================================================

def GaussS(A: np.ndarray, bvec: np.ndarray, 
            x0: np.ndarray, error: float=0.001) -> np.ndarray:
    ''' Linear equation solve with Gauss-Seidel method.
    Receives a matrix with the equations coefficients,
    a vector of the independent terms and a first kick value
    for the solution '''

    # Check convergence
    s =cvg.Converge(A)
    if s:
        # Initiate variables
        loop = True
        counter = 0
        N = len(A)
        xn = np.zeros_like(x0,dtype='float')

        #Iterate using GS method
        while loop:
            counter+=1
            for i in range(N):
                xn[i] = (bvec[i]-(np.dot(A,x0)[i]-np.diag(A)[i]*x0[i]))/np.diag(A)[i]
                x0[i] = xn[i] 

                # Check condition to keep looping
                check = abs(np.dot(A,xn)-bvec)
                if np.all(check<error):
                    loop = False   

        xf = xn
        print(f'\n{counter} steps taken to reach an error of {error}')
        print(f'x = {xf}\n')
        return xf

#============================================================================

def SOR_method(A: np.ndarray,bvec: np.ndarray,
               x0: np.ndarray, omega: float,
               error: float=0.01) -> np.ndarray:
    
    ''' Linear equation solve with Sucessive Relaxation method.
    Receives a matrix with the equations coefficients,
    a vector of the independent terms, a first kick value
    for the solution and the omega relaxation factor '''

    # Initiate variables
    loop = True
    counter = 0
    N = len(A)
    xn = np.zeros_like(x0,dtype='float')

    #Iterate using SOR method
    while loop:
        counter+=1
        for i in range(N):
            xn[i] = (1-omega)*x0[i] + omega*(bvec[i]-(np.dot(A,x0)[i]-np.diag(A)[i]*x0[i]))/np.diag(A)[i]
            x0[i] = xn[i] 

            # Check condition to keep looping
            check = abs(np.dot(A,xn)-bvec)
            if np.all(check<error):
                loop = False   
        if counter==1000:
            break

    xf = xn
    print(f'\n{counter} steps taken to reach an error of {error}')
    print(f'U = {xf}\n')
    return xf

#=================================================================================================================


#------------------------------------------
# SEEK ROOTS
#------------------------------------------


def find_crossing(ya,yb=0):
    if np.all(ya>yb):
        indexes = []
    else:
        indexes = np.where( abs(yb-ya)==min(abs(yb-ya)) )
    
    return indexes

#===========================================================================

def check_err(x0: float,xn: float,err: float) -> bool:
    """ Check error"""

    error = abs(x0-xn)
    if np.any(error>=err):
        return True
    else:
        return False
def check_err2(f: function,xn: float,err: float) -> bool:
    """ Check error"""

    error = abs(f(xn))
    if np.any(error>=err):
        return True
    else:
        return False

#============================================================================


def ZeroExists(f: function,a: float,b: float) -> bool:
    """Check existence of a zero in the function"""
    I = np.linspace(a,b,100)
    f_vec = f(I)
    
    m = np.any(f_vec<=0)
    n = np.any(f_vec>=0)

    if m and n:
        return True
    else:
        return False
def Bissec(f: function, a0: float, b0: float,
            x0: float=None, error: float=0.01) -> float:
    """ The function to be analized is 'f', 'a' and 'b' are the interval's
    start and finish. Function must be well behaved! """
    
    if x0==None:
        x0 = (b0+a0)/2
    a_new = a0
    b_new = b0

    loop = ZeroExists(f,a0,b0)
    count = 0
    while loop:    
        count += 1
        if f(x0)*f(b_new)<0:
            a_new = x0
            xn = a_new+(b_new-a_new)/2
            loop = check_err(x0,xn,error)
        else:
            b_new = x0
            xn = a_new+(b_new-a_new)/2
            loop = check_err(x0,xn,error)
        x0 = xn

    if count>0:
        print(f"zero founded at x={xn} in the interval [{a0},{b0}] within error < {error} in {count} iterations")
        return xn
    else:
        print(f"No zero founded at the interval [{a0},{b0}]")
        return None

#=================================================================================================================

def FpConv(g: function,a: float,b: float,pace: float=0.1) -> bool:
    """Check convergence of fixed-point method with g function
    in the interval [a,b]"""
    
    x_vec = np.arange(a,b+pace,pace)
    del_g = np.diff(g(x_vec))/np.diff(x_vec)

    if np.all(abs(del_g)<=1):
        return True
    else:
        return False
def FixedP(g: function, a: float, b: float,
           x0: float=None, error: float=0.01,
           loop: bool=False, max_count: int=300) -> float:
    """ Receives a function g(x), transformed from f(x)=0 into x=g(x) """

    if loop==False:
        loop = FpConv(g,a,b)

    if x0==None:
        x0 = (b+a)/2
        
    count = 0
    while loop:
        count += 1
        xn = g(x0)
        loop = check_err(x0,xn,error)
        x0 = xn
        if count > max_count:
            print(f"count exceeded {max_count} for some reason,breaking (did you check its convergence?)")
            return None

    if count==0:
        print(f"Convergence not guarantee")
        return None
    
    else: 
        print(f"zero founded at x={xn} in the interval [{a},{b}] within error < {error} in {count} iterations")
        return xn

#=================================================================================================================

def ZerosNewton(f: function,a: float,b: float,
                error: float=0.01, kicks: list=None,
                max_count: int=300) -> float:
                
    """Find function zeros using Newton's secant method"""

    g = lambda xi,xj: xj-f(xj)*(xi-xj)/(f(xi)-f(xj))

    if kicks==None:
        x0 = (2*a+b)/3
        x1 = (a+2*b)/3
    else:
        x0 = kicks[0]
        x1 = kicks[1]

    count = 0
    loop = True
    while loop:
        count += 1
        xn = g(x0,x1)
        loop = check_err2(f,xn,error)
        x0 = x1
        x1 = xn
        if count > max_count:
            print(f'count exceeded {max_count} for some reason,breaking...')
            return None
    print(f'Zero found in x = {xn} within error = {error}')
    return xn

#=================================================================================================================


#------------------------------------------
# EDO'S SOLVE
#------------------------------------------


class EDO():
    def __init__(self,F: list, T: float, dt: float=0.5, kind: str='base'):
        if not isinstance(dt,(float,int)):
            raise TypeError("h must be an integer or float")
        if not isinstance(T,(float,int)):
            raise TypeError("T must be an integer or float")
        if not all(hasattr(val, "__call__") for val in F):
            raise TypeError("F must be a list of callables (functions)")
        if not isinstance(kind,str) or kind not in ['base','mod','rk2','rk4','special']:
            raise TypeError(f"kind must be a string between: {['base','mod','rk2','rk4','special']}")

        self.__F = F
        self.__h = dt
        self.__kind = kind 
        self.__T = T
        self.__N = int(T//dt)
        self.__sys_size = len(F)


    def __call__(self,init: np.ndarray,kind: str=None) -> tuple:
        if kind!=None:
            if not isinstance(kind,str) or kind not in ['base','mod','rk2','rk4','special']:
                raise TypeError(f"new_kind must be a string between: {['base','mod','rk2','rk4','special']}")
            self.__kind = kind

        sys_size = self.__sys_size
        N = self.__N
        
        pace = init.copy()
        ys = np.zeros((N+1,sys_size))
        ys[0] = pace
        
        for i in range(1,N+1):
            ys[i] = self.step(pace)  
            pace = ys[i]
        
        t = np.linspace(0,self.__T,N+1)
        ys = ys.T
        return t,*ys 


    @property
    def h(self) -> float:
        return self.__h
    @property
    def T(self) -> float:
        return self.__T
    @property
    def kind(self) -> str:
        return self.__kind
    

    @h.setter
    def h(self,new_h: float) -> None:
        if not isinstance(new_h,(float,int)):
            raise TypeError("new_h must be an integer or float")
        self.__h = new_h
        self.__N = int(self.T//self.h)
    @T.setter
    def T(self,new_T: float) -> None:
        if not isinstance(new_T,(float,int)):
            raise TypeError("new_T must be an integer or float")
        self.__T = new_T
        self.__N = int(self.T//self.h)
    @kind.setter
    def kind(self,new_kind: str) -> None:
        if not isinstance(new_kind,str) or new_kind not in ['base','mod','rk2','rk4','special']:
            raise TypeError(f"new_kind must be a string between: {['base','mod','rk2','rk4','special']}")
        self.__kind = new_kind
                

    def step(self,pace: np.ndarray) -> np.ndarray:
        
        sys_size = self.__sys_size
        F = self.__F

        if self.__kind=='base':
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            yi = pace + k1
            return yi
        
        elif self.__kind=="mod":
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            pace1 = pace + k1
            k2 = np.array([self.h*F[i](*pace1) for i in range(sys_size)])
            yi = pace + (k1 + k2)/2
            return yi

        elif self.__kind=="rk2":
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            pace1 = pace + k1/2
            k2 = np.array([self.h*F[i](*pace1) for i in range(sys_size)])
            yi = pace + k2
            return yi

        elif self.__kind=="rk4":
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            pace1 = pace + k1/2
            k2 = np.array([self.h*F[i](*pace1) for i in range(sys_size)])
            pace2 = pace + k2/2
            k3 = np.array([self.h*F[i](*pace2) for i in range(sys_size)])
            pace3 = pace + k3
            k4 = np.array([self.h*F[i](*pace3) for i in range(sys_size)])  
            yi = pace + (1/6)*(k1 + 2*k2 + 2*k3 + k4) 
            return yi

        elif self.__kind=="special":
            yi = pace.copy()
            for j in range(sys_size):
                yi[j] = yi[j] + self.h*F[j](*yi)
            return yi

