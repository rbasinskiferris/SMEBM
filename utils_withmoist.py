import numpy as np
from numba import jit


#see note on remarkable 23 feb about boundary conditions
@jit
def odefunc_ebm_total_lat_firstordertaylor(T,t,aw,ai,S,A,B,D,dx,x,cw,ramp_on=0,F=0,drop=0.2,relhum=0,eps=0,psfc=0,e0=0,a=0,b=0,L=0,cp=0,T_init=1): 
    
    #optionally can be moist or dry
    #model writen in latitude space and used for latitude from -90 to 90
    r=1 # r=1 consistent with https://www.gfdl.noaa.gov/blog_held/40-playing-with-a-diffusive-energy-balance-model/ and https://npg.copernicus.org/articles/32/23/2025/
    #This function can solve either the dry or moist ebm. it defaults to a dry ebm if relhum is 0.
    # Only works for both hemispheres
    x=np.deg2rad(x)
    if ramp_on==0:
        alpha = aw*(T>0)+ai*(T<0)
    else:
        alpha = aw-ramp(T,drop)
    
    Src = S*alpha
    #instead of doing the matrix for divergence
    Tdot = np.zeros(x.shape)
    n=np.size(x)
    dx=np.diff(x)[0]

    if relhum>0: #moist
        if np.size(T_init)>1: #if you give T_init with size greater than 1, you operate the model in perturbation mode as in the original code from Kyle Armour (I think used in Bonan et al.) 
            # spec. hum, and theta_e
            theta_e=1/cp*dh_func(T_init)*T #Do a 1/cp because that's absorbed into the D.
        else:
            q = eps*relhum/psfc*e0*np.exp(a*(T)/(b+(T)));  #q=q(:); # here T is in oC. q is g kg-1
            theta_e = 1/cp*(cp*((T)+273.15) + L*q) #theta is mse divided by cp. note units of Kelvin are needed!!!. #This is h/cp because the D is in terms of that.
        
        
        for i in range(1,n-1):
            Tdot[i]=(D/((r**2)*dx**2))*(theta_e[i+1]-2*theta_e[i]+theta_e[i-1])-(D*np.tan(x[i])/(2*dx*r**2))*(theta_e[i+1]-theta_e[i-1])
            # use central difference
        
        Tdot[0] = 4*D*(theta_e[1]-theta_e[0])/dx**2 #see note on remarkable for how we got this boundary condition
        Tdot[-1] = -4*D*(theta_e[-1]-theta_e[-2])/dx**2

        if np.size(T_init)>1: #perturbation mode 
            f = (F - (B*(T)) + Tdot)/cw
        else: #full mode
            f = (Src +F-A - (B*T) + Tdot)/cw
    else: #dry
        for i in range(1,n-1):
            Tdot[i]=(D/((r**2)*dx**2))*(T[i+1]-2*T[i]+T[i-1])-(D*np.tan(x[i])/(2*dx*r**2))*(T[i+1]-T[i-1])
        # use central difference
        Tdot[0] = 4*D*(T[1]-T[0])/dx**2
        Tdot[-1] = -4*D*(T[-1]-T[-2])/dx**2
        if np.size(T_init)>1:
            f = (F - (B*(T)) + Tdot)/cw
        else:
            f = (Src +F-A - (B*T) + Tdot)/cw
    return f



@jit
def dh_func(T): #note this has a times cp that we dont use when we use h in diffusion because that's in the D
    relhum = 0.8   # relative humidity
    eps = 0.622    # moisture constant
    psfc = 9.8e4   # (Pa)
    e0 = 611.2     # vap. press (Pa)
    a = 17.67
    b = 243.5   # sat vap constants
    L = 2.45e6         # latent heat of vaporization (J kg-1)
    cp = 1004         # (J kg-1 K-1)
    dq=eps*relhum/psfc*e0*((a*b)/(b+T)**2)*np.exp(a*(T)/(b+(T)))
    theta_e = cp+ L*dq #this just got updated. april 23
    return theta_e


# ########## NEED THIS ONE TO CHECK FOR THE TAYLOR EXPANSION STUF
@jit
def odefunc_ebm_total_lat_variableD_plusterm_v4(T,t,aw,ai,S,A,B,D,dx,x,cw,ramp_on=0,F=0,drop=0.2,relhum=0,eps=0,psfc=0,e0=0,a=0,b=0,L=0,cp=0,T_init=1,dhdT=1,last_term=0,include_b=0): 
    Bval=B
    #optionally can be moist or dry
    #model writen in latitude space!
    r=1 # r=1 consistent with https://www.gfdl.noaa.gov/blog_held/40-playing-with-a-diffusive-energy-balance-model/ and https://npg.copernicus.org/articles/32/23/2025/
    #This function can solve either the dry or moist ebm. it defaults to a dry ebm if relhum is 0.
    # Only works for both hemispheres
    x = np.deg2rad(x.copy())
    if ramp_on==0:
        alpha = aw*(T>0)+ai*(T<0)
    else:
        alpha = aw-ramp(T,drop)
    
    Src = S*alpha
    #instead of doing the matrix for divergence
    Tdot = np.zeros(x.shape)
    n=np.size(x)
    dx=np.diff(x)[0]
    #dry
    for i in range(1,n-1):
        Tdot[i] = D*dhdT[i]*(T[i+1]-2*T[i]+T[i-1])/dx**2 - D*dhdT[i]*np.tan(x[i])/ (2*dx) * (T[i+1] - T[i-1])
    # use central difference
    Tdot[0] = 4*D*dhdT[0]*(T[1]-T[0])/dx**2
    Tdot[-1] = -4*D*dhdT[-1]*(T[-1]-T[-2])/dx**2
    if np.size(dhdT)>1: #calculates term b or not
        graddot = np.zeros_like(dhdT)
        for i in range(1,n-1):
            grad_h_plus = (dhdT[i+1] - dhdT[i]) / dx
            grad_T_plus = (T[i+1] - T[i]) / dx
            grad_h_minus = (dhdT[i] - dhdT[i-1]) / dx
            grad_T_minus = (T[i] - T[i-1]) / dx
    
            graddot[i] = 0.5 * (grad_h_plus * grad_T_plus + grad_h_minus * grad_T_minus)
        graddot[0]=(T[1]-T[0])*(dhdT[1]-dhdT[0])/dx**2
        graddot[-1]=(T[-1]-T[-2])*(dhdT[-1]-dhdT[-2])/dx**2

    if last_term>0: #this includes term c or not
        dhdTdot=np.zeros(x.shape)
        for i in range(1,n-1):
            dhdTdot[i]=(D/(dx**2))*(dhdT[i+1]-2*dhdT[i]+dhdT[i-1])-(D*np.tan(x[i])/(2*dx))*(dhdT[i+1]-dhdT[i-1])
        dhdTdot[0] = 4*D*(dhdT[1]-dhdT[0])/dx**2
        dhdTdot[-1] = -4*D*(dhdT[-1]-dhdT[-2])/dx**2
    if np.size(T_init)>1:
        if np.size(dhdT)>1:
            if include_b>0:
                f = (F - (Bval*(T)) + Tdot+2*D*graddot)/cw
            else:
                f = (F - (Bval*(T)) + Tdot)/cw #include only term a
        if last_term>0:
            f=(F - (Bval*(T)) + Tdot+2*D*graddot+dhdTdot*T)/cw
        elif np.size(dhdT)==1:
            f = (F - (Bval*(T)) + Tdot)/cw
    else:
        f = (Src +F-A - (Bval*T) + Tdot)/cw
    return f

@jit
def ramp(T,drop=0.2):
    #return np.where(T < -5, 0.2, np.where(T > 5, 0, 0.2 * (1 - (T + 5) / 10)))
    return np.where(T < -5, drop, np.where(T > 5, 0, drop * (1 - (T + 5) / 10)))


# Stochastic integration stuff
@jit
def noise_mat_white(i,num_time,cw,noise):
    np.random.seed(i)
    n=np.size(cw)
    sigma = noise
    noise_mat = np.zeros((n, num_time))

    for t in range(0,num_time):
        for i in range(0, n):
            noise_mat[i, t] = (sigma[i] / cw[i]) * np.random.randn()
    
    return noise_mat


def itoEuler_withnoisemat_explicit(f, noise_mat, y0, tspan,dx=1):
    #modified from code in sdeint package
    d = np.size(y0)  # Dimension of the initial state
    
    N = len(tspan)  # Number of time steps
    h = (tspan[-1] - tspan[0]) / (N - 1)  # Step size
    
    # Allocate space for result
    y = np.zeros((N, d), dtype=y0.dtype)
   
    # Initialize starting point
    y[0] = y0
    
    # Euler-Maruyama iteration
    for n in range(N - 1):
        tn = tspan[n]
        yn = y[n]
        
        y[n + 1] = yn + f(yn, tn) * h+np.sqrt(h/dx)*noise_mat[:,n] #see algorithm 10.8 in https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E025F475C24D116BF7633D7225D64C87/9781139017329c10_p431-481_CBO.pdf/semilinear_stochastic_pdes.pdf

    return y
