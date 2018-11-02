from numpy import zeros, linspace, pi, cos, array, sin, vectorize, add, square
import matplotlib.pyplot as plt
def plotSpring (X_0, V_0, hDenominator, plot):
    omega = 1
    P = 2*pi/omega
    h = P/hDenominator
    T = 5*P
    N_t = int(round(T/h))
    t = linspace(0, N_t*h, N_t+1)

    x = zeros(N_t+1)
    v = zeros(N_t+1)

    # Initial condition
    
    x[0] = X_0
    v[0] = V_0

    # Step equations forward in time
    for n in range(N_t):
        x[n+1] = x[n] + h*v[n]
        v[n+1] = v[n] - h*omega**2*x[n]

    #create both plots
    if(plot):    
        fig = plt.figure()
        l1, l2 = plt.plot(t, x, 'b-', t, X_0* cos(omega*t) +V_0 * sin(omega*t), 'r--')
        fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left')
        plt.ylabel('x')
        plt.title("Position with respect to time")
        plt.xlabel('t\n h = '+str(t[1]-t[0]))
        
        plt.show()
        
        
        fig = plt.figure()
        l1, l2 = plt.plot(t, v, 'b-', t, -X_0*omega * sin(omega*t) +V_0*omega * cos(omega*t), 'r--')
        fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left')
        plt.ylabel('v')
        plt.title("Velocity with respect to time")
        plt.xlabel('t\n h = '+str(t[1]-t[0]))
        
        plt.show()

    xError = X_0*cos(omega*t) + V_0*sin(omega*t) - x
    vError = -X_0*omega * sin(omega*t) +V_0*omega * cos(omega*t) - v

    return(xError, vError, x, v, t)

def computeEnergy(x,v, t):
    energy = add(square(x),square(v))
    plt.plot(t, energy)
    plt.ylabel('Energy')
    plt.title("Numeric evolution of energy")
    plt.xlabel('t\n h = '+str(t[1]-t[0]))    
    plt.show()
    
def globalError(x,v,t):
    plt.ylabel('Error')
    plt.title("Global error in position")
    plt.xlabel('t\n h = '+str(t[1]-t[0]))
    plt.plot(t, x)
    plt.show()
    plt.ylabel('Error')
    plt.title("Global error in velocity")
    plt.xlabel('t\n h = '+str(t[1]-t[0]))
    plt.plot(t, v)
    plt.show()

def plotSpringImplicit(X_0, V_0,hDenominator, plot):
    omega = 1
    P = 2*pi/omega
    h = P/hDenominator
    T = 5*P
    N_t = int(round(T/h))
    t = linspace(0, N_t*h, N_t+1)

    x = zeros(N_t+1)
    v = zeros(N_t+1)

    # Initial condition
    x[0] = X_0
    v[0] = V_0
    
    # Step equations forward in time
    for n in range(N_t):
        x[n+1] = x[n]/(1+h**2) + h*v[n]/(1+h**2)
        v[n+1] = v[n]/(1+h**2) - h*x[n]/(1+h**2)
    #create both plots
    if(plot):    
        fig = plt.figure()
        l1, l2 = plt.plot(t, x, 'b-', t, X_0* cos(omega*t) +V_0 * sin(omega*t), 'r--')
        fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left')
        plt.ylabel('x')
        plt.title("Position with respect to time")
        plt.xlabel('t\n h = '+str(h))
        
        plt.show()
        
        
        fig = plt.figure()
        l1, l2 = plt.plot(t, v, 'b-', t, -X_0*omega * sin(omega*t) +V_0*omega * cos(omega*t), 'r--')
        fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left')
        plt.ylabel('v')
        plt.title("Velocity with respect to time")
        plt.xlabel('t\n h = '+str(h))
        
        plt.show()

    xError = X_0*cos(omega*t) + V_0*sin(omega*t) - x
    vError = -X_0*omega * sin(omega*t) +V_0*omega * cos(omega*t) - v

    return(xError, vError, x, v, t)

def plotPhaseSpace(x,v,t):
    fig = plt.figure()
    l1, l2 = plt.plot(x, v, 'b-', x[0]* cos(t) +v[0] * sin(t),
                      -x[0]*sin(t) + v[0]*cos(t), 'r--')
    fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left') 
    plt.ylabel('Velocity (v)')
    plt.title("Phase portrait of velocity and time")
    plt.xlabel('Position (x)\n h = '+str(t[1]-t[0]))
        
    return plt

def plotSpringSymplectic(X_0, V_0, hDenominator, plot):
    omega = 1
    P = 2*pi/omega
    h = P/hDenominator
    T = 5*P
    N_t = int(round(T/h))
    t = linspace(0, N_t*h, N_t+1)

    x = zeros(N_t+1)
    v = zeros(N_t+1)

    # Initial condition
    
    x[0] = X_0
    v[0] = V_0

    # Step equations forward in time
    for n in range(N_t):
        x[n+1] = x[n] + h*v[n]
        v[n+1] = v[n] - h*omega**2*x[n+1]

    #create both plots
    if(plot):    
        fig = plt.figure()
        l1, l2 = plt.plot(t, x, 'b-', t, X_0* cos(omega*t) +V_0 * sin(omega*t), 'r--')
        fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left')
        plt.ylabel('x')
        plt.title("Position with respect to time")
        plt.xlabel('t\n h = '+str(t[1]-t[0]))
        
        plt.show()
        
        
        fig = plt.figure()
        l1, l2 = plt.plot(t, v, 'b-', t, -X_0*omega * sin(omega*t) +V_0*omega * cos(omega*t), 'r--')
        fig.legend((l1, l2), ('numerical', 'analytical'), 'upper left')
        plt.ylabel('v')
        plt.title("Velocity with respect to time")
        plt.xlabel('t\n h = '+str(h))
        
        plt.show()

    xError = X_0*cos(omega*t) + V_0*sin(omega*t) - x
    vError = -X_0*omega * sin(omega*t) +V_0*omega * cos(omega*t) - v

    return(xError, vError, x, v, t)


#plots the trajectory, energy, global errors, and phase space of the symplectic
#model
'''
plotSpringSymplectic(2,0,50, True)

computeEnergy(plotSpringSymplectic(2,0,50, False)[2],
              plotSpringSymplectic(2,0,50, False)[3],
              plotSpringSymplectic(2,0,50, False)[4])
globalError(plotSpringSymplectic(2,0,50, False)[0],
            plotSpringSymplectic(2,0,50, False)[1]
            , plotSpringSymplectic(2,0,50, False)[4])
plotPhaseSpace(plotSpringSymplectic(2,0,50, False)[2],
               plotSpringSymplectic(2,0,50, False)[3],
               plotSpringSymplectic(2,0,50, False)[4]).show()
'''
#plots phase spaces of all three approximations
'''
plotPhaseSpace(plotSpring(2,0,100, False)[2],plotSpring(2,0,100, False)[3],
               plotSpring(2,0,100, False)[4]).show()

plotPhaseSpace(plotSpringImplicit(2,0,100, False)[2],
               plotSpringImplicit(2,0,100, False)[3],
               plotSpringImplicit(2,0,100, False)[4]).show()

plotPhaseSpace(plotSpringSymplectic(2,0,100, False)[2],
               plotSpringSymplectic(2,0,100, False)[3],
               plotSpringSymplectic(2,0,100, False)[4]).show()

'''
#plots energy and global error of the implicit calculation
'''
globalError(plotSpringImplicit(2,0,1000, False)[0], plotSpringImplicit
            (2,0,1000, False)[1], plotSpringImplicit(2,0,1000, False)[4])

computeEnergy(plotSpringImplicit(2,0,1000, False)[2], plotSpringImplicit
              (2,0,1000, False)[3], plotSpringImplicit(2,0,1000, False)[4])
'''
#plots energy and global error for the explicit calculation
"""
computeEnergy(plotSpring(2,0,1000, False)[2], plotSpring(2,0,1000, False)[3],
              plotSpring(2,0,1000, False)[4])
globalError(plotSpring(2,0,1000, False)[0], plotSpring(2,0,1000, False)[1]
            , plotSpring(2,0,1000, False)[4])

"""

#the section of code below was used to generate the plots of error vs h
"""
hVals = linspace(8000,10000, 100)
fig = plt.figure()
vPlotSpring = vectorize(plotSpring)
xErr = []
vErr = []
for i in hVals:
    xErr.append(max(abs(plotSpring (2,0, i)[0])))
    vErr.append(max(abs(plotSpring (2,0, i)[1])))
    
plt.plot(hVals, xErr)
plt.ylabel('error')
plt.title("Error in position")
plt.xlabel('2Pi*h')
plt.show()
plt.ylabel('error')
plt.title("Error in velocity")
plt.xlabel('2Pi*h')
plt.plot (hVals, vErr)
plt.show()
"""
