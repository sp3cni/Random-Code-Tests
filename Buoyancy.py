import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def Trapz(value=np.array([[0.0,1.0]]),variable = np.array([0,0.01,1.0])):

    Area = lambda p0,p1,dt : ( p0 + p1 ) * dt /2

    p_right = [value[n] for n in range(len(value))]
    p_left = np.zeros_like(p_right)

    for n in range(1,len(value)):
        p_left[n-1] = value[n]

    t0,dt,tf = variable
    n = int((tf - t0)/dt)
    dt = (tf - t0)/n

    timespan = np.linspace(t0,tf,n)
    ts0 = np.zeros_like(timespan)

    for n in range(1,len(timespan)):
        ts0[n-1] = timespan[n]

    f = [Area(p_left[c],p_right[c],dt) for c in range(len(p_right))]
    l1 = np.sqrt((timespan - ts0)**2)
    l2 = np.sqrt((p_right - p_left)**2)
    F = np.sum(f)

    return 2 * np.round_(F,15)

def Teardrop(m=0,A=1,B=1,N=50):

    shapeEQ = lambda t,m,A,B : [-A * np.cos(t),B * np.sin(t) * np.sin(t/2)**m]

    tau = np.linspace(0,np.pi,N)

    x,y = shapeEQ(tau,m,A,B)

    x = x + A
    y = y

    #Rx = [x for x in reversed(x)]
    Ry = [x for x in reversed(y)] # to jest istotne dla odwrócenia kształtu...
    return x,Ry


def AreaOfShape(shape='rectangle',m = 0, A = 1, B=1, h0 = 0.0, hf = 1.0, N = 50):

    '''
    How to find an area of a shape? Derive it's volume? We want to know it's volume...
    integrate area of curve under it's edge? Why not...
    The method used is trapezoidal itegration only because it's using the least operations
    of the three define integration methods i know...
    '''
    if shape.lower() == 'rectangle':
        x = np.linspace(h0,hf,N)
        y = [A for H in x]
        return x,y,Trapz(y,[min(x),(max(x)-min(x))/N,max(x)])

    if shape.lower() == 'circle':
        x,y = Teardrop(0,A,A,N)
        return x,y,Trapz(y,[min(x),(max(x)-min(x))/N,max(x)])

    if shape.lower() == 'ellipse':
        x,y = Teardrop(0,A,B,N)
        return x,y,Trapz(y,[min(x),(max(x)-min(x))/N,max(x)])

    if shape.lower() == 'teardrop':
        x,y = Teardrop(m,A,B,N)
        return x,y,Trapz(y,[min(x),(max(x)-min(x))/N,max(x)])
    else:
        return 0,0,0

def CSline(shape = 'rectangle', R=1.0, h=0.1):

    if shape.lower() == 'rectangle':return 2*A

    if shape.lower() == 'circle':return 0.5 * np.sqrt(A*h - h**2)

    if shape.lower() == 'ellipse':return 0.5 * np.sqrt(A*h - h**2)

    if shape.lower() == 'teardrop':return 0.5 * np.sqrt(A*h - h**2)
    else:
        return 0,0,0

def FindSubmergedDepth(x,y,Aobj,
                       dLength = 0.1,
                       hf=10.0,
                       Rho_obj=1.00,
                       Mobj = 0.0,
                       Rho_fluid = 1.00,
                       G = 9.80665,
                       RealDepth = 0.0,
                       OuterForce = 0.0):

    Vobj = Aobj * dLength
    Wobj = Vobj * Rho_obj * G if Mobj <= 0.0 else Mobj * G

    Wobj = Wobj + OuterForce

    Vdisplaced = Wobj / Rho_fluid / G # What volume of fluid would bring neutral buoyancy

    Hsubmersed = Vdisplaced / dLength / ??

    dh = (max(x) - x[0])/len(x)

    hhf = 0.0
    Vobj2 = 0.0

    while Vobj2 <= Vdisplaced:



    # wersja iterująca pole przekoju do osiagnięcia tej wartości
    # ale istnieje coś co jest łatwiejsze do wyliczenia, dla kształtów innych niż kropla...

    while Vobj2 <= Vdisplaced:
        if hhf <= max(x) :
            n = int((hhf-0.0)/dh)
            yy = np.zeros(n)
            yy0 = np.zeros(n)
            xx = np.zeros(n)
            hhf += dh
        else:break
        if n < 2: continue # pominięcie poniższego kody jeśli jest mniej niż 2 próbki
        for n in range(0,n):
            try:
                xx[n]  =  x[n]
                yy[n]  =  y[n]
                yy0[n] =  -y[n]
            except: break
        try:
            Aobj2 = Trapz(yy,[0,max(xx)/len(xx),max(xx)])
            Vobj2 = Aobj2 * dLength
        except:continue
        Vdiplaced = Vdisplaced - \
                    (max(xx) + RealDepth)* max(yy) * dLength

    return max(xx)



## Teardrop:

#fig,ax = plt.subplots(1,1)

FrameLimit = 3.0


fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111) #[-30.0, -30.0, 30.0, 30.0]
ax.set_xlim(-3.0, 3.0), ax.set_xticks([])
ax.set_ylim(0, 2 * 3.0), ax.set_yticks([])

Body, = ax.plot([],[],c='k',lw=1.0)
waterline1, = ax.plot([0,0],[0,0],c='xkcd:lightblue',lw=0.50)
waterline2, = ax.plot([0,0],[0,0],c='xkcd:lightblue',lw=0.50)

x, y, Aobj = AreaOfShape(shape='teardrop',m=1,A=2.0,B=3.5,h0=0.0,hf=10.0,N=100)

ny = [-y for y in reversed(y)]
rx = [x for x in reversed(x)]

global xshow
global yshow

xshow = np.hstack((x,rx))
yshow = np.hstack((y,ny))


FrameLimit = 3.0

scattr = ax.scatter(0.1,0.1,marker='D')

#VolumeFluid = 10.0

FrameBottomArea = (2*FrameLimit)**2

FluidHeight = lambda V,A : V/A


VolumeFluid = np.random.uniform(10.0,20.0)

#Body, = ax.plot(yshow,xshow,c='k',lw=1.0)
#waterline1, = ax.plot([-3.0,3.0],[0,0],c='xkcd:lightblue',lw=0.50)
#waterline2, = ax.plot([-3.0,3.0],[0,0],c='xkcd:lightblue',lw=0.50)

def init():

    Body.set_ydata([np.nan]*len(xshow))

    return Body


while True:

    VolumeFluid = np.random.uniform(10.0,20.0)
    H = 0 * FluidHeight(VolumeFluid,FrameBottomArea) # how high a watercolumn is
    h = FindSubmergedDepth(x,y,Aobj,Rho_obj=1.0, Rho_fluid=np.random.uniform(0.8,1.5)) # offset from height of water column
    #print(yshow[4])
    xshow += H-h
    #waterline1.set_xdata([-FrameLimit,FrameLimit])
    #waterline1.set_ydata([H,H])
    Body.set_xdata(xshow)
    Body.set_ydata(xshow)





# animation = FuncAnimation(fig, FrameUpdate, interval=10,init_func=init,blit=True )
# plt.show()



'''
Buoyancy is of two components,
weight of displaced fluid and pressure of fluid column
acting against weight of a submersed object.

'''




