import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def GenerateFigure(R = 1.0,n = 3,h = 1.0,x0 = 0, y0 = 0, alpha = 0.0):

    '''
    Where R is a Radius of circle a figure is written in
    and n is the number of corners, edges, whatever,
    math tells us there are two possible ways to build a figure from a known circle,
    why circle? it needs only one variable.
    '''

    #a = 2 * R * np.sin(np.pi/n) # side length

    figure = np.zeros((n+1,2))#,dtype=[('xy',float,(2,)),('z',float)])

    figure[:n] =  [[x0 + R * np.cos(tau),y0 + R * np.sin(tau)] for tau in np.arange(alpha,alpha + 4*np.pi/2,np.deg2rad(360/n))]
    figure[-1] = figure[0]
    #figure['z'] = np.random.uniform(0,h)

    return figure

R = 2.0
n = 6

angle = np.deg2rad(360/n)

lilhex  = GenerateFigure(R=R,n=n,h=1.0,x0=0.0,y0=0.0)


a = 2 * R * np.sin(np.pi/n) # side length do we need that?

r = 0.5 * a / np.tan(np.pi/n)


plt.scatter(0,0,c='k')


# draws hex0
hexMesh = GenerateFigure(R=R,n=6,h=1.0,x0=0,y0=0)
plt.plot(hexMesh[:,0],hexMesh[:,1],c='tab:blue',lw=0.75)

# hex ring n = 1
N_mesh = GenerateFigure(R= 2*r,n=6,h=1.0, x0=0, y0=0, alpha = angle/2)
hexMesh = [GenerateFigure(R=R,n=6,h=1.0,x0=x,y0=y) for x,y in N_mesh]

for n in range(len(hexMesh)):

    plt.plot(hexMesh[n][:,0],hexMesh[n][:,1],c='tab:orange',lw=0.75)

#hex ring n = 2

for n in range(2):

    c = n%2
    if c == 0:
        N_mesh = GenerateFigure(R= 2 * R + a,n=6,h=1.0, x0=0, y0=0, alpha = 0 )
    if c == 1:
        N_mesh = GenerateFigure(R= 4*r,n=6,h=1.0, x0=0, y0=0, alpha = angle/2)


    hexMesh = [GenerateFigure(R=R,n=6,h=1.0,x0=x,y0=y) for x,y in N_mesh]

    for n in range(len(hexMesh)):
        plt.plot(hexMesh[n][:,0],hexMesh[n][:,1],c='tab:red',lw=0.75)

# hex ring n = 3

for n in range(4):

    c = n%2
    # if c == 0:
    #     N_mesh = GenerateFigure(R= 4 * R + a,n=6,h=1.0, x0=0, y0=0, alpha = angle/2 )
    if c == 1:
        N_mesh = GenerateFigure(R=a+ 4*r,n=6,h=1.0, x0=0, y0=0, alpha = angle/6)

    hexMesh = [GenerateFigure(R=R,n=6,h=1.0,x0=x,y0=y) for x,y in N_mesh]

    for n in range(len(hexMesh)):
        plt.plot(hexMesh[n][:,0],hexMesh[n][:,1],c='tab:green',lw=0.75)



plt.grid(which='both')
#plt.GridSpec()
plt.axis('equal')

plt.show()