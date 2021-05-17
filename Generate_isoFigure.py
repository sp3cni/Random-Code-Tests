from math import sin,cos,pi

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def GenerateFigure(R = 1.0,n = 3,h = 1.0):

    '''
    Where R is a Radius of circle a figure is written in
    and n is the number of corners, edges, whatever,
    math tells us there are two possible ways to build a figure from a known circle,
    why circle? it needs only one variable.
    '''

    #a = 2 * R * np.sin(np.pi/n) # side length

    figure = np.zeros(n+1,dtype=[('xy',float,(2,)),('z',float)])

    figure['xy'][:n,] =  [[R * np.cos(tau),R * np.sin(tau)] for tau in np.arange(0,4*np.pi/2,np.deg2rad(360/n))]
    figure['xy'][-1,] = figure['xy'][0]
    figure['z'] = np.random.uniform(0,h)

    return figure

R = 1.0
n = 6



lilhex  = GenerateFigure(R,n,10.0)


a = 2 * R * np.sin(np.pi/n) # side length

r = 0.5 * a / np.tan(np.pi/n)

Rcircle = np.array([(R * np.cos(tau),R * np.sin(tau)) for tau in np.linspace(0,2 * np.pi,360)])

rcircle = np.array([(r * np.cos(tau),r * np.sin(tau)) for tau in np.linspace(0,2 * np.pi,360)])

fig = plt.figure(figsize=[6,6],frameon=False)

ax = fig.add_axes([0,0,1,1])

ax.set_xlim(-2, 2), ax.set_xticks([])
ax.set_ylim(-2, 2), ax.set_yticks([])

ax.plot(lilhex['xy'][:,0],lilhex['xy'][:,1],lw=0.75)

ax.plot(Rcircle[:,0],Rcircle[:,1],c='tab:orange',lw=0.75)

ax.plot(rcircle[:,0],rcircle[:,1],c='tab:gray',lw=0.75)

ax.scatter(lilhex['xy'][:,0],lilhex['xy'][:,1],c='tab:blue',marker='o',s= 5.0)
ax.scatter(0,0,c='tab:red',s= 5.0,marker='o',)

ax.quiver(lilhex['xy'][0,0],lilhex['xy'][0,1],0,1,color='tab:blue',scale=4.0,scale_units='xy',width=0.003,headwidth=2.)
ax.quiver(lilhex['xy'][0,0],lilhex['xy'][0,1],1,0,color='tab:blue',scale=4.0,scale_units='xy',width=0.003,headwidth=2.)

ax.quiver(0,0,lilhex['xy'][0,0],lilhex['xy'][0,1],color='tab:red',scale=1.0,scale_units='xy',width=0.003,headwidth=2.)


def PointName(name,point):
    ax.annotate(name,xy = point,
                     xycoords='data',
                     xytext=point*20,
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->',color='black'))
    return

names = [f'P{n}' for n in range(1,len(lilhex))]

[PointName(name,point[0]) for name,point in zip(names,lilhex)]

ax.annotate('CG',xy = (0,0),
                 xycoords='data',
                 xytext=(20.0,-20.0),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->',color='black'))

ax.annotate('Rcg',xy = (0,0.1),
                 xycoords='data',
                 xytext=(1.0,0.0),
                 textcoords='offset points',
                 color='tab:red')


ax.quiver(0,0,rcircle[150,0],rcircle[150,1],color='tab:gray',scale=1.0,scale_units='xy',width=0.003,headwidth=2.)

ax.annotate('r',xy = (rcircle[150,0]/2,rcircle[150,1]/2),
                 xycoords='data',
                 xytext=(1.0,0.0),
                 textcoords='offset points',
                 color='tab:gray')


ax.annotate('This line \ndenotes area \nin which\ncollision \ndefinitely happened',xy = (rcircle[270,0],rcircle[270,1]),
                 xycoords='data',
                 xytext=(30.0,-120.0),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->',color='tab:gray'),
                 color='tab:gray')

N = 14 * n


ax.quiver(0,0,Rcircle[N,0],Rcircle[N,1],color='tab:orange',scale=1.0,scale_units='xy',width=0.003,headwidth=2.)

ax.annotate('R',xy = (Rcircle[N,0]/2,Rcircle[N,1]/2),
                 xycoords='data',
                 xytext=(3.0,3.0),
                 textcoords='offset points',
                 color='tab:orange')

ax.annotate('Inside this line\nwe start\nto check for collisions',
                 xy = (Rcircle[N,0],Rcircle[N,1]),
                 xycoords='data',
                 xytext=Rcircle[N,:]*40,
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->',color='tab:orange'),
                 color='tab:orange')



def VectorFromPoints(A,B):

    hypot = np.sqrt(np.sum((A-B)**2))

    return (B-A)/hypot

C = [lilhex['xy'][n-1,:] + 0.714 * VectorFromPoints(lilhex['xy'][n-1,:],lilhex['xy'][n,:]) for n in range(1,len(lilhex))]

# ax.text(C[0][0],C[0][1]+0.05,
#         'Just like this one here',
#         color='tab:blue'
#         ,ha="right", va="bottom",
#         rotation=-45, size=10,
#         bbox=dict(boxstyle="rarrow,pad=0.3", fc="w", ec="tab:blue", lw=0.75))
#
# ax.text(C[1][0]-0.025,C[1][1]-0.025,
#         'Or Here',
#         color='tab:blue'
#         ,ha="right", va="top",
#         rotation= 45, size=10,
#         bbox=dict(boxstyle="rarrow,pad=0.3", fc="w", ec="tab:blue", lw=0.75))
#
# D = lilhex['xy'][0,:] + VectorFromPoints(lilhex['xy'][0,:],lilhex['xy'][1,:])
#
# ax.text(D[0],D[1] + 0.05,
#         'Another Here',
#         color='tab:blue'
#         ,ha="right", va="bottom",
#         rotation= -45, size=10,
#         bbox=dict(boxstyle="rarrow,pad=0.3", fc="w", ec="tab:blue", lw=0.75))

# ax.annotate('Just like this one here',
#                  xy = C[0],
#                  xycoords='data',
#                  xytext=C[0]*30,
#                  textcoords='offset points',
#                  arrowprops=dict(arrowstyle='->',color='tab:blue'),
#                  color='tab:blue',
#                  ha="right", va="center")

plt.show()
