
import os
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.markers import MarkerStyle


## ODE 45 discrete solution:

def RK45_step(function,x,t,dt,*args):
    # krok
    k = dt
    K1 = function(t,x,*args)
    K2 = function(t + 0.5*k,x + k * 0.5 * K1,*args)
    K3 = function(t + 0.5*k,x + k * 0.5 * K2,*args)
    K4 = function(t + dt, x + k * K3, *args)
    return x+(K1+2*K2+2*K3+K4)/6




## define a particle:

class particle():

    def __init__(self,position= np.zeros(3),
                      angles  = np.zeros(3),
                      lin_vel = np.zeros(3),
                      ang_vel = np.zeros(3),
                      lin_acc = np.zeros(3),
                      ang_acc = np.zeros(3),
                      dt = 0.05,
                      Mass = 0.1,
                      DeltaMass = 0.0,
                      Inertia_Tensor = np.identity(3),
                      Gravity = 9.80665,
                      Radius = 0.005,):

        # kinematics:

        self.Position = np.array(position)
        self.Orientation = np.array(angles)
        self.LinearVelocity = np.array(lin_vel)
        self.AngularVelocity = np.array(ang_vel)
        self.LinearAcceleration = np.array(lin_acc)
        self.AngularAcceleration = np.array(ang_acc)

        self.BoundaryX = [-2.0,2.0]
        self.BoundaryY = [-2.0,2.0]
        self.BoundaryZ = [-2.0,2.0]

        self.Kinetic_Energy = np.array([0,0,0,0,0,0])
        self.Potential_Energy =np.array([0,0,0,0,0,0])

        # life:

        self.dt = dt
        self.TimeElapsed = 0.0
        self.TimeVector = [0.0]

        self.isAlive = True

        self.lifetime = 12.0 #random.randint(0,30) + random.randint(1,100)/100

        # shape:

        self.Radius = Radius

        # mass:

        self.Mass = Mass
        self.dm = DeltaMass
        self.InertiaTensor = Inertia_Tensor
        self.Gravity = Gravity #* self.dt * 0.25



    def ParticleState(self):
        return dict(position = self.Position, orientation = self.Orientation)

    def Inertia(self):

        '''
        Inertia tensor of a filled sphere
        '''

        self.InertiaTensor = np.identity(3) * 0.4 * self.Mass * self.Radius * self.Radius

        return

    def RateOfChange(self,t,x,Forces=np.zeros(3),Moments=np.zeros(3)):

        xdot = np.zeros(12)

        invert_mass = 1/self.Mass

        [u,v,w] = self.LinearVelocity

        [p,q,r] = self.AngularVelocity

        [phi,theta,psi] = self.Orientation

        [Xe,Ye,Ze] = self.Position

        [Fx,Fy,Fz] = Forces

        [L,M,N] = Moments

        cphi = np.cos(phi)  ; sphi = np.sin(phi)
        cthe = np.cos(theta) ; sthe = np.sin(theta)
        tanthe = np.tan(theta); secthe = 1/cthe if cthe != 0 else np.sign(cthe)*1.633e16
        cpsi = np.cos(psi)  ; spsi = np.sin(psi)

        Ixx,Iyy,Izz = np.matmul(self.InertiaTensor, np.ones(3))


        # change of linear velocity:

        self.LinearAcceleration[0] = invert_mass *  Fx + r * v - q * w - self.Gravity * sthe
        self.LinearAcceleration[1] = invert_mass *  Fy - r * u + p * w + self.Gravity * sphi * cthe
        self.LinearAcceleration[2] = invert_mass *  Fz + q * u - p * v - self.Gravity * cphi * cthe

        # change of angular velocity:

        self.AngularAcceleration[0] = 1/Ixx * (L + (Izz - Iyy) * q * r )
        self.AngularAcceleration[1] = 1/Iyy * (M + (Ixx - Izz) * p * r )
        self.AngularAcceleration[2] = 1/Izz * (N + (Iyy - Ixx) * p * q )

        # change of rotation:

        self.AngularVelocity[0] = p + (q * sphi + r * cphi) * tanthe
        self.AngularVelocity[1] = q * cphi - r * sphi
        self.AngularVelocity[2] = (q * sphi + r * cphi) * secthe

        Tr_ZYXe = np.array([[cthe*cpsi,-cphi*spsi + sphi*sthe*cpsi, sphi*spsi + cphi*sthe*cpsi],\
                            [cthe*spsi, cphi*cpsi + sphi*sthe*spsi,-sphi*cpsi + cphi*sthe*spsi],\
                            [-sthe    , sphi*cthe                 , cphi*cthe                 ]])

        vect = np.array([u,
                         v,
                         w])

        self.LinearVelocity = np.matmul(Tr_ZYXe,vect.T)

        xdot[:3] =  self.LinearAcceleration
        xdot[3:6] = self.AngularAcceleration
        xdot[6:9] = self.AngularVelocity
        xdot[9:] = self.LinearVelocity

        return xdot

    def DetectCollision(self,CollisionSource = [0.0,0.0,0.0] ):

        #if np.abs(self.Position - CollisionSource) <= self.Radius

        #self.Orientation[1] += np.deg2rad(random.randint(-180,180))
        return


    def BoundaryState(self):

        BoundaryX = 0.0
        BoundaryY = 0.0
        BoundaryZ = 0.0

        return

    def StateUpdate(self,*args):

        x = np.array([])

        for A in [self.LinearVelocity , self.AngularVelocity , self.Orientation, self.Position]:

            x = np.concatenate((x[:np.size(x)],A,x[np.size(x):]))

        k = self.dt
        t = self.TimeElapsed

        function = self.RateOfChange
        # K1 = function(t,x,*args)
        # K2 = function(t + 0.5*k,x + k * 0.5 * K1,*args)
        # K3 = function(t + 0.5*k,x + k * 0.5 * K2,*args)
        # K4 = function(t + k, x + k * K3, *args)
        # F = x+(K1+2*K2+2*K3+K4)/6

        K1 = function(t,x,*args)
        K2 = function(t + 0.6666 * k , x + 0.6666 * K1 ,*args)

        F = x + 0.25 * (K1 + 3* K2)*k

        self.TimeElapsed += self.dt
        self.TimeVector.append(self.TimeElapsed)
        self.Mass -= self.dm


        #F = self.BoundaryState()

        self.LinearVelocity = F[:3]
        self.AngularVelocity = F[3:6]
        self.Orientation = F[6:9]
        self.Position = F[9:]


        return #F

    def Living(self):

        if self.isAlive != False:
            if self.TimeElapsed > float(self.lifetime):
                self.isAlive = False
                #print('It has died')


            #self.isAlive = 0 # can be set outside too.

def RandomStart(margin):

    '''
    Generates a random int generated float value from range
    Used only for Rain() particles generation.
    '''

    return random.randint(-margin,margin) + random.randint(-100,100)*0.01

def RandomMass():
    return random.randint(0,1000)*0.0001 + 0.0001

n = 20
Height = 1


color1 = colors.to_rgba_array('xkcd:lightblue')

def Rain(n = 2,Height = 4.0):
    path1 = os.path.join(os.path.expanduser('~'),'Documents\\Python Scripts\\PhysicsEngineThings\\GifSet')
    '''
    Generate Rain effect of N-particles density per frame
    '''
    X = []

    FigNum = 1

    fig = plt.figure(FigNum)
    ax = fig.add_subplot(111)

    rain_drops = np.zeros(n, dtype=[('position', float, (2,)),
                                    ('size',     float),
                                    ('color',    float, (4,))])

    ax.set_xlim(-2,2)
    ax.set_ylim(0,Height)


    rain_drops['size'] = [20. for size in range(n)]
    rain_drops['color'][:,:3] = [color1[0,:3] for color in range(n)]
    rain_drops['color'][:,3:] = [color1[0,3:] for color in range(n)]

    scttr = ax.scatter(rain_drops['position'][:,0],rain_drops['position'][:,1],
                        s=rain_drops['size'], lw=0.5,
                        c=rain_drops['color'],marker = 'd',facecolors='none')

    H = 0.0

    V = 0.0

    xx = np.linspace(-2,2,2)

    hh = np.array([H,H])

    ln, = ax.plot(0,0,c='xkcd:blue',lw=0.75)
    timing = 0
    Imax = 0

    X = []#np.zeros(n)

    while plt.fignum_exists(FigNum):

        if len(X)<n : #populate list of instances
            X.append(particle(position = [0.0,0.0,Height],lin_vel = [np.random.uniform(-2,2),0.0,0.0],Mass = 0.01 ))
            #continue
        i = 0

        for instance in X:
            i = 0 if i >= n-1 else i

            instance.Living()
            if instance.Position[2] <= H :

                instance.isAlive = False
                V += instance.Mass
                H  = V/4.0

            if instance.isAlive != True:
                X.pop(i) # this should take out a drop that died, i'm not sure
                continue # we should get out of loop for this methink
            instance.StateUpdate()

            rain_drops['position'][i] = instance.Position[0::2]
            rain_drops['size'][i] += instance.dt
            rain_drops['color'][i,3:] = 1.0 - instance.TimeElapsed/instance.lifetime if instance.TimeElapsed < instance.lifetime else 0.0
            i += 1

        h = np.array([H,H])
        ax.lines.remove(ax.lines[0])
        ln, = ax.plot(xx,h,c='xkcd:blue',lw=0.75)


        ax.fill_between(x=[-2,2],y1=[0,0],y2=[H,H],color='xkcd:lightblue', alpha=0.3)
        scttr.set_color(rain_drops['color'])
        scttr.set_sizes(rain_drops['size'])
        scttr.set_offsets(rain_drops['position'])

        #print(f'Fluid Column is: {H} m high')


        fig.canvas.draw()

        plt.pause(0.1)

    plt.show()
    return

Rain(50,3.0)




    # fig2.savefig(f'{path1}\\frame[{i}].png', facecolor='w', edgecolor='w',
    #         orientation='portrait', format='png',
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)


