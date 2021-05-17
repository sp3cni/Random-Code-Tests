import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random

class Particle():

    def __init__(self,
                 position = np.zeros(4),
                 velocity = np.zeros(3),
                 mass = 1.0,
                 size = 0.1):

        self.position = np.zeros(1,dtype=[('x',float),('y',float),('z',float),
                                          #('phi',float), # rotation of Y on Z
                                          #('the',float), # rotation of X on Z
                                          ('psi',float)]) # rotation of X on Y
        self.velocity = np.zeros(1,dtype=[('u',float),
                                          #('v',float),
                                          ('w',float),
                                          #('p',float),
                                          #('q',float),
                                          ('r',float)])

        self.size = size
        self.cBoxRadius = 1.0
        self.cCheckRange = 1.5
        self.cBoxShape = 8 # how many sides does it have
        self.mass = mass
        self.inv_mass = 1/mass
        self.inertia = self.mass * self.size**2
        self.inv_inertia = 1/self.inertia

        self.Adrag = 0.0
        self.Gdrag = 0.0

        self.g = 9.80665 # having our own gravitational acceleration...

    # this should be outside
    def MotionEquations(self,x0,Force = 0.0,Moment = 0.0):

        dotx = np.zeros(7)
        u,w,r,psi = x0[0:4]
        cpsi = np.cos(psi)  ; spsi = np.sin(psi)
        #velocity:
        dotx[0] = Force * self.inv_mass      # du
        dotx[1] = -self.g                    # dw
        dotx[2] = Moment * self.inv_inertia  # dr
        #rotation:
        dotx[3] = r                          # dpsi
        #position:
        dotx[4] = cpsi * u                   # dy
        dotx[5] = spsi * u                   # dy
        dotx[6] = w                          # dz

        return dotx # do we need?

    def ModInternals(self,delta_mass,delta_size,dt):

        self.size += delta_size  * dt
        self.mass += delta_mass  * dt
        self.inertia = self.mass * self.size**2
        self.inv_mass = 1/self.mass
        self.inv_inertia = 1/self.inertia
        self.Adrag = 0.0
        self.Gdrag = 0.0


    # this perhaps outside too?
    # we don't need that many instances of the same thing, but hey... it's python,
    # with some effort we can make instances compute concurrently so...

    def StateIntegration(self,dt,*args):
        f = self.MotionEquations # yeah, refer to your own equations even though they are general...

        Xn =np.array([self.velocity['u'][0],
                      self.velocity['w'][0],
                      self.velocity['r'][0],
                      self.position['psi'][0],
                      self.position['x'][0],
                      self.position['y'][0],
                      self.position['z'][0]])

        K1 = f(Xn,*args)
        K2 = f(Xn + 0.66667 * K1,*args)
        Xnew = Xn + 0.25 * (K1 + 3 * K2) * dt

        self.velocity['u'] = Xnew[0] if Xnew[0] <= 100.0 else 100.0

        self.velocity['r'] = Xnew[2] if Xnew[2] <= 25.0 else 25.0
        self.position['psi'] = Xnew[3]
        self.position['x'] = Xnew[4]
        self.position['y'] = Xnew[5]

        if np.abs(Xnew[5]) > 4:
            self.position['y'] = Xnew[5]
            self.velocity['r'] = -np.sign(Xnew[5]) * np.random.uniform(0,20)
        else:



        if Xnew[6] > 0.0 :
            self.position['z'] = Xnew[6]
            self.velocity['w'] = Xnew[1]
        else:
            self.position['z'] = Xnew[6]
            self.velocity['w'] = -Xnew[1]

    def CollisionCheckSimple(self,collider):
        # First: Distance between our potential colliding bodies
        Sx = self.position['x'][0]
        Sy = self.position['y'][0]
        Sz = self.position['z'][0]
        Cx = collider.position['x'][0]
        Cy = collider.position['y'][0]
        Cz = collider.position['z'][0]

        DistanceCollision = np.sqrt((Sx - Cx )**2 + (Sy - Cy)**2 + (Sz - Cz)**2)

        if  not dDistanceCollision > (self.size + collider.size):
            deg = np.deg2rad(30)

            self.position['psi'] = np.random.uniform(-deg,deg)
            self.velocity['u']   = -self.velocity['u'] * np.sin(psi)




FigNum = 'Figure'

fig = plt.figure(num=FigNum,figsize = [4,4])
ax = fig.add_axes([0, 0, 1, 1],projection='3d')


#scattr = ax.scatter3D(xs=0,ys=0,zs=-1)

n = 2 # number of particles

# BallPit = [Particle(size = 1.0,mass=1.0) for n in range(n)] # this is namespace of our created particles,


p = Particle(size = 1.0,mass=1.0)

p.position['x'] = 0.0
p.position['y'] = 0.0
p.position['z'] = 0.0

pStateArray = np.zeros(1,dtype=[('position',float,(3,)),('size',float,(1,))])

pStateArray['position'] = [p.position['x'].item(),p.position['y'].item(),p.position['z'].item()]
pStateArray['size'] = p.size

#sctr = ax.scatter3D(p.position['x'].item(),p.position['y'].item(),p.position['z'].item(),s=p.size)

#plt.show()
path1 = os.path.join(os.path.expanduser('~'),'Documents\\Python Scripts\\PhysicsEngineThings\\GifSet')
i=0

p.velocity['u'] = np.random.uniform(1.0,5.0)
pp = 0
while plt.fignum_exists(FigNum):

    ax.cla()
    ax.scatter(p.position['x'].item(),p.position['y'].item(),s=p.size)

    ax.set_xlim(-1, 1), ax.set_xticks([])
    ax.set_ylim(-1, 1), ax.set_yticks([])
    ax.set_zlim( 0, 10), ax.set_zticks([])

    fig.canvas.draw()

    # end draw state n

    plt.pause(0.01)

    fig.savefig(f'{path1}\\frame[{i}].png', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


    # state n + 1
    i+=1
    #@run_once

    if pp > random.randint(0,10):
        p.velocity['r'] = np.random.uniform(-np.pi,np.pi)
        pp = 0
    else: pp += 1
    p.StateIntegration(0.01)






plt.show()


