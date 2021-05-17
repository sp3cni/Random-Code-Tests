import numpy as np

import Atmosphere

from wrapperScript import wrapper

from lxml import etree

#k = 1.2071744 # Cp/Cv dla mieszanki paliwowej
#Mol = 0.0185496345855 # masa molowa składników mieszanki Mol/kg
#T_chamber = 3300 + 273.15 # temperatura spalania swobodnego w K
#P_chamber = 20408000 # ciśnienie komory spalania w Pa


def numpy_strip_log(function):

    return np.exp(function)


def engine(Q,z):
# zdefiniujmy zachowanie silnika wzgledem otoczenia,
# istotnym parametrem bedzie wysokosc na jakiej sie znajdujemy
# przeplyw masowy
# reszta moze byc obliczona

# dane:
# T_chamber wynika z temperatury swobodnej reakcji
# P_chamber jest ciśnieniem w jakim zachodzi reakcja (ustalamy zewnętrznie)
# Q to przepływ masowy/ojętościowy mieszanki paliwowej
# ------------------------------------------------------------------------------
    P_immediate = Atmosphere.Model(z,'USstandard')
    #P_immediate = P_immediate[0]
    P_0 = Atmosphere.Model(0,'USstandard')
# ------------------------------------------------------------------------------
    R = 8.314462
    k = 1.2071744
    M = 0.0185496345855
    T_chamber = 3300 + 273.15
    P_chamber = 20408000
    #engeq = np.zeros(1)
    # optymalizujemy w taki sposób, że prędkość w zwężce osiąga Mach 1
    # ciśnienie zwężki
    P_throat = P_chamber * np.power((1 + ((k - 1) / 2)),-(k / (k-1)))
    # temperatura zwężki
    T_throat = T_chamber / (1+ ((k-1)/2))
    # powierzchnia zwężki
    A_throat = Q/P_throat * np.sqrt((R * T_throat) / (M * k))
# ------------------------------------------------------------------------------
    Mach_2 = (2 / (k-1)) * (np.power((P_chamber / P_0),(k-1) / k) - 1)
    A_exit = (A_throat / (np.sqrt(Mach_2))) * \
    np.power((1 + (((k-1) / 2) * Mach_2))/((k+1)/2),(k+1)/(2 * k - 2))
# ------------------------------------------------------------------------------
    P_total = P_chamber + 0 # zalozmy brak strat ze wzgledu na ksztalt
    T_total = T_chamber + 0 # zalozmy brak strat ze wzgledu na ubytek ciepla
# ------------------------------------------------------------------------------
    T_exit = T_total / ((1 + ((k-1)/2) * Mach_2))
    P_exit = P_total / np.power(((1 + ((k-1)/2) * Mach_2)),k / (k - 1))
# ------------------------------------------------------------------------------
    dM = A_throat * P_total / np.sqrt(T_total) * np.sqrt(k/R) * np.power((k+1)/2,\
    -(k+1)/(2*(k-1)))
    V_exit = np.sqrt(Mach_2) * np.sqrt(k * R * T_exit)
    ThrustEq = dM * V_exit + (P_exit - P_immediate) * A_exit
    #engeq[0] = ThrustEq # Ciąg silnika

    return V_exit,ThrustEq

n = 100

vel_exit = np.zeros(n)
Thrust = np.zeros(n)


ln = np.log

wet_mass = 1650 # [kg]
dry_mass = 650  # [kg]




mass_ratio = wet_mass/dry_mass

dm = 0.5 # [kg/s]

h = 0 # [m, above sea level]

vel_exit, Thrust = engine(dm,h)

g0 = 9.80655

delta_V = vel_exit * ln(mass_ratio)

Isp = delta_V/(g0 *mass_ratio)

C = Isp * g0

burn_time = wet_mass/dm * (1 - 1/np.exp(delta_V/C))

Par_names = ['dry_mass','wet_mass','mass_ratio','mass_flow','Thrust','mass_exit_velocity','DeltaV','Isp','C','Total_burn_time']
Par_values = [dry_mass,wet_mass,mass_ratio,dm,Thrust,vel_exit,delta_V,Isp,C,burn_time]


Rocket = {i:j for i,j in zip(Par_names,Par_values)}

# Problem we are set up with:
#
#
# We want a rocket of certain characterisitic
#
# Be it dry mass, engine performance and delta V to slove for necesary wet mass
# or some other instance with three known and one unknown parameter
# It is also possible to find other known derived things like:
#
# Total burn time (or of certain manouver,
# Specific Impulse
#






