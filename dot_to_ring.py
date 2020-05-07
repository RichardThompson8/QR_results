# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:11:00 2020

@author: Dick
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:22:53 2020

@author: tr1116
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp, simps, romberg
import matplotlib.pyplot as plt 
from tqdm import trange
from scipy.optimize import fsolve
from statistics import stdev, mean, median
from mpl_toolkits.mplot3d import Axes3D

ti = 0.0 #initial time
tdivs = 500 #number of time mesh nodes

ra = 0.0 #left boundary
rb = 4.0 #right boundary
rdivs = 200 #number of position mesh nodes
rmesh = np.linspace(ra, rb, rdivs)

rDi = 1.0 #initial dot radius
rT = 0.65 #top radius
theta = 0.96 #55 degrees. 
a = 0.245 #proportionality costant for the rate 
F = 1.0 #Flux
Ea = 7000 #Activation energy
R = 8.3 #Gas constant
T = 833 #Temperature


b = 2.25 #Guassian factor
w = 0.08 #Gaussian width
h_c = 0.08334 #Characteristic height
t_crossover = 0.15 * 2 #Guess for when max(h(r, t_crossover)) = h_c
tf_precise = (rDi - rT) * (np.tan(theta))/(a * F * np.exp(-Ea / (R * T))) #Time when rD = rT
tf = tf_precise  #Final time

m = 0.16 #Initial guess for gradient of r_0(t) at later times


iterations = 21

r0_guesses = []

plt.rcParams['font.size'] = 30


tmesh = np.linspace(ti, tf, tdivs)


def rD_value_at_t(t): #Function for dot radius with time
    if t <= tf_precise:
        return rDi - t * (a * F * np.exp(-Ea / (R * T)))/(np.tan(theta))
    else:
        return 0.0 
    
plt.figure(1) #plot of QD edge radius vs position at a fixed time.
plt.xlabel(r'$t$') 
plt.xlabel(r'$r_D(t)$') 
plt.plot(tmesh, [rD_value_at_t(i) for i in tmesh])

def hD(r, t):
    if r <= rT and r >= ra and t < tf and t >= ti:
        return np.tan(theta) * (rD_value_at_t(t) - rT)
    elif r <= rD_value_at_t(t) and r > rT:
        return np.tan(theta) * (rD_value_at_t(t) - r)
    else:
        return 0.0
    

#__________________________________________________________________________________


#----------------------------------------------------------------------------
 
def r0_guess_a(t): #Guess for the value of r0, the position of maximum growth
    rD_crossover = 1.2 * rD_value_at_t(t_crossover)
    if t < t_crossover:
        return 1.2 * rD_value_at_t(t)
    else:
        return rD_crossover + m * (t - t_crossover)

r0_guesses = []

r0_at_all_t_repeated = [[r0_guess_a(i) for i in tmesh]]
h_vs_r_at_tf_repeated = []
residuals_repeated = []
ave_residual_repeated = []

for r in trange(iterations):
    print(r)

#    r0_prev_guess = 
    
    def r0_guess(t):
       return r0_guess_a(t)

        
    h_vs_t_at_all_r = []
    
    def append_h_vs_t_at_r(r):
        def dhdt(h, t, r):
            if t < tf and r > ra:
                return b * a * F * np.exp(-Ea / (R * T)) * (rD_value_at_t(t) ** 2) * np.exp(-((r - r0_guess(t))**2)/(2*w**2)) / r# * np.exp(-h) # / (r ** 2)
                
            else:
                return 0.0
        
        hi = 0.0

        h_vs_t_at_r = odeint(dhdt, hi, tmesh, args = (r,))
        h_vs_t_at_all_r.append(list(h_vs_t_at_r))
    
    
    for i in range(len(rmesh)):
        append_h_vs_t_at_r(rmesh[i])
        
    def h_value_at_t_and_r(t, r): #Interpolation
        t_index = int(t*(tdivs - 1)/(tf - ti))
        r_index = int(r*(rdivs - 1)/(rb - ra))
        if t >= 0 and t < tf and r < rb and r >= ra:
            point_1 = np.array([tmesh[t_index], rmesh[r_index], h_vs_t_at_all_r[r_index][t_index][0]])
            point_2 = np.array([tmesh[t_index], rmesh[r_index + 1], h_vs_t_at_all_r[r_index + 1][t_index][0]])
            point_3 = np.array([tmesh[t_index + 1], rmesh[r_index], h_vs_t_at_all_r[r_index][t_index + 1][0]])
            normal = np.cross(point_2 - point_1, point_3 - point_1)
            value = (1/normal[2])*(normal[0]*tmesh[t_index] + normal[1]*rmesh[r_index] + normal[2]*h_vs_t_at_all_r[r_index][t_index][0] - normal[0]*t - normal[1]*r)    
            return value
        else:
            return 0.0
        
    
    h_vs_r_at_tf_repeated.append([h_value_at_t_and_r(tmesh[-2], i) for i in rmesh])
    
    r0_at_all_t = []
    hc_reached = 0
    
    def append_r0_at_t(t):
        t_index = int(t*(tdivs - 1)/(tf - ti))
        r_index_by_r = int((rdivs - 1)/(rb - ra))
        h_vs_r_at_t = [i[t_index][0] for i in h_vs_t_at_all_r]
        h0 = h_c
        
        if max(h_vs_r_at_t) < h0:
            r0 = 1.2 * rD_value_at_t(t)
        else:
            r0_index = int(r_index_by_r * 1.2 * rDi) + np.argmin([abs(h_vs_r_at_t[i] - h0) for i in range(int(r_index_by_r * 1.2 * rDi), len(h_vs_r_at_t))])
            r0_guess = rmesh[r0_index]
    
            def func_to_solve(r):
                return h_value_at_t_and_r(t, r) - h0    
            r0 = fsolve(func_to_solve, r0_guess, xtol = 1.49e-8)[0] 
            
        r0_at_all_t.append(r0)
        
    for i in range(len(tmesh)):
        append_r0_at_t(tmesh[i])
    
    r0_at_all_t_repeated.append(r0_at_all_t)
    residuals_list = [r0_at_all_t_repeated[-1][i] - r0_at_all_t_repeated[-2][i] for i in range(len(r0_at_all_t_repeated[-1]))]
    
    ave_residual = mean([abs(i) for i in residuals_list])
    ave_residual_repeated.append(ave_residual)
    
    print('ave_residual = ', ave_residual)
    
    def r0_value_at_t(t):
        if t < tf:
           t_index = int(t*(tdivs - 1)/(tf - ti))
           delta_t = tf/(tdivs - 1)
           return r0_at_all_t[t_index] + (t - tmesh[t_index])*(r0_at_all_t[t_index + 1] - r0_at_all_t[t_index])/delta_t
        else:
            return 0.0
    
        
    def new_r0_guess_a(t):
        return r0_value_at_t(t)
    
    r0_guess_a = new_r0_guess_a
    
#-----------------------------------------------------------------------------


plt.figure(82)
plt.xlabel(r'Iteration')
plt.ylabel(r'$\bar{l}$')
plt.xticks(list(range(0, 20, 2)))
plt.plot(range(iterations), ave_residual_repeated, 'rx', linestyle = 'None', markeredgewidth = 3, markersize=20)
plt.xticks(list(range(0,21,2)))
plt.yscale('log')


plt.figure(85)
plt.xlabel(r'$r / r_D(0)$')
plt.ylabel(r'$h^i(r / r_D(0) , t_f) / h_c$')
for i in [0, 2, 4, 6, iterations - 1]:
    label = r'h^i(r,t), i = ' + str(i)
    plt.plot(list(-rmesh)[::-1][:-1] + list(rmesh), [j/h_c for j in h_vs_r_at_tf_repeated[i]][::-1][:-1] + [j/h_c for j in h_vs_r_at_tf_repeated[i]], lw = 2.5, label = 'i = ' + str(i))
plt.ylim(0, 1.2 * max([j/h_c for j in h_vs_r_at_tf_repeated[i]]))
plt.legend(loc=9)

plt.figure(86)
plt.xlabel(r'$t / t_f$')
plt.ylabel(r'$r_0^i(t) / r_D(0)$')
for i in [0, 2, 4, 6, iterations - 1]:
    label = r'$r_0^i(t)$, i = ' + str(i)
    plt.plot(tmesh[:-1] / tf, r0_at_all_t_repeated[i][:-1], lw = 2.5, label = 'i = ' + str(i))
plt.legend()

hardcode_normalisation = 1.0
QD_end_time = tf - 0.01

h_vs_r_tf = [h_value_at_t_and_r(QD_end_time, i) for i in rmesh]
h_vs_r_tf_times_r = [h_value_at_t_and_r(QD_end_time, i) * i for i in rmesh]

V_QR = 2 * np.pi * simps(h_vs_r_tf_times_r, rmesh)
print('V_QR = ', V_QR)


y = list(rmesh)
plt.figure(6)
plt.xlabel(r'$r / r_D(0)$')
plt.ylabel(r'$h / h_c$')

plt.plot([0, rT, rD_value_at_t(ti)], [hD(0, ti) / h_c, hD(rT, ti) / h_c, 0], 'c--')
plt.plot([0, rT, rD_value_at_t(0.1*QD_end_time)], [hD(0, 0.1*QD_end_time) / h_c, hD(rT, 0.1*QD_end_time) / h_c, 0], 'm--')
plt.plot([0, rT, rD_value_at_t(0.25*QD_end_time)], [hD(0, 0.25*QD_end_time) / h_c, hD(rT, 0.25*QD_end_time) / h_c, 0], 'r--')
plt.plot([0, rT, rD_value_at_t(0.5*QD_end_time)], [hD(0, 0.5*QD_end_time) / h_c, hD(rT, 0.5*QD_end_time) / h_c, 0], 'g--')
plt.plot([0, rT, rD_value_at_t(0.75*QD_end_time)], [hD(0, 0.75*QD_end_time) / h_c, hD(rT, 0.75*QD_end_time) / h_c, 0], 'b--')
plt.plot([0, rT, rD_value_at_t(QD_end_time)], [0, 0, 0], 'k--')

plt.plot(rmesh, [hardcode_normalisation*h_value_at_t_and_r(ti, i) / h_c for i in rmesh], 'c-', label = r'$t = t_i$')
plt.plot(rmesh, [hardcode_normalisation*h_value_at_t_and_r(0.1*QD_end_time, i) / h_c for i in rmesh], 'm-', label = r'$t = \frac{1}{10}t_f$')
plt.plot(rmesh, [hardcode_normalisation*h_value_at_t_and_r(0.25*QD_end_time, i) / h_c for i in rmesh], 'r-', label = r'$t = \frac{1}{4}t_f$')
plt.plot(rmesh, [hardcode_normalisation*h_value_at_t_and_r(0.5*QD_end_time, i) / h_c for i in rmesh], 'g-', label = r'$t = \frac{1}{2}t_f$')
plt.plot(rmesh, [hardcode_normalisation*h_value_at_t_and_r(0.75*QD_end_time, i) / h_c for i in rmesh], 'b-', label = r'$t = \frac{3}{4}t_f$')
plt.plot(rmesh, [hardcode_normalisation*h_value_at_t_and_r(QD_end_time, i) / h_c for i in rmesh], 'k-', label = r'$t = t_f$')

plt.legend()

plt.rcParams['lines.linewidth'] = 3.5

plt.figure(7)
plt.xlabel(r'$r / r_D(0)$')
plt.ylabel(r'$h / h_c$')

plt.plot([-rD_value_at_t(ti), -rT, 0, rT, rD_value_at_t(ti)], [hD(0, ti) / h_c, hD(rT, ti) / h_c, 0][::-1][:-1] + [hD(0, ti) / h_c, hD(rT, ti) / h_c, 0], 'c--')
plt.plot([-rD_value_at_t(0.1*QD_end_time), -rT, 0, rT, rD_value_at_t(0.1*QD_end_time)], [hD(0, 0.1*QD_end_time) / h_c, hD(rT, 0.1*QD_end_time) / h_c, 0][::-1][:-1] + [hD(0, 0.1*QD_end_time) / h_c, hD(rT, 0.1*QD_end_time) / h_c, 0], 'm--')
plt.plot([-rD_value_at_t((1/3)*QD_end_time), -rT, 0, rT, rD_value_at_t((1/3)*QD_end_time)], [hD(0, 0.25*QD_end_time) / h_c, hD(rT, (1/3)*QD_end_time) / h_c, 0][::-1][:-1] + [hD(0, (1/3)*QD_end_time) / h_c, hD(rT, (1/3)*QD_end_time) / h_c, 0], 'r--')
plt.plot([-rD_value_at_t(0.5*QD_end_time), -rT, 0, rT, rD_value_at_t(0.5*QD_end_time)], [hD(0, 0.5*QD_end_time) / h_c, hD(rT, 0.5*QD_end_time) / h_c, 0][::-1][:-1] + [hD(0, 0.5*QD_end_time) / h_c, hD(rT, 0.5*QD_end_time) / h_c, 0], 'g--')
plt.plot([-rD_value_at_t(0.75*QD_end_time), -rT, 0, rT, rD_value_at_t(0.75*QD_end_time)], [hD(0, 0.75*QD_end_time) / h_c, hD(rT, 0.75*QD_end_time) / h_c, 0][::-1][:-1] + [hD(0, 0.75*QD_end_time) / h_c, hD(rT, 0.75*QD_end_time) / h_c, 0], 'b--')
plt.plot([-rD_value_at_t(QD_end_time), -rT, 0, rT, rD_value_at_t(QD_end_time)], [0, 0, 0, 0, 0], 'k--')

rmesh_tot = list(-rmesh[::-1]) + list(rmesh[1:])

plt.plot(rmesh_tot, [hardcode_normalisation*h_value_at_t_and_r(ti, i) / h_c for i in rmesh][::-1][:-1] + [hardcode_normalisation*h_value_at_t_and_r(ti, i) / h_c for i in rmesh], 'c-', label = r'$t = 0$')
plt.plot(rmesh_tot, [hardcode_normalisation*h_value_at_t_and_r(0.1*QD_end_time, i) / h_c for i in rmesh][::-1][:-1] + [hardcode_normalisation*h_value_at_t_and_r(0.1*QD_end_time, i) / h_c for i in rmesh], 'm-', label = r'$t = \frac{1}{10}t_f$')
plt.plot(rmesh_tot, [hardcode_normalisation*h_value_at_t_and_r((1/3)*QD_end_time, i) / h_c for i in rmesh][::-1][:-1] + [hardcode_normalisation*h_value_at_t_and_r((1/3)*QD_end_time, i) / h_c for i in rmesh], 'r-', label = r'$t = \frac{1}{3}t_f$')
plt.plot(rmesh_tot, [hardcode_normalisation*h_value_at_t_and_r(0.5*QD_end_time, i) / h_c for i in rmesh][::-1][:-1] + [hardcode_normalisation*h_value_at_t_and_r(0.5*QD_end_time, i) / h_c for i in rmesh], 'g-', label = r'$t = \frac{1}{2}t_f$')
plt.plot(rmesh_tot, [hardcode_normalisation*h_value_at_t_and_r(0.75*QD_end_time, i) / h_c for i in rmesh][::-1][:-1] + [hardcode_normalisation*h_value_at_t_and_r(0.75*QD_end_time, i) / h_c for i in rmesh], 'b-', label = r'$t = \frac{3}{4}t_f$')
plt.plot(rmesh_tot, [hardcode_normalisation*h_value_at_t_and_r(QD_end_time, i) / h_c for i in rmesh][::-1][:-1] + [hardcode_normalisation*h_value_at_t_and_r(QD_end_time, i) / h_c for i in rmesh], 'k-', label = r'$t = t_f$')
plt.ylim(0, 6.8)

plt.legend()







































