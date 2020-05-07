# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_bvp, odeint, solve_ivp, simps, romb, romberg, quad
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange


k = 100 #Reaction rate constant.
DG = 1.16 #Diffusion coefficient of Gallium.
DA = 0 #Diffusion coefficient of Arsenic.
F0 = 0.04 #Deposition flux of Arsenic.
F = 2 * F0 #Resdience time of Arsenic.
tau = 1/F0 #Resdience time of Arsenic.
GaAs_normalisation = 500
y1a = 1.0 #Surface adatom concentration of Gallium at r = rD.
y3a = 0.0 #Surface adatom concentration of Arsenic at r = rD.
CAs_c = F * tau #Surface adatom concentration of Arsenic for large r.
kappa_0 = k * y1a * CAs_c / 500 #Normalisation constant

rDi = 1.0#Initial droplet radius (first r-mesh point at t = 0).
rb = 10.0 #Last r-mesh point.
rdivs = 513 #484, 0.003, 20000, 165 #Number of r-mesh points.#300 when tol=0.01
tolerance = 1e-7 #Accuracy to which CGa and CAs are found
max_mesh_nodes = 4e7 #Number of mesh nodes algorithm for bvp solver uses to find concentration profiles

tf = 0.1473 #Final time
rD_cutoff = 0.003 #point at which radius stops decreasing to avoid small number errors
tdivs = 513 #Number of t-mesh points.
V_Ga = 1.0 #Ga primitive cell volume
V_GaAs = 1.0 #GaAs primitive cell volume
beta_theta = 1.0 #Contact angle constant
A = 2.57142857 #Constant coefficient of first term in drD/dt expression. 
B = 1.5 #Constant coefficient of second term in drD/dt expression.
b = 1.0 * rDi #Constant coefficient of Gaussian expression.
w = 0.15 * rDi #Width of Gaussian.
hi = 0.0 #Initial height of inner/outer ring (at all points).


plt.rcParams['font.size'] = 30

def dydr(r, y): 
    return np.vstack((y[1],
                      -y[1]/r + k * y[0] * F/(DG * (k * y[0] + 1/tau))))
    
def y_bc(ya, yb):
    return np.array([ya[0] - y1a, yb[0]])

rmeshi = np.linspace(rDi, rb, rdivs)
rmeshf = np.linspace(0, rb - rDi, rdivs)
y_at_ti_guess = np.zeros((2, rmeshi.size))
calc_y_at_ti_vs_r = solve_bvp(dydr, y_bc, rmeshi, y_at_ti_guess, tol = tolerance, max_nodes = max_mesh_nodes)
calc_y_at_ti_vs_r.status
    
def plot_y_at_ti_vs_r():
    plt.figure(1, figsize = (11,11))
    plt.plot(rmeshi, calc_y_at_ti_vs_r.sol(rmeshi)[0], 'r--', lw = '3', label = r'$C_{Ga}/C^0_{Ga}$')
    y3 = F / (k * calc_y_at_ti_vs_r.sol(rmeshi)[0] + 1/tau)
    plt.plot(rmeshi, y3, 'g', linestyle = (0, (1, 1)), lw = '3', label = r'$C_{As}/C^c_{As}$')
    plt.plot(rmeshi, k * calc_y_at_ti_vs_r.sol(rmeshi)[0] * y3 / kappa_0, color = 'k', linewidth = 5, label = r'$k_rC_{Ga}C_{As} / \kappa_0$')
    plt.plot([rmeshi[0], rmeshi[0]], [0, y1a], color = 'k', lw = '2', linestyle = (0, (1, 10)))
    plt.xlabel(r'$r/r_D(0)$')
    plt.legend()

plot_y_at_ti_vs_r() #This produces a graph of the concentration profiles.
  
"""
Droplet radius
"""

def y2_value_at_rD(rD): 
    def dydr(r, y): 
        return np.vstack((y[1],
                          -y[1]/r + k * y[0] * F/(DG * (k * y[0] + 1/tau))))
        
    def y_bc(ya, yb):
        return np.array([ya[0] - y1a, yb[0]])
    
    rmesh = np.linspace(rD, rb - (rDi - rD), rdivs)
    y_guess = np.zeros((2, rmesh.size))
    return solve_bvp(dydr, y_bc, rmesh, y_guess, tol = tolerance, max_nodes = max_mesh_nodes).sol(rD)[1]
   
def drDdt(t, rD):
    if rD > rD_cutoff:
        return (1 / rD) * (V_Ga / beta_theta) * (DG * A * y2_value_at_rD(rD[0]) - B * b * w / V_GaAs) #(Sould include DG dependence later because this is one of the parameters varied in the paper.)
    else:
        return 0.0
    
tmesh = np.linspace(0, tf, tdivs)
calc_rD_vs_t = solve_ivp(drDdt, [0.0, tf], [rDi], dense_output = True, atol = 1e-6)
rD_vs_t = calc_rD_vs_t.sol(tmesh)

y2a_vs_t = [y2_value_at_rD(i) for i in rD_vs_t[0]]

plt.figure(652, figsize = (11,11))
plt.xlabel(r'$t / t_f$')
plt.ylabel(r'$-\frac{dC_{Ga}(r_D, t)}{dr}$')
plt.yscale('log')
plt.plot(tmesh / tf, [-i for i in y2a_vs_t], color = 'r', lw = 4)



def plot_rD_vs_t():
        plt.figure(4, figsize = (11,11))
        plt.xlabel(r'$t / t_f$')
        plt.ylabel(r'$r_D(t) / r_D(0)$')
        x = np.linspace(0, tf, tdivs*500)
        x = tmesh
        plt.plot(x/tf, calc_rD_vs_t.sol(x)[0], color = 'r', lw = 4)

plot_rD_vs_t() #This produces a graph of droplet radius as a function of time.


def rD_value_at_t(t):
    if t <= tf:
        return calc_rD_vs_t.sol(t)[0]
    elif t > tf:
        return 0 

"""
Inner ring
"""

def dhIdt(t, r):
    if rD_value_at_t(t) > rD_cutoff:
        return b*np.exp(-((r - rD_value_at_t(t))**2)/(2*w**2))
    else:
       return 0.0

hIf_vs_r = []
for r in rmeshf:
    hOf_at_r = romberg(dhIdt, 0, tf, args = (r,))
    hIf_vs_r.append(hOf_at_r)
    
hI1_vs_r = []
for r in rmeshf:
    hOf_at_r = romberg(dhIdt, 0, tf*(1/4), args = (r,))
    hI1_vs_r.append(hOf_at_r)
    
hI2_vs_r = []
for r in rmeshf:
    hOf_at_r = romberg(dhIdt, 0, tf*(2/4), args = (r,))
    hI2_vs_r.append(hOf_at_r)
    
hI3_vs_r = []
for r in rmeshf:
    hOf_at_r = romberg(dhIdt, 0, tf*(3/4), args = (r,))
    hI3_vs_r.append(hOf_at_r)
    
        
plt.figure(908, figsize = (11,11))
plt.plot(rmeshf, hI1_vs_r, color = 'r', label = r'$t = \frac{1}{4}t_f$')
plt.plot(rmeshf, hI2_vs_r, color = 'g', label = r'$t = \frac{1}{2}t_f$')
plt.plot(rmeshf, hI3_vs_r, color = 'b', label = r'$t = \frac{3}{4}t_f$')
plt.plot(rmeshf, hIf_vs_r, color = 'k', label = r'$t = t_f$')
plt.xlabel(r'$r/r_D(0)$')
plt.yticks([])
plt.legend()

"""
Outer ring
"""

y_vs_r_at_all_t = []

def Append_solution(rD):
    def dydr(r, y): 
        return np.vstack((y[1],
                          -y[1]/r + k * y[0] * F/(DG * (k * y[0] + 1/tau))))
        
    def y_bc(ya, yb):
        return np.array([ya[0] - y1a, yb[0]])
    
    rmesh = np.linspace(rD, rb - (rDi - rD), rdivs)
    y_guess = np.zeros((2, rmesh.size))
    calc_y_vs_r_at_t = solve_bvp(dydr, y_bc, rmesh, y_guess, tol = tolerance, max_nodes = max_mesh_nodes).sol
    y_vs_r_at_all_t.append(calc_y_vs_r_at_t)
    return None

for i in trange(len(rD_vs_t[0])):
    Append_solution(rD_vs_t[0][i])


y1y3_values_mesh_points = []
for t in range(len(tmesh)):
    y1y3_over_rmesh_fixed_t = []
    for r in rmeshf:
        y1 = y_vs_r_at_all_t[t](r)[0]
        y3 = F / (k * y_vs_r_at_all_t[t](r)[0] + 1/tau)
        y1y3 = k * V_GaAs * y1 * y3
        if y1y3 > 1e-17 and rD_value_at_t(tmesh[t]) > rD_cutoff:
            y1y3_over_rmesh_fixed_t.append(y1y3)
        else:
            y1y3_over_rmesh_fixed_t.append(0.0)
    y1y3_values_mesh_points.append(y1y3_over_rmesh_fixed_t)
        
plt.figure(903)
for i in range(len(tmesh)):
    plt.plot(rmeshf, y1y3_values_mesh_points[i])
    
y1y3_values_mesh_points_r_vs_t = []
for r in rmeshf:
    y1y3_over_rmesh_fixed_r = []
    for t in range(len(tmesh)):
        y1 = y_vs_r_at_all_t[t](r)[0]
        y3 = F / (k * y_vs_r_at_all_t[t](r)[0] + 1/tau)
        y1y3 = k * V_GaAs * y1 * y3
        if y1y3 > 0 and rD_value_at_t(tmesh[t]) > rD_cutoff: #1e-17:
            y1y3_over_rmesh_fixed_r.append(y1y3)
        else:
            y1y3_over_rmesh_fixed_r.append(0.0)
    y1y3_values_mesh_points_r_vs_t.append(y1y3_over_rmesh_fixed_r)
        
plt.figure(904)
for i in range(len(rmeshf)):
    plt.plot(tmesh, y1y3_values_mesh_points_r_vs_t[i])

delta_t = tf/(tdivs - 1)

hOf_vs_r = []
for i in range(len(rmeshf)):
    hOf_at_r = simps(y1y3_values_mesh_points_r_vs_t[i], tmesh)
    hOf_vs_r.append(hOf_at_r)
    
hO1_vs_r = []
for i in range(len(rmeshf)):
    hOf_at_r = simps(y1y3_values_mesh_points_r_vs_t[i][:int((1/4) * tdivs)], tmesh[:int((1/4) * tdivs)])
    hO1_vs_r.append(hOf_at_r)
    
hO2_vs_r = []
for i in range(len(rmeshf)):
    hOf_at_r = simps(y1y3_values_mesh_points_r_vs_t[i][:int((2/4) * tdivs)], tmesh[:int((2/4) * tdivs)])
    hO2_vs_r.append(hOf_at_r)
    
hO3_vs_r = []
for i in range(len(rmeshf)):
    hOf_at_r = simps(y1y3_values_mesh_points_r_vs_t[i][:int((3/4) * tdivs)], tmesh[:int((3/4) * tdivs)])
    hO3_vs_r.append(hOf_at_r)
        

plt.figure(905)
plt.plot(rmeshf, hO1_vs_r, color = 'r', label = r'$t = \frac{1}{4}t_f$')
plt.plot(rmeshf, hO2_vs_r, color = 'g', label = r'$t = \frac{1}{2}t_f$')
plt.plot(rmeshf, hO3_vs_r, color = 'b', label = r'$t = \frac{3}{4}t_f$')
plt.plot(rmeshf, hOf_vs_r, color = 'k', label = r'$t = t_f$')
plt.xlabel(r'$r/r_D(0)$')
plt.yticks([])
plt.legend()

normalise = 5

h1_vs_r = np.array(hI1_vs_r) + normalise * np.array(hO1_vs_r)
h2_vs_r = np.array(hI2_vs_r) + normalise * np.array(hO2_vs_r)
h3_vs_r = np.array(hI3_vs_r) + normalise * np.array(hO3_vs_r)
hf_vs_r = np.array(hIf_vs_r) + normalise * np.array(hOf_vs_r)

h1_vs_r_T = np.insert(h1_vs_r, 0, h1_vs_r[::-1][:-1])
h2_vs_r_T = np.insert(h2_vs_r, 0, h2_vs_r[::-1][:-1])
h3_vs_r_T = np.insert(h3_vs_r, 0, h3_vs_r[::-1][:-1])
hf_vs_r_T = np.insert(hf_vs_r, 0, hf_vs_r[::-1][:-1])

rmeshf_T = np.linspace(-rmeshf[-1], rmeshf[-1], rdivs * 2 - 1)

plt.figure(906)
plt.plot(rmeshf_T, h1_vs_r_T, color = 'r', label = r'$t = \frac{1}{4}t_f$')
plt.plot(rmeshf_T, h2_vs_r_T, color = 'g', label = r'$t = \frac{1}{2}t_f$')
plt.plot(rmeshf_T, h3_vs_r_T, color = 'b', label = r'$t = \frac{3}{4}t_f$')
plt.plot(rmeshf_T, hf_vs_r_T, color = 'k', label = r'$t = t_f$')
plt.xlabel(r'$r/r_D(0)$')
plt.yticks([])
plt.legend()

hI1_vs_r_T = np.insert(np.array(hI1_vs_r), 0, np.array(hI1_vs_r)[::-1][:-1])
hI2_vs_r_T = np.insert(np.array(hI2_vs_r), 0, np.array(hI2_vs_r)[::-1][:-1])
hI3_vs_r_T = np.insert(np.array(hI3_vs_r), 0, np.array(hI3_vs_r)[::-1][:-1])
hIf_vs_r_T = np.insert(np.array(hIf_vs_r), 0, np.array(hIf_vs_r)[::-1][:-1])

hO1_vs_r_T = normalise * np.insert(np.array(hO1_vs_r), 0, np.array(hO1_vs_r)[::-1][:-1])
hO2_vs_r_T = normalise * np.insert(np.array(hO2_vs_r), 0, np.array(hO2_vs_r)[::-1][:-1])
hO3_vs_r_T = normalise * np.insert(np.array(hO3_vs_r), 0, np.array(hO3_vs_r)[::-1][:-1])
hOf_vs_r_T = normalise * np.insert(np.array(hOf_vs_r), 0, np.array(hOf_vs_r)[::-1][:-1])

plt.figure(888, figsize = (11,11))
plt.plot(rmeshf_T, hf_vs_r_T, color = 'k', lw = 8, label = r'$h(r/r_D(0), t = t_f)$')
plt.plot(rmeshf_T, hIf_vs_r_T, color = 'g', lw = 5, label = r'$h_I(r/r_D(0), t = t_f)$')
plt.plot(rmeshf_T, hOf_vs_r_T, color = 'r', lw = 5, label = r'$h_O(r/r_D(0), t = t_f)$')
plt.xlabel(r'$r/r_D(0)$')
plt.yticks([])
#plt.legend()

plt.figure(889, figsize = (11,11))
plt.plot(rmeshf_T, hf_vs_r_T, color = 'k', lw = 8, label = r'$h(r/r_D(0), t = t_f)$')
plt.ylim(0, 1.7 * max(hf_vs_r_T))
plt.xlabel(r'$r/r_D(0)$')
plt.ylabel(r'$h(r/r_D(0))$')
plt.yticks([])

#Following piece of code for 3D plot adapted from code written by ImportanceOfBeingErnest on https://stackoverflow.com/questions/47333811/how-do-i-create-a-surface-plot-with-matplotlib-of-a-closed-loop-revolve-about-an

x_values = rmeshf
# input xy coordinates
x_y_coordinates = [[x_values[i], hf_vs_r[i]] for i in range(len(x_values))]
xy = np.array(x_y_coordinates)
# radial component is x values of input
r = xy[:,0]
# angular component is one revolution of 60 steps
phi = np.linspace(0, 2*np.pi, 60)
# create grid
R,Phi = np.meshgrid(r,phi)
# transform to cartesian coordinates
X = R*np.cos(Phi)
Y = R*np.sin(Phi)
# Z values are y values, repeated 60 times
Z = np.tile(xy[:,1],len(Y)).reshape(Y.shape)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax2 = fig.add_axes([0.05,0.7,0.15,.2])
ax2.plot(xy[:,0],xy[:,1], color="k")

ax.plot_surface(X, Y, Z, alpha=0.5, color='gold', rstride=1, cstride=1)

plt.show()














