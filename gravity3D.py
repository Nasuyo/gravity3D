#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:38:17 2021

@author: schroeder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pyshtools as sh
import scipy

## Constants ------------------------------------------------------------------
G = sh.constants.G.value  # gravitational constant [m^3/(s^2 kg)]
c = scipy.constants.c  # speed of light [m/s]


## Point mass attraction at another specified point ---------------------------
mass = 1  # [kg]
mass_pos = np.array([0, 0, 0])  # [x, y, z] in [m]
point_pos = np.array([1, 4, 2])
l = np.linalg.norm(point_pos-mass_pos)  # [m]
F = G * mass / l**2  # [m/s^2]
F_x = - G * mass / l**3 * (point_pos[0] - mass_pos[0])  # [m/s^2]
F_y = - G * mass / l**3 * (point_pos[1] - mass_pos[1])  # [m/s^2]
F_z = - G * mass / l**3 * (point_pos[2] - mass_pos[2])  # [m/s^2] g (gravity acceleration in z-direction)
V = G * mass / l  # [m^2/s^2] gravity potential
ff = V / c**2  # unitless, fractional frequency difference due to potential


## Attraction of a mass volume in 3D ------------------------------------------
# This is done by scattering point masses within the volume, computing the 3D
# gravity acceleration in z-direction of it, and then shifting and accumulating
# the resulting 3D matrix in order to get the gravity from all the mass.

# FPQ building base example (neither final nor accurate)
fpq = np.array([[0, 0, 0], [40, 0, 0], [44, -14, 0], [74, -6, 0], [74, 6, 0],
                [50, 6, 0], [47, 18, 0], [0, 18, 0], [0, 0, 0]])  # [y, x, z] [m]

# measuring properties
extent = np.array([-30, 80, -20, 20])  # [x_min, x_max, y_min, y_max]  computation area
extent_z = np.array([-10, 6])
mheight = -3.9  # [m] measuring height; -3.9 would be 0.3m above cellar floor

# mass properties, e.g. a lorry next to the building
mass = 20000  # [kg] overall mass of the volume
L = 10  # [m] length (x)
W = 3  # [m] width (y)
H = 4  #[m] height (z)
mass_x = np.array([-11, -11+L])  # x-coordinates of the mass volume
mass_y = np.array([10, 10+W])  # y-coods
mass_z = np.array([0, 0+H])  # z-coords
spacing = 0.2  # [m] resolution
mass = (mass/(L*W*H))*spacing**3  # [kg/m^2]  # mass of a single mass point within the volume

# positions of the point masses in the volume
mass_pos = np.zeros((int(np.ceil(L*W*H/spacing**3)), 3))
l = 0
for i in np.arange(mass_x[0]+spacing/2, mass_x[1], spacing):
    for j in np.arange(mass_y[0]+spacing/2, mass_y[1], spacing):
        for k in np.arange(mass_z[0]+spacing/2, mass_z[1], spacing):
            mass_pos[l, :] = np.array([j, i, k])
            l += 1
            
# 3D gravity computation of one of the many point masses within the volume
N_x = int((extent[1] - extent[0]) / spacing)  # number of points along x-axis
N_y = int((extent[3] - extent[2]) / spacing)  # ...y-axis
N_z = int((extent_z[1] - extent_z[0]) / spacing)  # ...z-axis
g = np.zeros((N_y, N_x, N_z))  # [m/s^2] gravity
counter = 0
for k, z in enumerate(np.arange(extent_z[0]+spacing/2, extent_z[1], spacing)):
    print(counter)
    counter += 1
    for i, y in enumerate(np.arange(extent[2]+spacing/2, extent[3], spacing)):
        for j, x in enumerate(np.arange(extent[0]+spacing/2, extent[1], spacing)):
            point_pos = np.array([y, x, z])
            # prevent inf value when point_pos equals mass_pos
            if np.allclose(point_pos, mass_pos[0, :], atol=1e-5):
                g[i, j, k] = 0
                continue
            l = np.linalg.norm(point_pos-mass_pos[0, :])  # [m] distance between point and mass
            g[i, j, k] = - (G * mass / l**3 * (point_pos[2] - mass_pos[0, 2]))  # [m/s^2] gravity
            # for the gravity potential it would be the following formula:
            # V[i, j, k] = G * mass / l  # [m^2/s^2] gravity potential
            
# shift and accumulate g, so that the attraction of every of the many mass points in the volume is added
gg = np.zeros((N_y+int(W/spacing)-1, N_x+int(L/spacing)-1,
               N_z+int(H/spacing)-1))
for y in range(int(W/spacing)):  # shift along y-axis for the width of the volume
    for x in range(int(L/spacing)):  # x-axis
        for z in range(int(H/spacing)):  # z-axis
            gg[y:y+N_y, x:x+N_x, z:z+N_z] += g  # accumulate
            
# the borders of gg are uncomplete, cut them out
gg_core = gg[int(W/spacing)-1:N_y, int(L/spacing)-1:N_x,
             int(H/spacing)-1:N_z]  # cut off "uncomplete" cells
extent_core = extent + np.array([L-spacing, 0, W-spacing, 0])
extent_z_core = extent_z + np.array([H-spacing, 0])

# find out, at which z-level in gg_core the measurement took place
m_zindex = int((mheight - extent_z_core[0]) / spacing)  # index of measuring height within gg_core

# plot xy-plane at measurement level
mass_plane = np.array([[mass_x[0], mass_y[0]], [mass_x[0], mass_y[1]],
                      [mass_x[1], mass_y[1]], [mass_x[1], mass_y[0]], [mass_x[0], mass_y[0]]])
fig, ax = plt.subplots(figsize=(6, 12))
img = ax.imshow(np.flipud(gg_core[:, :, m_zindex]), extent=extent_core, interpolation='None', norm=colors.LogNorm())
fig.colorbar(img, fraction=0.020, pad=0.04, label='-g [m/s²]')
ax.plot(fpq[:, 0], fpq[:, 1], 'k')
ax.plot(mass_plane[:, 0], mass_plane[:, 1], 'k')
ax.set_aspect('equal')
ax.set_ylabel('[m]')
ax.set_xlabel('[m]')

# Disclaimer: The spacing of the mass points within the volume and the spacing
# of the computation points is the same here. This is easier to implement. But
# it has the downside that you can not compute the xy-plane at arbitrary levels.
# Here for example, the measuring height is supposed to be -3.9m, but at 1m
# spacing it is rounded to -4m.
# When setting a finer spacing, the computation time increases with N^3. 
# Therefore, this is practically very limited.
# If one is only interested in a specific plane/height, she/he can leave out
# the shifting in z-direction and make it a N^2 computation time increase.

    
## Scenario: mass changes coordinates in z-direction, e.g. an elevator --------
# This can be computed by simply shifting the 3D matrix by the desired change
# in z-direction and subtracting it from itself. Here, since we just want to
# plot one layer, it is enough to subtract two xy-planes from each other
fig, ax = plt.subplots(figsize=(6, 12))             #4.2 is the distance between ground floor and cellar floor
img = ax.imshow(np.flipud(gg_core[:, :, m_zindex+int(4.2/spacing)])-np.flipud(gg_core[:, :, m_zindex]),
                extent=extent_core, interpolation='None', norm=colors.SymLogNorm(linthresh=1e-10))
fig.colorbar(img, fraction=0.025, pad=0.04, label='-g [m/s²]')
ax.plot(fpq[:, 0], fpq[:, 1], 'k')
ax.plot(mass_plane[:, 0], mass_plane[:, 1], 'k')
ax.set_aspect('equal')
ax.set_ylabel('[m]')
