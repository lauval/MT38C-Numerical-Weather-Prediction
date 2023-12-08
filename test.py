import sys
sys.path.append('Functions') # add 'Functions' folder to system-recognised path in order to import .py file and functions from within a nested folder.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots
plt.style.use('science')

from Functions.grid_interpolation import *

# constants
phi_0 = 1e5
f_0   = 1e-4
L_y   = 5e5
A     = 0.0125
sigma = 5e5
x_0   = y_0 = 2.5e6

# set up some interpolation orders for experiments
interpolation_order = 3
test_order          = 1
second_test_order   = 5

# duration of simulation
duration  = 12 * 24 * 60 * 60 # 12 days in seconds
dt        = 60 * 60 # 1 hour in seconds
num_steps = duration/dt # number of timesteps

# common grid spacing
grid_space = 1e5

# adding 1e5 to both limits so that np.arrange() includes the upper limit
upper_x_limit = 2.5e7 + grid_space
upper_y_limit = 5e6 + grid_space

x = np.arange(0, upper_x_limit, grid_space)
y = np.arange(0, upper_y_limit, grid_space)

# define an X array where :
Y, X = np.meshgrid(y, x)

psi  = phi_0/f_0 * (1 - (A * (np.tanh((Y-y_0)/L_y))))

q_xy = np.exp(-((X-x_0)**2 + (Y-y_0)**2)/(2 * (sigma**2)))

def generate_velocity(streamfunction, y, x):
    N = np.shape(y)[1]  # length in the y direction
    M = np.shape(x)[0]
    dx = dy = y[0, 1] - y[0, 0]  # extract grid space step from y array
    
    # Implement centred difference scheme for u, ignoring the boundary conditions as u 
    # is not defined on the boundary
    u = -1 * ((streamfunction[:, 2:N] - streamfunction[:, 0:(N-2)]) / (2 * dy))

    # Define boundary conditions for v
    v_first = (streamfunction[1, 1:(N-1)] - streamfunction[-1, 1:(N-1)]) / (2 * dx)
    v_last = (streamfunction[0, 1:(N-1)] - streamfunction[-2, 1:(N-1)]) / (2 * dx)

    # Implement centred difference scheme for middle values of v
    v_interior = (streamfunction[2:M, 1:(N-1)] - streamfunction[0:(M-2), 1:(N-1)]) / (2 * dx)
    
    # Combine interior and boundary values for v
    v = np.vstack([v_first, v_interior, v_last])

    return u, v

U, V = generate_velocity(psi, Y, X)

def compute_dep_pts(u, v, dt, x, y, dx, max_iterations=100):

    # First we set a convergence threshold
    convergence_threshold = 0.01 * dx

    # Set an initial guess of wind at the midpoint between departure and arrival points to be the 
    # wind at the arrival point
    u_mid, v_mid = u, v

    # Set the first guess of departure points 
    x_dep = x - (u_mid * dt)
    y_dep = y - (v_mid * dt)

    # Iterate, updating departure points until convergence 
    for _ in range(max_iterations):
        # Interpolate wind to departure points using custom interpolation
        u_dep, v_dep = interpolate([u, v], x, y, x_dep, y_dep, interp_order=3, wrap=[True, False])

        # Estimate wind at midpoint
        u_mid = 0.5 * (u_dep + u)
        v_mid = 0.5 * (v_dep + v)

        # Compute new estimate departure points
        x_dep_new = x_dep - u_mid * dt
        y_dep_new = y_dep - v_mid * dt

        # Compute change from (x_dep, y_dep) to (x_dep_new, y_dep_new) across the grid
        max_change = np.max(np.sqrt((x_dep_new - x_dep)**2 + (y_dep_new - y_dep)**2))

        # Check for convergence
        if max_change < convergence_threshold * dx:
            break

        # Update departure points
        x_dep, y_dep = x_dep_new, y_dep_new

    return x_dep, y_dep

# Compute departure points
x_dep, y_dep = compute_dep_pts(u = U, v=V, dt=dt, x=X[:,1:-1], y=Y[:,1:-1], dx=grid_space)

def advect_tracer(num_steps, q, x, y, x_dep, y_dep):

    for time_step in range(int(num_steps)):

        # Interpolate q to departure points
        q_dep = interpolate([q[:,1:-1]], x[:,1:-1], y[:,1:-1], x_dep, y_dep, interp_order=3, wrap=[True, False])

        # Compute new q
        q[:,1:-1] = q_dep[0]

    return q

#advected_tracer_field = advect_tracer(num_steps, q_xy, X, Y, x_dep, y_dep)

fig, ax = plt.subplots(figsize=(5, 10))

# Function to update the plot at each animation frame
def update(frame):
    
    # Your existing code for the time stepping loop
    for time_step in range(int(num_steps)):

        # Interpolate q to departure points
        q_dep = interpolate([q_xy[:,1:-1]], X[:,1:-1], Y[:,1:-1], x_dep, y_dep, interp_order=3, wrap=[True, False])

        # Compute new q
        q_xy[:,1:-1] = q_dep[0]

        # Update the plot with the new data
        ax.clear()
        plt.imshow(q_xy, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        plt.pause(0.1)

# Set up the animation
animation = FuncAnimation(fig, update, frames=int(num_steps), repeat=False)

# Save the animation as an mp4 and gif
animation.save('tracer_advection.mp4', writer='ffmpeg', fps=120)
animation.save('tracer_advection.gif', writer='imagemagick', fps=60)
