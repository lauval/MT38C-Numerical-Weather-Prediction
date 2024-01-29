# imports
import time

import numpy as np

from grid_interpolation import *
from numpy.fft import rfft, rfftfreq, irfft 

# Functions:

def generate_tridiagonal_matrix(N, k, grid_spacing):
    """
    Generates a tridiagonal matrix solver for vorticity inversion in the 
    y-direction.

    Parameters:
    - N: int, size of the matrix.
    - k: float, wavenumber.
    - grid_spacing: float, spacing between grid points. Equivalent to dx or dy.

    Returns:
    - matrix: np.ndarray, tridiagonal matrix.
    """

    # define variables and constants
    dy = grid_spacing
    lambda_ = -(k**2)

    # initiate matrix
    matrix = np.zeros((N,N))
    F = 1/(dy**2)

    # fill the matrix with the correct values by looping through the rows 
    # first, then the columns:
    for i in range(N): 
        for j in range(N): 

            # fill the corners of the matrix with 1:
            if (i == j == 0) or (i == j == N-1): 
                matrix[i,j] = 1 
                
            # fill the rest of the matrix with the correct values    
            else:

                # if the row and column are the same, fill the diagonal 
                # with -2F
                if i == j: 
                    matrix[i,j] = lambda_ - (2*F)

                # if the row and column are next to each other, fill 
                # the matrix with F
                elif (i == j+1 or i == j-1) and (i != 0 and i != N-1):
                    matrix[i,j] = F

    return matrix

def compute_laplacian(streamfunction, y, x):
    """
    Computes the Laplacian of a given streamfunction.

    Parameters:
    - streamfunction: np.ndarray, 2D streamfunction field.
    - y: np.ndarray, 2D y-coordinates.
    - x: np.ndarray, 2D x-coordinates.

    Returns:
    - xi: np.ndarray, Laplacian of the streamfunction (vorticity).
    """
    
    N = np.shape(y)[1]  # num cols
    M = np.shape(x)[0]  # num rows
    dx = dy = y[0, 1] - y[0, 0]  # extract grid space step from y array
    
    # 3 point centred difference scheme for the interior points of vorticity
    d_y_2 = (streamfunction[:, 2:N] - 
             2 * streamfunction[:, 1:(N-1)] + 
             streamfunction[:, 0:(N-2)]) / (dy**2)

    d_x_2 = (streamfunction[2:M, 1:(N-1)] - 
             2 * streamfunction[1:(M-1), 1:(N-1)] + 
             streamfunction[0:(M-2), 1:(N-1)]) / (dx**2)

    # boundary conditions for the second derivative with respect to x
    # 
    d_x_2_first = (streamfunction[1, 1:(N-1)] - 
                   2*streamfunction[0, 1:(N-1)] + 
                   streamfunction[-1, 1:(N-1)]) / (2 * dx)
    
    d_x_2_last  = (streamfunction[0, 1:(N-1)] - 
                   2*streamfunction[-1, 1:(N-1)] + 
                   streamfunction[-2, 1:(N-1)]) / (2 * dx)

    # Stick the boundary conditions onto the interior values
    d_x_2 = np.vstack((d_x_2_first, d_x_2))
    d_x_2 = np.vstack((d_x_2, d_x_2_last))

    # Combine the second derivatives to calculate the vorticity
    xi = d_x_2 + d_y_2
    
    return xi

def compute_velocity(streamfunction, y, x):
    """
    Generates horizontal and vertical velocities from a 
    streamfunction.

    Parameters:
    - streamfunction: np.ndarray, 2D streamfunction field.
    - y: np.ndarray, 2D y-coordinates.
    - x: np.ndarray, 2D x-coordinates.

    Returns:
    - u, v: np.ndarray, 2D horizontal and vertical velocity 
                        fields.
    """

    N = np.shape(y)[1] 
    M = np.shape(x)[0]
    dx = dy = y[0, 1] - y[0, 0]  
    
    # Implement centred difference scheme for u, ignoring
    # the boundary conditions where u is undefined
    u = -1 * ((streamfunction[:, 2:N] - 
               streamfunction[:, 0:(N-2)]) / (2 * dy))

    # Define boundary conditions for v
    v_first = (streamfunction[1, 1:(N-1)] - 
               streamfunction[-1, 1:(N-1)]) / (2 * dx)
    
    v_last = (streamfunction[0, 1:(N-1)] - 
              streamfunction[-2, 1:(N-1)]) / (2 * dx)

    # Implement centred difference scheme for middle values
    # of v
    v_interior = (streamfunction[2:M, 1:(N-1)] - 
                  streamfunction[0:(M-2), 1:(N-1)]) / (2 * dx)
    
    # Combine interior and boundary values for v
    v = np.vstack([v_first, v_interior, v_last])

    return u, v

def compute_dep_pts(u, v, dt, x, y, dx, max_iterations=100):
    """
    Computes departure points based on a given wind field 
    V = (u, v).

    Parameters:
    - u, v: np.ndarray, 2D horizontal and vertical velocity 
                        fields.
    - dt: float, time step.
    - x, y: np.ndarray, 2D grid coordinates.
    - dx: float, grid spacing. Same as dy.
    - max_iterations: int, maximum iterations for updating 
                           departure points.

    Returns:
    - x_dep, y_dep: np.ndarray, 2D departure points.
    """

    # First we set a convergence threshold
    convergence_threshold = 0.01 * dx

    # Set an initial guess of wind at the midpoint between 
    # departure and arrival points to be the wind at
    # the arrival point
    u_mid, v_mid = u, v

    # Set the first guess of departure points 
    x_dep = x - (u_mid * dt)
    y_dep = y - (v_mid * dt)

    # Iterate, updating departure points until convergence 
    for _ in range(max_iterations):
        # Interpolate wind to departure points
        u_dep, v_dep = interpolate([u, v], 
                                    x, y,
                                    x_dep, y_dep, 
                                    interp_order=3, 
                                    wrap=[True, False])

        # Estimate wind at midpoint
        u_mid = 0.5 * (u_dep + u)
        v_mid = 0.5 * (v_dep + v)

        # Estimate new departure points
        x_dep_new = x_dep - u_mid * dt
        y_dep_new = y_dep - v_mid * dt

        # Compute change from (x_dep, y_dep) to (x_dep_new, y_dep_new) across
        # the grid
        max_change = np.max(np.sqrt((x_dep_new - x_dep)**2 + 
                                    (y_dep_new - y_dep)**2))

        # Check for convergence
        if max_change < convergence_threshold * dx:
            break

        # Else, update departure points
        x_dep, y_dep = x_dep_new, y_dep_new

    return x_dep, y_dep

# define constants and parameters
phi_0     = 1e5
f_0       = 1e-4
beta      = 1.6e-11
L_y       = 2.5e5
A         = 0.0125
B         = 1e-3
x_0_psi   = 1.25e7
y_0_psi   = 2.5e6
sigma_psi = 1e6

# duration of simulation
duration  = 6 * 24 * 60 * 60 # 6 days in seconds
dt        = 60 * 60          # 1 hour in seconds

# set up some time step sizes for experiments
step_sizes = [dt]

# set up some interpolation orders for experiments
# interpolation_orders = [1, 3, 5]
interpolation_orders = [5]

# define two streamfunction scenarios for experiments
# streamfunction_types = ['without_gaussian_bump', 'with_gaussian_bump']
streamfunction_types = ['random_noise']

# try different grid resolutions
grid_resolutions = [1e4]

# initalise the model

# loop through each grid resolution, timestep size, interpolation order and 
# streamfunction type and run the model for each combination (configuration)
for grid_space in grid_resolutions: 

    # Set up a 2D grid, with endpoints (limits) included
    upper_x_limit = 2.5e7 + grid_space
    upper_y_limit = 5e6 + grid_space

    x = np.arange(0, upper_x_limit, grid_space) # x coordinates
    y = np.arange(0, upper_y_limit, grid_space) # y coordinates
    y_centre = y_0 = np.mean(y)

    # define 2D X and Y arrays:
    Y, X   = np.meshgrid(y, x)
    Y_, X_ = Y[:, 1:-1], X[:, 1:-1] # subsets of X and Y for the quiver plots

    # and lastly, define some constant parameters for the ebv calculation
    f = f_0 + (beta * (Y - y_centre)) # planetary vorticity
    L_2_r = phi_0 / (f_0**2)          # Square of Rossby deformation radius

    # Define the two forms of the streamfunction
    psi_0_with_gaussian_bump = phi_0/f_0 * (1 - (A * np.tanh((Y-y_0)/L_y)) + 
            (B * np.exp(-((X-(x_0_psi))**2 + (Y-y_0_psi)**2
                          ) / (2 * (sigma_psi**2))
                        )))   
    

    psi_0_without_gaussian_bump = phi_0/f_0 * (1 - (A * np.tanh((Y-y_0)/L_y))) 

    # Create some reusable variables for the tridiagonal matrix solver 
    k = rfftfreq(len(x), grid_space/(2.0*np.pi)) # wavenumbers
    num_columns = len(y) # number of columns in the X and Y arrays (equal)

    for time_step_size in step_sizes:

        # calculate the number of timesteps corresponding to the duration based
        # on step size
        num_steps = int(duration/time_step_size) # number of timesteps

        # try different interpolation orders
        for interpolation_order in interpolation_orders:

            # loop through each test scenario: with gaussian bump and without 
            # gaussian bump and set the initial streamfunction, accordingly
            for streamfunction_type in streamfunction_types:
                if streamfunction_type == 'without_gaussian_bump':
                    psi_0 = psi_0_without_gaussian_bump
                elif streamfunction_type == 'with_gaussian_bump':
                    psi_0 = psi_0_with_gaussian_bump
                elif streamfunction_type == 'random_noise':
                    psi_0 = psi_0_without_gaussian_bump + np.random.rand(
                                                          len(x), len(y))

                print(f'\033[1;33m' +  # Start of ANSI escape code: bold yellow
                      f'\n'
                      f'Running model with the following configuration: \n'
                      f'resolution = {grid_space:.0e} metres, \n'
                      f'time step  = {int(time_step_size/dt)} dt, \n'
                      f'interpolation order = {interpolation_order}, \n'
                      f'streamfunction type = {streamfunction_type}' 
                      +  # End of ANSI escape code for bold yellow text
                      '\033[0m')  # End of ANSI escape code, reset to default 

                # calculate vorticity from whichever form of the streamfunction
                # we are using
                relative_vorticity = compute_laplacian(psi_0, Y, X)

                # compute the Cressman term
                cressman = - (psi_0 / L_2_r)

                # add all terms together to compute q_0, the initial guess for 
                # barotropic vorticity
                q_0 = f[:, 1:-1] + relative_vorticity + cressman[:, 1:-1]

                # run the model:
                for step in range(num_steps): # loop through each time step
                    
                    # start the timer
                    start = time.time()

                    # print progress
                    print(f'\n Step {step+1} of {num_steps},'
                          f'{((step+1)/num_steps)*100}% complete')

                    # if this is the first pass, use psi_0 and q_0 as the 
                    # initial values
                    if step == 0:
                        psi_n = psi_0
                        q_n = q_0
                    
                    # otherwise, use the previous time step's values
                    else:
                        q_n = q_n     
                        psi_n = psi_n 


                    # Step 3: Invert $q^n$ to get $\Psi^n$:
                    q_n_hat = rfft(q_n, axis=0)   
                    psi_hat = rfft(psi_n, axis=0) 

                    # compute the boundary conditions of the fourier 
                    # coefficients of the streamfunction
                    northern_boundary = psi_hat[:, 0]
                    southern_boundary = psi_hat[:, -1]

                    # add a dimension to so that the arrays can be
                    # concatenated
                    northern_boundary = northern_boundary[:, np.newaxis]
                    southern_boundary = southern_boundary[:, np.newaxis]

                    # then we concatenate the arrays together along the column 
                    # axis to form the full array of fourier coefficients
                    ebv_hat = np.concatenate(
                        (northern_boundary, q_n_hat, southern_boundary),
                        axis=1)

                    # Copy the vorticity array so that we keep the fourier 
                    # coefficients of the vorticity without overwriting 
                    psi_new = np.copy(ebv_hat)

                    # Loop through the number of columns and solve the 
                    # tridiagonal matrix for each y-value:
                    for i in range(num_columns):
                        N = len(y) 
                        matrix = generate_tridiagonal_matrix(N, k[i], 
                                                             grid_space)

                        # Solve the system of equations
                        psi_new[i, :] = np.linalg.solve(matrix, ebv_hat[i, :])

                    # Inverse Fourier transform the psi_hat_array to get the 
                    # streamfunction
                    inverted_psi = irfft(psi_new, axis=0, n=len(x))

                    # update psi_n to be the inverted psi
                    psi_n = inverted_psi

                    # Step 4: Compute the wind field from the stream function
                    U_n, V_n = compute_velocity(psi_n, Y, X)

                    # Step 5: Estimate the wind field at the next time step
                    if step==0:
                        U_minus_1 = U_n
                        V_minus_1 = V_n

                        U_n_plus_1 = (2 * U_n) - U_minus_1
                        V_n_plus_1 = (2 * V_n) - V_minus_1

                    else:
                        U_n_plus_1 = (2 * U_n) - U_minus_1
                        V_n_plus_1 = (2 * V_n) - V_minus_1

                    # Step 6: Calculate the departure points
                    x_dep, y_dep = compute_dep_pts(u = U_n_plus_1, 
                                                   v=V_n_plus_1, 
                                                   dt=dt, x=X[:,1:-1], 
                                                   y=Y[:,1:-1], dx=grid_space)

                    # Step 7: Interpolate $q^0$ to the departure points to get 
                    # $q^1$
                    q_n = interpolate([q_n], X[:,2:-2], Y[:,2:-2], 
                                       x_dep, y_dep,
                                       interp_order=interpolation_order,
                                       wrap=[True, False])
                    
                    q_n = q_n[0]
                    
                    print(f'data/{int(grid_space/1e4)}/'
                          f'{int(time_step_size/dt)}dt/'
                          f'{streamfunction_type}/'
                          f'order_{interpolation_order}/')

                    # save vorticity, streamfunction, u and v to file
                    np.save(f'data/{int(grid_space/1e4)}/'
                            f'{int(time_step_size/dt)}dt/'
                            f'{streamfunction_type}/'
                            f'order_{interpolation_order}/'
                            f'vorticity_{step+1:03d}.npy', q_n)
                    
                    np.save(f'data/{int(grid_space/1e4)}/'
                            f'{int(time_step_size/dt)}dt/'
                            f'{streamfunction_type}/'
                            f'order_{interpolation_order}/'
                            f'streamfunction_{step+1:03d}.npy', psi_n)

                    np.save(f'data/{int(grid_space/1e4)}/'
                            f'{int(time_step_size/dt)}dt/'
                            f'{streamfunction_type}/'
                            f'order_{interpolation_order}/'
                            f'u_{step+1:03d}.npy', U_n)

                    np.save(f'data/{int(grid_space/1e4)}/'
                            f'{int(time_step_size/dt)}dt/'
                            f'{streamfunction_type}/'
                            f'order_{interpolation_order}/'
                            f'v_{step+1:03d}.npy', V_n)

                    # store current u and v values as u_minus_1 and v_minus_1 
                    # for use in the next iteration
                    U_minus_1, V_minus_1 = U_n, V_n

                    # stop the timer
                    end = time.time()

                    duration = end-start

                    # print the time taken to run the model for this timestep
                    print(f'\033[1;32m' +  # bold green text
                          f'Time taken to run model for this timestep: '
                          f'{duration:.2f} seconds' +  # End of green text
                          '\033[0m')
                    
                    # save the duration of the run to file
                    np.save(f'data/{int(grid_space/1e4)}/'
                            f'{int(time_step_size/dt)}dt/'
                            f'{streamfunction_type}/'
                            f'order_{interpolation_order}/'
                            f'duration_{step+1:03d}.txt', duration)