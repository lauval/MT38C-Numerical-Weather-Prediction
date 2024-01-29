from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import imageio
plt.style.use('science')

def process_and_plot_data(data_type, grid_space, streamfunction_type, interpolation_order, step_size):
    
    # define grid 
    upper_x_limit = 2.5e7 + grid_space
    upper_y_limit = 5e6 + grid_space

    x = np.arange(0, upper_x_limit, grid_space)
    y = np.arange(0, upper_y_limit, grid_space)

    Y, X = np.meshgrid(y, x)

    # define subset of the grid for plotting wind barbs
    X_subset = X[:, 1:-1]
    Y_subset = Y[:, 1:-1]

    skip = 80  # set to 80 if grid resolution is 1e4

    # define input data directory
    input_dir = Path(f"data/{int(grid_space/1e4)}/{step_size/60/60:.0f}dt/{streamfunction_type}/order_{interpolation_order}")

    # define output data directory
    output_dir = input_dir / 'figs' / data_type

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created directory {output_dir}")

    print(f"Processing data in {input_dir} and saving figures to {output_dir}")

    # loop over all files in the input directory for the specified data type
    for i in range(144):

        if data_type == 'streamfunction':

            # load the data
            psi_file = f'{input_dir}/streamfunction_{i+1:03d}.npy'
            psi_n = np.load(psi_file)

            # load wind data
            u_file, v_file = f'{input_dir}/u_{i+1:03d}.npy', f'{input_dir}/v_{i+1:03d}.npy'
            U_n, V_n = np.load(u_file), np.load(v_file)

            # plot the streamfunction
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

            ax.set_aspect('equal')

            ax.contourf(X, Y, psi_n)
            ax.set_title(f'$\Psi$ (m$^2$ s$^{-1}$) at timestep {i+1} of 144')
            ax.set_ylabel('Y-Coordinate')
            ax.set_xlabel('X-Coordinate')         

            ax.quiver(X_subset[1:-1:skip, 1:-1:skip], Y_subset[1:-1:skip, 1:-1:skip], U_n[1:-1:skip, 1:-1:skip], V_n[1:-1:skip, 1:-1:skip], scale=700, color='black')  
            
            plt.savefig(f'{output_dir}/{data_type}_{i+1:03d}.png')
            
            plt.close()

        elif data_type == 'vorticity':

            # load the data
            q_file = f'{input_dir}/vorticity_{i+1:03d}.npy'
            q_n = np.load(q_file)

            # plot the streamfunction
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

            ax.set_aspect('equal')

            plot = ax.pcolormesh(q_n.T)
            ax.set_title(f'$q$ (s$^{-1}$) at timestep {i+1} of 144')
            ax.set_ylabel('Y-Coordinate')
            ax.set_xlabel('X-Coordinate')
            # colorbar = plt.colorbar(plot, ax=ax, pad=0.2, orientation='horizontal',  label='(s$^{-1}$)')

            plt.savefig(f'{output_dir}/{data_type}_{i+1:03d}.png')
            plt.close()

        elif data_type == 'combined':
            
            # load the data
            psi_file, vort_file = f'{input_dir}/streamfunction_{i+1:03d}.npy', f'{input_dir}/vorticity_{i+1:03d}.npy'
            psi_n, vort_n = np.load(psi_file), np.load(vort_file)

            # load wind data
            u_file, v_file = f'{input_dir}/u_{i+1:03d}.npy', f'{input_dir}/v_{i+1:03d}.npy'
            U_n, V_n = np.load(u_file), np.load(v_file)

            # plot the streamfunction with wind barbs
            fig, ax = plt.subplots(2, figsize=(7, 4), dpi=300)

            for axis in ax: # set equal aspect ratio
                axis.set_aspect('equal')
                axis.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

            ax[0].contourf(X, Y, psi_n)
            ax[0].set_title(f'$\Psi$ (m$^2$/s) at timestep {i+1} of 144')

            ax[0].quiver(X_subset[1:-1:skip, 1:-1:skip], Y_subset[1:-1:skip, 1:-1:skip], U_n[1:-1:skip, 1:-1:skip], V_n[1:-1:skip, 1:-1:skip], scale=700, color='black') # set scale to 700 if grid resolution is 1e4

            ax[1].pcolormesh(vort_n.T)
            ax[1].set_title(f'$q$ (/s) at timestep {i+1} of 144')

            plt.savefig(f'{output_dir}/{data_type}_{i+1:03d}.png')
            
            plt.close()
        
    print('Done\n')

def create_gif(data_type, grid_space, streamfunction_type, interpolation_order, step_size):
    
    input_dir = Path(f"data/{int(grid_space/1e4)}/{step_size/60/60:.0f}dt/{streamfunction_type}/order_{interpolation_order}/figs/{data_type}/")
    
    frames = []

    for i in range(144):
        filename = f'{input_dir}/{data_type}_{i+1:03d}.png'
        frames.append(imageio.v2.imread(filename))

    print(f'Creating file: {data_type}_{int(grid_space/1e4)}_{streamfunction_type}_order{interpolation_order}.gif')

    imageio.mimsave(f'new_gifs/{data_type}_{int(grid_space/1e4)}_{streamfunction_type}_order{interpolation_order}.gif', frames, 'GIF', duration=0.1)

    print('Done')

dt = 60 * 60          # 1 hour in seconds
# set up some time step sizes for experiments
step_sizes = [dt]

# set up some interpolation orders for experiments
interpolation_orders = [5]

# define two streamfunction scenarios for experiments
streamfunction_types = ['without_gaussian_bump', 'with_gaussian_bump', 'random_noise']

# try different grid resolutions
grid_resolutions = [1e4]

data_types = ['combined']

for data in data_types:
    for grid_space in grid_resolutions:
        for streamfunction_type in streamfunction_types:
            for interpolation_order in interpolation_orders:
                for step_size in step_sizes:
                    process_and_plot_data(data, grid_space, streamfunction_type, interpolation_order, step_size)
                    create_gif(data, grid_space, streamfunction_type, interpolation_order, step_size)