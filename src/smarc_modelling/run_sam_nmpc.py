#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import csv
import os
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np

# Assuming these imports work in your environment
try:
    from smarc_modelling.control.control import *
    from smarc_modelling.vehicles import *
    # from smarc_modelling.lib import plot # Removed to avoid \dotX crash
    from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
except ImportError:
    pass

def generate_helical_trajectory(dt, duration, type='helix'):
    """
    Generates a reference trajectory procedurally without CSV files.
    Returns:
        np.array: Matrix of shape (N_samples, 19) containing [state (13) + actuators (6)]
    """
    N_samples = int(duration / dt)
    t = np.linspace(0, duration, N_samples)
    
    trajectory = np.zeros((N_samples, 19))
    
    # 1. Trajectory Parameters
    u_vel = 1.0  # Forward speed [m/s]
    radius = 10.0 # Helix radius [m]
    period = 200.0 # Helix period [s]
    omega = 2 * np.pi / period
    z_center = 10.0 # Center depth [m]
    
    # 2. Generate Position (Helix)
    trajectory[:, 0] = u_vel * t
    trajectory[:, 1] = radius * np.sin(omega * t)
    trajectory[:, 2] = z_center + radius * np.cos(omega * t)
    
    # 3. Generate Orientation (Quaternion) - Level flight
    trajectory[:, 3] = 1.0 # qw
    trajectory[:, 4] = 0.0 # qx
    trajectory[:, 5] = 0.0 # qy
    trajectory[:, 6] = 0.0 # qz
    
    # 4. Generate Velocities (Body Frame)
    trajectory[:, 7] = u_vel
    trajectory[:, 8] = 0.0
    trajectory[:, 9] = 0.0
    
    # 5. Generate Actuator States (Trim Conditions)
    # [vbs, lcg, ds, dr, rpm1, rpm2]
    trajectory[:, 13] = 50.0  # VBS [%]
    trajectory[:, 14] = 50.0  # LCG [%]
    trajectory[:, 15] = 0.0   # Stern [rad]
    trajectory[:, 16] = 0.0   # Rudder [rad]
    trajectory[:, 17] = 200.0 # RPM1
    trajectory[:, 18] = 200.0 # RPM2
    
    return trajectory

def custom_plot_function(t, trajectory, simX, simU):
    """
    Replacement for smarc_modelling.lib.plot to avoid LaTeX errors.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # 1. Position (x, y, z)
    # trajectory is (N, 25), simX is (N, 19)
    axes[0].plot(t, simX[:, 0], label='x')
    axes[0].plot(t, simX[:, 1], label='y')
    axes[0].plot(t, simX[:, 2], label='z')
    axes[0].plot(t, trajectory[:, 0], 'k--', alpha=0.5, label='Ref x')
    axes[0].plot(t, trajectory[:, 1], 'k:', alpha=0.5, label='Ref y')
    axes[0].plot(t, trajectory[:, 2], 'k-.', alpha=0.5, label='Ref z')
    axes[0].set_ylabel('Position [m]')
    axes[0].legend(loc='right')
    axes[0].grid(True)
    axes[0].set_title('Position Tracking')

    # 2. Velocities (u, v, w)
    axes[1].plot(t, simX[:, 7], label='u')
    axes[1].plot(t, simX[:, 8], label='v')
    axes[1].plot(t, simX[:, 9], label='w')
    axes[1].set_ylabel('Velocity [m/s]')
    axes[1].legend(loc='right')
    axes[1].grid(True)
    axes[1].set_title('Body Velocities')

    # 3. Actuator States (VBS, LCG, RPM)
    axes[2].plot(t, simX[:, 13], label='VBS %')
    axes[2].plot(t, simX[:, 14], label='LCG %')
    axes[2].plot(t, simX[:, 17]/10, label='RPM1/10') # Scale for visibility
    axes[2].set_ylabel('Actuator States')
    axes[2].legend(loc='right')
    axes[2].grid(True)
    
    # 4. Control Inputs (Rates)
    # simU is (N, 6)
    axes[3].plot(t, simU[:, 0], label='VBS_dot')
    axes[3].plot(t, simU[:, 2], label='Stern_dot')
    axes[3].plot(t, simU[:, 3], label='Rudder_dot')
    axes[3].set_ylabel('Control Input (Rates)')
    axes[3].set_xlabel('Time [s]')
    axes[3].legend(loc='right')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()

def rmse(true, pred):
    """
    Compute Root Mean Square Error.
    """
    # Slice to match lengths if necessary
    min_len = min(len(true), len(pred))
    true = true[:min_len]
    pred = pred[:min_len]

    norm = np.sqrt(np.mean((np.linalg.norm(true[:,:3] - pred[:,:3],axis=1)) ** 2))

    x_true = np.array(true[:, 0])
    x_pred = np.array(pred[:, 0])
    y_true = np.array(true[:, 1])
    y_pred = np.array(pred[:, 1])   
    z_true = np.array(true[:, 2])
    z_pred = np.array(pred[:, 2])
    
    xrmse = np.sqrt(np.mean((x_true - x_pred) ** 2))
    yrmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    zrmse = np.sqrt(np.mean((z_true - z_pred) ** 2))
    print(f"x: {xrmse:.4f}\ny: {yrmse:.4f}\nz: {zrmse:.4f}\nnorm: {norm:.4f}\n")

def main():
    # Extract the CasADi model
    dt_val = 0.1
    sam = SAM_casadi(dt=dt_val)

    # create ocp object to formulate the OCP
    Ts = 0.1           # Sampling time
    N_horizon = 20      # Prediction horizon
    nmpc = NMPC(sam, Ts, N_horizon, update_solver_settings=True)
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length
    nc = 1

    # Run the MPC setup
    # x0 is set later based on trajectory start
    ocp_solver, integrator = nmpc.setup()

    case = "generated"
    
    # Simulation Loop
    for j in range(1):
        print(f"Generating trajectory for case: {case}")
        
        # --- REPLACED CSV LOADING WITH FUNCTION CALL ---
        duration = 40.0
        trajectory = generate_helical_trajectory(Ts, duration, type='helix')
        
        # Declare duration of sim. and the x_axis in the plots
        Nsim = (trajectory.shape[0])            
        x_axis = np.linspace(0, Ts*Nsim, Nsim)

        simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control derivative
        simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated states

        # Declare the initial state
        x0 = trajectory[0] 
        simX[0,:] = x0

        # Augment the trajectory and control input reference 
        Uref = np.zeros((trajectory.shape[0], nu)) 
        trajectory = np.concatenate((trajectory, Uref), axis=1) 

        # Initialize the state and control vector as David does
        for stage in range(N_horizon + 1):
            ocp_solver.set(stage, "x", x0)
        for stage in range(N_horizon):
            ocp_solver.set(stage, "u", np.zeros(nu,))

        # Array to store the time values
        t = np.zeros((Nsim))

        # closed loop - simulation
        print("Starting Simulation...")
        for i in range(Nsim):
            # extract the sub-trajectory for the horizon
            if i <= (Nsim - N_horizon):
                ref = trajectory[i:i + N_horizon, :]
            else:
                ref = trajectory[i:, :]

            # Update reference vector
            for stage in range(N_horizon):
                if stage < ref.shape[0]:
                     current_ref = ref[stage,:]
                else:
                     current_ref = ref[-1,:]
                
                try:
                    ocp_solver.set(stage, "p", current_ref) 
                except Exception:
                    pass

                # Set yref for the cost function (State + Control)
                ocp_solver.set(stage, "yref", current_ref)

            # Set the terminal state reference (State only)
            terminal_ref = ref[-1, :nx] if ref.shape[0] > 0 else trajectory[-1, :nx]
            ocp_solver.set(N_horizon, "yref", terminal_ref) 
    
            # Set current state constraint for the solver
            ocp_solver.set(0, "lbx", simX[i, :])
            ocp_solver.set(0, "ubx", simX[i, :])

            # solve ocp and get next control input
            if i % nc == 0 and i < Nsim - (nc-1):
                status = ocp_solver.solve()
                if status != 0:
                    print(f" Note: acados_ocp_solver returned status: {status}")

                # simulate system
                t[i] = ocp_solver.get_stats('time_tot')
                for k in range(nc):
                    simU[i+k, :] = ocp_solver.get(k, "u")
            
            # Simulate Plant (Integrator)
            noise_vector = np.zeros(19)
            simX[i+1, :] = integrator.simulate(x=simX[i, :]+noise_vector, u=simU[i, :])
        

        # evaluate timings
        t *= 1000  # scale to milliseconds
        print(f"median computation time: {np.median(t):.3f} ms")

        # plot results using custom function instead of broken external lib
        rmse(simX[:-1], trajectory)
        custom_plot_function(x_axis, trajectory, simX[:-1], simU)

if __name__ == '__main__':
    main()