import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from smarc_modelling.vehicles.BlueROV_nmpc_setup import setup_nmpc
from acados_template import AcadosSimSolver

def run_simulation():
    # 1. Setup
    # Initial state: [0,0,0] position, Identity quaternion, zero velocity
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Target state: Move 2m in X, 1m in Z, and Rotate 90 deg Yaw
    # Yaw 90 deg quaternion: [0.707, 0, 0, 0.707]
    x_ref_target = np.array([2.0, 1.0, 1.0, 0.707, 0.0, 0.0, 0.707, 0, 0, 0, 0, 0, 0])
    
    T_horizon = 3.0
    N_horizon = 30
    solver, model = setup_nmpc(x0, N_horizon, T_horizon)
    
    # Create an integrator for the "Real World" simulation
    # We use the same model, but in a discrete integrator wrapper
    sim_solver = AcadosSimSolver(solver.acados_ocp, json_file='acados_ocp.json')

    # Simulation params
    tf_sim = 10.0
    dt = T_horizon / N_horizon
    N_sim = int(tf_sim / dt)

    # Data logging
    x_history = np.zeros((N_sim+1, 13))
    u_history = np.zeros((N_sim, 6))
    x_history[0, :] = x0
    
    current_x = x0

    print("Starting NMPC Simulation...")
    for i in range(N_sim):
        
        # 1. Update Reference (Set Point)
        # We set the reference for all N stages to the target
        for j in range(N_horizon):
            yref = np.hstack((x_ref_target, np.zeros(6))) # State + Control (0)
            solver.set(j, "yref", yref)
        
        # Terminal reference
        solver.set(N_horizon, "yref", x_ref_target)

        # 2. Update Initial Condition constraint
        solver.set(0, "lbx", current_x)
        solver.set(0, "ubx", current_x)

        # 3. Solve OCP
        status = solver.solve()
        if status != 0:
            print(f"Solver failed at step {i} with status {status}")

        # 4. Get Control Action
        u_opt = solver.get(0, "u")
        u_history[i, :] = u_opt

        # 5. Simulate Plant (Integrate forward one step)
        sim_solver.set("x", current_x)
        sim_solver.set("u", u_opt)
        status_sim = sim_solver.solve()
        
        if status_sim != 0:
            print(f"Integrator failed at step {i}")
            
        current_x = sim_solver.get("x")
        
        # Normalize Quaternion to prevent drift in simulation
        q_norm = np.linalg.norm(current_x[3:7])
        current_x[3:7] /= q_norm
        
        x_history[i+1, :] = current_x

    print("Simulation finished. Plotting...")
    plot_results(x_history, u_history, dt)


def plot_results(x, u, dt):
    t = np.arange(x.shape[0]) * dt
    t_u = np.arange(u.shape[0]) * dt

    fig = plt.figure(figsize=(12, 10))

    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory')
    ax1.scatter(x[0,0], x[0,1], x[0,2], c='g', marker='o', label='Start')
    ax1.scatter(x[-1,0], x[-1,1], x[-1,2], c='r', marker='x', label='End')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('3D Position')
    ax1.legend()
    ax1.invert_zaxis() # Depth convention

    # 2. Position vs Time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, x[:, 0], label='x')
    ax2.plot(t, x[:, 1], label='y')
    ax2.plot(t, x[:, 2], label='z')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position [m]')
    ax2.set_title('Position States')
    ax2.grid(True)
    ax2.legend()

    # 3. Velocity vs Time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, x[:, 7], label='u')
    ax3.plot(t, x[:, 8], label='v')
    ax3.plot(t, x[:, 9], label='w')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_title('Body Velocities')
    ax3.grid(True)
    ax3.legend()

    # 4. Controls
    ax4 = fig.add_subplot(2, 2, 4)
    labels = ['Tx', 'Ty', 'Tz', 'Mx', 'My', 'Mz']
    for k in range(6):
        ax4.step(t_u, u[:, k], label=labels[k])
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Force/Torque')
    ax4.set_title('Control Inputs')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_simulation()