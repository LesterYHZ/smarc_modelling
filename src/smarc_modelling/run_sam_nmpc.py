import os
import sys
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from mpl_toolkits.mplot3d import Axes3D

# Import the provided SAM model
# Ensure SAM_casadi.py is in the same directory
try:
    from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
except ImportError:
    raise ImportError("Could not import SAM_casadi.py. Please ensure the file is in the current directory.")

def export_sam_model():
    """
    Wraps the SAM_casadi dynamics into an AcadosModel.
    """
    model = AcadosModel()
    
    # 1. Instantiate the provided CasADi model class
    # We set export=True to get the symbolic function suitable for optimization
    sam = SAM_casadi()
    
    # 2. Define Symbolic Variables for Acados
    # The SAM_casadi 'export=True' mode expects:
    # State x: 13 elements [eta(7), nu(6)]
    # Input u: 6 elements [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
    
    # Define symbols matching the dimensions in SAM_casadi
    x = ca.MX.sym('x', 13) 
    u = ca.MX.sym('u', 6)
    
    # 3. Get the dynamics function from the class
    # The dynamics() method returns a CasADi Function object
    f_dyn_casadi = sam.dynamics(export=True)
    
    # 4. evaluate the function with our symbolic variables to get the expression
    x_dot = f_dyn_casadi(x, u)
    
    # 5. Populate AcadosModel
    model.f_expl_expr = x_dot
    model.x = x
    model.u = u
    model.name = 'sam_auv'
    
    # Helper: Define state indices for clarity later
    # x: [x, y, z, qw, qx, qy, qz, u, v, w, p, q, r]
    # u: [vbs, lcg, delta_s, delta_r, rpm1, rpm2]
    
    return model

def create_ocp_solver(model, prediction_horizon, N_steps):
    """
    Configures the Optimal Control Problem (OCP) solver.
    """
    ocp = AcadosOcp()
    ocp.model = model
    
    # --- Dimensions ---
    ocp.dims.N = N_steps
    
    # --- Cost Function ---
    # We use a standard quadratic tracking cost:
    # J = Sum( (y - y_ref)^T * W * (y - y_ref) )
    # y = [x, u] (state and input concatenated)
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # Weight Matrix (Q for states, R for controls)
    # State indices: [x, y, z, qw, qx, qy, qz, u, v, w, p, q, r]
    # Control indices: [vbs, lcg, delta_s, delta_r, rpm1, rpm2]
    
    # Tuning Weights
    w_pos = 10.0   # High penalty on position error
    w_att = 5.0    # Moderate penalty on orientation
    w_vel = 1.0     # Low penalty on velocity error
    
    w_vbs = 0.1     # Cost on using VBS
    w_lcg = 0.1     # Cost on using LCG
    w_fin = 10.0    # Cost on moving fins (smoothness)
    w_rpm = 1e-5    # Cost on RPM (very low to allow thrust)
    
    # Diagonal Q matrix (13x13)
    Q_diag = np.array([
        w_pos, w_pos, w_pos,    # x, y, z
        w_att, w_att, w_att, w_att, # Quaternions (qw, qx, qy, qz)
        w_vel, w_vel, w_vel,    # u, v, w (surge, sway, heave)
        w_vel, w_vel, w_vel     # p, q, r
    ])
    
    # Diagonal R matrix (6x6)
    R_diag = np.array([
        w_vbs, w_lcg,           # VBS, LCG (%)
        w_fin, w_fin,           # Fins (rad)
        w_rpm, w_rpm            # Propellers
    ])
    
    W = np.diag(np.concatenate([Q_diag, R_diag]))
    W_e = np.diag(Q_diag) # Terminal cost (no control input at terminal stage)
    
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    
    # Mappings (Vx * x + Vu * u = y)
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)
    
    ocp.cost.Vx_e = np.eye(nx)
    
    # Initial References (will be updated in the loop)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)
    
    # --- Constraints ---
    # Read limits from the provided file documentation/comments
    
    # 1. VBS: 0 to 100%
    # 2. LCG: 0 to 100%
    # 3. Fins: +/- 7 degrees (~0.12 rad)
    # 4. RPM: +/- 1500
    
    deg2rad = np.pi / 180.0
    u_min = np.array([ 0.0,   0.0, -7*deg2rad, -7*deg2rad, -1500.0, -1500.0])
    u_max = np.array([100.0, 100.0,  7*deg2rad,  7*deg2rad,  1500.0,  1500.0])
    
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])
    
    # Set initial state (dummy, will be set in simulation)
    ocp.constraints.x0 = np.zeros(nx)
    
    # --- Solver Options ---
    ocp.solver_options.tf = prediction_horizon
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Robust QP solver
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Real-Time Iteration (fast)
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'      # Explicit Runge-Kutta
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 4
    
    # Compile
    json_file = 'acados_sam_ocp.json'
    solver = AcadosOcpSolver(ocp, json_file=json_file)
    
    # Create an integrator for the simulation "Plant"
    # This ensures the simulation uses the exact same model dynamics as the controller
    integrator = AcadosSimSolver(ocp, json_file=json_file)
    
    return solver, integrator

def generate_reference_trajectory(t_start, dt, N, type='helix'):
    """
    Generates a reference trajectory for the horizon.
    Returns an array of shape (N+1, nx+nu).
    """
    refs = []
    
    for i in range(N + 1):
        t = t_start + i * dt
        
        # Default: Stationary at origin
        x_ref = 0
        y_ref = 0
        z_ref = 2.0 # Target depth 2m
        
        # Quaternion identity (level flight) [qw, qx, qy, qz]
        q_ref = [1.0, 0.0, 0.0, 0.0] 
        
        # Velocities
        u_vel = 0.0
        
        if type == 'helix':
            # Forward speed 1 m/s
            u_vel = 0.02
            z_ref = u_vel * t
            
            # Spiral radius 2m, period 100s
            radius = 10.0
            omega = 2 * np.pi / 200.0
            x_ref = radius * np.sin(omega * t)
            y_ref = -10.0 + radius * np.cos(omega * t) # Center at depth 5m
            
            # Simple orientation heuristic
            q_ref = [1.0, 0.0, 0.0, 0.0]
            
        elif type == 'surge':
            u_vel = 1.5
            x_ref = u_vel * t
            z_ref = 2.0
        
        # Full state ref: [x, y, z, qw, qx, qy, qz, u, v, w, p, q, r]
        state_ref = np.array([
            x_ref, y_ref, z_ref,
            q_ref[0], q_ref[1], q_ref[2], q_ref[3],
            u_vel, 0, 0,
            0, 0, 0
        ])
        
        # Input ref: [vbs, lcg, d_s, d_r, rpm1, rpm2]
        input_ref = np.array([50.0, 50.0, 0.0, 0.0, 200.0, 200.0])
        
        # FIX: Always concatenate state and input, even for the terminal step.
        # This ensures the resulting numpy array is rectangular (N+1, 19).
        # The main loop will slice [:nx] for the terminal step, ignoring these inputs.
        refs.append(np.concatenate([state_ref, input_ref]))
            
    return np.array(refs)

def main():
    # --- Setup ---
    T_horizon = 2.0  # Prediction horizon in seconds
    N_steps = 20     # Number of steps in horizon
    dt = T_horizon / N_steps
    
    T_sim = 100.0      # Total simulation time
    n_sim_steps = int(T_sim / dt)
    
    # 1. Export Model
    model = export_sam_model()
    
    # 2. Create Solvers
    ocp_solver, plant_integrator = create_ocp_solver(model, T_horizon, N_steps)
    
    # --- Initialization ---
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    
    # Initial State: [0, 0, 5, 1, 0, 0, 0, ...] (Depth 5m, stationary)
    x0 = np.zeros(nx)
    x0[2] = 0.0 # z
    x0[3] = 1.0 # qw
    
    # Set initial condition in solver
    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)
    
    # Initialize guess (optional but recommended)
    for i in range(N_steps):
        ocp_solver.set(i, "x", x0)
        ocp_solver.set(i, "u", np.zeros(nu))
    ocp_solver.set(N_steps, "x", x0)
    
    # Storage for plotting
    sim_X = np.zeros((n_sim_steps + 1, nx))
    sim_U = np.zeros((n_sim_steps, nu))
    sim_X[0, :] = x0
    
    print("Starting NMPC Simulation...")
    
    # --- Simulation Loop ---
    current_x = x0
    ref_traj_hist = []
    for i in range(n_sim_steps):
        current_time = i * dt
        
        # 1. Update Reference Trajectory
        # We generate a trajectory starting from the current time into the future
        ref_traj = generate_reference_trajectory(current_time, dt, N_steps, type='helix')
        ref_traj_hist.append(ref_traj)

        for k in range(N_steps):
            ocp_solver.set(k, "yref", ref_traj[k])
        ocp_solver.set(N_steps, "yref", ref_traj[N_steps][:nx]) # Terminal cost uses state only
        
        # 2. Update Initial Condition for this iteration
        ocp_solver.set(0, "lbx", current_x)
        ocp_solver.set(0, "ubx", current_x)
        
        # 3. Solve OCP
        status = ocp_solver.solve()
        
        if status != 0:
            print(f"Acados returned status {status} at step {i}")
            # If solver fails, break or use previous control
            # break
        
        # 4. Get Control Input
        u_opt = ocp_solver.get(0, "u")
        sim_U[i, :] = u_opt
        
        # 5. Simulate Plant (Apply control)
        plant_integrator.set("x", current_x)
        plant_integrator.set("u", u_opt)
        
        plant_status = plant_integrator.solve()
        if plant_status != 0:
            print(f"Integrator failed at step {i}")
            
        current_x = plant_integrator.get("x")
        sim_X[i+1, :] = current_x
        
        # Optional: Normalize quaternion to prevent drift
        q_norm = np.linalg.norm(current_x[3:7])
        current_x[3:7] /= q_norm
        
    print("Simulation finished.")
    ref_traj_hist = np.array(ref_traj_hist)

    # --- Plotting ---
    t_span = np.linspace(0, T_sim, n_sim_steps + 1)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot 3D Position
    ax1 = axes[0]
    ax1.plot(sim_X[:, 0], sim_X[:, 1], label='Path (XY)')
    ax1.plot(ref_traj_hist[:,0,0], ref_traj_hist[:,0,1], 'r--', label='Reference Path (XY)')
    ax1.set_ylabel('Y [m]')
    ax1.set_xlabel('X [m]')
    ax1.set_title('XY Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Depth
    ax2 = axes[1]
    ax2.plot(t_span, sim_X[:, 2], color='tab:orange', label='Depth (Z)')
    ax2.plot(t_span[:-1], ref_traj_hist[:,0,2], 'r--', label='Reference Depth (Z)')
    ax2.invert_yaxis() # Depth increases downwards
    ax2.set_ylabel('Depth [m]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Depth Profile')
    ax2.grid(True)
    ax2.legend()
    
    # Plot Controls
    ax3 = axes[2]
    # Plot Propellers normalized
    ax3.plot(t_span[:-1], sim_U[:, 4], label='RPM 1', alpha=0.7)
    ax3.plot(t_span[:-1], sim_U[:, 5], label='RPM 2', alpha=0.7)
    ax3.set_ylabel('Control Inputs')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Actuation')
    ax3.legend()
    ax3.grid(True)
    # Plot Fins in degrees
    ax4 = axes[3]
    ax4.plot(t_span[:-1], np.degrees(sim_U[:, 2]), label='Stern (deg)', alpha=0.7)
    ax4.plot(t_span[:-1], np.degrees(sim_U[:, 3]), label='Rudder (deg)', alpha=0.7)
    
    ax4.set_ylabel('Control Inputs')
    ax4.set_xlabel('Time [s]')
    ax4.set_title('Actuation')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()