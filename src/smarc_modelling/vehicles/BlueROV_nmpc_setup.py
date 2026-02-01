from acados_template import AcadosOcp, AcadosOcpSolver
from smarc_modelling.vehicles.BlueROV_casadi import export_bluerov_model
import numpy as np
import scipy.linalg
import casadi as ca

def setup_nmpc(x0, N_horizon=30, T_horizon=3.0):
    # Load Model
    model = export_bluerov_model()
    
    ocp = AcadosOcp()
    ocp.model = model

    # Dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    # --- Horizon Settings ---
    # Increased Horizon to 3.0s to make the trajectory physically feasible
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    # --- Cost Function (Least Squares) ---
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # Symbolic expressions for cost (Required for NONLINEAR_LS)
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x

    # Weights
    # Smoother tuning: Less aggressive on position, more penalty on control usage
    Q_pos = np.array([100, 100, 100])     # Position (x,y,z)
    Q_rot = np.array([10, 10, 10, 10])     # Quaternion
    Q_vel = np.array([1, 1, 1, 1, 1, 1]) # Velocities
    
    # Increased R to 1.0 to prevent control saturation/chattering
    R_controls = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # Control inputs

    Q_diag = np.concatenate([Q_pos, Q_rot, Q_vel])
    
    W = scipy.linalg.block_diag(np.diag(Q_diag), np.diag(R_controls))
    W_e = np.diag(Q_diag) * 5.0 # Higher terminal cost to encourage convergence

    ocp.cost.W = W
    ocp.cost.W_e = W_e

    # References (placeholder, updated in simulation loop)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # --- Constraints ---
    F_max = 60.0 
    M_max = 20.0
    u_max = np.array([F_max, F_max, F_max, M_max, M_max, M_max])
    
    ocp.constraints.lbu = -u_max
    ocp.constraints.ubu =  u_max
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = x0

    # --- Solver Options ---
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    
    # Levenberg-Marquardt Regularization (Crucial for stability)
    ocp.solver_options.levenberg_marquardt = 1e-2
    
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return solver, model