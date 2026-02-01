from acados_template import AcadosModel
import casadi as ca
import numpy as np

def skew(vec):
    """Return skew symmetric matrix from a 3D vector"""
    return ca.vertcat(
        ca.horzcat(0, -vec[2], vec[1]),
        ca.horzcat(vec[2], 0, -vec[0]),
        ca.horzcat(-vec[1], vec[0], 0)
    )

def export_bluerov_model():
    model_name = 'bluerov'

    # --- Constants from BlueROV.py ---
    m = 13.5
    g = 9.81
    W = m * g
    B = W  # Neutral buoyancy assumption

    # Inertia
    Ix, Iy, Iz = 0.26, 0.23, 0.37
    I_mat = ca.diag(ca.vertcat(Ix, Iy, Iz))

    # Added Mass
    Xdu, Ydv, Zdw = 6.36, 7.12, 18.68
    Kdp, Mdq, Ndr = 0.189, 0.135, 0.222
    MA_diag = ca.vertcat(Xdu, Ydv, Zdw, Kdp, Mdq, Ndr)
    MA = ca.diag(MA_diag)

    # Rigid Body Mass
    MRB = ca.SX.zeros(6, 6)
    MRB[0:3, 0:3] = ca.SX.eye(3) * m
    MRB[3:6, 3:6] = I_mat
    
    # Total Mass
    M_total = MRB + MA
    Minv = ca.inv(M_total)

    # Damping Coefficients
    Xu, Yv, Zw = 13.7, 0.0, 33.0
    Kp, Mq, Nr = 0.0, 0.8, 0.0
    D_lin = ca.diag(ca.vertcat(Xu, Yv, Zw, Kp, Mq, Nr))

    Xuu, Yvv, Zww = 141.0, 217.0, 190.0
    Kpp, Mqq, Nrr = 1.19, 0.47, 1.5
    D_nl_coeffs = ca.vertcat(Xuu, Yvv, Zww, Kpp, Mqq, Nrr)

    # --- State & Control Symbols ---
    # p = [x, y, z]
    p = ca.SX.sym('p', 3)
    # q = [q0, q1, q2, q3] (Scalar first)
    q = ca.SX.sym('q', 4)
    # nu = [u, v, w, p, q, r]
    nu = ca.SX.sym('nu', 6)
    
    x = ca.vertcat(p, q, nu)

    # Controls: u = [Tx, Ty, Tz, Mx, My, Mz]
    tau = ca.SX.sym('tau', 6)
    u = tau

    # --- Dynamics Calculation ---
    
    # 1. Coriolis (C_RB + C_A)
    nu1 = nu[0:3]
    nu2 = nu[3:6]
    C_RB = ca.SX.zeros(6, 6)
    C_RB[0:3, 3:6] = -skew(m * nu1)
    C_RB[3:6, 0:3] = -skew(m * nu1)
    C_RB[3:6, 3:6] = -skew(I_mat @ nu2)

    # C_A (Added Mass)
    A11 = MA[0:3, 0:3]
    A12 = MA[0:3, 3:6]
    A21 = MA[3:6, 0:3]
    A22 = MA[3:6, 3:6]
    
    a1 = A11 @ nu1 + A12 @ nu2
    a2 = A21 @ nu1 + A22 @ nu2
    
    C_A = ca.SX.zeros(6,6)
    C_A[0:3, 3:6] = -skew(a1)
    C_A[3:6, 0:3] = -skew(a1)
    C_A[3:6, 3:6] = -skew(a2)

    C_total = C_RB + C_A

    # 2. Damping
    D_nl = ca.diag(D_nl_coeffs * ca.fabs(nu))
    D_total = D_lin + D_nl

    # 3. Restoring Forces (Gravity/Buoyancy) - FIXED HERE
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    
    # Rotation Matrix Body to NED (R_b_n)
    R_b_n = ca.vertcat(
        ca.horzcat(1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)),
        ca.horzcat(2*(q1*q2+q0*q3),   1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)),
        ca.horzcat(2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   1-2*(q1**2+q2**2))
    )
    
    # Transform Gravity/Buoyancy to Body Frame
    # R_n_b = R_b_n.T
    f_g_b = R_b_n.T @ ca.vertcat(0, 0, W)
    f_b_b = R_b_n.T @ ca.vertcat(0, 0, -B)
    
    # Restoring Forces (3x1)
    g_force = -(f_g_b + f_b_b)
    
    # Restoring Moments (3x1)
    # Since CG and CB are at [0,0,0], moments are zero.
    # If you had offsets: r_g = [x,y,z]; M_g = cross(r_g, f_g_b)
    g_moment = ca.SX.zeros(3, 1)

    # Stack to create 6x1 vector
    g_vec = ca.vertcat(g_force, g_moment)

    # 4. Kinematics
    p_dot = R_b_n @ nu1
    
    # Quaternion derivative
    T_q = 0.5 * ca.vertcat(
        ca.horzcat(-q1, -q2, -q3),
        ca.horzcat( q0, -q3,  q2),
        ca.horzcat( q3,  q0, -q1),
        ca.horzcat(-q2,  q1,  q0)
    )
    gamma = 100
    q_dot = T_q @ nu2 + (gamma / 2) * (1 - ca.dot(q, q)) * q

    # 5. Body Acceleration
    # All terms are now 6x1
    rhs = tau - C_total @ nu - D_total @ nu - g_vec
    nu_dot = Minv @ rhs

    f_expl = ca.vertcat(p_dot, q_dot, nu_dot)

    # --- Acados Model Struct ---
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = model_name
    
    return model