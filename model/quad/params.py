import typing as T
import numpy as np

def params_fcn(
    t_f: float,
    K: int,
    free_final_time: bool,
    time_dil: bool
) -> T.Dict[str, T.Any]:
    """
    Build and return all static parameters for the trajectory optimization.
    """
    # time‐scaling
    t_scp = t_f / (K - 1)

    # number of sub‐steps per interval
    N_dt = 10

    # dimensions
    n_x = 13  # 12 states + 1 slack
    n_u = 4

    # gravity
    g_0 = 9.806

    # vehicle limits
    max_ang = 45.0
    max_ang_rate = 100.0
    max_ang_rate_yaw = 100.0

    penalty_fcn_list = ['abs', 'max', 'none']  

    # initial / final states and inputs
    x_init = np.hstack(([-10.0, -10.0, 1.0], np.zeros(n_x - 3)))
    x_final = np.hstack(([ 10.0,  10.0, 1.0], np.zeros(n_x - 3)))
    u_init = np.array([0.0, 0.0, 0.0, g_0])
    u_final = u_init.copy()

    # index offsets
    dims = [
        n_x,            # states
        n_x * n_x,      # state × state terms
        n_x * n_u,      # state × input
        n_x * n_u,      # state × input
        n_x,            # states
        n_x             # states
    ]

    offsets = np.concatenate(([0], np.cumsum(dims))).astype(int)
    i0, i1, i2, i3, i4, i5, i6 = tuple(offsets)

    # lin‐spaced warm start
    X_last = np.linspace(x_init, x_final, K).T
    U_last = np.zeros((n_u, K))
    U_last[3, :] = g_0

    # time dilation warm start
    if time_dil:
        S_last = t_scp * np.ones((1, K - 1))
    else:
        S_last = np.array(t_f, dtype=np.float64)

    return {
        # basic settings
        "t_f": t_f,
        "K": K,
        "t_scp": t_scp,
        "N_dt":   N_dt,
        "free_final_time": free_final_time,
        "time_dil": time_dil,

        # dimensions
        "n_x": n_x,
        "n_u": n_u,

        # index offsets into flattened decision vector
        "i0": i0,
        "i1": i1,
        "i2": i2,
        "i3": i3,
        "i4": i4,
        "i5": i5,
        "i6": i6,

        # dynamics warm starts
        "X_last": X_last,
        "U_last": U_last,
        "S_last": S_last,

        # initial / final conditions
        "x_init": x_init,
        "x_final": x_final,
        "u_init": u_init,
        "u_final": u_final,

        # physics
        "g_0": g_0,
        "T_max": g_0 * 1.75,
        "T_min": g_0 * 0.06,
        "tau_max": 0.25,

        # input‐parameter interpolation mode
        "inp_param": ["ZOH", "FOH"][1],

        "add_minmax_sig": False,
        "min_S": t_scp * 0.1,
        "max_S": t_scp * 100.0,
        "w_S": 5.0,

        "add_inp": False,
        "add_inp_trq": False,
        "add_elv_rate": False,
        "w_inp": 1.0,
        "w_inp_trq": 1.0,
        "w_elv_r": 1.0,

        # discrete‐time penalty settings
        "f_dt_dim":  1,
        "w_dt_fcn":  [10],
        "pen_dt_fcn": [penalty_fcn_list[2]],   # or whatever penalty list index you want
        "name_dt_fcn": [['until_pos_spd', 'spd', 'none'][2]],  # must match the shape your solver expects

        # (and similarly, if you use continuous‐time penalties)
        "f_ct_dim":  1,
        "w_ct_fcn":  [1.0],
        "pen_ct_fcn": [penalty_fcn_list[2]],
        "name_ct_fcn": [['until_pos_spd', 'spd', 'none'][0]],

        # physical inertia (kg·m²)
        "I_x": 7e-2,
        "I_y": 7e-2,
        "I_z": 1.27e-1,

        # vehicle limits
        "vehicle_v_max": 20.0,
        "spd_lim":       2.0,
        "p_w":           np.array([2.5, -7.5, 0.0]),
        "r_w":           0.1,

        # vehicle limits
        "max_ang": max_ang,
        "max_ang_rate": max_ang_rate,
        "max_ang_rate_yaw": max_ang_rate_yaw,

        # converted to radians
        "phi_bd": np.deg2rad(max_ang),
        "theta_bd": np.deg2rad(max_ang),
        "phi_rate": np.deg2rad(max_ang_rate),
        "theta_rate": np.deg2rad(max_ang_rate),
        "yaw_rate": np.deg2rad(max_ang_rate_yaw),

        # altitude limits
        "add_min_alt": False,
        "add_max_alt": False,
        "min_alt": -5.0,
        "max_alt": 10.0,

        "yaw_fixed": False,
        "yaw_fixed_all": False,
        "yaw_fx_deg": 0.0,

        # state‐weight penalties
        "w_states_spd":  100.0,
        "w_states_alt":  100.0,

        "w_states_phi":  100.0,
        "w_states_tht":  100.0,

        "w_states_p":    100.0,
        "w_states_q":    100.0,
        "w_states_r":    100.0,

        # optimization settings
        "ite": 30,
        "w_con_dyn": 1e3,
        "w_con_stt": 1e2,

        "adaptive_step": True,
        "ptr_term": 1e-4,
        "w_ds": 10.0,
        "w_ptr": 100.0,
        "w_ptr_min": 1e-3,
        "r0": 0.01,
        "r1": 0.1,
        "r2": 0.9,

        # solver settings
        "generate_code": False,
        "use_generated_code": True,
        "convex_solver": "QOCO",  # or "ECOS", "CLARABEL", "MOSEK", etc.

        # integrator
        "rk4_steps": 10,

        # scaling
        "scale_fac": 10.0,

        # plotting
        "save_fig": True,
        "fig_format": "pdf",
        "fig_png_dpi": 600,
    }

def scale_params(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    """
    Scale certain parameters (positions, inputs, moments, etc.) by params['scale_fac'].
    """
    sc = params["scale_fac"]

    # divide‐by‐scale parameters
    to_divide = [
        "g_0", "T_max", "T_min", "vehicle_v_max",
        "spd_lim", "p_w", "r_w", "min_alt", "max_alt"
    ]
    for key in to_divide:
        params[key] /= sc

    # array‐specific slices
    params["x_init"][:6]   /= sc
    params["x_final"][:6]  /= sc
    params["X_last"][:6, :] /= sc

    params["u_init"][3]   /= sc
    params["u_final"][3]  /= sc
    params["U_last"][3, :] /= sc

    # multiply‐by‐scale parameters
    to_multiply = ["I_x", "I_y", "I_z", "tau_max"]
    for key in to_multiply:
        params[key] *= sc

    return params

def unscale_prox_results(
    prox_results: T.Dict[str, np.ndarray],
    params: T.Dict[str, T.Any]
) -> T.Dict[str, np.ndarray]:
    """
    Undo the scaling on the primal variables returned by the prox solver.
    """
    sc = params["scale_fac"]

    prox_results["X_new"][:6, :] *= sc
    prox_results["U_new"][3, :]  *= sc
    prox_results["U_new"][:3, :] /= sc

    return prox_results
