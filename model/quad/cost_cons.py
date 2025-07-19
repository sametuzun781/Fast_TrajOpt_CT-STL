from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import typing as T
import numpy as np
import cvxpy as cp
from utils import dict_append
from stl import conjunction, disjunction, UNTIL

def cvx_cost_fcn(
    X:    np.ndarray,
    U:    np.ndarray,
    S:    np.ndarray,
    nu:   np.ndarray,
    params: T.Dict[str, T.Any],
    npy: bool,
    cost_dict: T.Dict[str, T.Any] = None
) -> T.Union[float, T.Tuple[T.Any, T.Dict[str, T.Any]]]:
    
    # --------------------------------------------------------------------------------------------------------------
    
    convex_vehicle_cost = 0.
    if params['free_final_time']:
        if params['time_dil']:
            if npy:
                S_cost = params['w_S'] * np.linalg.norm(S[0, :], 1) / params['t_f']
            else:
                S_cost = params['w_S'] * cp.pnorm(S[0, :], 1) / params['t_f']
        else:
            if npy:
                S_cost = params['w_S'] * S / params['t_f']
            else:
                S_cost = params['w_S'] * cp.pnorm(S, 1) / params['t_f']

        convex_vehicle_cost += S_cost

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            cost_dict = dict_append(cost_dict, 'S_cost', S_cost)

    # --------------------------------------------------------------------------------------------------------------

    if params['add_inp']:
        if npy:
            input_cost = ( params['w_inp'] * np.sum(np.square(U[3,:])))
        else:
            input_cost = ( params['w_inp'] * cp.sum(cp.square(U[3,:])))

        convex_vehicle_cost += input_cost

    if params['add_inp_trq']:
        input_cost = 0
        if npy:
            input_cost += ( params['w_inp_trq'] * np.sum(np.square(U[0,:])))
            input_cost += ( params['w_inp_trq'] * np.sum(np.square(U[1,:])))
            input_cost += ( params['w_inp_trq'] * np.sum(np.square(U[2,:])))
        else:
            input_cost += ( params['w_inp_trq'] * cp.sum(cp.square(U[0,:])))
            input_cost += ( params['w_inp_trq'] * cp.sum(cp.square(U[1,:])))
            input_cost += ( params['w_inp_trq'] * cp.sum(cp.square(U[2,:])))

        convex_vehicle_cost += input_cost

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            cost_dict = dict_append(cost_dict, 'input_cost', input_cost)

    # --------------------------------------------------------------------------------------------------------------
    
    # Dont forget to update linear/nonlinear cost functions
    if params['add_elv_rate']:
        if npy:
            elv_rate_cost = ( params['w_elv_r'] * np.sum(np.square(X[5,:])))
        else:
            elv_rate_cost = ( params['w_elv_r'] * cp.sum(cp.square(X[5,:])))

        convex_vehicle_cost += elv_rate_cost

        if cost_dict is not None: # We compute the nonlin cost; otherwise lin
            cost_dict = dict_append(cost_dict, 'elv_rate_cost', elv_rate_cost)

    # --------------------------------------------------------------------------------------------------------------

    if cost_dict is not None: # We compute the nonlin cost; otherwise lin
        dyn_cost = params['w_con_dyn'] * np.linalg.norm(nu[0:12, :].reshape(-1), 1)
        state_cost = params['w_con_stt'] * np.linalg.norm(nu[12, :].reshape(-1), 1)
        cost_dict = dict_append(cost_dict, 'dyn_cost', dyn_cost)
        cost_dict = dict_append(cost_dict, 'state_cost', state_cost)

    # --------------------------------------------------------------------------------------------------------------

    if cost_dict is not None: # We compute the nonlin cost; otherwise lin
        return convex_vehicle_cost, cost_dict
    else:
        return convex_vehicle_cost

def cvx_cons_fcn(
    X:      np.ndarray,
    U:      np.ndarray,
    S:      np.ndarray,
    params: T.Dict[str, T.Any],
) -> T.List:
    
    vehicle_cons = []

    # --------------------------------------------------------------------------------------------------------------

    if params['free_final_time']:
        if params['time_dil']:
            if params['add_minmax_sig']:
                vehicle_cons += [
                        S[0, :] >= params['min_S'], 
                        S[0, :] <= params['max_S'], 
                    ]
            else:
                vehicle_cons += [
                        S[0, :] >= 1e-4, 
                    ]

    # --------------------------------------------------------------------------------------------------------------

    # CTCS
    n_states = 12
    vehicle_cons += [X[n_states, 0] == 0.]
    vehicle_cons += [X[n_states, 1:] - X[n_states, 0:-1] <= 1e-4]

    # --------------------------------------------------------------------------------------------------------------

    # Quadrotor

    vehicle_cons += [
        X[0:8, 0] == params['x_init'][0:8],
        X[0:8, -1] == params['x_final'][0:8],
        X[9:12, 0] == params['x_init'][9:12],
        X[9:12, -1] == params['x_final'][9:12],
    ]

    # Input conditions:
    vehicle_cons += [
        U[:, 0] == params['u_init'],
        U[:, -1] == params['u_final'],
        U[3, :] <= params['T_max'],
        U[3, :] >= params['T_min'],
        cp.abs(U[0:3, :]) <= params['tau_max'],
    ]

    # --------------------------------------------------------------------------------------------------------------
    
    return vehicle_cons

def ncvx_dt_fcn(
    X_last: np.ndarray,
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:

    z = jnp.sum(jnp.array(0.0) * X_last)
    if 'spd' in params['name_dt_fcn']:

        speeds = jnp.linalg.norm(X_last[3:6, :] + 1e-6, axis=0)
        spd_cost = speeds - params['spd_lim']
        # z += jnp.sum(jnp.maximum(0.0, spd_cost)**2)
        z += jnp.sum(jnp.sqrt(jnp.maximum(0.0, spd_cost)**2 + 1e-4) - jnp.sqrt(1e-4))

    return jnp.atleast_1d([z]) 

def ncvx_ct_fcn(
    X_CT:   np.ndarray,
    params: T.Dict[str, T.Any],
) -> jnp.ndarray:
    
    X = X_CT.reshape((params['n_x'], -1))

    speeds = jnp.linalg.norm(X[3:6, :] + 1e-8, axis=0)
    spd_cost = speeds - params['spd_lim']

    z = jnp.sum(jnp.array(0.0) * X)
    if 'until_pos_spd' in params['name_ct_fcn']:

        pos = jnp.linalg.norm(X[0:3, :] - params['p_w'][:, None] + 1e-8, axis=0) - params['r_w']

        # z += -UNTIL(c=1e-1, 
        #                 p=1, 
        #                 w_f=jnp.ones_like(speeds), 
        #                 w_g=jnp.ones_like(pos), 
        #                 w_fg=np.ones(2), 
        #                 f=-100*spd_cost, 
        #                 g=-10*pos)

        # z += -UNTIL(c=0.00005, 
        #                 p=1, 
        #                 w_f=jnp.ones_like(speeds), 
        #                 w_g=jnp.ones_like(pos), 
        #                 w_fg=np.ones(2), 
        #                 f=-10*spd_cost, 
        #                 g=-1*pos)
        
        z += -UNTIL(c=0.005, 
                        p=1, 
                        w_f=jnp.ones_like(speeds), 
                        w_g=jnp.ones_like(pos), 
                        w_fg=np.ones(2), 
                        f=-10*spd_cost, 
                        g=-1*pos)


        # z += -UNTIL(c=0.01, 
        #                 p=1, 
        #                 w_f=jnp.ones_like(speeds), 
        #                 w_g=jnp.ones_like(pos), 
        #                 w_fg=np.ones(2), 
        #                 f=-100/5*spd_cost, 
        #                 g=-10/5*pos)
        
            
    elif 'spd' in params['name_ct_fcn']:

        # z += jnp.sum(jnp.maximum(0.0, speeds - params['spd_lim'])**2)

        z += jnp.sum(jnp.sqrt(jnp.maximum(0.0, spd_cost)**2 + 1e-4) - jnp.sqrt(1e-4)) # / ((params['K'] - 1) * params['rk4_steps'])

        # w = np.ones(((params['K'] - 1) * params['rk4_steps']))
        # z += -conjunction(1e-6, 1, w, -spd_cost)

        # # Since start and the end of the traj have the half importance 
        # w = np.ones(((params['K'] - 1) * params['rk4_steps'])) * 2
        # w[0] = 1
        # w[-1] = 1
        # z += -conjunction(1e-6, 1, w, -10*spd_cost)

    return jnp.atleast_1d([z])
