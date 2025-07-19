from jax import config, jit, jacfwd
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import typing as T
from utils import rotation_matrix

def dynamics(params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    """
    JIT-compiled dynamics and its Jacobians:
      - f_func(x,u):   state derivative
      - A_func(x,u):   ∂f/∂x
      - B_func(x,u):   ∂f/∂u
    """

    def f(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Quadrotor continuous‐time dynamics plus CTCS penalty.
        State x = [pos(3), vel(3), euler(3), omega(3), CTCS]
        Input u = [τ₁, τ₂, τ₃, thrust]
        """

        rot = rotation_matrix(x[6:9])
        
        dx = jnp.zeros_like(x)
        dx = dx.at[0:3].set(x[3:6])  # velocities

        # Rotation application for force in body z direction
        dx = dx.at[3].set(u[3] * rot[0, 2])
        dx = dx.at[4].set(u[3] * rot[1, 2])
        dx = dx.at[5].set(u[3] * rot[2, 2] - params['g_0'])

        # Angular velocities
        dx = dx.at[6].set(
            x[9] +
            jnp.sin(x[6]) * jnp.tan(x[7]) * x[10] +
            jnp.cos(x[6]) * jnp.tan(x[7]) * x[11]
        )
        dx = dx.at[7].set(jnp.cos(x[6]) * x[10] - jnp.sin(x[6]) * x[11])
        dx = dx.at[8].set(
            jnp.sin(x[6]) / jnp.cos(x[7]) * x[10] +
            jnp.cos(x[6]) / jnp.cos(x[7]) * x[11]
        )

        # Torques
        dx = dx.at[9].set((params['I_y'] - params['I_z']) / params['I_x'] * x[10] * x[11] + u[0] / params['I_x'])
        dx = dx.at[10].set((params['I_z'] - params['I_x']) / params['I_y'] * x[9] * x[11] + u[1] / params['I_y'])
        dx = dx.at[11].set((params['I_x'] - params['I_y']) / params['I_z'] * x[9] * x[10] + u[2] / params['I_z'])

        dx = dx.at[12].set(0)
        
        spd = jnp.linalg.norm(x[3:6] + 1e-8)
        dx = dx.at[12].add( params['w_states_spd'] * jnp.maximum(0, spd - params['vehicle_v_max'])**2 )

        dx = dx.at[12].add( params['w_states_alt'] * jnp.maximum(0, x[2] - params['max_alt'])**2 )
        dx = dx.at[12].add( params['w_states_alt'] * jnp.maximum(0, -x[2] + params['min_alt'])**2 )
        
        dx = dx.at[12].add( params['w_states_phi'] * jnp.maximum(0,  x[6] - params['phi_bd'])**2 )
        dx = dx.at[12].add( params['w_states_phi'] * jnp.maximum(0, -x[6] - params['phi_bd'])**2 )

        dx = dx.at[12].add( params['w_states_tht'] * jnp.maximum(0,  x[7] - params['theta_bd'])**2 )
        dx = dx.at[12].add( params['w_states_tht'] * jnp.maximum(0, -x[7] - params['theta_bd'])**2 )

        dx = dx.at[12].add( params['w_states_p'] * jnp.maximum(0,  x[9] - params['phi_rate'])**2 )
        dx = dx.at[12].add( params['w_states_p'] * jnp.maximum(0, -x[9] - params['phi_rate'])**2 )

        dx = dx.at[12].add( params['w_states_q'] * jnp.maximum(0,  x[10] - params['theta_rate'])**2 )
        dx = dx.at[12].add( params['w_states_q'] * jnp.maximum(0, -x[10] - params['theta_rate'])**2 )

        dx = dx.at[12].add( params['w_states_r'] * jnp.maximum(0, x[11] - params['yaw_rate'])**2 )
        dx = dx.at[12].add( params['w_states_r'] * jnp.maximum(0, -x[11] - params['yaw_rate'])**2 )

        ## - CTCS Updates Params: x_init and n_x + Dyn + CTCS Cons + f_obj

        return dx

    params['f_func'] = jit(f)
    params['A_func'] = jit(jacfwd(f, argnums=0))
    params['B_func'] = jit(jacfwd(f, argnums=1))

    return params
