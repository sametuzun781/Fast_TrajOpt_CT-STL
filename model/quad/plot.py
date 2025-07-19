import os
import numpy as np
from tqdm import tqdm

import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

mpl.rcParams['figure.dpi'] = 120 
mpl.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']}) ## for Palatino and other serif fonts use: #rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:red', 'k', 'tab:green', 'tab:blue', 'tab:grey'])
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'legend.fontsize': 11})
mpl.rcParams.update({'axes.xmargin': 0})
mpl.rcParams.update({'lines.solid_capstyle': 'round'})
mpl.rcParams.update({'lines.solid_joinstyle': 'round'})
mpl.rcParams.update({'lines.dash_capstyle': 'round'})
mpl.rcParams.update({'lines.dash_joinstyle': 'round'})
mpl.rcParams.update({'text.latex.preamble': r"\usepackage{bm}"})

def plt_drone_fcn(ax, center, z_dir, length_drone, head_angle):
    
    def cyl(ax, p0, p1, rad_drone, clr=None, clr2=None):

        # Vector in direction of axis
        v = p1 - p0

        # Find magnitude of vector
        mag = np.linalg.norm(v + 1e-6)

        # Unit vector in direction of axis
        v = v / mag

        # Make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])

        # Make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= np.linalg.norm(n1 + 1e-6)

        # Make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)

        # Surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, 100)
        rsample = np.linspace(0, rad_drone, 2)

        # Use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)

        rsample, theta = np.meshgrid(rsample, theta)

        # Generate coordinates for surface
        # "Tube"
        X, Y, Z = [p0[i] + v[i] * t + rad_drone * np.sin(theta2) * n1[i] 
                   + rad_drone * np.cos(theta2) * n2[i] for i in [0, 1, 2]]
        # "Bottom"
        X2, Y2, Z2 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] 
                      + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        # "Top"
        X3, Y3, Z3 = [p0[i] + v[i]*mag + rsample[i] * np.sin(theta) * n1[i] 
                      + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]

        ax.plot_surface(X, Y, Z, color=clr, zorder=9)
        ax.plot_surface(X2, Y2, Z2, color=clr, zorder=9)
        ax.plot_surface(X3, Y3, Z3, color=clr, zorder=9)

        if clr2:
            phi = np.linspace(0,2*np.pi, 50)
            theta = np.linspace(0, np.pi, 25)

            dx = 3 * rad_drone*np.outer(np.cos(phi), np.sin(theta))
            dy = 3 * rad_drone*np.outer(np.sin(phi), np.sin(theta))
            dz = 3 * rad_drone*np.outer(np.ones(np.size(phi)), np.cos(theta))

            ax.plot_surface(p1[0]+dx, p1[1]+dy, p1[2]+dz, 
                            cstride=1, rstride=1, color=clr2, zorder=10)
            ax.plot_surface(p0[0]+dx, p0[1]+dy, p0[2]+dz, 
                            cstride=1, rstride=1, color=clr2, zorder=10)

    # Rodrigues' rotation formula
    def rotate_vec(v, d, alpha):
        ada = v * np.cos(alpha) + np.cross(d, v) * np.sin(alpha) + d * np.dot(d, v) * (1 - np.cos(alpha))
        return ada

    z_dir = z_dir / np.linalg.norm(z_dir + 1e-6)
    rad_drone = length_drone * 0.02

    l1_axis = rotate_vec(head_angle, z_dir, np.pi/4)
    p0 = center - l1_axis * length_drone/2
    p1 = center + l1_axis * length_drone/2

    l2_axis = rotate_vec(l1_axis, z_dir, np.pi/2)
    p2 = center - l2_axis * length_drone/2
    p3 = center + l2_axis * length_drone/2

    # Body
    cyl(ax, p0, p1, rad_drone, clr='black', clr2='yellow')
    cyl(ax, p2, p3, rad_drone, clr='black', clr2='yellow')

    # Head
    p6 = center
    p7 = center + head_angle * length_drone/4
    cyl(ax, p6, p7, rad_drone/1.5, clr='gray')

    p8 = center
    p9 = center + z_dir * length_drone/2
    
    # Trust
    cyl(ax, p8, p9, rad_drone*0.8, clr='red')

def plot(results, params):

    xy_fs = 15
    scatter_sc = 15
    c_up_lim = 'green'
    c_low_lim = 'purple'
    c_trig = 'red'
    c_plt = 'blue'

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d',computed_zorder=False)

    spd_norm = np.linalg.norm(results['x_all'][:,3:6], axis=1)

    ax.scatter(results['x_nmpc_all'][0, :, 0], results['x_nmpc_all'][0, :, 1], 
               results['x_nmpc_all'][0, :, 2], c='black', zorder=11, s=20)
    
    points = np.array([results['x_all'][:,0], results['x_all'][:,1], results['x_all'][:,2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, 
                        cmap=plt.cm.rainbow, 
                        norm=plt.Normalize(0., spd_norm.max()), 
                        array=spd_norm, 
                        lw=3,
                        zorder=-2,
                        )

    ax.add_collection(lc)
    cbar = fig.colorbar(lc, aspect=50, pad=0.0, shrink=0.75, orientation='horizontal')
    cbar.set_label("Speed [m s$^{-1}$]", fontsize=xy_fs, labelpad=5)

    # Plot the quadrotor
    # for k in np.linspace(0, results['x_all'].shape[0]-1, 12, dtype=int):

    #     z_dir = ((rotation_matrix(results['x_all'][k, 6:9].copy()).reshape(3,3)) @ np.array([[0.], [0.], [1.]]))[:,0]
    #     head_angle = ((rotation_matrix(results['x_all'][k, 6:9].copy()).reshape(3,3)) @ np.array([[1.], [0.], [0.]]))[:,0]

    #     plt_drone_fcn(ax = ax,
    #                   center = results['x_all'][k, 0:3],
    #                   z_dir = z_dir,
    #                   length_drone = 0.4,
    #                   head_angle = head_angle )

    # Plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    xy_fs = 15
    ax.set_xlabel('$r^x$ [m]', size=xy_fs)
    ax.set_ylabel('$r^y$ [m]', size=xy_fs)
    ax.set_zlabel('$r^z$ [m]', size=xy_fs)

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 11
    ax.zaxis.labelpad = 0.5

    xy_fs = 14
    ax.xaxis.set_tick_params(labelsize=xy_fs)
    ax.yaxis.set_tick_params(labelsize=xy_fs)
    ax.zaxis.set_tick_params(labelsize=xy_fs)
    ax.set_zticks(np.arange(0, 2.5, 0.5))

    ax.set_xlim(-10.2, 10.2)
    ax.set_ylim(-10.2, 10.2)
    ax.set_zlim(-0.2, 2.2)

    # ax.view_init(10, 220)
    ax.view_init(90, 270)
    ax.set_aspect('equal')
    if params['save_fig']: fig.savefig('qf_pos.' + params['fig_format'], bbox_inches='tight', dpi=params['fig_png_dpi'])

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    fig, axs = plt.subplots(4, 4, gridspec_kw={'height_ratios': [3, 3, 3, 0.001]}, figsize=(9, 5))

    for ax in axs[-1, :]:
        ax.remove()

    # ---------------------------------------------------------------------------------
    
    axa = axs[0,0]
    x_time = np.dstack((results['times_all'][:-1], results['times_all'][1:])).reshape(-1)
    axa.plot(x_time, results['u_all'][:,3,:].reshape(-1, order='F'), c='blue')
    axa.axhline(y=params['T_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('Thrust, $T$ [N]', fontsize=xy_fs)

    axa = axs[0,1]
    x_time = np.dstack((results['times_all'][:-1], results['times_all'][1:])).reshape(-1)
    axa.plot(x_time, results['u_all'][:,0,:].reshape(-1, order='F'), c='blue')
    axa.axhline(y=params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.axhline(y=-params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\tau_x$ [N m]', fontsize=xy_fs)

    axa = axs[0,2]
    x_time = np.dstack((results['times_all'][:-1], results['times_all'][1:])).reshape(-1)
    axa.plot(x_time, results['u_all'][:,1,:].reshape(-1, order='F'), c='blue')
    axa.axhline(y=params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.axhline(y=-params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\tau_z$ [N m]', fontsize=xy_fs)

    axa = axs[0,3]
    x_time = np.dstack((results['times_all'][:-1], results['times_all'][1:])).reshape(-1)
    axa.plot(x_time, results['u_all'][:,2,:].reshape(-1, order='F'), c='blue')
    axa.axhline(y=params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.axhline(y=-params['tau_max'], c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\tau_z$ [N m]', fontsize=xy_fs)
    
    # ---------------------------------------------------------------------------------

    axa = axs[1,0]
    spd_norm = np.linalg.norm(results['x_all'][:,3:6], axis=1)
    axa.plot(results['times_all'], spd_norm, c=c_plt)
    spd_node = np.linalg.norm(results['x_nmpc_all'][0, :, 3:6], axis=1)
    axa.scatter(results['times_nodes'], 
                spd_node, s=scatter_sc, c='black')
    axa.axhline(y=params['vehicle_v_max'], c=c_up_lim, linestyle='dashed')
    axa.set_xlabel(r'time', fontsize=xy_fs)
    axa.set_ylabel('Speed, $v$ [m/s]', fontsize=xy_fs)

    axa = axs[1,1]
    axa.plot(results['times_all'], results['x_all'][:,6].reshape(-1, order='F')*180/np.pi, c='blue')
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 6]*180/np.pi, s=scatter_sc, c='black')
    axa.axhline(y=-params['phi_rate']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.axhline(y= params['phi_rate']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$p$ [deg s$^{-2}$]', fontsize=xy_fs)

    axa = axs[1,2]
    axa.plot(results['times_all'], results['x_all'][:,7].reshape(-1, order='F')*180/np.pi, c='blue')
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 7]*180/np.pi, s=scatter_sc, c='black')
    axa.axhline(y=-params['theta_rate']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.axhline(y= params['theta_rate']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$q$ [deg s$^{-2}$]', fontsize=xy_fs)

    axa = axs[1,3]
    axa.plot(results['times_all'], results['x_all'][:,8].reshape(-1, order='F')*180/np.pi, c='blue')
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 8]*180/np.pi, s=scatter_sc, c='black')
    axa.axhline(y=-params['yaw_rate']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.axhline(y= params['yaw_rate']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$r$ [deg s$^{-2}$]', fontsize=xy_fs)
    
    # ---------------------------------------------------------------------------------

    axa = axs[2,0]
    axa.plot(results['times_all'], results['x_all'][:,2], c=c_plt)
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 2], s=scatter_sc, c='black')
    axa.axhline(y=params['min_alt'], c=c_up_lim, linestyle='dashed')
    if params['add_max_alt']:
        axa.axhline(y=params['max_alt'], c='purple', linestyle='dashed')
    if params['add_min_alt']:
        axa.axhline(y=params['min_alt'], c='purple', linestyle='dashed')
    axa.set_xlabel(r'time', fontsize=xy_fs)
    axa.set_ylabel('Altitude, $z$ [m]', fontsize=xy_fs)

    axa = axs[2,1]
    axa.plot(results['times_all'], results['x_all'][:,6].reshape(-1, order='F')*180/np.pi, c='blue')
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 6]*180/np.pi, s=scatter_sc, c='black')
    axa.axhline(y=-params['phi_bd']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.axhline(y= params['phi_bd']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\phi$ [deg]', fontsize=xy_fs)

    axa = axs[2,2]
    axa.plot(results['times_all'], results['x_all'][:,7].reshape(-1, order='F')*180/np.pi, c='blue')
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 7]*180/np.pi, s=scatter_sc, c='black')
    axa.axhline(y=-params['theta_bd']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.axhline(y= params['theta_bd']*180/np.pi, c=c_up_lim, linestyle='dashed')
    axa.set_ylabel('$\\theta$ [deg]', fontsize=xy_fs)

    axa = axs[2,3]
    axa.plot(results['times_all'], results['x_all'][:,8].reshape(-1, order='F')*180/np.pi, c='blue')
    axa.scatter(results['times_nodes'], 
                results['x_nmpc_all'][0, :, 8]*180/np.pi, s=scatter_sc, c='black')
    axa.set_ylabel('$\\psi$ [deg]', fontsize=xy_fs)

    # ---------------------------------------------------------------------------------

    if params['save_fig']: fig.savefig('qf_oth.' + params['fig_format'], bbox_inches='tight', dpi=params['fig_png_dpi'])

def animation(results, params, delt=2):
    
    newpath = r'sim' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    K_in = results['x_all'].shape[0]

    temp_1 = np.zeros((3, K_in))
    temp_1[:,:] = None
    temp_2 = np.zeros((K_in, 1))
    temp_2[:,:] = None
    temp_3 = np.zeros((K_in, 1))
    temp_3[:,:] = None
    temp_5 = np.zeros((K_in, 1))
    temp_5[:,:] = None
    temp_6 = np.zeros((K_in, 1))
    temp_6[:,:] = None
    temp_7 = np.zeros((K_in, 1))
    temp_7[:,:] = None
    temp_8 = np.zeros((K_in, 1))
    temp_8[:,:] = None

    lat_s = 12
    fig_gap_3d = 5
    xyz_lim_min = results['x_all'][:, 0:3].min(axis=0)
    xyz_lim_max = results['x_all'][:, 0:3].max(axis=0)

    xyz_lim_min = xyz_lim_min - fig_gap_3d
    xyz_lim_max = xyz_lim_max + fig_gap_3d

    if params['add_min_alt']:
        xyz_lim_min[2] = params['min_alt'] - fig_gap_3d
    if params['add_max_alt']:
        xyz_lim_max[2] = params['max_alt'] + fig_gap_3d

    lenght_drone_scale = np.linalg.norm(xyz_lim_max - xyz_lim_min)
    length_drone =  0.05 * lenght_drone_scale

    for k in tqdm(range(0, K_in, delt)):
        
        temp_1[:, :k+1] = results['x_all'][:k+1, 6:9].T*180/np.pi
        temp_2[:k+1, 0] = np.linalg.norm(results['x_all'][:k+1, 3:6], axis=1)
        temp_3[:k+1, 0] = results['x_all'][:k+1, 2]

        k_max = min(results['u_all'][0, 0, :].shape[0], k+1)
        temp_5[:k_max, 0] = results['u_all'][0, 3, :k_max]
        temp_6[:k_max, 0] = results['u_all'][0, 0, :k_max]
        temp_7[:k_max, 0] = results['u_all'][0, 1, :k_max]
        temp_8[:k_max, 0] = results['u_all'][0, 2, :k_max]
    
        fig = plt.figure(figsize=(24, 21.6))

        gs1 = gridspec.GridSpec(nrows=6, ncols=6, left=0.01, 
                        right=0.99, top=0.95, bottom=0.05)

        plots = [gs1[3:6, 0:4], 
                    gs1[0:3, 0:4], 
                    gs1[0, 4], gs1[1, 4], 
                    gs1[2, 4], gs1[3, 4], 
                    gs1[4, 4], gs1[5, 4], 
                    gs1[0, 5], gs1[1, 5], 
                    gs1[2, 5], gs1[3, 5], 
                    gs1[4, 5], gs1[5, 5]
                    ]
                                
        cts = 0
        for ada_plt in plots:
            
            if cts < 2:
                if cts == 0:
                    ax = fig.add_axes([0.02, -0.02, 
                                       0.72,  0.49], projection='3d')
                else:
                    ax = fig.add_subplot(ada_plt, projection='3d')

                ax.plot(results['x_all'][0, 0],  results['x_all'][0, 1],   results['x_all'][0, 2], 
                        marker = 'X', markersize=10,  c='yellow', label='Start Point', zorder=10)
                
                ax.plot(results['x_all'][-1, 0], results['x_all'][-1, 1],  results['x_all'][-1, 2], 
                        marker = 'X', markersize=10,  c='limegreen', label='End Point', zorder=10)

                ax.plot(results['x_all'][:k+1, 0], results['x_all'][:k+1, 1], results['x_all'][:k+1, 2], 
                        c='orange', lw=2, label='Optimal Trajectory', zorder=10)

                z_dir = ((rotation_matrix(results['x_all'][k, 6:9].copy()).reshape(3,3)) @ np.array([[0.], [0.], [1.]]))[:,0]
                head_angle = ((rotation_matrix(results['x_all'][k, 6:9].copy()).reshape(3,3)) @ np.array([[1.], [0.], [0.]]))[:,0]

                plt_drone_fcn(ax = ax,
                                center = results['x_all'][k, 0:3],
                                z_dir = z_dir,
                                length_drone = length_drone,
                                head_angle = head_angle )
                
                ax.set_aspect('equal')
                if cts == 0:
                    ax.view_init(elev=90, azim=-90)
                    ax.set_zticks([])
                elif cts == 1:
                    ax.legend(prop = { "size": 15 })
                    ax.view_init(elev=30, azim=215)

                sz = 13
                ax.set_xlabel('X [m]', size=sz)
                ax.set_ylabel('Y [m]', size=sz)
                ax.set_zlabel('Z [m]', size=sz)

                ax.set_xlim(xyz_lim_min[0], xyz_lim_max[0])
                ax.set_ylim(xyz_lim_min[1], xyz_lim_max[1])
                ax.set_zlim(xyz_lim_min[2], xyz_lim_max[2])

            if cts == 2:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_3, color='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('Altitude [$m$]', size=lat_s, c='black')
                if params['add_max_alt']:
                    ax.axhline(y = params['max_alt'] - params['quad_radius'], color = 'purple', label='Max Altitude', linestyle = '-')
                else:
                    ax.axhline(y = 0., color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 3:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_2, c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('Speed [$m/s$]', size=lat_s, c='black')
                ax.axhline(y = params['vehicle_v_max'], color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 4:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_1[0,:], c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('Roll angle (deg)', size=lat_s, c='black')
                ax.axhline(y =  params['phi_bd']*180/np.pi, color = 'r', linestyle = '-')
                ax.axhline(y = -params['phi_bd']*180/np.pi, color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 5:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_1[1,:], c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('Pitch angle (deg)', size=lat_s, c='black')
                ax.axhline(y =  params['theta_bd']*180/np.pi, color = 'r', linestyle = '-')
                ax.axhline(y = -params['theta_bd']*180/np.pi, color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 6:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_1[2,:], c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('Yaw angle (deg)', size=lat_s, c='black')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 9:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_5, c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('Thrust, $T$ [N]', size=lat_s, c='black', labelpad=-2)
                ax.axhline(y=params['T_max'], color = 'r', linestyle = '-')
                ax.axhline(y=params['T_min'], color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 10:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_6, c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('$\\tau_x$ [N m]', size=lat_s, c='black', labelpad=-6)
                ax.axhline(y=params['tau_max'], color = 'r', linestyle = '-')
                ax.axhline(y=-params['tau_max'], color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 11:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_7, c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('$\\tau_y$ [N m]', size=lat_s, c='black', labelpad=-6)
                ax.axhline(y=params['tau_max'], color = 'r', linestyle = '-')
                ax.axhline(y=-params['tau_max'], color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            if cts == 12:
                ax = fig.add_subplot(ada_plt)
                ax.plot(results['times_all'], temp_8, c='blue')
                ax.set_xlabel('time [$s$]', size=lat_s, c='black')
                ax.set_ylabel('$\\tau_z$ [N m]', size=lat_s, c='black', labelpad=-6)
                ax.axhline(y=params['tau_max'], color = 'r', linestyle = '-')
                ax.axhline(y=-params['tau_max'], color = 'r', linestyle = '-')
                ax.set_xlim(results['times_all'][0], results['times_all'][-1])
                ax.grid(color='black', linestyle='-', linewidth=0.5)

            cts = cts + 1

        kf = str(k).zfill(4)
        fig.savefig('sim/anim_' + kf)
        plt.close(fig)
