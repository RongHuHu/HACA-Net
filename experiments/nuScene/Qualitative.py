#%matplotlib inline
import sys
sys.path.append('../../Transformer')
import os
import numpy as np
import torch
import dill
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
import visualization

nuScenes_data_path = "../../data/v1.0"
nuScenes_devkit_path = './devkit/python-sdk/'
sys.path.append(nuScenes_devkit_path)
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name='boston-seaport')
bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')

line_colors = ['#375397','#80CBE5','#ABCB51','#F05F78', '#C8B0B0']
#line_colors = ['#56B4E9', '#E69F00', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']
#line_colors = ['r','g','c','m','y','k','gold','lime']

with open('../data1/nuScenes_test_map_full.pkl', 'rb') as f:
    eval_env = dill.load(f, encoding='latin1')
eval_scenes = eval_env.scenes

ph = 6
log_dir = './models'

model_dir = os.path.join(log_dir, '/data/lirh/experiments/nuSc/best/his_map_robot/his4-pre8/his_map_robot_car') 
# model_dir = os.path.join(log_dir, '/data/lirh/trajectron/vel_ee') 
eval_stg, hyp = load_model(model_dir, eval_env, ts=12)

model_dir2 = os.path.join(log_dir, '/data/lirh/experiments/nuSc/best/his_map_robot/his4-pre8/his_map_robot_car') 
# model_dir = os.path.join(log_dir, '/data/lirh/trajectron/vel_ee') 
eval_stg2, hyp2 = load_model(model_dir2, eval_env, ts=2)

scene = eval_scenes[16]
# for i in range(26): 
#     print('scenes[',i,']x_min',eval_scenes[i].x_min)
#     print('scenes[',i,']x_max',eval_scenes[i].x_max)
#     print('scenes[',i,']y_min',eval_scenes[i].y_min)
#     print('scenes[',i,']y_max',eval_scenes[i].y_max)

scene.name

# Define ROI in nuScenes Map
#scene20
# x_min = 472.0
# x_max = 669.0
# y_min = 1447.0
# y_max = 1613.0
#scene25
# x_min = 773.0
# x_max = 1100.0
# y_min = 1231.0
# y_max = 1510.0
#scene0
# x_min = 138.0
# x_max = 355.0
# y_min = 809.0
# y_max = 997.0
#scene19
# x_min = 540.0
# x_max = 826.0
# y_min = 1351.0
# y_max = 1567.0
#scene18
# x_min = 848.0
# x_max = 1219.0
# y_min = 1080.0
# y_max = 1375.0
#scene17
# x_min = 1044.0
# x_max = 1329.0
# y_min = 1009.0
# y_max = 1255.0
#scene16
x_min = 1131.0
x_max = 1416.0
y_min = 933.0
y_max = 1238.0

layers = ['drivable_area',
          'road_segment',
          'lane',
          'ped_crossing',
          'walkway',
          'stop_line',
          'road_divider',
          'lane_divider']

ph = 6
with torch.no_grad():
    timesteps = np.array([2])
    # predictions, _ = eval_stg.predict(scene,
    #                                    timestep,
    #                                    ph)
    predictions_g = eval_stg.predict(scene,
                                    timesteps, 
                                    ph,
                                    num_samples=250)

    predictions_mm_g = eval_stg.predict(scene,
                                      timesteps,
                                      ph,
                                      num_samples=1,
                                      gmm_mode=True)
    predictions = eval_stg2.predict(scene,
                                    timesteps, 
                                    ph,
                                    num_samples=250)

    predictions_mm = eval_stg2.predict(scene,
                                      timesteps,
                                      ph,
                                      num_samples=1,
                                      gmm_mode=True)                 


    # Plot predicted timestep for random scene in map
    my_patch = (x_min, y_min, x_max, y_max)
    #fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.1, render_egoposes_range=False)
    #my_patch = (440, 1320, 530, 1400)
    fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.15, render_egoposes_range=False)


    # ax.plot([], [], 'ko-',
    #         zorder=620,
    #         markersize=4,
    #         linewidth=2, alpha=0.7, label='Ours (MM)')

    # ax.plot([],
    #         [],
    #         'w--o', label='Ground Truth',
    #         linewidth=3,
    #         path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        
    
    plot_vehicle_nice(ax,
                      predictions,
                      scene.dt,
                      max_hl=4,
                      ph=ph,
                      map=None, x_min=x_min, y_min=y_min)

#     plot_vehicle_mm(ax,
#                     predictions_mm,
#                     scene.dt,
#                     max_hl=4,
#                     ph=ph,
#                     map=None, x_min=x_min, y_min=y_min)

    plot_gt(ax,
            predictions_mm_g,
            scene.dt,
            max_hl=4,
            ph=ph,
            map=None, x_min=x_min, y_min=y_min)

    # ax.set_ylim((1385, 1435))
    # ax.set_xlim((850, 900))
    # ax.set_ylim((1320, 1400))
    # ax.set_xlim((440, 530))
    # ax.set_ylim((1140, 1200))
    # ax.set_xlim((1080, 1150))
    # x_min = 1131.0
    # x_max = 1416.0
    # y_min = 933.0
    # y_max = 1238.0
    ax.set_ylim((1020,1083))
    ax.set_xlim((1270,1333))     
    # ax.set_ylim((1012,1102))
    # ax.set_xlim((1260,1350))    

    leg = ax.legend(loc='upper right', fontsize=20, frameon=True)
    ax.axis('off')
    for lh in leg.legendHandles:
        lh.set_alpha(.5)
    ax.get_legend().remove()
    fig.show()
    fig.savefig('plots/qual_nuScenes.pdf', dpi=300, bbox_inches='tight')
###############################################################################
    my_patch = (x_min, y_min, x_max, y_max)
    fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.15, render_egoposes_range=False)   
    plot_vehicle_nice_g(ax,
                      predictions_g,
                      scene.dt,
                      max_hl=4,
                      ph=ph,
                      map=None, x_min=x_min, y_min=y_min)

#     plot_vehicle_mm_g(ax,
#                     predictions_mm_g,
#                     scene.dt,
#                     max_hl=4,
#                     ph=ph,
#                     map=None, x_min=x_min, y_min=y_min)

    plot_gt(ax,
            predictions_mm_g,
            scene.dt,
            max_hl=4,
            ph=ph,
            map=None, x_min=x_min, y_min=y_min)

    ax.set_ylim((1020,1083))
    ax.set_xlim((1270,1333))      

    leg = ax.legend(loc='upper right', fontsize=20, frameon=True)
    ax.axis('off')
    for lh in leg.legendHandles:
        lh.set_alpha(.5)
    ax.get_legend().remove()
    fig.show()
    fig.savefig('plots/qual_nuScenes-best.pdf', dpi=300, bbox_inches='tight')
###############################################################################
    my_patch = (0, 0, 1, 1)
    fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(1, 1), alpha=0.5, render_egoposes_range=False)
    ax.plot([], [], 
            'o',
            color=line_colors[0 % len(line_colors)],
            markersize=3,
            alpha=0.7, 
            path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()],
            label='Ground Truth')
        #     label='Ours (ML)')
    ax.plot([], [], 
            'o',
            color=line_colors[1 % len(line_colors)],
            markersize=3,
            alpha=0.7, 
            path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    ax.plot([], [], 
            'o',
            color=line_colors[2 % len(line_colors)],
            markersize=3,
            alpha=0.7, 
            path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    ax.plot([], [], 
            'o',
            color=line_colors[3 % len(line_colors)],
            markersize=3,
            alpha=0.7, 
            path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    ax.plot([], [], 
            'o',
            color=line_colors[4 % len(line_colors)],
            markersize=3,
            alpha=0.7, 
            path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])                                        

#     ax.plot([],
#             [],
#             'w^', 
#             markersize=3,
#             path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()],
#             label='Ground Truth')
    leg = ax.legend(loc='upper left', fontsize=30, frameon=True)
    for lh in leg.legendHandles:
        lh.set_alpha(.5)
    ax.axis('off')
    ax.grid('off')
    fig.savefig('plots/qual_nuScenes_legend.pdf', dpi=300, bbox_inches='tight')
    #fig.savefig('plots/qual_nuScenes.pdf', dpi=300, bbox_inches='tight')

