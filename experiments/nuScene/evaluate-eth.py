import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../Transformer")

import utils
import evaluation
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument(
    "--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+',
                    help="prediction horizon", type=int, default=None)
args = parser.parse_args()


def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape(
        (old_shape[0], old_shape[1], old_shape[2]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)

    return num_viol_trajs


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cuda:8')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cuda:8')

    trajectron.set_environment(env)
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                ' ')
            env.attention_radius[(node_type1, node_type2)
                                 ] = float(attention_radius)

    scenes = env.scenes

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            ############### MOST LIKELY ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            print("-- Evaluating GMM Grid Sampled (Most Likely)")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                timesteps = np.arange(scene.timesteps)
                history_pred = eval_stg.predict(scene,
                                                timesteps, 
                                                ph, 
                                                min_history_timesteps=hyperparams['minimum_history_length'],
                                                min_future_timesteps=ph,
                                                num_samples=1,
                                                gmm_mode=True)
                if not history_pred:
                    continue
                    
                batch_error_dict = evaluation.compute_batch_statistics(
                    history_pred,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    node_type_enum=env.NodeType,
                    map=None,
                    prune_ph_to_future=True,
                    kde=False)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                #eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_most_likely.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_most_likely.csv'))    
            #pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_kde_full.csv'))

            ############### BEST OF 20 ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            print("-- Evaluating best of 20")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t, t + 10)  
                    history_pred = eval_stg.predict(scene,
                                                    timesteps, 
                                                    ph, 
                                                    min_history_timesteps=hyperparams['minimum_history_length'],
                                                    min_future_timesteps=ph,
                                                    num_samples=20,
                                                    gmm_mode=False)
                    if not history_pred:
                        continue

                    batch_error_dict = evaluation.compute_batch_statistics(
                        history_pred,
                        scene.dt,
                        max_hl=max_hl,
                        ph=ph,
                        node_type_enum=env.NodeType,
                        map=None,
                        best_of=True,
                        prune_ph_to_future=True,
                        kde=True)
                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                    eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

                
            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
                        ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_best_of.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
                        ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_best_of.csv'))
            pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}
                        ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_kde_best_of.csv'))
                    



