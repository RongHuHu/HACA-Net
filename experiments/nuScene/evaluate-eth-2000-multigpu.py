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
import torch.multiprocessing as mp

sys.path.append("../../Transformer")

import utils
import evaluation
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar

# 初始化设置
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()

def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255
    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)
    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))
    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1], old_shape[2]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)
    return num_viol_trajs

def load_model(model_dir, env, ts=100, device='cuda:0'):
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment(env)
    return trajectron, hyperparams

def process_gpu(gpu_id, scene_indices_subset, args):
    # 初始化CUDA上下文
    torch.cuda.init()
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    print(f"Process started on GPU {gpu_id} | CUDA visible: {torch.cuda.device_count()} devices")

    # 独立加载环境数据
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    
    # 根据索引获取场景
    scenes_subset = [env.scenes[idx] for idx in scene_indices_subset]
    
    # 加载模型到指定GPU
    eval_stg, hyperparams = load_model(args.model, env, args.checkpoint, device=device)

    # 处理预测时间范围
    for ph in args.prediction_horizon:
        eval_kde_nll = np.array([])
        print(f"GPU {gpu_id} | Processing PH={ph}")
        
        # 处理分配的场景
        for scene_idx, scene in enumerate(scenes_subset):
            print(f"GPU {gpu_id} | Scene {scene_idx+1}/{len(scenes_subset)} (Global ID: {env.scenes.index(scene)})")
            
            for t in tqdm(range(0, scene.timesteps, 10), desc=f"PH {ph}", leave=False):
                timesteps = np.arange(t, t + 10)
                
                # 执行预测
                history_pred = eval_stg.predict(
                    scene,
                    timesteps,
                    ph,
                    min_history_timesteps=hyperparams['minimum_history_length'],
                    min_future_timesteps=ph,
                    num_samples=100,
                    gmm_mode=False
                )
                
                if not history_pred:
                    continue
                
                # 计算结果
                batch_error_dict = evaluation.compute_batch_statistics(
                    history_pred,
                    scene.dt,
                    max_hl=hyperparams['maximum_history_length'],
                    ph=ph,
                    node_type_enum=env.NodeType,
                    map=None,
                    best_of=True,
                    prune_ph_to_future=True,
                    kde=True
                )
                
                # 保存结果
                if args.node_type in batch_error_dict:
                    eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))
                
                # 显存清理
                del history_pred
                torch.cuda.empty_cache()
        
        # 保存临时结果
        if len(eval_kde_nll) > 0:
            output_filename = f"{args.output_tag}_ph{ph}_gpu{gpu_id}.csv"
            pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}).to_csv(
                os.path.join(args.output_path, output_filename)
            )

if __name__ == "__main__":
    # 加载基础环境数据
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    
    # 指定使用的GPU列表
    target_gpus = [3, 4, 5, 6]
    num_workers = len(target_gpus)
    
    # 分割场景索引
    all_scene_indices = list(range(len(env.scenes)))
    scene_indices_subsets = np.array_split(all_scene_indices, num_workers)
    
    # 创建进程池
    mp.set_start_method('spawn', force=True)
    processes = []
    
    # 启动子进程
    for worker_idx, gpu_id in enumerate(target_gpus):
        p = mp.Process(
            target=process_gpu,
            args=(
                gpu_id,
                scene_indices_subsets[worker_idx].tolist(),
                args
            )
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并结果文件
    for ph in args.prediction_horizon:
        merged_df = pd.DataFrame()
        
        # 收集各GPU结果
        for gpu_id in target_gpus:
            temp_file = os.path.join(args.output_path, f"{args.output_tag}_ph{ph}_gpu{gpu_id}.csv")
            if os.path.exists(temp_file):
                df = pd.read_csv(temp_file)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                os.remove(temp_file)  # 删除临时文件
        
        # 保存最终结果
        if not merged_df.empty:
            final_output = os.path.join(args.output_path, f"{args.output_tag}_{ph}_kde_best_of.csv")
            merged_df.to_csv(final_output, index=False)
            print(f"Merged results saved to: {final_output}")
        else:
            print(f"No results found for prediction horizon {ph}")