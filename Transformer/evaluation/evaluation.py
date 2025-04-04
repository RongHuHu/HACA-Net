import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from utils import *
import visualization
from matplotlib import pyplot as plt


def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(
        predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = 0.5
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]
    # where_are_nan = np.isnan(predicted_trajs)
    # where_are_inf = np.isinf(predicted_trajs)
    # predicted_trajs[where_are_nan] = 0
    # predicted_trajs[where_are_inf] = 0
    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(
                    kde.logpdf(
                        gt_traj[timestep].T),
                    a_min=log_pdf_lower_bound,
                    a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(
                                             obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs

def compute_miss_rate(predicted_trajs, gt_traj, dist_thresh: float = 2):

    dist = gt_traj - predicted_trajs
    #print('dist0',dist)
    dist = np.power(dist, 2)
    #print('dist1',dist)
    dist = np.sum(dist, axis=3)
    #print('dist2',dist)
    dist = np.power(dist, 0.5)
    #print('dist3',dist)
    #dist[masks_rpt] = -np.inf
    dist = np.max(dist, axis=2)
    #print('dist4',dist)
    dist = np.min(dist, axis=1)
    #print('dist5',dist)
    m_r = [np.sum(dist > dist_thresh) / len(dist)]
    #print('m_r',m_r)
    return m_r

def compute_batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict, _, futures_dict) = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {
            'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list(),  'miss_rate': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            #print('futures_dict[t][node]',futures_dict[t][node])
            ade_errors = compute_ade(
                prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(
                prediction_dict[t][node], futures_dict[t][node])
            miss_rate = compute_miss_rate(
                prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(
                    prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(
                    prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                miss_rate = np.min(miss_rate, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))
            batch_error_dict[node.type]['miss_rate'].extend(list(miss_rate))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])
            
    return batch_error_dict


def lane_compute_batch_statistics(prediction_output_dict,
                                  dt,
                                  max_hl,
                                  ph,
                                  node_type_enum,
                                  kde=False,
                                  obs=False,
                                  map=None,
                                  prune_ph_to_future=False,
                                  best_of=True):

    (prediction_dict, _, futures_dict) = lane_prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {
            'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        kde_ll = 0
        obs_viols = 0
        for node in prediction_dict[t].keys():
            ade_error = np.array([])
            fde_error = np.array([])
            for lane in prediction_dict[t][node].keys():
                pred = prediction_dict[t][node][lane]
                gt = futures_dict[t][node][lane]
                ade_error = np.append(ade_error, compute_ade(
                    pred, gt))
                fde_error = np.append(fde_error, compute_fde(
                    pred, gt))
                if kde:
                    kde_ll = compute_kde_nll(
                        pred, gt)
                else:
                    kde_ll = 0
                if obs:
                    obs_viols = compute_obs_violations(
                        pred, map)
                else:
                    obs_viols = 0

            if best_of:
                ade_error = np.min(ade_error, keepdims=True)
                fde_error = np.min(fde_error, keepdims=True)
                kde_ll = np.min(kde_ll)

            batch_error_dict[node.type]['ade'].extend(list(ade_error))
            batch_error_dict[node.type]['fde'].extend(list(fde_error))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])

    return batch_error_dict


def log_batch_errors(
        batch_errors_list,
        log_writer,
        namespace,
        curr_iter,
        bar_plot=[],
        box_plot=[]):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                log_writer.add_histogram(
                    f"{node_type.name}/{namespace}/{metric}",
                    metric_batch_error,
                    curr_iter)
                log_writer.add_scalar(
                    f"{node_type.name}/{namespace}/{metric}_mean",
                    np.mean(metric_batch_error),
                    curr_iter)
                log_writer.add_scalar(
                    f"{node_type.name}/{namespace}/{metric}_median",
                    np.median(metric_batch_error),
                    curr_iter)

                if metric in bar_plot:
                    pd = {'dataset': [namespace] * len(metric_batch_error),
                          metric: metric_batch_error}
                    kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_barplots(
                        ax, pd, 'dataset', metric)
                    log_writer.add_figure(
                        f"{node_type.name}/{namespace}/{metric}_bar_plot",
                        kde_barplot_fig,
                        curr_iter)

                if metric in box_plot:
                    mse_fde_pd = {
                        'dataset': [namespace] *
                        len(metric_batch_error),
                        metric: metric_batch_error}
                    fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_boxplots(
                        ax, mse_fde_pd, 'dataset', metric)
                    log_writer.add_figure(
                        f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean",
                    np.mean(metric_batch_error))
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median",
                    np.median(metric_batch_error))
