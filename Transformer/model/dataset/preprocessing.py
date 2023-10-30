import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
container_abcs = collections.abc


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if isinstance(data, bytes):
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if len(
                elem) == 4:  # We assume those are the maps, map points, headings and patch_size

            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(
                scene_map, scene_pts=torch.tensor(
                    np.array(scene_pts)), patch_size=patch_size[0], rotation=heading_angle)
            return map
        elif len(elem) == 3:
            lanes_t_points, y_lane, lane_mask = zip(*batch)
            return [lanes_t_points, y_lane, lane_mask]

        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return dill.dumps(
            neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
    return default_collate(batch)


def get_relative_robot_traj(
        env,
        state,
        node_traj,
        robot_traj,
        node_type,
        robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    #print('robot_traj',robot_traj)
    #print('node_traj',node_traj)
    _, std = env.get_standardize_params(
        state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(robot_traj,
                                    state[robot_type],
                                    node_type=robot_type,
                                    mean=node_traj,
                                    std=std)
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type],
                           node.type, mean=rel_state, std=std)
    # If we predict position we do it relative to current pos
    if list(pred_state[node.type].keys())[0] == 'position':
        y_st = env.standardize(
            y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams['edge_encoding']:
        # Scene Graph
        scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    timestep_range_r = np.array([t, t + max_ft])
    if hyperparams['incl_robot_node']:
        x_node = node.get(timestep_range_r, state[node.type])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(
            timestep_range_r, state[robot_type], padding=0.0)
        robot_traj_st_t = get_relative_robot_traj(
            env, state, x_node, robot_traj, node.type, robot_type)

    # Map
    map_tuple = None
    lane_tuple = None
    if hyperparams['map_cnn_encoding'] or hyperparams['map_vit_encoding']:

        if node.non_aug_node is not None:
            x = node.non_aug_node.get(np.array([t]), state[node.type])
        me_hyp = hyperparams['map_encoder']['cnn_param']
        if 'heading_state_index' in me_hyp:
            heading_state_index = me_hyp['heading_state_index']
            # We have to rotate the map in the opposit direction of the agent
            # to match them
            if isinstance(heading_state_index,
                          list):  # infer from velocity or heading vector
                heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                            x[-1, heading_state_index[0]]) * 180 / np.pi
            else:
                heading_angle = -x[-1, heading_state_index] * 180 / np.pi
        else:
            heading_angle = None
        
        scene_map = scene.map[node.type]
        map_point = x[-1, :2]

        patch_size = hyperparams['map_encoder']['cnn_param']['patch_size']
        map_tuple = (scene_map, map_point, heading_angle, patch_size)

    #print('neighbors_edge_value',neighbors_edge_value)
    #print('neighbors_data_st',neighbors_data_st)
    return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, lane_tuple, map_tuple)


def get_timesteps_data(
        env,
        scene,
        t,
        node_type,
        state,
        pred_state,
        edge_types,
        min_ht,
        max_ht,
        min_ft,
        max_ft,
        hyperparams):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    nodes_per_ts = scene.present_nodes(
        t,
        type=node_type,
        min_history_timesteps=min_ht,
        min_future_timesteps=max_ft,
        return_robot=not hyperparams['incl_robot_node'])
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
        scene_graph = None  # scene.get_scene_graph(timestep,
        #                      env.attention_radius,
        #                      hyperparams['edge_addition_filter'],
        #                      hyperparams['edge_removal_filter'])
        present_nodes = nodes_per_ts[timestep]
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            batch.append(
                get_node_timestep_data(
                    env,
                    scene,
                    timestep,
                    node,
                    state,
                    pred_state,
                    edge_types,
                    max_ht,
                    max_ft,
                    hyperparams,
                    scene_graph=scene_graph))

    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps
