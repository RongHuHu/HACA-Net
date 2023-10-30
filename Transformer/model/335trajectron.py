import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore


class Trajectron(object):
    def __init__(self, model_registrar, hyperparams, log_writer, device):
        super().__init__()

        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams["minimum_history_length"]
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]
        self.state = self.hyperparams["state"]
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams["pred_state"]

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()
        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            #if node_type == "VEHICLE": # "PEDESTRIAN":
            #    continue
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(
                    env,
                    node_type,
                    self.model_registrar,
                    self.hyperparams,
                    self.device,
                    edge_types,
                    log_writer=self.log_writer,
                )
                #print('self.node_models_dict["PEDESTRIAN"]',self.node_models_dict["PEDESTRIAN"])
                #print('self.node_models_dict[node_type]:',dir([self.node_models_dict[node_type]]))
                

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def train_loss(self, batch, node_type):
        (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st, neighbors_edge_value, robot_traj_st_t, _, map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)

        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if isinstance(map, torch.Tensor):
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss, reg_loss = model.train_loss(
            inputs=x,
            inputs_st=x_st_t,
            labels=y,
            labels_st=y_st_t,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph, 
            neighbors=restore(neighbors_data_st),
            neighbors_edge_value=restore(neighbors_edge_value),                      
        )
        return loss, reg_loss

    def eval_loss(self, batch, node_type):

        (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st, neighbors_edge_value, robot_traj_st_t, _, map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)

        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if isinstance(map, torch.Tensor):
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss, reg_loss = model.eval_loss(
            node_type,
            inputs=x,
            inputs_st=x_st_t,
            labels=y,
            labels_st=y_st_t,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=self.ph,
            neighbors=restore(neighbors_data_st),
            neighbors_edge_value=restore(neighbors_edge_value),                       
        )

        return loss.cpu().detach().numpy(), reg_loss.cpu().detach().numpy()

    def predict(
            self,
            scene,
            timesteps,
            ph,
            min_future_timesteps=0,
            min_history_timesteps=1,
            num_samples=1):

        history_pred_dict = dict()
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(
                env=self.env,
                scene=scene,
                t=timesteps,
                node_type=node_type,
                state=self.state,
                pred_state=self.pred_state,
                edge_types=model.edge_types,
                min_ht=min_history_timesteps,
                max_ht=self.max_ht,
                min_ft=min_future_timesteps,
                max_ft=min_future_timesteps,
                hyperparams=self.hyperparams,
            )
            # There are no nodes of type present for timestep
            if batch is None:
                continue

            (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st, neighbors_edge_value, robot_traj_st_t,
             _, map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if isinstance(map, torch.Tensor):
                map = map.to(self.device)
            # Run forward pass
            history_pred = model.predict(
                inputs=x,
                inputs_st=x_st_t,
                map=map,
                prediction_horizon=ph,
                robot=robot_traj_st_t,
                neighbors=restore(neighbors_data_st),
                neighbors_edge_value=restore(neighbors_edge_value), 
                num_samples=num_samples,                                
            )

            # [bs, timestep, feature_dim] -> [1, bs, timestep, feature_dim]
            if self.hyperparams["autoregressive"]:
                history_pred = history_pred.unsqueeze(0).cpu().detach().numpy()
            else:
                history_pred = history_pred.cpu().detach().numpy()

            for i, ts in enumerate(timesteps_o):
                if ts not in history_pred_dict.keys():
                    history_pred_dict[ts] = dict()
                history_pred_dict[ts][nodes[i]] = np.transpose(
                    history_pred[:, [i]], (1, 0, 2, 3))

        return history_pred_dict
