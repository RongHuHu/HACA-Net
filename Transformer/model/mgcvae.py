import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from model.components import *
from model.model_utils import (
    L2_norm,
    obs_violation_rate,
    classification_loss,
    generate_square_subsequent_mask,
    generate_mask,
    rgetattr,
    rsetattr,
)
from model.model_utils import *
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch import ViT
from environment.scene_graph import DirectedEdge
import model.dynamics as dynamic_module
from torch.distributions import MultivariateNormal
class MultimodalGenerativeCVAE(object):
    def __init__(
            self,
            env,
            node_type,
            model_registrar,
            hyperparams,
            device,
            edge_types,
            log_writer=None):

        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [
            edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.class_input = []
        self.node_modules = dict()

        self.min_hl = self.hyperparams["minimum_history_length"]
        self.max_hl = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]
        self.state = self.hyperparams["state"]
        self.pred_state = self.hyperparams["pred_state"][node_type]
        self.state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [DirectedEdge.get_str_from_types(
            *edge_type) for edge_type in self.edge_types]
        self.create_submodule(edge_types_str)

        # self.x_size = 256
        # dynamic_class = getattr(
        #     dynamic_module, hyperparams['dynamic'][self.node_type]['name'])
        # dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        # self.dynamic = dynamic_class(
        #     self.env.scenes[0].dt,
        #     dyn_limits,
        #     device,
        #     self.model_registrar,
        #     self.x_size,
        #     self.node_type)

        self.memory = None
        self.memory_mask = None

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(
            name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_models(self):
        #######################
        # Transformer Encoder #
        #######################
        self.add_submodule(
            self.node_type + "/node_history_encoder",
            model_if_absent=Encoder(
                ninp=self.state_length,
                nlayers=self.hyperparams["transformer"]["nlayers"],
                nhead=self.hyperparams["transformer"]["nhead"],
                in_dim=self.hyperparams["transformer"]["in_dim"],
                fdim=self.hyperparams["transformer"]["fdim"],
            ),
        )

        if self.hyperparams['edge_encoding']:
            ##############################
            #   Edge Influence Encoder   #
            ##############################
            self.add_submodule(self.node_type + '/edge_influence_encoder',
                                model_if_absent=AdditiveAttention(
                                encoder_hidden_state_dim=self.hyperparams["transformer"]["output_size"],
                                decoder_hidden_state_dim=self.hyperparams["transformer"]["in_dim"]))

        map_output_size = None

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams["map_vit_encoding"]:
            me_params = self.hyperparams["map_encoder"]["vit_param"]
            map_output_size = me_params["output_size"]
            self.add_submodule(
                self.node_type + "/map_encoder",
                model_if_absent=ViT(
                    image_size=me_params["image_size"],
                    patch_size=me_params["patch_size"],
                    num_classes=me_params["output_size"],
                    dim=me_params["dim"],
                    depth=me_params["deep"],
                    heads=me_params["heads"],
                    mlp_dim=me_params["mlp_dim"],
                    dropout=me_params["dropout"],
                    emb_dropout=me_params["emb_dropout"],
                ),
            )
        elif self.hyperparams["map_cnn_encoding"]:
            me_params = self.hyperparams["map_encoder"]["cnn_param"]
            map_output_size = me_params["output_size"]
            self.add_submodule(
                self.node_type + "/map_encoder",
                model_if_absent=CNNMapEncoder(
                    me_params["map_channels"],
                    me_params["hidden_channels"],
                    me_params["output_size"],
                    me_params["masks"],
                    me_params["strides"],
                    me_params["patch_size"],
                ),
            )
        
        ##########################
        #   Fusion Multi-Input   #
        ##########################
        if self.hyperparams['edge_encoding']:
            if self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"]:
                if self.hyperparams['incl_robot_node']: 
                    fusion_layer_size = 3 * self.hyperparams["transformer"]["output_size"] + map_output_size 
                else:
                    fusion_layer_size = 2 * self.hyperparams["transformer"]["output_size"] + map_output_size
            elif self.hyperparams['incl_robot_node']:
                fusion_layer_size = 3 * self.hyperparams["transformer"]["output_size"] 
            else:
                fusion_layer_size = 2 * self.hyperparams["transformer"]["output_size"]
        else:
            if self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"]:
                if self.hyperparams['incl_robot_node']: 
                    fusion_layer_size = 2 * self.hyperparams["transformer"]["output_size"] + map_output_size 
                else:
                    fusion_layer_size = self.hyperparams["transformer"]["output_size"] + map_output_size
            elif self.hyperparams['incl_robot_node']:
                fusion_layer_size = 2 * self.hyperparams["transformer"]["output_size"] 
            else:
                fusion_layer_size = self.hyperparams["transformer"]["output_size"]

        self.add_submodule(
            self.node_type + "/fusion/MLP",
            model_if_absent=Mlp(
                in_channels=fusion_layer_size,
                output_size=self.hyperparams["transformer"]["output_size"],
                layer_num=self.hyperparams["fusion_hist_map_layer"],
                mode="regression",
            ),
        )            
            
        self.add_submodule(
            self.node_type + "/fusion/Times_Channel_Squeeze",
            model_if_absent=Times_Channel_Squeeze(
                in_channels=fusion_layer_size,
                output_size=fusion_layer_size,
                layer_num=self.hyperparams["fusion_hist_map_layer"],
                mode="regression",
            ),
        ) 

        ###################
        #   Decoder MLP   #
        ###################
        if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']: 
            layer_num=1
        else:
            layer_num=2

        self.add_submodule(
            self.node_type + "/decoder/MLP",
            model_if_absent=Mlp(
                in_channels=fusion_layer_size, 
                output_size=self.pred_state_length,
                layer_num=layer_num, 
                mode="regression",
            ),
        )
        
        self.add_submodule(
            self.node_type + "/GMM/MLP",
            model_if_absent=Mlp(
                in_channels=self.pred_state_length, 
                output_size=32,
                layer_num=1, 
                mode="regression",
            ),
        )

        #######################
        # Transformer Decoder #
        #######################
        self.add_submodule(
            self.node_type + "/decoder/transformer_decoder",
            model_if_absent=Trajectory_Decoder(
                nlayers=self.hyperparams["transformer"]["nlayers"],
                tgt_inp=self.pred_state_length,
                in_dim=self.hyperparams["transformer"]["in_dim"],
                nhead=self.hyperparams["transformer"]["nhead"],
                fdim=self.hyperparams["transformer"]["fdim"],
                noutput=self.hyperparams["transformer"]["output_size"],
            ),
        )

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(
            self.node_type +
            '/decoder/state_action',
            model_if_absent=nn.Sequential(
                nn.Linear(
                    self.state_length,
                    self.pred_state_length)))

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_log_pis',
            model_if_absent=Mlp(
                in_channels=32, 
                output_size=1,
                layer_num=1, 
                mode="regression",
            )
        )

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_mus',
            model_if_absent=Mlp(
                in_channels=32, 
                output_size=2,
                layer_num=1, 
                mode="regression",
            )
        )

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_log_sigmas',
            model_if_absent=Mlp(
                in_channels=32, 
                output_size=2,
                layer_num=1, 
                mode="regression",
            )
        )

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_corrs',
            model_if_absent=Mlp(
                in_channels=32, 
                output_size=1,
                layer_num=1, 
                mode="regression",
            )
        )

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('->')[1]].values()]))
            
            self.add_submodule(
                edge_type + "/edge_history_encoder",
                model_if_absent=Edge_encoder(
                    ninp=self.state_length + neighbor_state_length,
                    nlayers=self.hyperparams["transformer"]["nlayers"],
                    nhead=self.hyperparams["transformer"]["nhead"],
                    in_dim=self.hyperparams["transformer"]["in_dim"],
                    fdim=self.hyperparams["transformer"]["fdim"],
                ),
            )    


    def create_submodule(self, edge_types):
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_models()

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams['edge_encoding']:
            self.create_edge_models(edge_types)
            
        for name, module in self.node_modules.items():
            module.to(self.device)

    def obtain_encoded_tensors(
            self,
            mode,
            inputs,
            inputs_st,
            map,
            robot,
            neighbors,
            neighbors_edge_value            
            ):

        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]
        
        node_history_st = inputs_st#
        node_present_state_st = inputs_st[:, -1]
        
        ##################
        # Encode History #
        ##################
        memory_padding_mask = generate_mask(node_history_st).to(self.device)
        memory_src_mask = generate_square_subsequent_mask(node_history_st.size()[-2], self.device)
        transformer_encoder = self.node_modules[self.node_type + "/node_history_encoder"]

        if self.hyperparams['incl_robot_node']: 
            robot_padding_mask = generate_mask(robot).to(self.device)
            robot_src_mask = generate_square_subsequent_mask(robot.size()[-2], self.device)           
        else:
            robot_padding_mask = None
            robot_src_mask = None 

        memory, robot = transformer_encoder(
                node_history_st, robot, memory_src_mask, memory_padding_mask, robot_padding_mask, robot_src_mask)

        ############################
        # Map Information encoding #
        ############################
        encoded_map = None
        if self.hyperparams["map_cnn_encoding"]:
            encoded_map = self.node_modules[self.node_type + "/map_encoder"](
                map * 2.0 - 1.0, (mode == ModeKeys.TRAIN))
            do = self.hyperparams["map_encoder"]["cnn_param"]["dropout"]
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))
            
        elif self.hyperparams["map_vit_encoding"]:
            encoded_map = self.node_modules[self.node_type + "/map_encoder"](map * 2.0 - 1.0)  

        # #############################
        # Encode Node Edges per Type #
        # #############################
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                # Encode edges for given edge type
                # print('edge_type',edge_type) #(VEHICLE, PEDESTRIAN)，(VEHICLE, VEHICLE)
                encoded_edges_type = self.encode_edge(node_history_st,
                                                      edge_type,
                                                      neighbors[edge_type],
                                                      neighbors_edge_value[edge_type],
                                                      batch_size)
                node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]

            #####################
            # Encode Node Edges #
            #####################
            total_edge_influence = self.encode_total_edge_influence(node_edges_encoded,
                                                                    memory,
                                                                    batch_size)
        else:
            total_edge_influence = None

        return memory, memory_src_mask, memory_padding_mask, node_present_state_st, node_pos, encoded_map, robot, total_edge_influence

    def encode_edge(self,
                    node_history_st,
                    edge_type,
                    neighbors,
                    neighbors_edge_value, 
                    batch_size):

        max_hl = self.hyperparams['maximum_history_length']
        max_neighbors = 0 
        for neighbor_states in neighbors:
            max_neighbors = max(max_neighbors, len(neighbor_states))     

        edge_states_list = list()  
        for i, neighbor_states in enumerate(neighbors):  
            if len(neighbor_states) == 0: 
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                )
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))       

        op_applied_edge_states_list = list()
        for neighbors_state in edge_states_list:
            op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
        
        if self.hyperparams['dynamic_edges'] == 'yes':
            op_applied_edge_mask_list = list()
            for edge_value in neighbors_edge_value:
                op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),  #torch.sum
                                                                            dim=0, keepdim=True), max=1.))
            combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)
        
        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        transformer_encoder = self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + "/edge_history_encoder"]
        joint_padding_mask = generate_mask(joint_history).to(self.device)
        joint_src_mask = generate_square_subsequent_mask(joint_history.size()[-2], self.device)

        ret = transformer_encoder(joint_history, joint_src_mask, joint_padding_mask)
        
        if self.hyperparams['dynamic_edges'] == 'yes':
            return ret * combined_edge_masks
        else:
            return ret

    def encode_total_edge_influence(self, encoded_edges, node_history_encoder, batch_size):
        if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.hyperparams["transformer"]["output_size"]), device=self.device)
        else:
                combined_edges = torch.stack(encoded_edges, dim=1)
                avg_pool = nn.AdaptiveAvgPool1d(1)
                combined_edges = combined_edges.permute(0,2,1)
                combined_edges = avg_pool(combined_edges)
                combined_edges = combined_edges.permute(0,2,1)
        return combined_edges

    def at_dec(
            self,
            inputs_st,
            memory,
            memory_mask,
            memory_key_padding_mask,
            n_s_t0,
            map_feature,
            prediction_horizon,
            robot,
            total_edge_influence
            ):
        mem_size = memory.size()
        history_timestep = memory.size()[-2]
        init_pos = n_s_t0[:, 0:2]
        batch_size = init_pos.size()[0]
        pred_state = init_pos.size()[1]
   
        history_pred = None
        max_hl=self.max_hl//2

        transformer_decoder = self.node_modules[self.node_type + "/decoder/transformer_decoder"]
        mlp = self.node_modules[self.node_type + "/decoder/MLP"]
        history_input = init_pos.view(batch_size, 1, pred_state)
        if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']:
            fusion = self.node_modules[self.node_type + "/fusion/Times_Channel_Squeeze"]
        inputs_st=inputs_st[:, :, 0:2]
        memory_dec=inputs_st[:, -max_hl:, :]            
        ph=0
        for _ in range(prediction_horizon):
            ph=ph+1
            tgt_mask = generate_square_subsequent_mask(ph, self.device)
            h = transformer_decoder(
                ph,
                tgt=inputs_st,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            h_state=h[:, [-1], :]
            if self.hyperparams['edge_encoding']:
                h_state = torch.cat([h_state, total_edge_influence], dim=-1)
            if self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"]:
                h_state = torch.cat([h_state, map_feature.unsqueeze(-2)], dim=-1)
            if self.hyperparams['incl_robot_node']:    
                h_state = torch.cat([h_state, robot], dim=-1)                
            if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']:
                h_state = fusion(h_state) 
            new_state = mlp(h_state) + inputs_st[:, [-1], :]
            inputs_st = torch.cat([inputs_st, new_state], dim=-2)
        history_pred = inputs_st[:, -prediction_horizon:, :]
        return history_pred

    def project_to_GMM_params(self,
                              tensor) -> (torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor):

        log_pis = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules[self.node_type + '/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def GMM_dec(
            self,
            mode,
            inputs_st,
            memory,
            memory_mask,
            memory_key_padding_mask,
            n_s_t0,
            map_feature,
            prediction_horizon,
            robot,
            total_edge_influence,
            num_samples=1,
            num_components=1,
            gmm_mode=False
            ):

        mem_size = memory.size()
        history_timestep = memory.size()[-2]
        init_pos = n_s_t0[:, 0:2]
        batch_size = init_pos.size()[0]
        pred_state = init_pos.size()[1]

        history_pred = None
        max_hl=self.max_hl//2

        transformer_decoder = self.node_modules[self.node_type + "/decoder/transformer_decoder"]
        mlp = self.node_modules[self.node_type + "/decoder/MLP"]
        mlp_g = self.node_modules[self.node_type + "/GMM/MLP"]
        history_input = init_pos.view(batch_size, 1, pred_state)
        history_input = history_input.repeat(num_samples * num_components, 1, 1)
        if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']:
            fusion = self.node_modules[self.node_type + "/fusion/Times_Channel_Squeeze"]
        inputs_st = inputs_st[:, :, 0:2]
        a_t = inputs_st.repeat(num_samples * num_components, 1, 1) 
        pred_dim = self.pred_state_length          
        ph=0
        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []
        memory_key_padding_mask=memory_key_padding_mask.repeat(num_samples * num_components, 1)
        memory_mask=memory_mask.repeat(num_samples * num_components, 1)
        memory=memory.repeat(num_samples * num_components, 1, 1)

        for _ in range(prediction_horizon):
            ph=ph+1
            tgt_mask = generate_square_subsequent_mask(ph, self.device)
            h = transformer_decoder(
                ph,
                tgt=a_t,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            h_state=h[:, [-1], :]
            if self.hyperparams['edge_encoding']:
                h_state = torch.cat([h_state, total_edge_influence.repeat(num_samples * num_components, 1, 1)], dim=-1)
            if self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"]:
                h_state = torch.cat([h_state, map_feature.unsqueeze(-2).repeat(num_samples * num_components, 1, 1)], dim=-1)
            if self.hyperparams['incl_robot_node']:    
                h_state = torch.cat([h_state, robot.repeat(num_samples * num_components, 1, 1)], dim=-1)                
            if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']:
                h_state = fusion(h_state)                                       
            new_state = mlp(h_state) + a_t[:, [-1], :]
            a_t = torch.cat([a_t, new_state], dim=-2)

        a_t = a_t[:, -prediction_horizon:, :]
        log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(mlp_g(a_t))
        # mus = a_t
        # bs, state_length, _ = a_t.size()
        # a_t = a_t.reshape(-1, 2)
        # log_pis = torch.ones(bs * state_length, 1).to(self.device)
        # mus = torch.zeros(bs * state_length, 2).to(self.device)
        # log_sigmas = torch.zeros(bs * state_length, 2).to(self.device)
        # corrs = torch.zeros(bs * state_length, 1).to(self.device)
        # for i in range(bs * state_length):
        #     mean = a_t[i].to(self.device)
        #     cov = torch.eye(2).to(self.device)
        #     dist = MultivariateNormal(mean, cov)
        #     mus[i] = mean
        #     #log_sigmas[i] = torch.log(torch.sqrt(dist.covariance_matrix.diag()))
        #     scale_tril = dist.scale_tril.to(self.device)
        #     log_sigmas[i] = torch.log(torch.sqrt(torch.sum(scale_tril ** 2, dim=-1)))
        #     corrs[i] = (scale_tril[0, 1] / scale_tril[0, 0]).item()
        # log_pis = dist.log_prob(mus).to(self.device)
        log_pis.append(torch.ones_like(corr_t.reshape(
                                        num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1)))
        mus.append(mu_t.reshape(num_samples, num_components, -1, 2
                                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
        log_sigmas.append(log_sigma_t.reshape(num_samples, num_components, -1, 2
                                              ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
        corrs.append(corr_t.reshape(num_samples, num_components, -1
                                    ).permute(0, 2, 1).reshape(-1, num_components))
  
        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
 
        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, - 1, ph, num_components]), torch.reshape(
                    mus, [num_samples, - 1, ph, num_components *pred_dim]), torch.reshape(
                        log_sigmas, [num_samples, - 1, ph, num_components * pred_dim]), torch.reshape(corrs, [num_samples, - 1, ph, num_components]))

        #print('a_dist',a_dist)
        # if self.hyperparams['dynamic'][self.node_type]['distribution']:
        #     y_dist = self.dynamic.integrate_distribution(a_dist, x)
        # else:
        y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            #sampled_future = self.dynamic.integrate_samples(a_sample, x)
            sampled_future = a_sample
        else:
            sampled_future = a_dist.rsample()

        return y_dist, sampled_future, a_t

    def train_decoder(
        self,
        mode,
        inputs_st,
        autoregressive,
        memory,
        memory_mask,
        memory_key_padding_mask,
        labels_st,
        n_s_t0,
        encoded_map,
        prediction_horizon,
        robot,
        total_edge_influence,
        num_samples,
        num_components,
        gmm_mode
    ):

        if autoregressive: 
            history_pred = self.at_dec(
                inputs_st,
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0, 
                encoded_map,
                prediction_horizon,
                robot,
                total_edge_influence
            )
            return history_pred 
        else:
            y_dist, sampled_future, history_pred = self.GMM_dec(
                mode,
                inputs_st,
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0, 
                encoded_map,
                prediction_horizon,
                robot,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            return y_dist, sampled_future, history_pred  

        

    def test_decoder(
        self,
        mode,
        inputs_st,
        autoregressive,
        memory,
        memory_mask,
        memory_key_padding_mask,
        n_s_t0,
        encoded_map,
        prediction_horizon,
        robot,
        total_edge_influence,
        num_samples,
        num_components,
        gmm_mode
    ):

        if autoregressive: 
            history_pred = self.at_dec(
                inputs_st,
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0, 
                encoded_map,
                prediction_horizon,
                robot,
                total_edge_influence
            )
            return history_pred 
        else:
            y_dist, sampled_future, history_pred = self.GMM_dec(
                mode,
                inputs_st,
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0, 
                encoded_map,
                prediction_horizon,
                robot,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            return y_dist, sampled_future, history_pred  

    def train_loss(
            self,
            inputs,
            inputs_st,
            labels,
            labels_st,
            robot,
            map,
            prediction_horizon,
            neighbors,
            neighbors_edge_value,             
            ) -> torch.tensor:
        
        mode = ModeKeys.TRAIN
        num_samples = 1
        num_components = 1
        gmm_mode = False

        (memory,
         memory_src_mask,
         memory_key_padding_mask,
         n_s_t0,
         rel_state,
         encoded_map,
         robot_enc,
         total_edge_influence
         ) = self.obtain_encoded_tensors(mode=mode,
                                         inputs=inputs,
                                         inputs_st=inputs_st,
                                         map=map,
                                         robot=robot,
                                         neighbors=neighbors,
                                         neighbors_edge_value=neighbors_edge_value)

        if self.hyperparams["autoregressive"]:
            history_pred = self.train_decoder(
                mode,
                inputs_st,
                self.hyperparams["autoregressive"],
                memory,
                memory_src_mask,
                memory_key_padding_mask,
                labels_st,
                n_s_t0,
                encoded_map,
                prediction_horizon,
                robot_enc,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            reg_loss = 0
            history_reg_loss = L2_norm(history_pred, labels_st)
            history_reg_loss = torch.sum(
                history_reg_loss, dim=-1) / prediction_horizon  
            mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss
            reg_loss = torch.sum(reg_loss) / torch.sum(mask)
            loss = reg_loss   
        else:
            y_dist, our_sampled_future, history_pred = self.train_decoder(
                ModeKeys.PREDICT,
                inputs_st,
                self.hyperparams["autoregressive"],
                memory,
                memory_src_mask,
                memory_key_padding_mask,
                labels_st,
                n_s_t0,
                encoded_map,
                prediction_horizon,
                robot_enc,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            # means = nn.Parameter(torch.randn(num_components, history_pred.size()[-2], history_pred.size()[-1]))
            # print('means',means)
            # means_batch = torch.index_select(means, dim=0, index=labels_st)
            # print('means_batch',means_batch)
            # N=history_pred.size()[0]
            # lambda_=1.0
            # alpha=1.0
            # likelihood_reg_loss = lambda_ * (torch.sum((history_pred - means_batch)**2) / 2) * (1. / N)

            # log_p_yt = torch.clamp(y_dist.log_prob(labels_st), max=6) # 1, 128, 12
            # log_p_y = torch.sum(log_p_yt, dim=2) # 1, 128
            # log_p_y_mean = torch.mean(log_p_y, dim=0) # 128
            # log_likelihood = torch.mean(log_p_y_mean)
            # nll = -log_likelihood
            
            reg_loss = 0
            history_reg_loss = L2_norm(our_sampled_future, labels_st)
            history_reg_loss = torch.sum(
                history_reg_loss, dim=-1) / prediction_horizon  
            mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss
            reg_loss = torch.sum(reg_loss) / torch.sum(mask)
            
            lambda_=1
            alpha=0          
            loss = reg_loss # + alpha * nll

        if self.log_writer is not None:
            self.log_writer.add_scalar(
                "%s/%s" % (str(self.node_type), "loss"), loss, self.curr_iter)
        
        return loss, reg_loss

    def eval_loss(
        self,
        node_type,
        inputs,
        inputs_st,
        labels,
        labels_st,
        robot,
        map,
        prediction_horizon,
        neighbors,
        neighbors_edge_value,        
    ) -> torch.tensor:

        mode = ModeKeys.EVAL
        num_samples = 1
        num_components = 1
        gmm_mode = False
        node_pos = inputs[:, [-1], 0:2]  
        bs, _, feature_dim = node_pos.size()

        (memory,
         memory_mask,
         memory_key_padding_mask,
         n_s_t0,
         rel_state,
         encoded_map,
         robot_enc,
         total_edge_influence
         ) = self.obtain_encoded_tensors(mode=mode,
                                         inputs=inputs,
                                         inputs_st=inputs_st,
                                         map=map,
                                         robot=robot,
                                         neighbors=neighbors,
                                         neighbors_edge_value=neighbors_edge_value)

        if self.hyperparams["autoregressive"]:
            history_pred = self.test_decoder(
                mode,
                inputs_st,
                self.hyperparams["autoregressive"],
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0,
                encoded_map,
                prediction_horizon,
                robot_enc,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            reg_loss = 0
            history_pred = history_pred * 80 + node_pos
            history_reg_loss = L2_norm(history_pred, labels)
            history_reg_loss = torch.sum(
                history_reg_loss, dim=-1) / prediction_horizon 
            mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss
            reg_loss = torch.sum(reg_loss) / torch.sum(mask)
            loss = reg_loss  
        else:
            y_dist, our_sampled_future, a_t = self.test_decoder(
                ModeKeys.PREDICT,
                inputs_st,
                self.hyperparams["autoregressive"],
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0,
                encoded_map,
                prediction_horizon,
                robot_enc,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )

            # log_p_yt = torch.clamp(y_dist.log_prob(labels_st), max=6)
            # log_p_y = torch.sum(log_p_yt, dim=2)
            # log_p_y_mean = torch.mean(log_p_y, dim=0)  
            # log_likelihood = torch.mean(log_p_y_mean)
            # nll = -log_likelihood
            # loss = nll  # loss

            reg_loss = 0
            history_pred = our_sampled_future * 80 + node_pos
            history_reg_loss = L2_norm(history_pred, labels)
            history_reg_loss = torch.sum(
                history_reg_loss, dim=-1) / prediction_horizon 
            mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss
            reg_loss = torch.sum(reg_loss) / torch.sum(mask)  # reg_loss
            loss = reg_loss
        
        return loss, reg_loss

    def predict(
            self,
            inputs,
            inputs_st,
            robot,
            map,
            prediction_horizon,
            neighbors,
            neighbors_edge_value,
            num_samples,
            gmm_mode,
            ):

        node_pos = inputs[:, [-1], 0:2] 
        bs, _, feature_dim = node_pos.size()
        mode = ModeKeys.PREDICT
        num_components = 1

        (memory,
         memory_mask,
         memory_key_padding_mask,
         n_s_t0,
         rel_state,
         encoded_map,
         robot_enc,
         total_edge_influence
         ) = self.obtain_encoded_tensors(mode=mode,
                                         inputs=inputs,
                                         inputs_st=inputs_st,
                                         map=map,
                                         robot=robot,
                                         neighbors=neighbors,
                                         neighbors_edge_value=neighbors_edge_value)

        if self.hyperparams["autoregressive"]:
            history_pred = self.test_decoder(
                mode,
                inputs_st,
                self.hyperparams["autoregressive"],
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0,
                encoded_map,
                prediction_horizon,
                robot_enc,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            history_pred = history_pred * 80 + node_pos
        else:
            _, our_sampled_future, _ = self.test_decoder(
                mode,
                inputs_st,
                self.hyperparams["autoregressive"],
                memory,
                memory_mask,
                memory_key_padding_mask,
                n_s_t0,
                encoded_map,
                prediction_horizon,
                robot_enc,
                total_edge_influence,
                num_samples,
                num_components,
                gmm_mode
            )
            #print('our_sampled_future',our_sampled_future)
            history_pred = our_sampled_future * 80 + node_pos
        
        return history_pred
