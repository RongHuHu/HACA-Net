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

        if self.hyperparams["lane_cnn_encoding"]:
            ###################
            #   lane Encoder  #
            ###################
            me_params = self.hyperparams["lane_encoder"][self.node_type]
            self.add_submodule(self.node_type + "/lane_encoder",
                model_if_absent=Lane_Encoder(
                    me_params["nlayers"],
                    me_params["embedding_size"],
                    me_params["map_channels"],
                    me_params["output_size"],
                    me_params["kernel_size"],
                    me_params["strides"],
                ),
            )

            fusion_layer_size = self.hyperparams["transformer"]["in_dim"] + \
                me_params["output_size"]

            self.add_submodule(
                self.node_type + "/fusion/lane_hist",
                model_if_absent=Mlp(
                    in_channels=fusion_layer_size,
                    output_size=self.hyperparams["transformer"]["in_dim"],
                    layer_num=self.hyperparams["fusion_hist_map_layer"],
                    mode="regression",
                ),
            )
            ###################
            #  MLP + softmax  #
            ###################
            self.add_submodule(
                self.node_type + "/Lane/MLP_Softmax",
                model_if_absent=Mlp(
                    in_channels=me_params["output_size"] * 3,
                    output_size=self.hyperparams["max_lane_num"],
                    layer_num=self.hyperparams["mlp_layer"],
                    mode="classification",
                ),
            )
        else:
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
                in_channels=2, 
                output_size=32,
                layer_num=2, 
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
                lane_inp=self.hyperparams["lane_encoder"]["VEHICLE"]["output_size"],
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
            model_if_absent=nn.Linear(32,1))

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_mus',
            model_if_absent=nn.Linear(32,2))

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_log_sigmas',
            model_if_absent=nn.Linear(32,2))

        self.add_submodule(
            self.node_type +
            '/decoder/proj_to_GMM_corrs',
            model_if_absent=nn.Linear(32,1))


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
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
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
            inputs_lane,
            map,
            robot,
            neighbors,
            neighbors_edge_value            
            ):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        candidate_lane = inputs_lane
        #print('candidate_lane',candidate_lane.size())
        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_pos = inputs[:, -1, 0:2]#torch.Size([128, 2])
        node_vel = inputs[:, -1, 2:4]#torch.Size([128, 2])
        # print('node_pos',node_pos.size())
        # print('node_vel',node_vel.size())
        
        node_history_st = inputs_st#torch.Size([128, 9, 8])
        # print('node_history_st',node_history_st.size())
        node_present_state_st = inputs_st[:, -1]#torch.Size([128, 8])
        # print('node_present_state_st',node_present_state_st.size())
        
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
        #torch.Size([128, 9, 256])
        # print('memory',memory.size())

        ############################
        # Map Information encoding #
        ############################
        encoded_map = None
        encoded_lane = None
        if self.hyperparams["map_cnn_encoding"]:
            encoded_map = self.node_modules[self.node_type + "/map_encoder"](
                map * 2.0 - 1.0, (mode == ModeKeys.TRAIN))
            do = self.hyperparams["map_encoder"]["cnn_param"]["dropout"]
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))
            #torch.Size([128, 32])
            # print('cnn_encoded_map',encoded_map.size())
            
        elif self.hyperparams["map_vit_encoding"]:
            encoded_map = self.node_modules[self.node_type + "/map_encoder"](map * 2.0 - 1.0)
            # torch.Size([128, 32])
            # print('vit_encoded_map',encoded_map.size())
            
        elif self.hyperparams["lane_cnn_encoding"]:
            lane_num = self.hyperparams["max_lane_num"]
            sample_num = self.hyperparams["sample_num"]
            embedding_size = self.hyperparams["lane_encoder"][self.node_type]["embedding_size"]
            candidate_lane = candidate_lane.view(batch_size * lane_num, embedding_size, sample_num)
            # print("candidate_lane :",candidate_lane)
            encoded_lane = self.node_modules[self.node_type + "/lane_encoder"](candidate_lane).view(
                batch_size, lane_num, self.hyperparams["lane_encoder"][self.node_type]["output_size"]
            )

        # print('neighbors',neighbors)
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

        return memory, memory_src_mask, memory_padding_mask, node_present_state_st, node_pos, encoded_lane, encoded_map, robot, total_edge_influence

    def encode_edge(self,
                    node_history_st,
                    edge_type,
                    neighbors,
                    neighbors_edge_value, 
                    batch_size):

        max_hl = self.hyperparams['maximum_history_length']
        max_neighbors = 0    
        # print(len(self.state[edge_type[1]].values()))
        for neighbor_states in neighbors:
            #print('len(neighbor_states)',len(neighbor_states))
            max_neighbors = max(max_neighbors, len(neighbor_states))     

        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                )
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))       
            # Used in Structural-RNN to combine edges as well.
        op_applied_edge_states_list = list()
        for neighbors_state in edge_states_list:
            op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))#torch.sum
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
        #print('combined_neighbors',combined_neighbors.size()) # torch.Size([128, 5, 6])
        
        if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
            op_applied_edge_mask_list = list()
            for edge_value in neighbors_edge_value:
                op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),  #torch.sum
                                                                            dim=0, keepdim=True), max=1.))
            combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)
        #print('combined_edge_masks',combined_edge_masks.size()) # torch.Size([128, 1])
        
        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)
        #print('joint_history',joint_history.size()) # torch.Size([128, 5, 14])

        #transformer_encoder = self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + "/edge_history_encoder"]
        transformer_encoder = self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + "/edge_history_encoder"]
        joint_padding_mask = generate_mask(joint_history).to(self.device)
        joint_src_mask = generate_square_subsequent_mask(joint_history.size()[-2], self.device)

        ret = transformer_encoder(joint_history, joint_src_mask, joint_padding_mask)
        #print('ret',ret.size()) # torch.Size([128, 32])
        
        if self.hyperparams['dynamic_edges'] == 'yes':
            return ret * combined_edge_masks
        else:
            return ret

    def encode_total_edge_influence(self, encoded_edges, node_history_encoder, batch_size):
        if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.hyperparams["transformer"]["output_size"]), device=self.device)
        else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                combined_edges = torch.stack(encoded_edges, dim=1)
                # combined_edges, _ = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges,
                #                                                                                   node_history_encoder)
                # print('combined_edges',combined_edges.size()) # torch.Size([128, 2, 32])
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
            lane_feature,
            map_feature,
            prediction_horizon,
            robot,
            total_edge_influence
            ):
        
        # print('n_s_t0',n_s_t0.size())#torch.Size([128, 8])
        mem_size = memory.size()
        history_timestep = memory.size()[-2]
        init_pos = n_s_t0[:, 0:2]
        batch_size = init_pos.size()[0]
        pred_state = init_pos.size()[1]
   
        lane_pred = []
        history_pred = None
        max_hl=self.max_hl//2
        # inference model by lane feature and history feature
        if self.hyperparams["lane_cnn_encoding"]:
            max_lane_num = lane_feature.size()[-2]
            lane_decoder = self.node_modules[self.node_type +
                                             "/decoder/transformer_decoder"]
            fusion = self.node_modules[self.node_type + "/fusion/lane_hist"]
            mlp = self.node_modules[self.node_type + "/decoder/MLP"]

            for i in range(max_lane_num):
                lane_input = init_pos.view(batch_size, 1, pred_state)
                lane_feature = lane_feature[:, [i], :].repeat(
                    1, history_timestep, 1)
                lane_memory = torch.cat([memory, lane_feature], dim=-1)
                lane_hist = fusion(lane_memory)
                for i in range(prediction_horizon):
                    tgt_mask = generate_square_subsequent_mask(
                        lane_input.size()[-2], self.device)
                    h_state = lane_decoder(
                        tgt=lane_input,
                        memory=lane_hist,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                    )
                    if i == prediction_horizon - 1:
                        self.class_input.append(h_state)
                    delta_pos = mlp(h_state[:, -1, :])
                    new_state = lane_input[:, -1, :] + delta_pos
                    lane_input = torch.cat(
                        [lane_input, new_state.view(batch_size, 1, pred_state)], dim=-2)
                lane_pred.append(lane_input[:, 1:, :])
        else:  # inference model by only history feature
            transformer_decoder = self.node_modules[self.node_type + "/decoder/transformer_decoder"]
            mlp = self.node_modules[self.node_type + "/decoder/MLP"]
            # mlp_dec = self.node_modules[self.node_type + "/decoder/MLP_dec"]
            history_input = init_pos.view(batch_size, 1, pred_state)
            if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']:
                fusion = self.node_modules[self.node_type + "/fusion/Times_Channel_Squeeze"]
                # fusion2 = self.node_modules[self.node_type + "/fusion/hist_map"]
            # print('history_input',history_input.size())   #torch.Size([128, 1, 2])
            # print('memory',memory.size())   #torch.Size([128, 9, 256])
            # print('memory_mask',memory_mask.size())   #torch.Size([9, 9])
            # print('memory_key_padding_mask',memory_key_padding_mask.size())   #torch.Size([128, 9])
            # print('robot',robot.size())#torch.Size([128, 13, 256])
            inputs_st=inputs_st[:, :, 0:2]
            memory_dec=inputs_st[:, -max_hl:, :]            
            ph=0
            for _ in range(prediction_horizon):
                ph=ph+1
                tgt_mask = generate_square_subsequent_mask(history_input.size()[-2], self.device)
                # tgt = torch.cat([memory_dec,history_input], dim=-2)
                # print('tgt_mask',tgt_mask.size())#[1, 1]~[8, 8]
                h = transformer_decoder(
                    ph,
                    tgt=inputs_st,#[128, 1, 2]
                    memory=memory,#128, 9, 256]
                    tgt_mask=tgt_mask,#[1, 1]~[8, 8]
                    memory_mask=memory_mask,#[9, 9]
                    memory_key_padding_mask=memory_key_padding_mask,#[128, 9]
                )#[128, 1, 32]~[128, 8, 32]
                h_state=h[:, [-1], :]
                # print('h_state',h_state.size())#[128, 1, 32]
                if self.hyperparams['edge_encoding']:
                    h_state = torch.cat([h_state, total_edge_influence], dim=-1)
                if self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"]:
                    h_state = torch.cat([h_state, map_feature.unsqueeze(-2)], dim=-1)
                if self.hyperparams['incl_robot_node']:    
                    h_state = torch.cat([h_state, robot], dim=-1)                
                if self.hyperparams['edge_encoding'] or self.hyperparams["map_cnn_encoding"] or self.hyperparams["map_vit_encoding"] or self.hyperparams['incl_robot_node']:
                    h_state = fusion(h_state)                                       
                    #print('h_state',h_state.size())
                new_state = mlp(h_state) + history_input[:, [-1], :]
                # print('new_state',new_state.size())#[128, 1, 2]
                history_input = torch.cat([history_input, new_state], dim=-2)
                # print('history_input',history_input.size())#[128, 2, 2]~[128, 9, 2]
            history_pred = history_input[:, 1:, :]
            # print('history_pred',history_pred.size())#[128, 8, 2]
        return history_pred, lane_pred

    def project_to_GMM_params(self,
                              tensor) -> (torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules[self.node_type +
                                    '/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules[self.node_type +
                                '/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules[self.node_type +
                                       '/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(
            self.node_modules[self.node_type + '/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def GMM(
            self,
            mode,
            x,
            history_pred,
            n_s_t0,
            prediction_horizon,
            num_samples=1,
            num_components=1,
            gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length
        mlp = self.node_modules[self.node_type + "/GMM/MLP"]
        # z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        # zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        # cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        # initial_h_model = self.node_modules[self.node_type +
        #                                     '/decoder/initial_h']

        # initial_state = initial_h_model(zx)
        history_pred = mlp(history_pred)
        #print('history_pred',history_pred.size())
        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type +
                                '/decoder/state_action'](n_s_t0)

        # state = initial_state
        # if self.hyperparams['incl_robot_node']:
        #     input_ = torch.cat([zx, a_0.repeat(num_samples *
        #                                        num_components, 1), x_nr_t.repeat(num_samples *
        #                                                                          num_components, 1)], dim=1)
        # else:
        #     input_ = torch.cat(
        #         [zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = history_pred[:, [j], :].repeat(num_samples * num_components, 1, 1)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)
            # print('log_pi_t',log_pi_t) # torch.Size([128, 1, 1])
            # print('mu_t',mu_t) # torch.Size([128, 1, 2])
            # print('log_sigma_t',log_sigma_t) # torch.Size([128, 1, 2])
            # print('corr_t',corr_t) # torch.Size([128, 1, 1])

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]
            
            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            # if num_components > 1:
            #     if mode == ModeKeys.PREDICT:
            #         log_pis.append(
            #             self.latent.p_dist.logits.repeat(num_samples, 1, 1))
            #     else:
            #         log_pis.append(
            #             self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            # else:
            log_pis.append(
                torch.ones_like(corr_t.reshape(
                num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1)))

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            # if self.hyperparams['incl_robot_node']:
            #     dec_inputs = [zx, a_t, y_r[:, j].repeat(
            #         num_samples * num_components, 1)]
            # else:
            #     dec_inputs = [zx, a_t]
            # input_ = torch.cat(dec_inputs, dim=1)
            # state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        # print('log_pis',log_pis) 
        # print('mus',mus) 
        # print('log_sigmas',log_sigmas) 
        # print('corrs',corrs)       

        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -
                                               1, ph, num_components]), torch.reshape(mus, [num_samples, -
                                                                                            1, ph, num_components *
                                                                                            pred_dim]), torch.reshape(log_sigmas, [num_samples, -
                                                                                                                                   1, ph, num_components *
                                                                                                                                   pred_dim]), torch.reshape(corrs, [num_samples, -
                                                                                                                                                                     1, ph, num_components]))
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
            return y_dist, sampled_future
        else:
            return y_dist

    def train_decoder(
        self,
        inputs_st,
        autoregressive,
        memory,
        memory_mask,
        memory_key_padding_mask,
        labels_st,
        n_s_t0,
        encoded_lane,
        encoded_map,
        prediction_horizon,
        robot,
        total_edge_influence
    ):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        pred_lane_index = None

        history_pred, lane_pred = self.at_dec(
            inputs_st,
            memory,
            memory_mask,
            memory_key_padding_mask,
            n_s_t0, encoded_lane,
            encoded_map,
            prediction_horizon,
            robot,
            total_edge_influence
        )

        if self.hyperparams["lane_cnn_encoding"]:
            lane_inp = torch.cat(self.class_input, dim=-1)
            pred_lane_index = self.classification_lane(lane_inp)
            self.class_input = []

        return history_pred, lane_pred, pred_lane_index

    def test_decoder(
        self,
        inputs_st,
        autoregressive,
        memory,
        memory_mask,
        memory_key_padding_mask,
        n_s_t0,
        encoded_lane,
        encoded_map,
        prediction_horizon,
        robot,
        total_edge_influence
    ):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        pred_lane_index = None
        history_pred, lane_pred = self.at_dec(
            inputs_st,
            memory, 
            memory_mask, 
            memory_key_padding_mask, 
            n_s_t0, encoded_lane,
            encoded_map, 
            prediction_horizon,
            robot,
            total_edge_influence
        )

        if self.hyperparams["lane_cnn_encoding"]:
            lane_inp = torch.cat(self.class_input, dim=-1)
            pred_lane_index = self.classification_lane(lane_inp)
            self.class_input = []

        return history_pred, lane_pred, pred_lane_index

    def classification_lane(self, encoded_lane):

        mlp = self.node_modules[self.node_type + "/Lane/MLP_Softmax"]
        pred_lane_index = mlp(encoded_lane)

        return pred_lane_index

    def train_loss(
            self,
            inputs,
            inputs_st,
            inputs_lane,
            lane_label,
            lane_t_mask,
            labels,
            labels_st,
            robot,
            map,
            prediction_horizon,
            neighbors,
            neighbors_edge_value,             
            ) -> torch.tensor:
        
        mode = ModeKeys.TRAIN

        (memory,
         memory_src_mask,
         memory_key_padding_mask,
         n_s_t0,
         rel_state,
         encoded_lane,
         encoded_map,
         robot_enc,
         total_edge_influence
         ) = self.obtain_encoded_tensors(mode=mode,
                                         inputs=inputs,
                                         inputs_st=inputs_st,
                                         inputs_lane=inputs_lane,
                                         map=map,
                                         robot=robot,
                                         neighbors=neighbors,
                                         neighbors_edge_value=neighbors_edge_value)

        history_pred, lane_pred, lane_attn = self.train_decoder(
            inputs_st,
            self.hyperparams["autoregressive"],
            memory,
            memory_src_mask,
            memory_key_padding_mask,
            labels_st,
            n_s_t0,
            encoded_lane,
            encoded_map,
            prediction_horizon,
            robot_enc,
            total_edge_influence
        )
        # using hyperparam to choose loss mode
        reg_loss = 0
        cls_loss = 0
        con_loss = 0
        viol_rate = 0

        if self.hyperparams["lane_cnn_encoding"]:
            lane_pred = torch.stack(lane_pred, dim=1)
            lane_reg_loss = L2_norm(lane_pred, torch.unsqueeze(
                labels_st, 1))  # [bs, lane_num, timestep]
            lane_reg_loss = torch.sum(
                lane_reg_loss, dim=-1) / prediction_horizon  # [bs, lane_num]
            lane_min_loss = torch.min(lane_reg_loss, dim=-1)[0]
            lane_mask = torch.ones(lane_min_loss.size(), device=self.device)
            # cls_loss = classification_loss(lane_label, lane_attn)
            # cls_loss = cls_loss / torch.sum(lane_mask)
            reg_loss = lane_min_loss
        else:
            history_reg_loss = L2_norm(history_pred, labels_st)
            history_reg_loss = torch.sum(
                history_reg_loss, dim=-1) / prediction_horizon  # [nbs]
            lane_mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss

        reg_loss = torch.sum(reg_loss) / torch.sum(lane_mask)
        loss = reg_loss + cls_loss + con_loss

        if self.log_writer is not None:
            self.log_writer.add_scalar(
                "%s/%s" % (str(self.node_type), "loss"), loss, self.curr_iter)

        y_dist, _ = self.GMM(ModeKeys.PREDICT, memory, history_pred, n_s_t0,
            prediction_horizon, num_samples=1, num_components=1)

        # We use unstandardized labels to compute the loss
        log_p_yt = torch.clamp(y_dist.log_prob(labels), max=6)
        log_p_y = torch.sum(log_p_yt, dim=2)
        log_p_y_mean = torch.mean(log_p_y, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_mean)
        nll = -log_likelihood

        return loss, reg_loss

    def eval_loss(
        self,
        node_type,
        inputs,
        inputs_st,
        inputs_lane,
        lane_label,
        lane_t_mask,
        labels,
        labels_st,
        robot,
        map,
        prediction_horizon,
        neighbors,
        neighbors_edge_value,        
    ) -> torch.tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        (memory,
         memory_mask,
         memory_key_padding_mask,
         n_s_t0,
         rel_state,
         encoded_lane,
         encoded_map,
         robot_enc,
         total_edge_influence
         ) = self.obtain_encoded_tensors(mode=mode,
                                         inputs=inputs,
                                         inputs_st=inputs_st,
                                         inputs_lane=inputs_lane,
                                         map=map,
                                         robot=robot,
                                         neighbors=neighbors,
                                         neighbors_edge_value=neighbors_edge_value)

        history_pred, lane_pred, lane_attn = self.test_decoder(
            inputs_st,
            self.hyperparams["autoregressive"],
            memory,
            memory_mask,
            memory_key_padding_mask,
            n_s_t0,
            encoded_lane,
            encoded_map,
            prediction_horizon,
            robot_enc,
            total_edge_influence
        )

        node_pos = inputs[:, [-1], 0:2]  # get init state
        bs, _, feature_dim = node_pos.size()

        reg_loss = 0
        cls_loss = 0
        con_loss = 0
        viol_rate = 0
        # using hyperparam to choose loss mode
        if self.hyperparams["lane_cnn_encoding"]:
            lane_pred = torch.stack(lane_pred, dim=1)
            max_lane_num = lane_pred.size()[1]
            lane_pred = lane_pred * 80 + \
                node_pos.view(bs, 1, 1, feature_dim).repeat(
                    1, max_lane_num, 1, 1)
            lane_reg_loss = L2_norm(lane_pred, torch.unsqueeze(
                labels, 1))  # [bs, lane_num, timestep]
            lane_reg_loss = torch.sum(
                lane_reg_loss, dim=-1) / prediction_horizon  # [bs, lane_num]
            lane_min_loss = torch.min(lane_reg_loss, dim=-1)[0]
            lane_mask = torch.ones(lane_min_loss.size(), device=self.device)
            # cls_loss = classification_loss(lane_label, lane_attn)
            # cls_loss = cls_loss / torch.sum(lane_mask)
            reg_loss = lane_min_loss
        else:
            history_pred = history_pred * 80 + node_pos
            history_reg_loss = L2_norm(history_pred, labels)
            history_reg_loss = torch.sum(
                history_reg_loss, dim=-1) / prediction_horizon  # [nbs]
            lane_mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss

        reg_loss = torch.sum(reg_loss) / torch.sum(lane_mask)
        loss = reg_loss + cls_loss + con_loss

        y_dist, _ = self.GMM(ModeKeys.PREDICT, memory, history_pred, n_s_t0,
                            prediction_horizon, num_samples=1, num_components=1)

        # We use unstandardized labels to compute the loss
        log_p_yt = torch.clamp(y_dist.log_prob(labels), max=6)
        log_p_y = torch.sum(log_p_yt, dim=2)
        log_p_y_mean = torch.mean(log_p_y, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_mean)
        nll = -log_likelihood
    
        return loss, reg_loss
        #return nll, nll

    def predict(
            self,
            inputs,
            inputs_st,
            inputs_lane,
            lane_t_mask,
            robot,
            map,
            prediction_horizon,
            neighbors,
            neighbors_edge_value,
            num_samples,
            ):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        node_pos = inputs[:, [-1], 0:2]  # get init state
        bs, _, feature_dim = node_pos.size()
        mode = ModeKeys.EVAL

        (memory,
         memory_mask,
         memory_key_padding_mask,
         n_s_t0,
         rel_state,
         encoded_lane,
         encoded_map,
         robot_enc,
         total_edge_influence
         ) = self.obtain_encoded_tensors(mode=mode,
                                         inputs=inputs,
                                         inputs_st=inputs_st,
                                         inputs_lane=inputs_lane,
                                         map=map,
                                         robot=robot,
                                         neighbors=neighbors,
                                         neighbors_edge_value=neighbors_edge_value)
        history_pred, lane_pred, lane_attn = self.test_decoder(
            inputs_st,True, memory, memory_mask, memory_key_padding_mask, n_s_t0, encoded_lane, encoded_map, prediction_horizon,robot_enc,total_edge_influence)
        #print('history_pred1',history_pred)
        if self.hyperparams["lane_cnn_encoding"]:
            lane_pred = torch.stack(lane_pred, dim=1)
            max_lane_num = lane_pred.size()[1]
            lane_pred = lane_pred * 80 + \
                node_pos.view(bs, 1, 1, feature_dim).repeat(
                    1, max_lane_num, 1, 1)
        else:
            history_pred = history_pred * 80 + node_pos
            # print('history_pred',history_pred)
            #print('history_pred',len(history_pred))

        _, our_sampled_future = self.GMM(ModeKeys.PREDICT, memory, history_pred, n_s_t0,
                            prediction_horizon, num_samples=num_samples, num_components=1)

        #return history_pred, lane_pred
        return our_sampled_future, lane_pred
