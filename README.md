This repository contains the code for "Multi-scale Feature Aggregation Autoregressive Network for Multi-Agent Trajectory Prediction" by Ronghao Li, Yin Shen.

# Installation
## Environment
**Tested OS**: Windows10, Ubuntu 20.04  
Python >= 3.7.1  
torch == 1.11.0+cu113  
nuscenes-devkit == 1.1.10

**Install the dependencies**:   
```
conda create --name Multimodal_Transformer python=3.7 -y  
source activate Multimodal_Transformer  
pip install -r requirements.txt
```
## Datasets
**For the ETH/UCY dataset,** 
we've already included preprocessed data splits for the ETH and UCY Pedestrian datasets in this repository, you can see them in experiments/pedestrians/raw. In order to process them into a data format that our model can work with, execute the follwing.
```
cd data/pedestrians
python process_data.py
```
**For the nuScenes dataset, the following steps are required:**   
a. Download the trainval metadata of Full dataset(v1.0) on the [nuScenes website](https://www.nuscenes.org/ "nuScenes").  
b. Checkout the instructions [here](https://github.com/nutonomy/nuscenes-devkit "nuscenes-devkit") to install devkit for nuScenes.
  `pip install nuscenes-devkit`   
c. Follow the instructions of [nuScenes prediction challenge](https://www.nuscenes.org/prediction?externalData=all&mapData=all&modalities=Any "prediction challenge"). Download and install the map expansion pack (v1.3). Unzip the metadata and map expansion to your data folder, the file organization structure is as follows:
```
${MFAA-Net root}
├── data
    `-- ├── v1.0
      `-- ├── maps --> map expansionmap pack unzip here
          ├── v1.0-trainval --> metadata unzip here
```  
d. Run our script to obtain a processed version of the nuScenes dataset:  
```
cd experiments/nuScene
python nuScenes_process_data.py --version=v1.0-trainval --output_path=../processed_data --data=../../data/v1.0/
```
# Model Training
When training and testing the model, pay attention to the following: **on the ETH/UCY datasets**, the parameters of `argument_parser.py` and `config.json` are used; while the parameters of `nu-argument_parser.py` and `nu-config.json` are used **on the nuScenes dataset**. `argument_parser.py` and `nu-argument_parser.py` are under the `Transformer/`,  `config.json` and `nu-config.json` are under the `experiments/nuScene/config/`.

## ETH/UCY
To train a model on the ETH/UCY datasets, you can execute one of the following commands from within the `Transformer/` directory, depending on which pedestrian scene you choose for your training. The following commands are all based on `Base` mode. If you wish to use the `Base + Social-graphs` mode commands, please remove `--no_edge_encoding` from each command.

Dataset  | Command
 ---- | -----
 ETH  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/pedestrian  --log_tag eth --augment --device "cuda:0" --data_name eth --no_edge_encoding` 
 HOTEL  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/pedestrian  --log_tag hotel --augment --device "cuda:0" --data_name hotel --no_edge_encoding`  
 UNIV  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/pedestrian  --log_tag univ --augment --device "cuda:0" --data_name univ --no_edge_encoding`  
 ZARA1  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/pedestrian  --log_tag zara1 --augment --device "cuda:0" --data_name zara1 --no_edge_encoding`  
 ZARA2  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/pedestrian  --log_tag zara2 --augment --device "cuda:0" --data_name zara2 --no_edge_encoding` 

## nuScenes
To train a model on the nuScenes dataset, you can execute one of the following commands from within the `Transformer/` directory, depending on the model version you desire. In particular, if the model version is Base + Social-graphs + Ego-robot + Maps, change `"learning_rate": 0.003` in `nu-config.json` under `experiments/nuScene/config/` to `"learning_rate ": 0.0045`.
Model  | Command
 ---- | -----
 Base  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag his --augment --device "cuda:0" --no_edge_encoding`
 Base + Ego-robot  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag his_robot --augment --device "cuda:0" --incl_robot_node --no_edge_encoding`  
 Base + Maps  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag his_map --augment --device "cuda:0" --map_vit_encoding --no_edge_encoding`  
 Base + Ego-robot, Maps  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag his_map_robot --augment --device "cuda:0" --incl_robot_node --map_vit_encoding --no_edge_encoding` 
 Base + Social-graphs  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag soc --augment --device "cuda:0"` 
 Base + Social-graphs, Ego-robot  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag soc_robot --augment --device "cuda:0" --incl_robot_node`  
 Base + Social-graphs, Maps  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag soc_map --augment --device "cuda:0" --map_vit_encoding`  
 Base + Social-graphs, Ego-robot, Maps  | `python train.py --node_freq_mult_train --log_dir ../experiments/nuScene/models/nuscene  --log_tag soc_map_robot --augment --device "cuda:0" --incl_robot_node --map_vit_encoding` 

# Model Evaluation
To evaluate a trained model, you can execute one of the following commands from within the `experiments/nuScene/` directory, depending on which pedestrian scene you choose for your evaluation. 
**NOTE: Before evaluation, please comment out `x = self.norm(x)` on line 400 of `Transformer_model.py` under `Transformer/model/components/`.** This is used to speed up training and has an adverse impact on the evaluation results.  
## ETH/UCY
To evaluate a model on the nuScenes dataset, you can execute one of the following commands from within the `experiments/nuScene/` directory, depending on which pedestrian scene you choose for your evaluation. You can choose the future horizon by modifying the parameter value in the following command (`prediction_horizon=12/8` corresponds to predicting the 4.8s/3.2s future respectively).

Dataset  | Command
 ---- | -----
 ETH  | `python evaluate-eth.py --model models/pedestrian/eth --checkpoint=best_epoch --data ../processed_data/eth_test_map_full.pkl --output_path results --output_tag eth --node_type PEDESTRIAN --prediction_horizon 12`    
 HOTEL  | `python evaluate-eth.py --model models/pedestrian/hotel --checkpoint=best_epoch --data ../ETH/hotel_test_map_full.pkl --output_path results --output_tag hotel --node_type PEDESTRIAN --prediction_horizon 12`    
 UNIV  | `python evaluate-eth.py --model models/pedestrian/univ --checkpoint=best_epoch --data ../ETH/univ_test_map_full.pkl --output_path results --output_tag univ --node_type PEDESTRIAN --prediction_horizon 12`     
 ZARA1  | `python evaluate-eth.py --model models/pedestrian/zara1 --checkpoint=best_epoch --data ../ETH/zara1_test_map_full.pkl --output_path results --output_tag zara1 --node_type PEDESTRIAN --prediction_horizon 12`     
 ZARA2  | `python evaluate-eth.py --model models/pedestrian/zara2 --checkpoint=best_epoch --data ../ETH/zara2_test_map_full.pkl --output_path results --output_tag zara2 --node_type PEDESTRIAN --prediction_horizon 12`   

`best_epoch` is the number of epochs during 150-epoch-training that performs best on the validation set. Based on observation and analysis, we provide the following numbers as references for `best_epoch` of the five datasets:  
`Base` mode
Dataset | ETH | HOTEL | UNIV | ZARA1 | ZARA2
---- | ----- | ----- | ----- | ----- | -----
best_epoch | 145 | 150 | 150 | 148 | 150

`Base + Social-graphs` mode:
Dataset | ETH | HOTEL | UNIV | ZARA1 | ZARA2
---- | ----- | ----- | ----- | ----- | -----
best_epoch | 150 | 150 | 132 | 146 | 148

## nuScenes
To evaluate a model on the nuScenes dataset, you can execute one of the following commands from within the `experiments/nuScene/` directory, depending on the model version you desire. You can choose the future horizon by modifying the parameter value in the following command (`prediction_horizon=12/8/6/4/2` corresponds to predicting the 6s/4s/3s/2s/1s future respectively).

Model  | Command
 ---- | -----
 Base  | `python evaluate-nu.py --model models/nuscene/his --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag his --node_type VEHICLE --prediction_horizon 12`
 Base + Ego-robot  | `python evaluate-nu.py --model models/nuscene/his_robot --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag his_robot --node_type VEHICLE --prediction_horizon 12`  
 Base + Maps  | `python evaluate-nu.py --model models/nuscene/his_map --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag his_map --node_type VEHICLE --prediction_horizon 12`  
 Base + Ego-robot, Maps  | `python evaluate-nu.py --model models/nuscene/his_map_robot --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag his_map_robot --node_type VEHICLE --prediction_horizon 12`  
 Base + Social-graphs  | `python evaluate-nu.py --model models/nuscene/soc --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag soc --node_type VEHICLE --prediction_horizon 12` 
 Base + Social-graphs, Ego-robot  | `python evaluate-nu.py --model models/nuscene/soc_robot --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag soc_robot --node_type VEHICLE --prediction_horizon 12`   
 Base + Social-graphs, Maps  | `python evaluate-nu.py --model models/nuscene/soc_map --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag soc_map --node_type VEHICLE --prediction_horizon 12`  
 Base + Social-graphs, Ego-robot, Maps  | `python evaluate-nu.py --model models/nuscene/soc_map_robot --checkpoint=9 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag soc_map_robot --node_type VEHICLE --prediction_horizon 12`
 
# Quantitative Results
## ETH/UCY
Run `pes_quantitative.py` under the `experiments/nuScene/` folder to obtain quantitative results. You can obtain the results of different pedestrian datasets by changing `eth` in line 7 of the code to any one of `eth, hotel, univ, zara1, zara2`.
If you want to get quantitative results under different prediction horizons, please modify the number in brackets in line 9 of the code `for ph in [12]:`.
## nuScenes
Run `nu_quantitative.py` under the `experiments/nuScene/` folder to obtain quantitative results. You can obtain the results of different model versions by changing `his` in line 7 of the code to any one of `his, his_robot, his_map, his_map_robot, soc, soc_robot, soc_map, soc_map_robot`.
If you want to get quantitative results under different prediction horizons, please modify the number in brackets in line 9 of the code `for ph in [12]:`.

# Acknowledgments
Without this repo, I could not complete my project:
[Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)
