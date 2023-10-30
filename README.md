# Installation
## Environment
**Tested OS**: Windows, Linux  
Python == 3.7  
**Install the dependencies**:   
```
conda create --name Multimodal_Transformer python=3.7 -y  
source activate Multimodal_Transformer  
pip install -r requirements.txt
```
# Datasets
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
c. Follow the instructions of [nuScenes prediction challenge](https://www.nuscenes.org/prediction?externalData=all&mapData=all&modalities=Any "prediction challenge"). Download and install the map expansionmap pack    (v1.3). Unzip the metadata and map expansion to your data folder, the file organization structure is as follows:
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
## ETH/UCY Pedestrian datasets
To train a model on the ETH/UCY datasets, you can execute one of the following commands from within the `Transformer/` directory, depending on which pedestrian scene you choose for your training.

Dataset  | Command
 ---- | ----- | ------  
 ETH  | 单元格内容 
 HOTEL  | 单元格内容 
 UNIV  | 单元格内容 
 ZARA1  | 单元格内容 
 ZARA2  | 单元格内容
 
## nuScenes dataset
To train a model on the nuScenes dataset, you can execute one of the following commands from within the `Transformer/` directory, depending on the model version you desire.
