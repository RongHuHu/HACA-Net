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
* For the ETH/UCY dataset, we've already included preprocessed data splits for the ETH and UCY Pedestrian datasets in this repository, you can see them in experiments/pedestrians/raw. In order to process them into a data format that our model can work with, execute the follwing.
```
cd pedestrian_data/pedestrians
python process_data.py
```
* For the nuScenes dataset, the following steps are required:  
  a. Download the trainval metadata of Full dataset(v1.0) on the [nuScenes website](https://www.nuscenes.org/ "nuScenes").  
  b. Checkout the instructions [here](https://github.com/nutonomy/nuscenes-devkit "nuscenes-devkit") to install devkit for nuScenes.
  `pip install nuscenes-devkit`   
  c. Follow the instructions of [nuScenes prediction challenge](https://www.nuscenes.org/prediction?externalData=all&mapData=all&modalities=Any "prediction challenge"). Download and install the map expansionmap pack    (v1.3). Unzip the metadata and map expansion to your data folder, the file organization structure is as follows:
```
${root}
├── data
    `-- ├── v1.0
      `-- ├── maps --> map expansionmap pack unzip here
          ├── v1.0-trainval --> metadata unzip here
```
  c. Run our script to obtain a processed version of the nuScenes dataset under datasets/nuscenes_pred:  
    `python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>`
