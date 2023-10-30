# Installation
## Environment
**Tested OS**: Windows, Linux  
Python == 3.7  
**Install the dependencies**:   
`conda create --name MFAANet python=3.7 -y`  
`source activate MFAANet`  
`pip install -r requirements.txt`

# Datasets
* For the ETH/UCY dataset, we already included a converted version compatible with our dataloader under datasets/eth_ucy.
* For the nuScenes dataset, the following steps are required:  
  a. Download the orignal nuScenes dataset. Checkout the instructions here.  
  b. Follow the instructions of nuScenes prediction challenge. Download and install the map expansion.  
  c. Run our script to obtain a processed version of the nuScenes dataset under datasets/nuscenes_pred:  
    `python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>`
