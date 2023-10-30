# Installation
## Environment
**Tested OS**: Windows, Linux  
Python >= 3.7  
**Install the dependencies**:   
`pip install -r requirements.txt`

# Datasets
* For the ETH/UCY dataset, we already included a converted version compatible with our dataloader under datasets/eth_ucy.
* For the nuScenes dataset, the following steps are required:
  >Download the orignal nuScenes dataset. Checkout the instructions here.
  >Follow the instructions of nuScenes prediction challenge. Download and install the map expansion.
  >Run our script to obtain a processed version of the nuScenes dataset under datasets/nuscenes_pred:
    'python data/process_nuscenes.py --data_root <PATH_TO_NUSCENES>'
