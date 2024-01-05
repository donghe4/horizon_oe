# horizon_oe

## Quickstart

### 0. Setting Horizon Env.
```python 
$bash run_docker.sh /data gpu
```
### 1. Prepare Dataset 

pack nuscenes dataset

```python
# To pack train set
$python3 ./tools/nuscenes_packer.py --src-data-dir ./data/nuscenes/ --pack-type lmdb --target-data-dir ./tmp_data/nuscenes/v1.0-trainval --version v1.0-trainval --split-name train
# To pack val set
$python3 ./tools/nuscenes_packer.py --src-data-dir ./data/nuscenes/ --pack-type lmdb --target-data-dir ./tmp_data/nuscenes/v1.0-trainval --version v1.0-trainval --split-name val
```

The tree strcture of nuscenes packed is as following:
```
|-- tmp_data 
|   |-- nuscenes 
|   |   |-- v1.0-trainval
|   |   |   |-- train_lmdb  #packed train set
|   |   |   |   |-- data.mdb
|   |   |   |   `-- lock.mdb
|   |   |   `-- val_lmdb   #packed val set
|   |   |   |   |-- data.mdb
|   |   |   |   `-- lock.mdb
```

Decompress `v1.0-trainval_meta.tar` and `nuscenes-map-extension-v1.3.zip`, then reconstruct as following:

```
|-- tmp_data 
|   |-- nuscenes 
|   |   |-- meta
|   |   |   |-- maps  
|   |   |   |   |-- 36092f0b03a857c6a3403e25b4b7aab3.png
|   |   |   |   |-- ...
|   |   |   |   |-- 93406b464a165eaba6d9de76ca09f5da.png
|   |   |   |   |-- prediction
|   |   |   |   |-- basemap
|   |   |   |   |-- expansion
|   |   |   |-- v1.0-trainval  
|   |   |       |-- attribute.json
|   |   |           ...
|   |   |       |-- visibility.json
|   |   `-- v1.0-trainval 
|   |   |   |-- train_lmdb  
|   |   |   `-- val_lmdb   
```
### 2. Train 
modify configs according to your device
```python 
$python3 tools/train.py --config configs/bev_ipm_efficientnetb0_multitask_nuscenes.py --stage float
```

### 3. Evaluation

```python 
$python3 tools/predic.py --config configs/bev_ipm_efficientnetb0_multitask_nuscenes.py --stage float
```

### 4. Infer
```python
python3 tools/infer.py --config configs/bev/bev_mt_ipm_efficientnetb0_nuscenes.py --model-inputs imagedir:${imagedir},homo:${homography.npy}
```


## For more details, please find in https://sapeon.atlassian.net/wiki/spaces/~6413f81a57f0c028e2f53406/pages/edit-v2/284229893
