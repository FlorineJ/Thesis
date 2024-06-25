# Port Land Use detection on Satellite imagery 
# Based on repository from Joel de Plaen @joel-deplaen-ivm

Pipeline for training data creation, training and inference of a MaskRCNN to detect port land uses in Europe using Google Satellite - 1 m resolution

## 0. Installation

### Instalation guide:

- conda: environment.yml
- Procedure torch, torchvision, detectron2

        pip3 install \
        torch==1.10.2 \
        torchvision==0.11.3 -extra-index-url https://download.pytorch.org/whl/cu113
        python -m pip install detectron2 -f \
        https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

- On cluster use:

        module load 2022
        module load CUDA/11.3.1

- Test environment by importing:

        import detectron2        
        import torch             
        import cv2 as cv         
        import numpy as np       
        from osgeo import gdal   
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg

- Verify torch, torchvision, cuda compatibility by running:

        python -m detectron2.utils.collect_env

        See: https://stackoverflow.com/questions/70831932/cant-connect-to-gpu-when-building-pytorch-projects
        or
        Python -c "import uutils; uutils.torch_uu.gpu_test()
        see: https://stackoverflow.com/questions/66992585/how-does-one-use-pytorch-cuda-with-an-a100-gpu

## 1. Data Preperation
- Aim at the preperation of the imagery and annotation for DL training

### 1.2 tiling_v02.ipynb
- Create tiles of satelite imagery and annotation for DL model training
- Should be added to overide the gdal .ini file in conda env:  

        osmconf.ini

- Also in *subs_detection/scripts/extract_osm_sub.py*:

        gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join('..',"osmconf.ini"))"
### 1.3 convert_tif_split_dataset_v02.ipynb
### 1.4 create_jsons_nso.ipynb or create_jsons_nso_no-annotations_variation.ipynb
## 2 Train Model
### 2.1 config_train_evaluate.ipynb
### 2.2 train.py
## 3. Run Model
### 3.1 inference_and_stiching.ipynb
