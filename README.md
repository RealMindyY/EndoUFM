# EndoUFM
This is a PyTorch implementation for **EndoUFM: Utilizing Foundation Models for Monocular Depth Estimation of endoscopic images**

## Setup
We ran our experiments with CUDA 11.8, PyTorch 1.11.0, and Python 3.8.

Download pretrained model from: [depth_anything_vitb14](https://github.com/LiheYoung/Depth-Anything). 
Create a folder named ```pretrained_model``` in this repo and place the downloaded model in it.

Download pretrained model from: [medsam_vit_b](https://github.com/bowang-lab/MedSAM). 
Place the downloaded model in the folder ```segment_anything```.

## Data Preparation

1. Download the dataset: [SCARED](https://endovissub2019-scared.grand-challenge.org/Home/).
2. The training/test split for SCARED in our works is defined in the `splits/endovis` and further preprocessing is available in [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner).

## Training
```
python train_end_to_end.py --data_path <your_data_path> --log_dir './results'
```

## Evaluation
1. Download pretrained checkpoint from: [depth_model](https://drive.google.com/drive/folders/1Nyi9E_LDHeWW8AnkraxpTyWuUwG8BS0p?usp=drive_link). Create a folder named ```pretrained_checkpoints``` in this repo and place the downloaded model in it.
2. Evaluate the model:
```
python evaluate_depth_new.py --data_path <your_data_path> --load_weights_folder './pretrained_checkpoints'
```

## Acknowledgement
Thanks the authors for their excellent works:

[AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner)

[IID-SfMLearner](https://github.com/bobo909/IID-SfmLearner)

[EndoDAC](https://github.com/BeileiCui/EndoDAC)

[DepthAnything](https://github.com/LiheYoung/Depth-Anything)

[SegmentAnything](https://github.com/facebookresearch/segment-anything)

[MedSAM](https://github.com/bowang-lab/MedSAM)
