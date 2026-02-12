# EndoUFM
This is a PyTorch implementation for **EndoUFM: Utilizing Foundation Models for Monocular Depth Estimation of endoscopic images**

## Evaluation
1. Download pretrained checkpoint from: [depth_model](https://drive.google.com/drive/folders/1Nyi9E_LDHeWW8AnkraxpTyWuUwG8BS0p?usp=drive_link). Create a folder named ```pretrained_checkpoints``` in this repo and place the downloaded model in it.
2. Evaluate the model
```
python evaluate_depth_new.py --data_path <your_data_path> --load_weights_folder './pretrained_checkpoints'
```
