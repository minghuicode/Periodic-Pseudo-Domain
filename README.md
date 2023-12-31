# Periodic-Pseudo-Domain
High Efficient Anchor-Free Oriented Small Objects Detection for Remote Sensing Images via Periodic Pseudo-Domain

### (left) HBB: Horizontal Bounding Boxes; (right) OBB: Oriented Bounding Boxes
![](task.jpg)

## Running Enviroment

```
ubuntu 18.04
cuda-11.0
python 3.9.0
torch 2.0.0
```

## Installation

 1. download code

```
git clone https://github.com/minghuicode/Periodic-Pseudo-Domain
```

2. create conda enviroment

```
conda env create -f conda_env.yaml
source activate light
pip install -r pip_env.txt
```

3. enable skew-Iou based NMS in GPU, otherwise, skew-IoU based NMS only run in CPU

```
cd Periodic-Pseudo-Domain/utils
sh build.sh
```

## Model Training 

perpare training dataset

```
cd Periodic-Pseudo-Domain
conda activate light
python train.py
```
