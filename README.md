# HOPE2Net

HOPE2Net (**H**ist**O**logy and **P**osition **E**mbedding **Net**work) is a multilayer perceptron that properly learns the balance between histology features and positions embeddings for prediction tasks such as gene expressions and pathway activities. HOPE2Net is a useful computational tool for analyses of spatial transcriptomics data, which builds the bridge between H&E-stained histology images and gene expressions. HOPE2Net can be applied to multiple tissue sections with or without images.

## Reproducibility

We described and introduced HOPE2Net in our [methodological paper](https://www.biorxiv.org/). To find code to reproduce the results we generated in that paper, please visit the result folder [github repository](https://github.com/suke18/HOPE2Net/tree/master/results), which provides all code (including that for other methods) necessary to reproduce our results.

## Usage

A [tutorial jupyter notebook](https://drive.google.com/drive/folders/1uEULKNCi20HQpTHQceQwcehW9ztVTiSR?usp=sharing), together with a dataset, is publicly downloadable.

```python
import torch
from MLP_model import plNet
Hope2Net = plNet(n_feature = 1000, n_hidden1 = 128, n_hidden2 = 64, n_output=1000)
x1 = torch.Tensor(200, 1000)
x2 = torch.Tensor(200, 1000)
Hope2Net(x1, x2)
```
This is an illustration of simple feedforward for Hope2Net. <img src="https://render.githubusercontent.com/render/math?math=X_{1}">
 mimics the image feature matrix and <img src="https://render.githubusercontent.com/render/math?math=X_{2}"> represents the position embeddings. When computing these two modalities of features, we enforce them with same dimension for the purpose of allocating weights. The weights will further be learned during the training process. 

## Feature extraction
We use transfer learning approach to extract deep image features for each tile where the barcoded spot is centered. Three state-of-art DL architectures are considered: VGG19, ResNet50, and ViT16. The pre-trained weights on imageNet1k dataset can be accessed by `keras.applicaiton` or `pytorch_pretrained_vit` libraries. Free free to utilize the implementation at [Tile_features.py](https://github.com/suke18/HOPE2Net/blob/master/scripts/Tile_features.py).


## Software Requirements

- Python >= 3.8
- Torch >= 2.0.1
- scanpy >= 1.5.1
- pytorch\_lightning
