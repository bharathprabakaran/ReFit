# MIDL_BoundaryCAM

<img src="img/frame.pdf" width="800" height="447"/>

## Abstract
Weakly Supervised Semantic Segmentation (WSSS) with only image-level supervision is
a promising approach to deal with the need of Segmentation networks, especially for
generating large number of pixel-wise masks in a given dataset. However, most state-
of-the-art image-level WSSS techniques lack an understanding of the geometric features
embedded in the images since the network cannot derive any object boundary information
from just image-level labels. We define a boundary here as the line separating the object and
background. To address this drawback, we propose our novel BoundaryCAM framework,
which deploys state-of-the-art class activation maps combined with various post-processing
techniques in order to achieve fine-grained higher-accuracy segmentation masks. To achieve
this, we investigate a wide-range of state-of-the-art unsupervised semantic segmentation
networks that can be used to construct a boundary map, which enables BoundaryCAM
to predict object locations with sharper boundaries. By applying our method to WSSS
predictions, we were able to achieve up to 10% improvements even to the benefit of the
current state-of-the-art WSSS methods for medical imaging.


## Getting Started

### Minimum requirements

1. Dependencies :

matplotlib 2.2.2
numpy 1.14.5
Pillow 5.2.0
scikit-image 0.14.0
scikit-learn 0.19.1
scipy 1.1.0
tensorboardX 1.2
torch 1.4.0
torchvision 0.5.0
nibabel

 2. Hardware :

### Download data

#### Dataset

1. The BraTS-2020 dataset can downloaded from this [link](https://www.med.upenn.edu/cbica/brats2020/data.html)
2. The preprocessed and 3-fold cross-validation split of prostate DECATHALON dataset WSS-CMER's link. [link](https://github.com/gaurav104/WSS-CMER).

Basic dataset folder structure, using Prostate dataset as an exemplary. (Note: Make sure to change the dataset directory accordingly inside the config file )


## Train Decathlon Data
Please set all paths as mentioned at the top of every program.

1.
```python
python deca_Classifier.py
```
2.
```
python deca_USS.py
```
3.
```
python deca_GradCAM.py
```
4.
```
python deca_BOUNDARY_FIT.py
```
5.
```
python deca_eval.py
```
