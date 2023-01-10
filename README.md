# MIDL_BoundaryCAM

<img src="img/frame.pdf" width="800" height="447"/>

## Abstract
Reliable classification and detection of certain medical conditions in images with state-of-the-art semantic segmentation networks require vast amounts of pixel-wise annotation. However, the public availability of such datasets is minimal. Therefore, weakly supervised semantic segmentation presents a promising alternative to this problem. Nevertheless, few works focus on applying weakly supervised semantic segmentation to the medical sector. Due to their complexity and the small number of training examples of many datasets in the medical sector, classifier-based weakly supervised networks like class activation maps (CAMs) struggle to extract useful information from them. However, most state-of-the-art approaches rely on them to achieve their improvements. Therefore, we propose our EnsembleCAM framework that can still utilize the low-quality CAM predictions of complicated datasets to improve the accuracy of our results. Towards that, EnsembleCAM exploits our observations: first, low-quality CAM predictions, on low enough thresholds, often covers the target object completely, and second, our observation that the false positives of different low-quality CAMs vary from CAM to CAM. We performed exhaustive experiments on the popular multi-modal BRATS and prostate DECATHLON segmentation challenge datasets. Using the proposed framework, we have demonstrated an improved dice score of up to 8\% on BRATS and 6\% on DECATHLON compared to the previous state-of-the-art.


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
Please set all paths as mentioned at the top of every program 
```python
python deca_Classifier.py
python deca_USS.py
python deca_GradCAM.py
python deca_BOUNDARY_FIT.py
python deca_eval.py

```

```
