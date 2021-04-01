# Wireless Capsule Endoscopy Classification 

## Datasets
I am using publicly available [KID datasets](https://mdss.uth.gr/datasets/endoscopy/kid/) for training and testing purpose.  

KID1 dataset consists of nine abnormalities: Angioectasias, Aphthae, Bleeding, ChylousCysts, Lymphangectasias, Polypoids, Stenoses, Ulcers, and Villous Oedemas. 

KID2 dataset contains four abnormalities: Ampulla of Vater, Inflammatory, Polypoids, and Vascular. 

Since annotated data is inadequate to get good classification accuracy, I am using [Albumentations](https://albumentations.ai/docs/) for data augmentation. 

## Using ResNet 

![Train Images](./sample/trainimages.png)
*Figure 1: Images used for training the model.*
![Test Images](./sample/testimages.png)
*Figure 2: Images used for testing the model.*
![Filtered Images](./sample/filtered.png)
*Figure 3: Outputs of the filters performing various image processing techniques like edge detection, colour invervion etc.*
![Learning rate](./sample/lr_finder.png)
*Figure 4: Learning rate finder - around 1x10^{-3} loss is mimimum.*

## Using Vision Transformers

Implementation of [Vision Transformers](https://arxiv.org/abs/2010.11929) paper for the classification task of the trails generated by a Wireless Capsule Endoscopy (WCE).

![Figure 5 from paper](./sample/vit_model.png)
*Figure 5: Illustrates the overview of the model. The images are first divided into patches followed by position embedding, which then passed to the Transformer encoder. To perform classification, an extra learnable “classification token” is added to the sequence.*

## Results
Abnormalities: Ampulla of Vater, Inflammatory, Polypids, and Vascular. 

Train images = 478 | Validation images = 54 | Testing images = 61 

*Table 1: Comparative analysis of multiclass classification of ResNet and Vision Transformer models on KID2.*

| Model              | #Images | Accuracy |
|--------------------|---------|----------|
| Resnet50           | 593     | 0.84     |
| Vision Transformer | 593     | 0.85     |

---
### Important Links:

[Official Vision Transformer Github Repo](https://github.com/google-research/vision_transformer)
|
[Pytorch Model of Vision Transformer](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models)

---