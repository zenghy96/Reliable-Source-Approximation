# Reliable source approxmation (RSA)
Official implementation of Early Accepted MICCAI'2024 paper [Reliable Source Approximation: Source-Free Unsupervised Domain Adaptation for Vestibular Schwannoma MRI Segmentation]().

## Introduction
Source-Free Unsupervised Domain Adaptation (SFUDA) has recently become a focus in the medical image domain adaptation, as it only utilizes the source model and does not require annotated target data. However, current SFUDA approaches cannot tackle the complex segmentation task across different MRI sequences, such as the vestibular schwannoma segmentation. To address this problem, we proposed Reliable Source Approximation (RSA), which can generate source-like and structure-preserved images from the target domain for updating model parameters and adapting domain shifts. Specifically, RSA deploys a conditional diffusion model to generate multiple source-like images under the guidance of varying edges of one target image. An uncertainty estimation module is then introduced to predict and refine reliable pseudo labels of generated images, and the prediction consistency is developed to select the most reliable generations. Subsequently, all reliable generated images and their pseudo labels are utilized to update the model. Our RSA is validated on vestibular schwannoma segmentation across multi-modality MRI.  The experimental results demonstrate that RSA consistently improves domain adaptation performance over other state-of-the-art SFUDA methods.

![Alt text](pics/RSA_flow.png)


## Requirements

## Data preparation
We used MR images included contrast-enhanced T1-weighted (ceT1) images and high-resolution T2-weighted (hrT2) images. The original dataset can be found [here](https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-seg/), and you can use the [official data preprocess code](https://github.com/KCL-BMEIS/VS_Seg).

Alternatively, you can directly use the [data we have  preprocessed and split](https://drive.google.com/file/d/1eV_23hgthHMx7TK3H-_Q54rvF-oEwl8_/view?usp=sharing). 

Download above data and organize the dataset directory structure as follows:
```
data/
    T1/
        training/
        testing/
        validation/
     T2/
        training/
        testing/
        validation/
```


## Pretrain on source dataset
Pretrain diffusion model:

Pretrain segmentation model:

## Domain adaptation
Translate T2 images to T1 using edge:
```asdasd```

Select reliable samples:

Finetune pretrained segmentation model:

## Testing