# Random Deep Learning experiments

- See training code and inference results in [notebooks](https://github.com/einstalek/DL-models-experiments/tree/master/notebooks)
- Implemented models are in [models](https://github.com/einstalek/DL-models-experiments/tree/master/models)

## Object Detection and Classification on [Cats-and-Dogs dataset](https://www.kaggle.com/andrewmvd/dog-and-cat-detection)

No postprocessing added to Unet-based models

Model  |  cat AP  |  dog AP  | mAP
:--:|:--:|:--:|:--:
Unet  | 51.62 | 67.31 | 59.47
Unet + SEBlocks  | 95.19 | 86.73 | 90.96
RetinaNet | 46.38 | 68.65 | 57.52

Result samples from **SEUnet**

<img src="./images/cat_dog_seunet.png " width="384" height="384">

## Image generation on [Facades dataset](https://www.kaggle.com/balraj98/facades-dataset)
Model  |  pix2pix | sagan
:--:|:--:|:--:
Samples | <img src="./images/pix2pix_samples.png " width="256" height="256"> | <img src="./images/sagan_samples.png " width="256" height="256">

## Deep Fashion on [Viton dataset](https://github.com/xthan/VITON)

Reproduced models and pipeline from [VITON: An Image-based Virtual Try-on Network](https://arxiv.org/pdf/1711.08447.pdf)

Result samples:

<img src="./images/viton.png " width="448" height="448">
