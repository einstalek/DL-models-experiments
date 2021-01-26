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

Samples for SEUnet

<img src="./images/cat_dog_seunet.png " width="384" height="384">

## Image generation on [Facades dataset](https://www.kaggle.com/balraj98/facades-dataset)
Model  |  pix2pix | sagan
:--:|:--:|:--:
Samples | <img src="./images/pix2pix_samples.png " width="256" height="256"> | <img src="./images/sagan_samples.png " width="256" height="256">

## Deep Fashion on [Viton dataset](https://github.com/xthan/VITON)

The model is complex and consists of multiple stages

Stage | Result 
:--:|:--:
Coarse transform | <img src="./images/unet_viton_coarse.png " width="384" height="384">
