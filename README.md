# Random Deep Learning experiments

- See training code and inference results in [notebooks](https://github.com/einstalek/DL-models-experiments/tree/master/notebooks)
- Implemented models are in [models](https://github.com/einstalek/DL-models-experiments/tree/master/models)

## Object Detection and Classification on [Cats-and-Dogs dataset](https://www.kaggle.com/andrewmvd/dog-and-cat-detection)
Model  |  cat AP  |  dog AP  | mAP
:--:|:--:|:--:|:--:
Unet  | 51.62 | 67.31 | 59.47
Unet + SEBlocks  | 95.19 | 86.73 | 90.96
RetinaNet | 46.38 | 68.65 | 57.52

## Image generation on [Facades dataset](https://www.kaggle.com/balraj98/facades-dataset)
Model  |  pix2pix | sagan
:--:|:--:|:--:
Samples | <img src="./images/pix2pix_samples.png " width="256" height="256"> | <img src="./images/sagan_samples.png " width="256" height="256">

## Deep Fashion on [Viton dataset](https://github.com/xthan/VITON)
Stage | result 
:--:|:--:
Coarse transform | <img src="./images/unet_viton_coarse.png " width="384" height="384">
