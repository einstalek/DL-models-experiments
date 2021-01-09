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
Model  |  Generated samples
:--:|:--:
pix2pix  |  <img src="./images/pix2pix_samples.png " width="224" height="224">
