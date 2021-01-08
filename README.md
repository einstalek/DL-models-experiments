# Random Deep Learning experiments

- See training code and inference results in [notebooks](https://github.com/einstalek/DL-models-experiments/tree/master/notebooks)
- Implemented models are in [models](https://github.com/einstalek/DL-models-experiments/tree/master/models)

## Object Detection and Classification on [Cats-and-Dogs](https://www.kaggle.com/andrewmvd/dog-and-cat-detection) dataset
Model  |  cat AP  |  dog AP  | mAP
:--:|:--:|:--:|:--:
Unet  | 51.62 | 67.31 | 59.47
Unet + SEBlocks  | 95.19 | 86.73 | 90.96
RetinaNet | 46.38 | 68.65 | 57.52 
