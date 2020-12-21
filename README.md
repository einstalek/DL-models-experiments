## Random Deep Learning experiments

- See training code and inference results in [notebooks](https://github.com/einstalek/DL-models-experiments/tree/master/notebooks)
- Implemented models are in [models](https://github.com/einstalek/DL-models-experiments/tree/master/models)

### Object Detection and Classification on [Cats-and-Dogs](https://www.kaggle.com/andrewmvd/dog-and-cat-detection) dataset
Model          |  mAP
:-------------------------:|:-------------------------:
Unet, Resnet50 backbone  | 51.62% cat AP, 67.31% dog AP 
Unet, SEResnet50 backbone  | 95.19% cat AP, 86.73% dog AP 
