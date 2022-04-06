# BCN 20000 tools for assembling and training

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons Lizenzvertrag" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a>

This repository gives access to the tools to train the models presented at the BCN 20000's dataset scientific publication. The dataset itself is available for download at the [ISIC archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%22meta.datasetId%7CBCN_20000%22%5D).

<hr>


## Cropping

The code for the cropping technique used on the dermatoscopies can be found at:
- [`preprocessing/image_cropper.py`](preprocessing/image_cropper.py):

The csv's with the image filename must be passed as a `--csv_dir` argument when executing the code.

![Original (a), (b) and (c) and cropped images (d), (e) and (f)](https://github.com/CarlosHernandezP/BCN-20k/blob/main/Cropped_uncropped_figure.png)


<hr>

## Training
In order to train a model, one should set the model's name at [`utils/settings.yaml`](utils/settings.yaml) for one of the following:
| **Settings name** |      Model      |   
|-------------------|-----------------|
| res18             | ResNet 18       |  
| res34             | ResNet 34       |  
| res50             | ResNet 50       |   
| effb0             | EfficientNet b0 |  
| effb1             | EfficientNet b1 | 
| effb2             | EfficientNet b2 


In the same file you can change the proposed `learning_rate` and `regularization` values. The code will save a model everytime it surpasses the highest balanced accuracy of the validation set. The checkpoints are saved at `saved_models/`.
<hr>
