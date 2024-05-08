# BCN 20000 tools for assembling and training

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons Lizenzvertrag" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a>

This repository gives access to the tools to train the models presented at the BCN 20000's dataset scientific publication. The dataset itself is available for download at [FigShare](https://figshare.com/articles/journal_contribution/BCN20000_Dermoscopic_Lesions_in_the_Wild/24140028/1).

![Diagram of dataset formation](https://github.com/imatge-upc/BCN20000/blob/main/Final_diagram.png)

<hr>

# Generating the Master Split File for the BCN_20K Dataset

This repository includes a Python script that generates a master split file for the BCN_20K dataset. The master split file provides a reproducible way to split the dataset into training, validation, and testing sets across multiple folds.

The master split file used to obtain the publication results can be found on this repo. If the user wants to generate again, it can run the script from the command line by providing the path to your input CSV file and the desired output path for the master split CSV file.

```bash
python create_master_split.py path/to/bcn_20k_train.csv path/to/output/master_split_file.csv
```

## Installation

First, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repository-url.git
cd your-repository-directory
pip install -r requirements.txt
```

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
