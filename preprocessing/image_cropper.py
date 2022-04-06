import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple

def obtain_bb(threshed_image : np.ndarray) -> Tuple[int, int, int, int]:
    """
    Obtains the bounding box of the lesion

    Inputs:
    ----------------

    Outputs:
    ----------------
    x : int : initial horizontal coordinate of the bounding box
    y : int : initial vertical coordinate of the bounding box
    w : int : width of the bounding box
    h : int : height of the bounding box
    """

    contours, _ = cv2.findContours(threshed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100:
            continue
        else:
            x,y,w,h = rect
    return x,y,w,h

def intensity_ratio(gray_image : np.ndarray, x : int, y : int, w : int, h : int) -> float :
    """
    Computes the intensity ratio between the whole image
    and the image within the bounding box

    Inputs:
    ----------------
    gray_image : numpy.ndarray : grayscale image
    x : int : initial horizontal coordinate of the bounding box
    y : int : initial vertical coordinate of the bounding box
    w : int : width of the bounding box
    h : int : height of the bounding box

    Outputs:
    ---------------
    cii/fii : float : intensity ratio
    """

    cropped_img = gray_image[y:y+h, x:x+w]
    cropped_img.shape
    intensity = 0
    for i in range(cropped_img.shape[0]):
        for j in range(cropped_img.shape[1]):
            intensity += cropped_img[i][j]

    cii = intensity/(cropped_img.shape[0]+cropped_img.shape[1])

    intensity_f = 0
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            intensity_f += gray_image[i][j]
    fii =intensity_f/(gray_image.shape[0]+gray_image.shape[1])
    return cii/fii


def crop_files(df : pd.DataFrame):
    """
    Iterates over a DataFrame to perform the cropping of images with black background

    Inputs:
    -------------------
    df : pd.DataFrame : must contain a column named 'filename' with the path to the images

    Outputs:
    -------------------
    Overwrites the images deemed for cropping 
    """
    for i, filename in enumerate(tqdm(df['filename'])):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,60,255,0)
        try:
            rect = obtain_bb(thresh)

            x,y,w,h = rect
            int_ratio = intensity_ratio(gray_image, x, y, w, h)

            if float(int_ratio) > 1.1:
                print('resaving {}'.format(filename))
                img = img[y:y+h, x:x+w]
                cv2.imwrite(filename, img)
        except:
           print('Image {} was not found'.format(filename))
           continue

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Configuration for dermatoscopy cropping')
    parser.add_argument('--csv_dir', type=str, default=None, required=True,
                        help='Path for the csv with the file locations')
    args = parser.parse_args()
   
    df = pd.read_csv(args.csv_dir)
    crop_files(df)


