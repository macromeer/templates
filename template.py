#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import cv2
from napari.utils import io as napari_io
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm


# Argument Parsing
parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
args = parser.parse_args()


# Define utility functions

def process_image(image_path):
    try:

        # processing workflow copied from napari

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_gb = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
        image_to = cle.threshold_otsu(image_gb)
        image_l = cle.connected_components_labeling_box(image_to)
        image_S = nsbatwm.split_touching_objects(image_l, 9.0)
        image_labeled = cle.connected_components_labeling_box(image_S)

        return image_labeled
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Process images
image_folder = os.path.join(args.input, "DAPI")
for filename in os.listdir(image_folder):
    print(f"Processing image: {filename}")
    labeled_image = process_image(os.path.join(image_folder, filename))
    if labeled_image is not None:
        napari_io.imsave(os.path.join(image_folder, f"{filename[:-4]}_labels.tif"), labeled_image)

