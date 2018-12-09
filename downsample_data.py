#!/usr/bin/env python

import os
import numpy as np
from shutil import copyfile, rmtree
import tqdm

# From the original true dataset; what percent
# to use as a hold-out set? 
#
# `retrain.py` says 10, so let's use that.

base_directory = "flower_photos"

def get_image_files (directory, base_directory=base_directory):
    directory = os.path.join(base_directory, directory)
    exts      = [".jpg", ".png", ".jpeg"]

    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)

        if not os.path.isfile(full_path):
            continue

        ext = os.path.splitext(f)[1]

        if ext.lower() not in exts:
            continue

        yield full_path


def main (amounts,
            seed                  = 271828182,
            holdout_percent       = 10,
            holdout_directory     = "holdout",
            experiments_directory = "experiments"):
    # For reproducibility
    np.random.seed(seed)

    dirs    = [ d for d in os.listdir(base_directory) 
                  if os.path.isdir(os.path.join(base_directory, d)) ]

    images  = { d: get_image_files(d) for d in dirs }

    if os.path.isdir(holdout_directory):
        rmtree(holdout_directory)

    os.mkdir(holdout_directory)

    if os.path.isdir(experiments_directory):
        rmtree(experiments_directory)
    
    os.mkdir(experiments_directory)

    for category, images in tqdm.tqdm(images.items()):
        images = np.array(list(images))
        count  = len(images)

        np.random.shuffle(images)

        # 1. Take `holdout_percent` away from _each folder_; keep
        #    them for safe-keeping.
        holdout        = int(count * (holdout_percent / 100))
        holdout_images = images[:holdout]

        newdir = holdout_directory + "/" + category
        os.mkdir(newdir)
        
        for image in holdout_images:
            copyfile(image, newdir + "/" + os.path.basename(image))

        # 2. Copy the rest to the respective folders.
        for amount in tqdm.tqdm(amounts):
            # If we've got a string here, we actually want that
            # exact amount, instead of percentage.
            if type(amount) == str:
                items = int(amount)
            else:
                items = int((count - holdout) * (amount / 100))

            training_images = images[holdout:holdout + items]
            
            newdir = "{}/{}/{}".format(experiments_directory, amount, category)
            os.makedirs(newdir)

            for image in tqdm.tqdm(training_images):
                copyfile(image, newdir + "/" + os.path.basename(image))

if __name__ == "__main__":
    amounts = ["1", "3", 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]
    main(amounts)
