### Setup

1. `conda create -n how-much-data-experiments python=3`
1. `pip install tensorflow numpy pandas` (tested with TensorFlow 1.12 & 1.8.0)
1. `curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz`
1. `tar xzf flower_photos.tgz`
1. `pip install tensorflow-hub` (tested with version 0.1.1)
1. `pip install six scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
1. `pip install imgaug`
1. `pip install vega`
1. `pip install jupyter`
1. `pip install tqdm`

### Running one-off experiments

Open "Experiment.ipynb" in a notebook environment. Run through the cells in
order; modifying the parameters at the top as you wish. There's a little
busywork you'll need to do if you change foldernames.

[Run this notebook on Google
Colaboratory](https://colab.research.google.com/drive/1VWxeyhOGMTHHVZFe33s5zaHIKXdDUkA2).

### Running everything

```
./downsample_data.py  # Runs fast
./augment_data.py     # (Optional) Takes an hour or so 
./train_everything.sh
./infer_folders.sh
```

### Reviewing

Run the `Analysis` notebook in `./notebooks`.


