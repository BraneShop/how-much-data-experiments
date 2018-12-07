
### Setup

1. `conda create -n how-much-data-experiments python=3`
1. `pip install tensorflow numpy pandas` (tested with TensorFlow 1.12 & 1.8.0)
1. `curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz`
1. `tar xzf flower_photos.tgz`
1. `curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py`
1. `pip install tensorflow-hub` (tester with version 0.1.1)
1. `pip install six scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
1. `pip install imgaug`
1. `pip install vega`
1. `pip install jupyter`
1. `pip install tqdm`


### Running

```
./downsample_data.py  # Runs fast
./augment_data.py     # (Optional) Takes an hour or so 
./train_everything.sh
./infer_folders.sh
```

### Reviewing

Run the `Analysis` notebook in `./notebooks`.
