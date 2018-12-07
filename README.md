### Steps

1. `conda create -n how-much-data-experiments python=3`
2. `pip install tensorflow` (tested with TensorFlow 1.12)
3. `curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz`
4. `tar xzf flower_photos.tgz`
5. `curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py`
6. `pip install tensorflow-hub` (tester with version 0.1.1)
7. `pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
8. `pip install imgaug`
8. `pip install pandas`
8. `pip install vega`


Run stuff.

TODO:

  - ~Log-Level Stuff~
  - Train all the modes 5 ... 100
  - Run all the images through the model post-training
  - Maybe use the 'nohash' thing so that a given bunch are with-held properly?
  - Compare them


VIS TODO:

  - 1. Plot the train/test/val graphs?
  - 2. Plot a little grid of the inferred images?

"Before you read the rest of this blog, have some guesses about
the following:

For a 5-class classification problem, ... what would you expect
to be able to do with 3 images of each class?"

