# Adapted from "label_image.py" from: https://www.tensorflow.org/hub/tutorials/image_retraining

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from downsample_data import get_image_files

import numpy      as np
import tensorflow as tf
import pandas     as pd

import tqdm
import argparse
import os

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                sess,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    result = sess.run(normalized)

    return result


if __name__ == "__main__":
    input_height = 299
    input_width  = 299
    input_mean   = 0
    input_std    = 255
    input_layer  = "Placeholder"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="folder to be processed", required=True)
    parser.add_argument("--graph", help="graph/model to be executed", required=True)
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--prefix", help="prefix of output csv", required=True)
    args = parser.parse_args()

    model_file = args.graph
    folder     = args.folder
    prefix     = args.prefix

    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)

    inferences = []
    sess       = tf.Session()
    
    with tf.Session(graph=graph) as sess:
        labels = list(sorted(os.listdir(folder)))
        images = { d: get_image_files(d, folder) for d in labels }

        for label, images in tqdm.tqdm(images.items()):
            for file_name in tqdm.tqdm(images):
                t = read_tensor_from_image_file(
                        file_name,
                        sess,
                        input_height=input_height,
                        input_width=input_width,
                        input_mean=input_mean,
                        input_std=input_std)

                input_name       = "import/" + input_layer
                output_name      = "import/" + output_layer
                input_operation  = graph.get_operation_by_name(input_name)
                output_operation = graph.get_operation_by_name(output_name)

                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                    })

                results = np.squeeze(results)
                k       = 5
                top_k   = results.argsort()[-k:][::-1]

                data = {}
                data["file_name"]  = file_name
                data["true_label"] = label

                for i in top_k:
                    l = labels[i]
                    data[f"predicted_{l}"] = results[i]
                inferences.append(data)

    df = pd.DataFrame(inferences)
    df.to_csv(f"csvs/{prefix}-results.csv")

