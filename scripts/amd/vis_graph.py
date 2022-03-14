# https://gist.github.com/KosukeArase/092a04b5e0c8e69fac9c13ee26bfbafa
import os
import sys

from google.protobuf import text_format

import tensorflow as tf
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()


with tf.compat.v1.Session() as sess:
    model_filename = sys.argv[1]
    with gfile.FastGFile(model_filename, 'r') as f:
        graph_def = tf.compat.v1.GraphDef()
        text_format.Merge(f.read(), graph_def)
        g_in = tf.import_graph_def(graph_def)
LOGDIR = os.path.join(os.path.dirname(model_filename))
train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)