import tensorflow as tf
from text_batch_util import EncodeDecodeUtil

import numpy as np
from text_batch_util import BatchUtil
from tensorflow.models.rnn import rnn_cell, seq2seq
from tensorflow.models.rnn import rnn_cell, seq2seq





"""Tensorflow sources of RNN and seq2seq models:
Library for building sequence-to-sequence models.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/seq2seq.py

Neural translation sequence - to - sequence model.
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py

Helper functions for preparing translation data.
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py

Binary that trains and runs the translation model.
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/translate.py
"""

tf.ops.reset_default_graph()
sess = tf.InteractiveSession()

enc_decod = EncodeDecodeUtil(True)
input = enc_decod.encode()

cell = tf.nn.seq2seq.rnn_cell()
model = tf.nn.seq2seq.basic_rnn_seq2seq()

sess.close()



