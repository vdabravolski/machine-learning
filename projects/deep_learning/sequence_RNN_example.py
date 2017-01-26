import tensorflow as tf
import numpy as np
from text_batch_util import BatchUtil
from tensorflow.models.rnn import rnn_cell, seq2seq
import tensorflow.nn.seq2seq


tf.ops.reset_default_graph()
sess = tf.InteractiveSession()

