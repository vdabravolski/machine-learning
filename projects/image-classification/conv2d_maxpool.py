#import cnn_network
import tensorflow as tf
import problem_unittests as tests
from math import ceil
from tensorflow import nn

# def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
#     """
#     Apply convolution then max pooling to x_tensor
#     :param x_tensor: TensorFlow Tensor
#     :param conv_num_outputs: Number of outputs for the convolutional layer
#     :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
#     :param conv_strides: Stride 2-D Tuple for convolution
#     :param pool_ksize: kernal size 2-D Tuple for pool
#     :param pool_strides: Stride 2-D Tuple for pool
#     : return: A tensor that represents convolution and max pooling of x_tensor
#     """
#     # basic parameters
#     batch_size, in_height, in_width, in_depth = x_tensor.get_shape().as_list()
#     batch_size_tensor = tf.shape(x_tensor)[0]
#     stride_height = conv_strides[0]
#     stride_width = conv_strides[1]
#
#     #initiate filter
#     W = tf.Variable(tf.truncated_normal([conv_num_outputs, conv_ksize[0], conv_ksize[1], in_depth], stddev=0.1))
#     B = tf.Variable(tf.truncated_normal([conv_num_outputs], stddev=0.1))
#
#     # CONVOLUTION LAYER
#     # Utilizing SAME padding, here is a size of output layer
#     conv_out_height = ceil(float(in_height) /float(conv_strides[0]))
#     conv_out_width = ceil(float(in_width) / float(conv_strides[1]))
#     conv_layer_list=[] # a workaround according to http://stackoverflow.com/questions/37697747/typeerror-tensor-object-does-not-support-item-assignment-in-tensorflow
#
#
#     # calculate paddings
#     pad_along_height = max(((conv_out_height - 1) * conv_strides[0] + conv_ksize[0] - in_height), 0)
#     pad_along_width = max(((conv_out_width - 1) * conv_strides[1] + conv_ksize[1] - in_width), 0)
#     pad_top = pad_along_height // 2
#     pad_bottom = pad_along_height - pad_top
#     pad_left = pad_along_width // 2
#     pad_right = pad_along_width - pad_left
#
#     # Actually padding the layer
#     if batch_size is None: # this is a workaround for undefined dimension "None"
#         batch_size_norm = 1
#     else:
#         batch_size_norm = batch_size
#
#     x_tensor_reshaped = tf.reshape(x_tensor, [batch_size_norm, in_height, in_width, in_depth])
#     paddings = tf.constant([[0, 0], [pad_top, pad_bottom], [pad_left, pad_right],[0, 0]])  # here [0,0] means that we are not padding depth axis
#     x_tensor_padded = tf.pad(x_tensor, paddings, "CONSTANT", name="conv")
#
#     counter = 0
#     for batch in range(batch_size_norm): # iterate over batches
#         for k in range(conv_num_outputs): # apply specified number of filter (K hyperparameter)
#             for y in range(conv_out_height):
#                 for x in range(conv_out_width):
#                         temp = x_tensor_padded[batch, y*conv_strides[0]:(y*conv_strides[0]+conv_ksize[0]),
#                                x * conv_strides[1]:(x * conv_strides[1] + conv_ksize[1]), :]
#                         conv_layer_list.append(tf.reduce_sum(tf.matmul(temp, W[k,:,:,:],transpose_b=True)) + B[k])
#                         counter += 1
#     print(counter)
#     conv_layer = tf.reshape(tf.stack(conv_layer_list),shape=[batch_size_norm, conv_out_height, conv_out_width, conv_num_outputs])
#
#
#     # POOLING LAYER
#     pool_layer_list = []
#     counter = 0
#     # calculate POOL size
#     # TODO: currently code doesn't handle strides & filter combinations which doesn't fit perfectly
#     pool_out_height = (conv_out_height - pool_ksize[0])//pool_strides[0] + 1
#     pool_out_width = (conv_out_width - pool_ksize[1])//pool_strides[1] + 1
#     pool_out_depth = conv_num_outputs
#
#     for batch in range(batch_size_norm): # iterate over batches
#         for k in range(conv_num_outputs): # apply specified number of filter (K hyperparameter)
#             for y in range(pool_out_height):
#                 for x in range(pool_out_width):
#                     temp = conv_layer[batch, y*pool_strides[0]:(y*pool_strides[0]+pool_ksize[0]),
#                                x * pool_strides[1]:(x * pool_strides[1] + pool_ksize[1]), :]
#                     pool_layer_list.append(tf.reduce_max(temp))
#                     counter += 1
#
#     pool_layer = tf.reshape(tf.stack(pool_layer_list), shape=(batch_size_norm, pool_out_height, pool_out_width, pool_out_depth))
#
#
#     if batch_size is None:
#         pool_layer = tf.expand_dims(None, pool_layer)
#         #pool_layer.set_shape([batch_size, pool_out_height, pool_out_width, pool_out_depth])
#         pool_layer_d = tf.reshape(pool_layer, [batch_size_tensor, pool_out_height, pool_out_width, pool_out_depth])
#
#
#     return pool_layer_d


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_con_pool(conv2d_maxpool)


import tensorflow as tf

#
# def flatten(x_tensor):
#     """
#     Flatten x_tensor to (Batch Size, Flattened Image Size)
#     : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
#     : return: A tensor of size (Batch Size, Flattened Image Size).
#     """
#     batch, height, width, channels = x_tensor.get_shape().as_list()
#     batch_tensor = tf.shape(x_tensor)[0]
#     x_tensor_flat = tf.reshape(x_tensor,[batch_tensor, height*width*channels])
#     return x_tensor_flat
#
#
# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """
# tests.test_flatten(flatten)


# def fully_conn(x_tensor, num_outputs):
#     """
#     Apply a fully connected layer to x_tensor using weight and bias
#     : x_tensor: A 2-D tensor where the first dimension is batch size.
#     : num_outputs: The number of output that the new tensor should be.
#     : return: A 2-D tensor where the second dimension is num_outputs.
#     """
#     batch, image = x_tensor.get_shape().as_list()
#
#     W = tf.Variable(tf.truncated_normal([image, num_outputs], stddev=0.1))
#     B = tf.Variable(tf.zeros([num_outputs]))
#
#     conn_layer = tf.matmul(x_tensor, W) + B
#     conn_layer = tf.nn.relu(conn_layer)
#     return conn_layer
#
#
# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """
# tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch, image = x_tensor.get_shape().as_list()

    W = tf.Variable(tf.truncated_normal([image, num_outputs], stddev=0.1))
    B = tf.Variable(tf.zeros([num_outputs]))

    out_layer = tf.matmul(x_tensor, W) + B
    return out_layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)

tf.nn.dropout
tf.truncated_normal
tf.nn.max_pool