#import cnn_network
import tensorflow as tf
import problem_unittests as tests
from math import ceil
from tensorflow import nn

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # basic parameters
    batch_size, in_height, in_width, in_depth = x_tensor.get_shape().as_list()
    stride_height = conv_strides[0]
    stride_width = conv_strides[1]

    #initiate filter
    W = tf.Variable(tf.truncated_normal([conv_num_outputs, conv_ksize[0], conv_ksize[1], in_depth], stddev=0.1))
    B = tf.Variable(tf.truncated_normal([conv_num_outputs], stddev=0.1))

    # CONVOLUTION LAYER
    # Utilizing SAME padding, here is a size of output layer
    conv_out_height = ceil(float(in_height) /float(conv_strides[0]))
    conv_out_width = ceil(float(in_width) / float(conv_strides[1]))
    conv_layer_list=[] # a workaround according to http://stackoverflow.com/questions/37697747/typeerror-tensor-object-does-not-support-item-assignment-in-tensorflow
    #tf.zeros([conv_out_height, conv_out_width, conv_num_outputs])

    # calculate paddings
    pad_along_height = max(((conv_out_height - 1) * conv_strides[0] + conv_ksize[0] - in_height), 0)
    pad_along_width = max(((conv_out_width - 1) * conv_strides[1] + conv_ksize[1] - in_width), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # Actually padding the layer
    if batch_size is None: # this is a workaround for undefined dimension "None" # TODO: have to give another thought to this
        x_tensor = tf.reshape(x_tensor, [in_height, in_width, in_depth])
        paddings = tf.constant([[pad_top,pad_bottom],[pad_left,pad_right], [0, 0]]) # here [0,0] means that we are not padding depth axis
        padded_x_tensor = tf.pad(x_tensor, paddings, "CONSTANT", name="conv")
    else:
        raise "unhandled dimension size"

    counter = 0
    for k in range(conv_num_outputs): # apply specified number of filter (K hyperparameter)
        for y in range(conv_out_height):
            for x in range(conv_out_width):
                    temp = padded_x_tensor[y*conv_strides[0]:(y*conv_strides[0]+conv_ksize[0]),
                           x * conv_strides[1]:(x * conv_strides[1] + conv_ksize[1]), :]
                    conv_layer_list.append(tf.reduce_sum(tf.matmul(temp, W[k,:,:,:],transpose_b=True)) + B[k])
                    counter += 1
    print(counter)
    conv_layer = tf.reshape(tf.stack(conv_layer_list),shape=[conv_out_height, conv_out_width, conv_num_outputs])


    # POOLING LAYER
    pool_layer_list = []
    counter = 0
    # calculate POOL size
    # TODO: currently code doesn't handle strides & filter combinations which doesn't fit perfectly
    pool_out_height = (conv_out_height - pool_ksize[0])//pool_strides[0] + 1
    pool_out_width = (conv_out_width - pool_ksize[1])//pool_strides[1] + 1
    pool_out_depth = conv_num_outputs
    for k in range(conv_num_outputs): # apply specified number of filter (K hyperparameter)
        for y in range(pool_out_height):
            for x in range(pool_out_width):
                temp = conv_layer[y*pool_strides[0]:(y*pool_strides[0]+pool_ksize[0]),
                           x * pool_strides[1]:(x * pool_strides[1] + pool_ksize[1]), :]
                pool_layer_list.append(tf.reduce_max(temp))
                counter += 1

    pool_layer = tf.reshape(tf.stack(pool_layer_list), shape=(pool_out_height, pool_out_width, pool_out_depth))


    if batch_size is None:
        pool_layer = tf.expand_dims(None, pool_layer)


    return pool_layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)

tf.nn.conv2d()