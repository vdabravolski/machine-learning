import tensorflow as tf
import problem_unittests as tests



def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    image_shape = list(image_shape)
    image_shape = [None]+image_shape
    X = tf.placeholder(dtype=tf.float32, shape=tuple(image_shape), name='x')
    return X


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    label_shape = [None]
    label_shape.append(n_classes)
    Y = tf.placeholder(dtype=tf.float32, shape=tuple(label_shape), name='y')
    return Y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')
    return keep_prob


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)