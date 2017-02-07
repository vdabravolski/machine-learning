"""what needs to be done:
1. load mnist data
2. combine randomly

3. Following  labels to be created (1-hot all):
    - target length
    - 1-5 classifiers
"""

from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

empty_space = -1  # to encode it in the sequence


def generate_sequences(images, labels, out_size, max_length=5):
    """
    This function is used to generate a sequences of digits out of individual images.
    The function assumes uniform distribution of sequence legths.

    Inputs:
        images - individual digits as np.arrays
        labels  - labels for each individual digit
        out_size - number of output images (sequences of individual images)
        max_lenght - max number of digits in sequence.

    Outputs:
        sequences - numpy arrays with sequences. All images has the same size (image_size * sequence_length)
        sequence_labels - 2D numpy array (out_size * sequence_length). "-1" designates empty space for digit.
    """
    seq_length = out_size // max_length  # the number of generated synthetic image for each length.

    sequences = np.empty(shape=(out_size, 28, 28 * max_length), dtype=np.float32)
    sequence_labels = np.empty(shape=(out_size, max_length), dtype=np.int32)

    for j in xrange(seq_length):
        for i in xrange(max_length):
            index = i * seq_length + j

            sequences[index], sequence_labels[index] = \
                get_sequence_image(images, labels, real_digits=(i), noisy_digits=(max_length - i))

    return sequences, sequence_labels

def get_sequence_image(x, y, real_digits, noisy_digits):
    length = (real_digits + noisy_digits)
    output_image = np.empty(shape=(28, 28 * length), dtype=np.float32)
    output_label = np.empty(shape=(length), dtype=np.int32)

    for i in xrange(real_digits):
        rand_index = np.random.choice(np.shape(x)[0])
        output_image[:, 28 * i:(28 * (i + 1))] = x[rand_index, :, :]
        output_label[i] = y[rand_index]

    for j in xrange((real_digits + 1), (real_digits + noisy_digits)):
        output_image[:, 28 * j:(28 * (j + 1))] = get_rand_image()
        output_label[j] = empty_space

    return output_image, output_label


def get_rand_image(x_dim=28, y_dim=28):
    return np.random.rand(x_dim, y_dim)

seq, seq_label = generate_sequences(X_train, y_train, 50)



"""
bug 1: cannot generate sequence for max langht with real digits.

"""
plt.imshow(seq[0])

print("done")
