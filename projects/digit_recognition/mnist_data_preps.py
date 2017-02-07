"""what needs to be done:
1. load mnist data
2. combine randomly

3. Following  labels  to be created (1-hot all):
    - target length
    - 1-5 classifiers
"""

from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

empty_space = -1  # empty spaces will be encoded as "-1"
nb_classes = 11  # include 10 digits and "-1" which designates empty space in digit sequence.


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
        sequence_length - length of the sequence (include only real digits)
    """
    # seq_length = out_size // max_length  # the number of generated synthetic image for each length.

    sequences = np.empty(shape=(out_size, 28, 28 * max_length), dtype=np.float32)
    sequence_labels = np.empty(shape=(out_size, max_length, nb_classes), dtype=np.int32)
    sequence_lengths = np.empty(shape=(out_size), dtype=np.int32)

    index = 0

    for _ in xrange(out_size):

        # get random number of digits for given sequence
        real_digits = np.random.choice(max_length) + 1 # TODO: this implies that each sequence will have betweeen 1 and 5 real digits (by default)
        noisy_digits = max_length - real_digits

        sequences[index], tmp_labels = \
            get_sequence_image(images, labels, real_digits, noisy_digits)

        sequence_labels[index] = np_utils.to_categorical(tmp_labels, nb_classes) # 1-ho
        sequence_lengths[index] = real_digits

        index += 1

    return sequences, sequence_labels, sequence_lengths

def get_sequence_image(x, y, real_digits, noisy_digits):
    length = (real_digits + noisy_digits)
    output_image = np.empty(shape=(28, 28 * length), dtype=np.float32)
    output_label = np.empty(shape=(length), dtype=np.int32)

    for i in xrange(real_digits):
        rand_index = np.random.choice(np.shape(x)[0])
        output_image[:, 28 * i:(28 * (i + 1))] = x[rand_index, :, :]
        output_label[i] = y[rand_index]

    for j in xrange((real_digits), (real_digits + noisy_digits)):
        output_image[:, 28 * j:(28 * (j + 1))] = get_rand_image()
        output_label[j] = empty_space

    return output_image, output_label


def get_rand_image(x_dim=28, y_dim=28):
    return np.random.rand(x_dim, y_dim)


# Test.
#seq, seq_label, _ = generate_sequences(X_train, y_train, 50)
#plt.imshow(seq[0])

#print("done")
