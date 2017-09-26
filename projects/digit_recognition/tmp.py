import numpy as np
from svhm_grid_runner import data
from matplotlib import pyplot as plt
import matplotlib as mpl

def test_data_method():
    """
    To test the quality of generated to data by keras generator and specifically matchin of labels with actual images.
    """

    X, Y = data(batch_size=25) # TODO: this returns generator object, not the sample of images itself

    ### Sampling predictions
    fig = plt.figure()


    mpl.rcParams['axes.titlesize'] = 'small'
    plt.axis('off')

    X_sample = X.next()
    Y_sample = Y.next()

    for idx, el in enumerate(X_sample):
        # Create subplot
        a = fig.add_subplot(5, 5, (idx+1))
        imgplot = plt.imshow(el)

        true_sequence = ''
        for i in xrange(np.argmax(Y_sample[0][idx])):
            label = str(np.argmax(Y_sample[i + 1][idx]))
            true_sequence += label
        a.set_title("label:" + true_sequence)

    plt.show()

test_data_method()