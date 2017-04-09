from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle
import re

def get_os_indices(filenames, keras_indices):
    """
    This method is to overcome differences in image files sorting between keras and OS X default.
    :return:
    os_indices
    """
    files = [filenames[i] for i in keras_indices]

    os_indices = []

    for _, el in enumerate(files):
        m = re.search(r'[^/-]*', el)
        os_indices.append(int(m.group(0)))


    return os_indices


def custom_image_generator(flow_from_directory_gen, labels_pickle):
    label_data = pickle.load(open(labels_pickle,'rb'))
    # TODO: p2 avoid loading pickle multiple times.

    for X, indices in flow_from_directory_gen:
        os_indices = get_os_indices(flow_from_directory_gen.filenames, indices)

        # labels = [label_data[i] for i in os_indices]

        # Verify that the labels are in-line wiht the records in pickle file. Otherwise, throw the error
        labels = []
        for i in os_indices:
            try:
                labels.append(label_data[i])
            except IndexError:
                print("\n"+str(i)+": index out of range")
#            if not str((i+1)) in label_data[i][0]:
#                raise Exception('Label and image is not aligned. Image:'+label_data[i][0]+" and class:"+str(i))



        # Init the placeholders
        length = np.zeros(shape=(len(X), 6))
        coord = np.zeros(shape=(len(X), 20))

        first = np.zeros(shape=(len(X), 11))
        second = np.zeros(shape=(len(X), 11))
        third = np.zeros(shape=(len(X), 11))
        forth = np.zeros(shape=(len(X), 11))
        fifth = np.zeros(shape=(len(X), 11))

        for idx, el in enumerate(labels):
            length[idx, :] = el[2]
            coord[idx, :] = np.asarray(el[1]).flatten()
            first[idx, :] = el[3][0]
            second[idx, :] = el[3][1]
            third[idx, :] = el[3][2]
            forth[idx, :] = el[3][3]
            fifth[idx, :] = el[3][4]
        Y = [length, first, second, third, forth, fifth, coord]

        yield (X, Y)