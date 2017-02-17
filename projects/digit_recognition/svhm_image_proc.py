import numpy as np
import pickle
import os
from scipy import misc


class ImageProcessor(object):
    def __init__(self, data_path="data/SVHM/train/", extension=1.3, cropped_size=64):
        self.data_raw = pickle.load(open((data_path + "data.p"), "rb"))
        self.data_path = data_path

        # containers for cropped images
        self.dataset_cropped = []
        self.image_extension = extension
        self.cropped_size = cropped_size

    def _findBBOX(self, data_record):
        """
        :param
            data_record - choose between 'train' and 'test'
        :return: updated_BBOX - coordinates of new BBOX.
        """

        min_top = np.inf
        min_left = np.inf
        max_right = -np.inf
        max_bottom = -np.inf

        for el in data_record['boxes']:
            if el["height"] < min_top:
                min_top = el["height"]

            if el["left"] < min_left:
                min_left = el["left"]

            if (el['left']+el["width"]) > max_right:
                max_right = el['left']+el["width"]

            if (el["height"] + el['top']) > max_bottom:
                max_bottom = (el["height"] + el['top'])

        crop_coord = [min_top, min_left, max_bottom, max_right]

        return crop_coord

    def _getUpdatedRecord(self, crop_coord, data_record):
        """
        Method to return update relative coordinates of the individual digits' bboxes.

        :param crop_coordinate:
        :param data_record:
        :return:
        """


        rel_top = 0.
        rel_left = 0.
        rel_bottom = 0.
        rel_right = 0.

        # transformed features
        coord_relative = []
        sequence = ''
        length = 0

        top = crop_coord[0].astype('float32')
        bottom = crop_coord[2].astype('float32')
        left = crop_coord[1].astype('float32')
        right = crop_coord[3].astype('float32')


        for el in data_record['boxes']:
            rel_top = (el['top']-top) / (bottom-top)
            rel_bottom = (el['top']+el['height']-top) / (bottom-top)
            rel_left = (el['left']-left) / (right-left)
            rel_right = (el['left']+el['width']-left) / (right-left)

            coord_relative.append([rel_top, rel_left, rel_bottom, rel_right])
            tmp = np.zeros(11) # 10 digits + empty space (encoded as 10)
            tmp[int(el['label'])] = 1
            sequence += tmp
            length += 1

        tmp = np.zeros(6)
        tmp[length] = 1
        length = tmp
        #TODO: handle for empty digits. It's recommended to place their coordinates outside of the bbox

        return coord_relative, sequence, length



    def _cropResizeImage(self, image_name, crop_coord):
        """
        Method to crop individual image and then resize. Doesn't support random cropping in order to increase dataset.
        :param image:
        :param extension:
        :param cropped_size:
        :return: cropped_image
        """

        image = misc.imread(self.data_path+image_name)

        # extension of coord
        top = crop_coord[0].astype('float32') / (self.image_extension / 2)
        bottom = crop_coord[2].astype('float32') * (self.image_extension / 2)
        left = crop_coord[1].astype('float32') / (self.image_extension / 2)
        right = crop_coord[3].astype('float32') * (self.image_extension / 2)

        # cropping
        image_cropped = image[top:bottom, left:right, :]

        # resizing
        #image_res = misc.imresize(image_cropped, size=[self.cropped_size, self.cropped_size])
        image_res = misc.imresize(image_cropped, size=(64,64)) #TODO: parametrize it one day

        image_res = image_res.astype('float32')
        image_res = image_res / 255  # normalize the image

        return image_res

    def saveProcessed(self, name_prefix="processed_", rewrite=True):
        """
        This method creates pickle files for processed test and train sets of SVHM images.
        :return:
        """
        processed_filename = self.data_path + name_prefix + "data.p"
        if rewrite:
            if os.path.isfile(processed_filename):
                os.remove(processed_filename)

        for i in xrange(len(self.data_raw)):
            image_name = self.data_raw[i]['filename']
            crop_coord = self._findBBOX(self.data_raw[i])
            image_resized = self._cropResizeImage(image_name, crop_coord)
            coord_updated, sequence_label, length = self._getUpdatedRecord(crop_coord, self.data_raw[i])

            self.dataset_cropped.append([image_resized, coord_updated, length, sequence_label])


        #TODO: save to pickle and test.


proc = ImageProcessor(data_path="data/SVHM/test/")
proc.saveProcessed()









