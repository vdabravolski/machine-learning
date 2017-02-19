import numpy as np
import pickle
import os
from scipy import misc


class ImageProcessor(object):
    def __init__(self, data_path="data/SVHM/train", extension=0.3, cropped_size=64):
        """
        :param data_path: location of the data stored. Can be absolute or relative.
        :param extension: percentage of image extension prior to cropping.
        :param cropped_size: size of the image in pixel (both vertical and horizontal) after cropping.
        """
        self.data_raw = pickle.load(open((data_path + "/data_upd.p"), "rb"))
        self.data_path = data_path

        # containers for cropped images
        self.dataset_cropped = []
        self.image_extension = extension
        self.cropped_size = cropped_size

    def _findBBOX(self, data_record):
        """
        Method find coordinate of the smallest bounding box which can fit all the individual digits.
        :param data_record: data record
        :return: cordinates of the bounding box.
        """

        min_top = np.inf
        min_left = np.inf
        max_right = -np.inf
        max_bottom = -np.inf

        for el in data_record['boxes']:
            if el["top"] < min_top:
                min_top = el["top"]

            if el["left"] < min_left:
                min_left = el["left"]

            if (el['left']+el["width"]) > max_right:
                max_right = el['left']+el["width"]

            if (el["height"] + el['top']) > max_bottom:
                max_bottom = (el["height"] + el['top'])
            print("ind digit coord:"+str([min_top, min_left, max_bottom, max_right]))

        BBOX_coordinates = [min_top, min_left, max_bottom, max_right]
        print("overal digits coordinates:"+str(BBOX_coordinates))
        return BBOX_coordinates

    def _getUpdatedRecord(self, digits_coord, data_record):
        """
        method returns an updated record for for specific image
        :param digits_coord:
        :param data_record:
        :return: relative coordinates of each digit, 1-hot encoded length and digits
        """

        # transformed features
        coord_relative = []
        sequence = []
        length = 0

        top = float(digits_coord[0])
        bottom = float(digits_coord[2])
        left = float(digits_coord[1])
        right = float(digits_coord[3])


        for el in data_record['boxes']:
            rel_top = (el['top']-top) / (bottom-top)
            rel_bottom = (el['top']+el['height']-top) / (bottom-top)
            rel_left = (el['left']-left) / (right-left)
            rel_right = (el['left']+el['width']-left) / (right-left)

            print("relative coord:"+str([rel_top, rel_left, rel_bottom, rel_right]))
            coord_relative.append([rel_top, rel_left, rel_bottom, rel_right])
            tmp = np.zeros(11) # 10 digits + empty space (encoded as 10)
            tmp[int(el['label'])] = 1
            sequence.append(tmp)
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

        image = misc.imread(self.data_path+"/"+image_name)
        image_height = np.shape(image)[0]
        image_width = np.shape(image)[1]

        # extension of coord TODO: what happens if we get negative coordinate?
        width_ext = self.image_extension * (crop_coord[3] - crop_coord[1]) / 2
        height_ext = self.image_extension * (crop_coord[2] - crop_coord[0]) / 2
        print("height ext:" + str(height_ext))
        print("width ext:"+str(width_ext))
        top = max(0, crop_coord[0] - height_ext)
        bottom = min(image_height, crop_coord[2] + height_ext)
        left = max(0, crop_coord[1] - width_ext)
        right = min(image_width, crop_coord[3] + width_ext)

        top = int(top)
        bottom = int(bottom)
        left = int(left)
        right = int(right)

        crop_coord_upd = [top, left, bottom, right] # updating crop coordinates to account for extension.
        print("crop coordinate after extension:"+str(crop_coord_upd))

        # cropping
        image_cropped = image[top:bottom, left:right, :]

        # resizing
        image_res = misc.imresize(image_cropped, size=(64,64))

        image_res = image_res.astype('float32')
        image_res = image_res / 255  # normalize the image

        return image_res, crop_coord_upd

    def saveProcessed(self, name_prefix="processed_", rewrite=True, save_image=True):
        """
        This method creates pickle files for processed test and train sets of SVHM images.
        :return: reference and filenames of the pickle
        """
        processed_filename = self.data_path + name_prefix + "data.p"
        if rewrite:
            if os.path.isfile(processed_filename):
                os.remove(processed_filename)

        for i in xrange(len(self.data_raw)):
            image_name = self.data_raw[i]['filename']
            print(str(self.data_raw[i]['boxes']))
            digits_coord = self._findBBOX(self.data_raw[i])
            image_resized, digits_coord = self._cropResizeImage(image_name, digits_coord)
            coord_updated, sequence_label, length = self._getUpdatedRecord(digits_coord, self.data_raw[i])

            self.dataset_cropped.append([image_name, image_resized, coord_updated, length,sequence_label])
            #record = {'image_name': image_name, 'image': image_resized, 'coordinates': coord_updated, 'length': length, 'label': sequence_label}
            #self.dataset_cropped.append(record)


            if save_image:
                misc.imsave(self.data_path+"_cropped/"+image_name, image_resized)

        datafile = self.data_path+"_cropped/"+"dataset.p"
        pickle.dump(self.dataset_cropped, open(datafile, "wb"))
        return datafile




proc_test = ImageProcessor(data_path="data/SVHM/test")
cropped_test = proc_test.saveProcessed()

proc_train = ImageProcessor(data_path="data/SVHM/train")
cropped_train = proc_train.saveProcessed()










