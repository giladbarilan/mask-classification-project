import os
import numpy as np
import cv2
from random import shuffle
from math import floor
from keras.utils.np_utils import to_categorical


class DataReceiver:
    # we add a folder direction.
    @staticmethod
    def __join_full_path(li, path):
        return [path + x for x in li]

    # the generator uses this function to load an image.
    # the list of the file names and the labels.
    # the index of the image we want to load.
    @staticmethod
    def __load_image_in_data_generator(li, i):
        img = cv2.imread(li[i][0], 1)
        img = img / 255
        return img

    # the function creates a generator.
    # li -> the list of the file names and the labels.
    # batch_size -> the batch_size we use for the generator.
    @staticmethod
    def __load_images_generator(li, batch_size):
        print(len(li))
        li = li[:len(li) - len(li) % batch_size]

        while True:
            for batch_offset in range(0, len(li), batch_size):
                x_list = []
                y_list = []

                for idx in range(batch_size):
                    aligned_index = batch_offset + idx
                    x_list.append(DataReceiver.__load_image_in_data_generator(li, aligned_index))
                    y_list.append(li[aligned_index][1])

                x_list = np.array(x_list)
                y_list = np.array(to_categorical(y_list, num_classes=4, dtype='uint8'))

                yield x_list, y_list

    @staticmethod
    # we receive the images and labels we need for each step.
    # level_name -> the level we want to load (train, test, validation).
    # we use this level name to know from which set of directories we should take the classes.
    # batch_size -> the batch size is needed for calculating the steps and calculating the amount
    # of data we send in the generator per yield.
    def receive_for_level(level_name, batch_size):
        mask_files = os.listdir(level_name + '\\mask\\')
        half_mask_files = os.listdir(level_name + '\\half_mask\\')
        bottom_mask_files = os.listdir(level_name + '\\bottom_mask\\')
        unmasked_files = os.listdir(level_name + '\\unmasked\\')

        # we take the min length in order to have an equal amount of pictures for each label.
        length_to_take = min([len(mask_files), len(half_mask_files), len(bottom_mask_files), len(unmasked_files)])

        # we take the exact amount for each label.
        mask_files = mask_files[:length_to_take]
        half_mask_files = half_mask_files[:length_to_take]
        bottom_mask_files = bottom_mask_files[:length_to_take]
        unmasked_files = unmasked_files[:length_to_take]

        lvl = DataReceiver.__join_full_path(mask_files,
                                            level_name + '\\mask\\') + \
              DataReceiver.__join_full_path(half_mask_files,
                                            level_name + '\\half_mask\\') + \
              DataReceiver.__join_full_path(unmasked_files,
                                            level_name + '\\unmasked\\') + \
              DataReceiver.__join_full_path(bottom_mask_files,
                                            level_name + '\\bottom_mask\\')

        solution_length = floor(len(lvl) / 4)  # we use math.floor in case we get a floating point number.
        # we give for each of the images a parallel label.
        lvl_labels = solution_length * [0] + solution_length * [1] + solution_length * [2] + solution_length * [3]

        """
        # we do it for all three levels.
        DataReceiver.__load_list_images(lvl)

        print('end load...')  # identifies we ended loading the images for the specified step.
        lvl, lvl_labels = DataReceiver.__shuffle_lists(lvl, lvl_labels)

        return list(lvl), list(lvl_labels)
        """

        # we shuffle the lists.
        zipped_obj = list(zip(lvl, lvl_labels))
        shuffle(zipped_obj)

        return DataReceiver.__load_images_generator(zipped_obj, batch_size), floor(len(lvl_labels) / batch_size)

