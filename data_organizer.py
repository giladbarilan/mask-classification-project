import os
import math
import shutil
import threading
from image_processor import resize_image


'''
the function builds the necessary directories for the train test validation.
'''


def build_directories():
    __override_if_exists('train/mask/')
    __override_if_exists('train/bottom_mask/')
    __override_if_exists('train/unmasked/')
    __override_if_exists('train/half_mask/')
    __override_if_exists('test/mask/')
    __override_if_exists('test/bottom_mask/')
    __override_if_exists('test/unmasked/')
    __override_if_exists('test/half_mask/')
    __override_if_exists('validation/mask/')
    __override_if_exists('validation/bottom_mask/')
    __override_if_exists('validation/unmasked/')
    __override_if_exists('validation/half_mask/')


'''
Override existing folder if exists.
'''

# the function checks if "directory" exists.
# if it does it overrides it and create empty folders instead.
# if it does not exist it creates empty folders as well.
# directory -> the directory we want to override.
def __override_if_exists(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)


# the function resizes all images in Image directory to the ResizedImages directory.
def resize_images_on_directory():
    __override_if_exists('ResizedImagesPOC/')

    directory = 'ResizedImagesPOC/'
    images = os.listdir('Images/')  # lists all image names

    for file in images:
        resize_image('Images/' + file, directory + file)


# the function splits all of the photos from the ResizedImages directory into classes.
# and returns the lists.
def __split_mask_no_mask():
    mask = []  # the list of images with mask.
    half_mask = []  # the list of images with half mask.
    bottom_mask = []  # the list of images with mask on chin.
    no_mask = []  # the list of images without mask
    images = os.listdir('ResizedImages/')

    print('Grouping into subjects...')

    for file in images:
        mask_type = file.split('_')[1]

        if mask_type == '1':
            mask.append((file, False))
        elif mask_type == '2':
            half_mask.append((file, False))
        elif mask_type == '3':
            bottom_mask.append((file, False))
        else:
            no_mask.append((file, False))

    return mask, half_mask, bottom_mask, no_mask


# copies list of files to destination
def copy_files_to_dst(li, dst):
    for file in li:
        shutil.copyfile('ResizedImages/' + file, os.path.join(dst, file))


# gets the right necessary amount of data to train/test/validation
def __get_data_for_each_steps(mask, half_mask, bottom_mask, no_mask, percent, step_name):
    print(f'Start organizing step: {step_name}')

    mask_ = __take_from_list(mask, percent)  # takes the percent from masks.
    half_mask_ = __take_from_list(half_mask, percent)  # takes the percent from half masks.
    bottom_mask_ = __take_from_list(bottom_mask, percent) # takes the precent from bottom masks.
    no_mask_ = __take_from_list(no_mask, percent)  # takes the percent from no masks.

    directory_to_write_to = step_name + '\\'

    print(f'Start mask in {step_name}')
    mask_thread = threading.Thread(target=copy_files_to_dst, args=(mask_, directory_to_write_to + 'mask'))
    mask_thread.start()

    print(f'Start bottom_mask in {step_name}')
    bottom_mask_thread = threading.Thread(target=copy_files_to_dst,
                                          args=(bottom_mask_, directory_to_write_to + 'bottom_mask'))
    bottom_mask_thread.start()

    print(f'Start half_mask in {step_name}')
    half_mask_thread = threading.Thread(target=copy_files_to_dst,
                                        args=(half_mask_, directory_to_write_to + 'half_mask'))
    half_mask_thread.start()

    print(f'Start unmasked in {step_name}')
    unmasked_thread = threading.Thread(target=copy_files_to_dst, args=(no_mask_, directory_to_write_to + 'unmasked'))
    unmasked_thread.start()

    # we wait to all of the threads before we move on.
    mask_thread.join()
    bottom_mask_thread.join()
    half_mask_thread.join()
    unmasked_thread.join()

    print(f'Finished organizing step: {step_name}')


# divides the content of ResizedImages folder to train, test, validation folders.
# train -> the percentage size we are taking for the train step.
# test -> the percentage size we are taking for the test step.
# validation -> the percentage size we are taking for the validation step.
def divide_to_train_test_validation(train=0.7, test=0.2, validation=0.1):
    if (train + test + validation) > 1:
        raise ValueError("train, test, validation doesn't is more than one. Invalid arguments.")

    mask, half_mask, bottom_mask, no_mask = __split_mask_no_mask()  # lists of the images for each of the mask status classes.

    __get_data_for_each_steps(mask, half_mask, bottom_mask, no_mask, train, 'train')
    __get_data_for_each_steps(mask, half_mask, bottom_mask, no_mask, test, 'test')
    __get_data_for_each_steps(mask, half_mask, bottom_mask, no_mask, validation, 'validation')


# the function takes a percent of the list
# li -> the list we are taking the precents from.
# percent -> the amount of precents we are taking for the step.
def __take_from_list(li: list, percent):
    taken_list = []  # the list of the taken elements.
    amount_of_elements_to_take = math.floor(len(li) * percent)  # the amount of elements we take.
    li_index = 0  # the index on the li list we currently check.
    amount_of_elements_taken = 0  # the amount of elements taken at a time.
    # note -> we could also check the len(li) but we want to be more efficient.

    # while we did not take the whole elements.
    while li_index < len(li) and amount_of_elements_taken < amount_of_elements_to_take:
        if not li[li_index][1]:  # if the element was not taken.
            li[li_index] = li[li_index][0], True
            taken_list.append(li[li_index][0])
            amount_of_elements_taken += 1

        li_index += 1

    return taken_list
