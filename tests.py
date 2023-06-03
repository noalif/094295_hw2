from augmentations import *
import albumentations as A
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import random
train_dir = '/home/student/hw2_094295/data/train'
val_dir = '/home/student/hw2_094295/data/val'


def num_images():
    sum_train = 0
    # print('train')
    for number in os.listdir(train_dir):
        if number == '.DS_Store':
            continue
        number_path = f"{train_dir}/{number}"

        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]
        # print(len(number_images))
        sum_train += len(number_images)

    sum_val = 0
    # print('val')
    for number in os.listdir(val_dir):
        if number == '.DS_Store':
            continue
        number_path = f"{val_dir}/{number}"

        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]
        # print(len(number_images))
        sum_val += len(number_images)

    print('sum train: ' + str(sum_train))
    print('sum val: ' + str(sum_val))
    print('sum images: ' + str(sum_train + sum_val))


def divide_train_val(p=0.2):
    for number in os.listdir(train_dir):
        files = os.listdir(train_dir + '/' + str(number))
        no_of_files = int(len(files) * p)
        for file_name in random.sample(files, no_of_files):
            os.rename(train_dir + '/' + str(number) + '/' + file_name, val_dir + '/' + str(number) + '/' + file_name)


def copy_folder(src_folder, dest_folder):
    shutil.copytree(src_folder, dest_folder)


def restore_to_starting_data():
    src_folder = '/home/student/hw2_094295/data_without_garbage'
    dest_folder = '/home/student/hw2_094295/data'

    shutil.rmtree(dest_folder)

    copy_folder(src_folder, dest_folder)

def base_test():
    restore_to_starting_data()
    divide_train_val()
    num_images()

def aug_old_then_split():
    restore_to_starting_data()
    albumenate_image_noam(train_dir)
    divide_train_val()
    num_images()

def split_aug_old_both():
    restore_to_starting_data()
    divide_train_val()
    albumenate_image_noam(train_dir)
    albumenate_image_noam(val_dir)
    num_images()

def split_aug_old_train():
    restore_to_starting_data()
    divide_train_val()
    albumenate_image_noam(train_dir)
    num_images()

def split_aug_old_train_old_val():
    restore_to_starting_data()
    divide_train_val()
    albumenate_image_noam(train_dir)
    albumenate_image_val()
    num_images()

def split_aug_train():
    restore_to_starting_data()
    divide_train_val(p=0.5)
    aug_data(train_dir, num=7)
    num_images()

def split_aug_both():
    restore_to_starting_data()
    divide_train_val()
    aug_data(train_dir, num=4)
    aug_data(val_dir, num=4)
    num_images()


def main():
    split_aug_train()



if __name__ == '__main__':
    main()



