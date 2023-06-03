import albumentations as A
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import random

train_dir = '/home/student/hw2_094295/data/train'
val_dir = '/home/student/hw2_094295/data/val'

# Define the augmentation transforms
translation_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0),
])

gaussian_noise_transform = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0)),
])


def pick_aug():
    blur = A.AdvancedBlur(always_apply=False, p=1.0, blur_limit=(3, 49), sigmaX_limit=(0.2, 5.03),
                          sigmaY_limit=(0.2, 5.16),
                          rotate_limit=(-210, 210), beta_limit=(0.5, 73.36), noise_limit=(0.9, 30.6))
    drop_out = A.CoarseDropout(always_apply=False, p=1.0, max_holes=12, max_height=9, max_width=15, min_holes=1,
                               min_height=1, min_width=8, fill_value=0, mask_fill_value=None)
    gauss_noise = A.GaussNoise(always_apply=False, p=1.0, var_limit=(0.0, 233.55), per_channel=True, mean=-7.24)
    rotate = A.Rotate(always_apply=False, p=1.0, limit=(-45, 45), interpolation=3, border_mode=1, value=(0, 0, 0),
                      mask_value=None, rotate_method='largest_box', crop_border=False)
    translation = A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit_x=(-0.04, 0.38),
                                     shift_limit_y=(-0.04, 0.38),
                                     scale_limit=(-0.51, 0.3900000000000001), rotate_limit=(-50, 50), interpolation=0,
                                     border_mode=0, value=(255, 255, 255), mask_value=None, rotate_method='largest_box')
    down_scale = A.Downscale(always_apply=False, p=1.0, scale_min=0.14, scale_max=0.47)
    grid_distortion = A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3),
                                       interpolation=0,
                                       border_mode=0, value=(0, 0, 0), mask_value=None, normalized=False)

    aug_list = [blur, drop_out, gauss_noise, rotate, translation, down_scale, grid_distortion]
    # random_aug = random.sample(aug_list, k=3)
    return random.sample(aug_list, k=3)


def aug_data(dir, num=3):
    for number in os.listdir(dir):
        if number == '.DS_Store':
            continue
        number_path = f"{dir}/{number}"
        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]

        for image_path in number_images:
            image = Image.open(number_path + '/' + image_path)
            for i in range(num - 1): # each image, were duplicating 3 more times with random augs
                random_aug_list = pick_aug()
                transform = A.Compose(random_aug_list)
                aug_img = transform(image=np.array(image))['image']
                aug_img = Image.fromarray(aug_img)
                aug_img.save(os.path.join(number_path, 'aug_' + str(i) + '_' + image_path))

            # adding one more augmentation:
            # rotate + shear + scale
            max_shear_x = 0.2  # Maximum horizontal shear angle in radians
            max_shear_y = 0.2
            shear_x = random.uniform(-max_shear_x, max_shear_x)
            shear_y = random.uniform(-max_shear_y, max_shear_y)
            sheared_image = image.transform(image.size, Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0))

            if str(number) not in ('iv', 'vi'):
                rotation_angle = random.choice([90, 270])
                sheared_image = sheared_image.rotate(rotation_angle, expand=True)

            min_scale = 0.5  # Minimum scaling factor
            max_scale = 2.0  # Maximum scaling factor
            scale_factor = random.uniform(min_scale, max_scale)
            scaled_size = (int(sheared_image.width * scale_factor), int(sheared_image.height * scale_factor))
            scaled_image = sheared_image.resize(scaled_size)
            scaled_image.save(os.path.join(number_path, "shear_" + image_path))


def albumenate_image_cut_flip_rotate(dir):
    for number in os.listdir(dir):
        if number == '.DS_Store':
            continue
        number_path = f"{dir}/{number}"
        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]

        for image_path in number_images:

            image = Image.open(number_path + '/' + image_path)

            # Define the cutout augmentation
            cutout = A.Cutout(num_holes=1, max_h_size=20, max_w_size=20, fill_value=0, always_apply=True)

            # Apply the cutout augmentation to the image
            augmented_image = cutout(image=np.array(image))['image']

            # Save the augmented image
            augmented_image = Image.fromarray(augmented_image)
            augmented_image.save(os.path.join(number_path, "augmented_" + image_path))

            # Horizontal Flip
            if str(number) in ("ix", "iiiv", "iiv"):
                modified_img = image.transpose(Image.FLIP_LEFT_RIGHT)
                modified_img.save(os.path.join(number_path, "augmented_" + image_path))

            # Vertical Flip
            if str(number) in ("i", "ii", "iii", "iv", "v", "vi"):
                modified_img = image.transpose(Image.FLIP_TOP_BOTTOM)
                modified_img.save(os.path.join(number_path, "augmented_" + image_path))

            # ROTATE 90 270  +
            # Randomly select between 90 and 270 degrees rotation
            if str(number) not in ('iv', 'vi'):
                rotation_angle = random.choice([90, 270])
                modified_img = image.rotate(rotation_angle, expand=True)
                modified_img.save(os.path.join(number_path, "augmented_" + image_path))


# create albumentations on data set
def albumenate_image_noam(dir):
    for number in os.listdir(dir):
        number_path = f"{dir}/{number}"
        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]

        num_orig_data = len(number_images)
        num_agmented_images = max(999 - num_orig_data, 0)

        for image_path in number_images:
            # making sure we keeo the number of images for each digit 1000
            # need to make sure that every time we save an image, we subtract 1 from num_agmented_images

            image = Image.open(number_path + '/' + image_path)

            # 1
            # rotate + shear + scale
            max_shear_x = 0.2  # Maximum horizontal shear angle in radians
            max_shear_y = 0.2
            shear_x = random.uniform(-max_shear_x, max_shear_x)
            shear_y = random.uniform(-max_shear_y, max_shear_y)
            sheared_image = image.transform(image.size, Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0))

            if str(number) not in ('iv', 'vi'):
                rotation_angle = random.choice([90, 270])
                sheared_image = sheared_image.rotate(rotation_angle, expand=True)

            min_scale = 0.5  # Minimum scaling factor
            max_scale = 2.0  # Maximum scaling factor
            scale_factor = random.uniform(min_scale, max_scale)
            scaled_size = (int(sheared_image.width * scale_factor), int(sheared_image.height * scale_factor))
            scaled_image = sheared_image.resize(scaled_size)
            scaled_image.save(os.path.join(number_path, "augmented_" + image_path))
            # print('saved 1 - augmented')

            # 2
            # translation
            translated_image = translation_transform(image=np.array(image))['image']
            translated_image = Image.fromarray(translated_image)
            translated_image.save(os.path.join(number_path, 'translation_' + image_path))
            # print('saved 2 - translation_')

            # 3
            # gauss + trans
            if str(number) not in ('iv'):
                gaus_image = gaussian_noise_transform(image=np.array(image))['image']
                gaus_image = Image.fromarray(gaus_image)
                trans_gaus_image = translation_transform(image=np.array(gaus_image))['image']
                trans_gaus_image = Image.fromarray(trans_gaus_image)
                trans_gaus_image.save(os.path.join(number_path, 'gaus_trans_' + image_path))
                # print('saved 3 - gaus_trans_')
                # print(number)

            # 4
            # scale + translation + gaus + shear + rotate
            if str(number) not in ('iv', 'i', 'ix'):
                gaus_image = gaussian_noise_transform(image=np.array(scaled_image))['image']
                gaus_image = Image.fromarray(gaus_image)
                trans_gaus_image = translation_transform(image=np.array(gaus_image))['image']
                trans_gaus_image = Image.fromarray(trans_gaus_image)
                trans_gaus_image.save(os.path.join(number_path, 'gaus_trans_sh_sc_ro' + image_path))
                # print('saved 4 - gaus_trans_sh_sc_ro')
                # print(number)


# create albumentations on data set
def albumenate_image_st():
    for number in os.listdir(train_dir):
        number_path = f"{train_dir}/{number}"
        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]

        # num_orig_data = len(number_images)
        # num_agmented_images = max(999 - num_orig_data, 0)
        # print(number)
        # print(num_agmented_images / num_orig_data)
        # continue

        # rotate + shear + scale
        for image_path in number_images:
            # making sure we keeo the number of images for each digit 1000
            # need to make sure that every time we save an image, we subtract 1 from num_agmented_images
            # if num_agmented_images == 0:
            #   break

            if random.random() < 0.7:
                image = Image.open(number_path + '/' + image_path)

                max_shear_x = 0.2  # Maximum horizontal shear angle in radians
                max_shear_y = 0.2
                shear_x = random.uniform(-max_shear_x, max_shear_x)
                shear_y = random.uniform(-max_shear_y, max_shear_y)
                sheared_image = image.transform(image.size, Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0))

                if str(number) not in ('iv', 'vi'):
                    rotation_angle = random.choice([90, 270])
                    sheared_image = sheared_image.rotate(rotation_angle, expand=True)

                min_scale = 0.5  # Minimum scaling factor
                max_scale = 2.0  # Maximum scaling factor
                scale_factor = random.uniform(min_scale, max_scale)
                scaled_size = (int(sheared_image.width * scale_factor), int(sheared_image.height * scale_factor))
                scaled_image = sheared_image.resize(scaled_size)
                scaled_image.save(os.path.join(number_path, "augmented_" + image_path))


def albumenate_image_val():
    for number in os.listdir(val_dir):
        if number == '.DS_Store':
            continue
        number_path = f"{val_dir}/{number}"
        number_images = [file for file in os.listdir(number_path) if str(file).lower().endswith(".png")]

        for image_path in number_images:

            image = Image.open(number_path + '/' + image_path)

            # Define the cutout augmentation
            cutout = A.Cutout(num_holes=1, max_h_size=20, max_w_size=20, fill_value=0, always_apply=True)

            # Apply the cutout augmentation to the image
            augmented_image = cutout(image=np.array(image))['image']

            # Save the augmented image
            augmented_image = Image.fromarray(augmented_image)
            augmented_image.save(os.path.join(number_path, "augmented_" + image_path))

            # Horizontal Flip
            if str(number) in ("ix", "iiiv", "iiv"):
                modified_img = image.transpose(Image.FLIP_LEFT_RIGHT)
                modified_img.save(os.path.join(number_path, "augmented_" + image_path))

            # Vertical Flip
            if str(number) in ("i", "ii", "iii", "iv", "v", "vi"):
                modified_img = image.transpose(Image.FLIP_TOP_BOTTOM)
                modified_img.save(os.path.join(number_path, "augmented_" + image_path))

            # ROTATE 90 270  +
            # Randomly select between 90 and 270 degrees rotation
            if str(number) not in ('iv', 'vi'):
                rotation_angle = random.choice([90, 270])
                modified_img = image.rotate(rotation_angle, expand=True)
                modified_img.save(os.path.join(number_path, "augmented_" + image_path))


def main():
    num_images()


if __name__ == '__main__':
    main()
