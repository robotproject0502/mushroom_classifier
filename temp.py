import glob
import numpy as np
import shutil

train_nor_image_list = glob.glob(r'G:\기계학습프로젝트데이터\기계학습프로젝트데이터_train\식용버섯\*\*\*')
for i in range(0, len(train_nor_image_list), 10):
    train_nor_image = train_nor_image_list[i]
    new_image_name = '_'.join(train_nor_image.split('\\')[-2:])
    # print(new_image_name)
    shutil.move(train_nor_image, r'G:\기계학습프로젝트데이터\기계학습프로젝트데이터_vad\식용버섯\{}.jpg'.format(new_image_name))

train_posion_image_list = glob.glob(r'G:\기계학습프로젝트데이터\기계학습프로젝트데이터_train\독버섯\*\*\*')
for i in range(0, len(train_posion_image_list), 10):
    train_posion_image = train_posion_image_list[i]
    new_image_name = '_'.join(train_posion_image.split('\\')[-2:])
    # print(new_image_name)
    shutil.move(train_posion_image, r'G:\기계학습프로젝트데이터\기계학습프로젝트데이터_vad\독버섯\{}.jpg'.format(new_image_name))