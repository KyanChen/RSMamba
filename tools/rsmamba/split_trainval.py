import argparse
import glob
import os
import random

import mmengine


def main():
    folder = '/path_to_data/NWPU-RESISC45'
    save_folder = 'datainfo/NWPU'
    mmengine.mkdir_or_exist(save_folder)
    split_trainval_ratio = 0.7
    # split_trainval_ratio = 0.5
    select_n_classes = 1000
    # get the subfolder list
    subfolders = glob.glob(folder + '/*')
    subfolders = [f for f in subfolders if os.path.isdir(f)]
    # select n classes
    random.shuffle(subfolders)
    subfolders = subfolders[:select_n_classes]
    cls_names = [os.path.basename(f) for f in subfolders]

    train_split = []
    val_split = []
    for i, f in enumerate(subfolders):
        img_files = glob.glob(f + '/*.jpg')
        img_files = [os.path.basename(os.path.dirname(file_img)) + '/' + os.path.basename(file_img) for file_img in img_files]
        # shuffle the image file list
        random.shuffle(img_files)
        # split the train and val
        split = int(len(img_files) * split_trainval_ratio)
        train_files = img_files[:split]
        val_files = img_files[split:]
        train_split.extend(train_files)
        val_split.extend(val_files)
    # save the train and val list to file
    with open(save_folder + '/train.txt', 'w') as f:
        f.write('\n'.join(train_split))
    with open(save_folder + '/val.txt', 'w') as f:
        f.write('\n'.join(val_split))
    with open(save_folder + '/cls_names.txt', 'w') as f:
        f.write('\n'.join(cls_names))
    print(f'train: {len(train_split)}, val: {len(val_split)}')
    # print the class names with \" and \"
    print(', '.join([f'\"{cls_name}\"' for cls_name in cls_names]))


if __name__ == '__main__':
    main()
