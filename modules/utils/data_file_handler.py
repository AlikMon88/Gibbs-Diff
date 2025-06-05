import numpy as np
import os

def tiny_imagenet_file_handler(source_path, return_path = True):
    train_path = os.path.join(source_path, 'tiny-imagenet-200/train')
    test_path = os.path.join(source_path, 'tiny-imagenet-200/test/images')

    train_class_path = [os.path.join(train_path, class_id + '/images') for class_id in os.listdir(train_path)]
    
    train_image_path = []
    for tp in train_class_path:
        for image_id in os.listdir(tp):
            train_image_path.append(os.path.join(tp, image_id))

    test_image_path = []
    for image_id in os.listdir(test_path):
        test_image_path.append(os.path.join(test_path, image_id))

    print('#train_images: ', len(train_image_path))
    print('#test_images: ', len(test_image_path))

    if return_path:
        return train_image_path, test_image_path

def cosmo_data_file_handler():
    pass

if __name__ == '__main__':
    print('running __data_file_handler.py__')