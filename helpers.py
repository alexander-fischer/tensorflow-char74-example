import os

import PIL
from PIL import Image
import numpy as np

alphabet = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
            'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22,
            'x': 23, 'y': 24, 'z': 25}


def char_to_num(char):
    num = alphabet[char]
    return num


def num_to_char(num):
    for key in alphabet:
        if alphabet[key] == num:
            return key


def load_chars74k_data(dir="chars74k-lite"):
    filenames = []
    label_list = []

    for path, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                file = path + '/' + file
                filenames.append(file)

                label = path[-1:]
                label_list.append(label)

    return filenames, label_list


def create_dataset(file_paths, label_set):
    data_x = []
    data_y = []

    for path in file_paths:
        single_x = np.asarray(PIL.Image.open(path)).flatten()
        data_x.append(single_x)

    for l in label_set:
        l_to_num = char_to_num(l)
        data_y.append(l_to_num)

    np_data_x = np.array(data_x)
    np_data_y = np.array(data_y)
    return np_data_x, np_data_y
