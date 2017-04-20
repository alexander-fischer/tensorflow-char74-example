import numpy as np
import PIL
from PIL import Image


def sliding_window(path):
    from itertools import islice

    img_arr = np.asarray(PIL.Image.open(path)).flatten()

    def window(seq, n):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    slides = []

    for w in window(img_arr, 400):
        count_white = w.count(255)
        if count_white < 400:
            slides.append(np.array(w))

    slide_array = np.array(slides)

    return slide_array
    #images = slide_array.reshape(slide_array.shape[0], 20, 20, 1)
    #images = images.astype('float32')

    #return images

