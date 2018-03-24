import numpy as np

from keras.preprocessing.image import load_img, img_to_array, array_to_img

BOS = np.zeros(shape=(256, 256, 1))
PAD = np.ones(shape=(256, 256, 1))

def load_img_batch(file_pathes, batch_size):
    imgs = np.empty(shape=(batch_size, 256, 256, 1))

    for i, file_path in enumerate(file_pathes):
        imgs[i] = load_scaled_img(file_path)

    eos_start = i + 1
    for i in range(eos_start, batch_size):
        imgs[i] = PAD

    return imgs

def load_scaled_img(filepath):
    img = load_img(filepath, grayscale=True, target_size=(256, 256), interpolation='lanczos')
    img = img_to_array(img)
    img /= 255
    return img

def save_img(img, filepath):
    img = array_to_img(img, scale=True)
    img.save(filepath, 'JPEG', quality=100, optimize=True)