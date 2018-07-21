from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def process(images):
    # TODO: fix means and variances of images
    pass


if __name__ == '__main__':
    images = np.array([mpimg.imread(f'gen_img{i}.png').flatten() for i in range(5)])
    # TODO: Fit Fast ICA and get new images

    new_images = None
    new_images = new_images.reshape((3, 800, 600, 4))
    new_images = process(new_images)

    for i in new_images:
        plt.imshow(i.clip(min=0, max=1))
        plt.show()
