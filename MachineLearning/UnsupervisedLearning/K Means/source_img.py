from k_means import KMeans, KMeansPlusPlus
import numpy as np
import argparse
from matplotlib import image as mpimg
import matplotlib.pyplot as plt


class AlgorithmSelectionAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, eval(value))


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path of img')
    parser.add_argument('-k', help='number of clusters', type=int)
    parser.add_argument('--algorithm', choices=['KMeans', 'KMeansPlusPlus'],
                        action=AlgorithmSelectionAction)
    parser.add_argument('--name', help='type your name')
    return parser.parse_args()


def main(args):
    img = mpimg.imread(args.path)
    shp = img.shape
    tp = img.dtype
    k = args.k
    kmeans = args.algorithm(args.k)
    if len(shp) == 2:
        img = img.reshape(list(img.shape) + [1])
    kmeans.fit(img)
    plt.axis('off')
    new_img = kmeans.predict(img)[1]
    new_img = new_img.reshape(shp).astype(tp)
    plt.imshow(new_img)
    plt.savefig('{}_{}.jpg'.format(args.path.split('.')[0], args.name))
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
