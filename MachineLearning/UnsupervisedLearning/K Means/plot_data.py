import numpy as np
import argparse
import matplotlib.pyplot as plt


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path of data')
    return parser.parse_args()


def main(args):
    data = np.load(args.data)
    plt.scatter(data[:,0], data[:, 1], alpha=0.5)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
