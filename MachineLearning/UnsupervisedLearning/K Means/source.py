from k_means_solved import KMeans, KMeansPlusPlus
import numpy as np
import argparse
import matplotlib.pyplot as plt


class AlgorithmSelectionAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, eval(value))


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path of data')
    parser.add_argument('-k', help='number of clusters', type=int)
    parser.add_argument('--algorithm', choices=['KMeans', 'KMeansPlusPlus'],
                        action=AlgorithmSelectionAction)
    parser.add_argument('--name', help='type your name')
    return parser.parse_args()


def main(args):
    data = np.load(args.data)
    kmeans = args.algorithm(args.k)
    kmeans.fit(data)
    labels, means = kmeans.predict(data)
    colors = 'bgrcmyk'

    clusters = {i: [] for i in range(args.k)}

    for i, l in enumerate(labels):
        clusters[l].append(i)

    for i in range(args.k):
        plt.scatter(data[clusters[i]][:, 0], data[clusters[i]][:, 1],
                    c=colors[i], alpha=0.5)
    plt.axis('off')
    plt.savefig('K_Means_results_{}'.format(args.name))
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
