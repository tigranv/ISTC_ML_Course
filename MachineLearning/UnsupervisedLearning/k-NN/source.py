import argparse
import numpy as np
import json
from k_nn import K_NN


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', help='path of file to train on')
    parser.add_argument('--test-data', help='path of file to test on')
    parser.add_argument('-k', help='number of nearest neigbours to use')
    return parser.parse_args()


def main(args):
    with open(args.train_data, 'r') as rfile:
        train_data = np.array(json.load(rfile))

    with open(args.test_data, 'r') as rfile:
        test_data = np.array(json.load(rfile))

    k_nn_classifier = K_NN(args.k)
    k_nn_classifier.fit( )

    answers_train = []
    correct_answers_train = []
    for klass, data in enumerate(train_data):
        for point in data:
            answers_train.append(k_nn_classifier.predict(*point))
            correct_answers_train.append(klass)

    answers = np.array(answers_train)
    correct_answers = np.array(correct_answers_train)
    num_correct_answers_train = np.sum(correct_answers == answers)

    print('Accuracy on train data: {}%'.format(
        num_correct_answers_train * 100 / len(answers)))

    answers_test = []
    correct_answers_test = []
    for klass, data in enumerate(test_data):
        for point in data:
            answers_test.append(k_nn_classifier.predict(*point))
            correct_answers_test.append(klass)

    answers = np.array(answers_test)
    correct_answers = np.array(correct_answers_test)
    num_correct_answers_test = np.sum(correct_answers == answers)

    print('Accuracy on test data: {}%'.format(
        num_correct_answers_test * 100 / len(answers)))

    all_answers = answers_train + answers_test
    all_correct_answers = correct_answers_train + correct_answers_test

    answers = np.array(all_answers)
    correct_answers = np.array(all_correct_answers)
    num_correct_answers_all = np.sum(correct_answers == answers)
    print('Overall accuracy: {}%'.format(
        num_correct_answers_all * 100 / len(answers)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
