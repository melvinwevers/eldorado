import os
from util import Corpus
import logging
import argparse


def main(args):
    data_path = os.path.join(args.path)
    newspaper = Corpus(data_path, args.title, args.type)  
    newspaper.process()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='../../../datasets/newspapers_clean/',
                        help='Path to corpus data.')
    parser.add_argument('--title', type=str, default=None,
                       help='newspaper title.') # TODO: give list of options
    parser.add_argument('--type', type=str, default=None,
                       help='articles or ads.')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    main(args)
