import nltk
import pickle
import argparse
from collections import Counter

CAP_FILE = 'captions/cap.{}.train.json'
DICT_OUTPUT_FILE = 'captions/dict.{}.json'

class CaptionsProcessor(object):
    def __init__(self):
        self.text = []

    def load_captions(self, cap_file):
        """Join all the captions into a single text"""
        import json
        data = json.load(open(cap_file, 'r'))

        text = []

        for i in range(len(data)):
            captions = data[i]['captions']
            for caption in captions:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                text = text + tokens + '.'

        self.text = text
        return
    
    def most_common_words(self):
        cnt = Counter()
        cnt.update(self.text)
        return cnt.most_common()


def main(args):
    proc = CaptionsProcessor()
    proc.load_captions(cap_file=CAP_FILE.format(args.data_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='dress')
    args = parser.parse_args()
    main(args)