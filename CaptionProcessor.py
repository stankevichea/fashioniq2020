import nltk
import argparse
import enchant
from nltk.corpus import stopwords
# You need to run this to get the stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from collections import defaultdict

CAP_FILE = 'captions/cap.{}.train.json'
OUTPUT_FILE = 'captions/dict.{}.json'
FEATURE_FILE = 'recognized_features.{}.json'
NEGATION_FILE = 'negations.json'

class CaptionsProcessor(object):
    def __init__(self):
        self.text = []
        self.labels = defaultdict(list)
        self.captions = defaultdict(list)
        return

    def load_captions(self, cap_file):
        import json
        data = json.load(open(cap_file, 'r'))

        self.text = []

        for i in range(len(data)):
            captions = data[i]['captions']
            target = data[i]['target']
            self.captions[target] += captions
            for c in captions:
                self.text += c
        return

    def split_captions(self, captions):
        res_captions = defaultdict(list)
        for target in captions:
            for caption in captions[target]:
                res_captions[target] += word_tokenize(caption.lower())
        return res_captions

    def spellcheck_captions(self, captions):
        d = enchant.Dict("en_US")
        res_captions = defaultdict(list)

        def correct_word(word):
            if d.check(word):
                return word
            sugg = d.suggest(word)
            return sugg[0] if sugg else word

        for target in captions:
            res_captions[target] += [correct_word(w) for w in captions[target]]

        return res_captions


    def lemmatize_captions(self, captions):
        wnl = WordNetLemmatizer()
        res_captions = defaultdict(list)

        def lemmatize_word(word):
            for tag in ('n', 'v', 'a', 's', 'r'):
                sugg = wnl.lemmatize(word, tag)
                if sugg != word:
                    return sugg
            return word

        for target in captions:
            res_captions[target] += [lemmatize_word(w) for w in captions[target]]
 
        return res_captions

    def map_unkown_words(self, dictionary, captions):
        res_captions = defaultdict(list)

        def map_word(word):
            if(word in dictionary.all):
                return word
            else:
                return "<dummy>"

        for target in captions:
            res_captions[target] += [map_word(w) for w in captions[target]]
        return res_captions

    def create_labels(self, dictionary, captions):
        for target in captions:
            target_labels = []
            for ngram in ngrams(["<dummy>"] + captions[target], 2):
                if ngram[0] in dictionary.negations and ngram[1] in dictionary.labels:
                    label = "NOT_" + ngram[1]
                    target_labels.append(label)
                elif ngram[1] in dictionary.labels:
                    label = ngram[1]
                    target_labels.append(label)
            self.labels[target] += target_labels
        return

    def process_captions(self, dictionary):
        splitted = self.split_captions(self.captions)
        spellchecked = self.spellcheck_captions(splitted)
        lemmatized = self.lemmatize_captions(spellchecked)
        filtered = self.map_unkown_words(dictionary, lemmatized)
        self.create_labels(dictionary, filtered)
        return

    def most_common_words(self):
        cnt = Counter()
        cnt.update(self.text)
        return cnt.most_common()

    def save(self, file_name):
        data = {}

        for key in self.labels:
            self.labels[key] = list(set(self.labels[key]))

        data['labels'] = self.labels
        import json
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
        return

class SimpleDictionary(object):
    def __init__(self, label_file, negation_file):
        self.load_labels(label_file)
        self.load_negations(negation_file)
        self.all = set()
        self.all.update(self.labels)
        self.all.update(self.negations)
        return

    def load_labels(self, label_file):
        import json
        data = json.load(open(label_file, 'r'))

        colors = data['color']
        lengths = data['length']
        parts = data['part']

        self.labels = set(colors + lengths + parts)
        return

    def load_negations(self, negation_file):
        import json
        data = json.load(open(negation_file, 'r'))

        self.negations = set(data["negations"])
        return


def main(args):
    proc = CaptionsProcessor()
    dictionary = SimpleDictionary(label_file=FEATURE_FILE.format(args.data_set), negation_file=NEGATION_FILE.format(args.data_set))
    proc.load_captions(cap_file=CAP_FILE.format(args.data_set))
    proc.process_captions(dictionary)
    proc.save(file_name=OUTPUT_FILE.format(args.data_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='dress')
    args = parser.parse_args()
    main(args)
