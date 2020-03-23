import nltk
import argparse
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

        text = []
        stop_words = set(stopwords.words('english'))
        wnl = WordNetLemmatizer()

        def correct_word(word):
            if wnl.check(word):
                return word

            sugg = wnl.suggest(word)
            return sugg[0] if sugg else word

        def lemmatize_word(word):
            for tag in ('n', 'v', 'a', 's', 'r'):
                sugg = wnl.lemmatize(word, tag)
                if sugg != word:
                    return sugg
            return word

        for i in range(len(data)):
            captions = data[i]['captions']
            target = data[i]['target']
            target_captions = []
            for caption in captions:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                corrected_words = [correct_word(w) for w in tokens]
                filtered_tokens = [w for w in corrected_words if not w in stop_words]
                lemmatized_tokens = [lemmatize_word(w) for w in filtered_tokens]
                text = text + lemmatized_tokens
                target_captions = target_captions + lemmatized_tokens

            # apparently there are no repetitions in the training set
            self.captions[target] = target_captions

        self.text = text
        return

    def load_captions_simple(self, cap_file):
        import json
        data = json.load(open(cap_file, 'r'))

        self.text = []

        for i in range(len(data)):
            captions = data[i]['captions']
            target = data[i]['target']
            self.captions[target] += captions
        return

    def process_captions(self, labeller):
        wnl = WordNetLemmatizer()

        for target in self.captions:
            target_labels = []
            for caption in self.captions[target]:
                # TODO: get some approximate comparison
                filtered_caption = [wnl.lemmatize(w) for w in word_tokenize(caption) if (wnl.lemmatize(w) in labeller.labels) or (wnl.lemmatize(w) in labeller.negations)]
                for ngram in ngrams(["<dummy>"] + filtered_caption, 2):
                    if ngram[0] in labeller.negations:
                        label = "NOT_" + ngram[1]
                    elif ngram[1] in labeller.labels:
                        label = ngram[1]
                    target_labels.append(label)
            self.labels[target] += target_labels
        return

    def process_captions_simple(self, labeller):
        for target in self.captions:
            self.captions[target] = [w for w in self.captions[target] if w in labeller.labels]
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

class SimpleLabeller(object):
    def __init__(self, label_file, negation_file):
        self.load_labels(label_file)
        self.load_negations(negation_file)
        return

    def load_labels(self, label_file):
        import json
        data = json.load(open(label_file, 'r'))

        wnl = WordNetLemmatizer()
        colors = [wnl.lemmatize(w) for w in data['color']]
        lengths = [wnl.lemmatize(w) for w in data['length']]
        parts = [wnl.lemmatize(w) for w in data['part']]

        self.labels = set(colors + lengths + parts)
        return

    def load_negations(self, negation_file):
        import json
        data = json.load(open(negation_file, 'r'))

        self.negations = set(data["negations"])
        return


def main(args):
    proc = CaptionsProcessor()
    labeller = SimpleLabeller(label_file=FEATURE_FILE.format(args.data_set), negation_file=NEGATION_FILE.format(args.data_set))
    proc.load_captions_simple(cap_file=CAP_FILE.format(args.data_set))
    proc.process_captions(labeller)
    proc.save(file_name=OUTPUT_FILE.format(args.data_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='dress')
    args = parser.parse_args()
    main(args)
