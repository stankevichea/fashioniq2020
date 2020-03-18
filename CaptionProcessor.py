import nltk
import argparse
from nltk.corpus import stopwords 
# You need to run this to get the stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from collections import Counter

CAP_FILE = 'captions/cap.{}.train.json'
OUTPUT_FILE = 'captions/dict.{}.json'

class CaptionsProcessor(object):
    def __init__(self):
        self.text = []
        self.captions = dict()

    def load_captions(self, cap_file):
        import json
        data = json.load(open(cap_file, 'r'))

        text = []
        stop_words = set(stopwords.words('english')) 
        wnl = WordNetLemmatizer()

        for i in range(len(data)):
            captions = data[i]['captions']
            target = data[i]['target']
            target_captions = []
            for caption in captions:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                filtered_tokens = [w for w in tokens if not w in stop_words] 
                lemmatized_tokens = [wnl.lemmatize(w) for w in filtered_tokens]
                text = text + lemmatized_tokens
                target_captions = target_captions + lemmatized_tokens
                
            self.captions[target] = list(set(target_captions))

        self.text = text
        return
    
    def most_common_words(self):
        cnt = Counter()
        cnt.update(self.text)
        return cnt.most_common()

    def save(self, file_name):
        data = {}
        data['counter'] = self.most_common_words()
        data['captions'] = self.captions      
        import json
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
        return


def main(args):
    proc = CaptionsProcessor()
    proc.load_captions(cap_file=CAP_FILE.format(args.data_set))
    proc.save(file_name=OUTPUT_FILE.format(args.data_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='dress')
    args = parser.parse_args()
    main(args)