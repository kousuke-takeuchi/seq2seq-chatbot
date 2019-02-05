import nltk
import numpy as np
from tqdm import tqdm


class PreProcessing:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def tokenize(self, sentense):
        text = nltk.word_tokenize(sentense)
        return text
        # return nltk.pos_tag(text)

    # 単語辞書、インデックス化された入力データを作成する
    def create_word_vectors(self, sentenses, max_len=50, blank='', dictionary=None, filename=None):
        results = []
        if dictionary is None:
            dictionary = {
                'words': set([blank]),
                'word2idx': {0: blank},
                'idx2word': {blank: 0},
            }
        for sentense in sentenses:
            tokenized = []
            tokens = self.tokenize(sentense)
            for token in tokens[:max_len]:
                if not token in dictionary['words']:
                    if token == '':
                        continue
                    wordslen = len(dictionary['words'])
                    dictionary['words'].add(token)
                    dictionary['word2idx'][token] = wordslen
                    dictionary['idx2word'][wordslen] = token
                tokenized.append(dictionary['word2idx'][token])
            blanks = [0 for _ in range(max(0, max_len-len(tokenized)))]
            tokenized += blanks
            results.append(tokenized)
        return np.array(results), dictionary

    def generate_token_vectors(self, dataset, max_len=50, iteration=False):
        class Iterator:
            def __init__(this, l):
                this.d = None
                this.i = 0
                this.l = l
            def __iter__(this):
                return this
            def __next__(this):
                if this.i == len(this.l):
                    raise StopIteration()
                vectors, this.d = self.create_word_vectors(this.l[this.i], max_len=max_len, dictionary=this.d)
                this.i += 1
                return indexes, this.d
            def __len__(this):
                return len(this.l)

        iter = Iterator(dataset)
        if iteration:
            return iter

        data = []
        dictionary = None
        for indexes, d in tqdm(iter):
            data.append(indexes)
            dictionary = d
        return data, dictionary
