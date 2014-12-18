from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from nltk import word_tokenize
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class StemTokenizer(object):
    def __init__(self, stemmer_type='Porter'):
        self.stemmer_type = stemmer_type
        if self.stemmer_type == 'Porter':
            self.stemmer = PorterStemmer()
        elif self.stemmer_type == 'Lancaster':
            self.stemmer = LancasterStemmer()
        else:
            raise Exception('Invalid stemmer_type = {0}'.format(stemmer_type))

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]

class SimilarityFinder:
    def __init__(self, corpus, duplicates):
        self.corpus = corpus
        self.duplicates = duplicates
        self.vectorizer = self.get_vectorizer()
        self.models = {}
        self.X_train = {}
        self.sim = {}

    def get_vectorizer(self):
        return TfidfVectorizer(min_df=1,
                               ngram_range=(1, 2),
                               stop_words='english',
                               tokenizer=StemTokenizer(),
                               # tokenizer=LemmaTokenizer(),
                               strip_accents='unicode',
                               norm='l2',
                               decode_error='ignore')

    def compute_similarity_scores(self):
        X = np.array([''.join(text) for text in self.corpus])
        X = self.vectorizer.fit_transform(X)
        self.similarity_scores = cosine_similarity(X, X)


    def get_similarity_score(self, docid_1, docid_2):
        return self.similarity_scores[docid_1][docid_2]

if __name__ == '__main__':
    N = int(raw_input())

    corpus = [raw_input() for _ in xrange(N)]

    sf = SimilarityFinder(corpus)
    sf.compute_similarity_scores()

    T = int(raw_input())
    for t in xrange(T):
        docid_1 = int(raw_input())
        docid_2 = int(raw_input())

        print docid_1, docid_2, sf.get_similarity_score(docid_1, docid_2)
